"""Pre-RoPE K + V compression sweep.

V-only optimization (results_cross_validation.json) is lossless to ~3.2x
because V is position-independent and can be pooled. K failed previously
because RoPE bakes position into K, making pooled post-RoPE K targets
unmatchable across positions.

This run adds K back, but in **pre-RoPE space**:

  1. Extract post-RoPE K from the target forward pass.
  2. Apply inverse RoPE at the target positions -> pre-RoPE K, which is
     position-independent and therefore safe to pool (just like V).
  3. Pool the pre-RoPE K targets (perplexity-weighted, same as V).
  4. During optimization, extract K from the virtual-token forward,
     apply inverse RoPE at the virtual tokens' interpolated positions,
     and match against the pooled pre-RoPE K targets.
  5. During inference, change nothing: the model applies RoPE to the
     virtual tokens' K at their interpolated positions automatically.

Loss = V loss (unchanged) + K loss (pre-RoPE space). Both terms are
position-independent and per-layer mag^2-normalized.

Verified RoPE interface (transformers 5.8.0, Qwen2):
  apply_rotary_pos_emb: x_embed = x*cos + rotate_half(x)*sin
  inverse (negated angle): x = x_embed*cos - rotate_half(x_embed)*sin
  model.model.rotary_emb(x, position_ids) -> (cos, sin) of shape
  (B, seq, head_dim), dtype following x.

Conditions per turn: full_turn, raw_embeddings, kv_optimized at
{1.6, 2.4, 3.2, 4.8, 6.4}x, v_only_4.8x (direct baseline), no_turn.

Turns 8 / 18 / 24, same questions and scorers as the prior runs.

Standalone -- does not import from any other project file.
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------- constants -----------------------------

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
TRANSCRIPT_PATH = Path(__file__).parent / "transcript.jsonl"
RESULTS_PATH = Path(__file__).parent / "results_kv_optimization.json"

PREFIX_TURN_INDEX = 0  # 0-indexed; turn 1 (system prompt)
COMPRESSION_RATIOS = [1.6, 2.4, 3.2, 4.8, 6.4]
V_ONLY_RATIO = 4.8  # the ratio at which we also run a V-only baseline

LR = 0.003
BASE_STEPS = 300
BASE_TOKENS = 250
MIN_STEPS = 100
LOG_EVERY = 50
MAX_NEW_TOKENS = 256

EXTRA_STOP_TOKEN_STRINGS = ["<|im_end|>", "<|endoftext|>"]

# Hardcoded V-only prior scores (correct facts out of 3) from
# results_feasibility_pplpool.json (turn 8) and results_cross_validation
# .json (turns 18, 24). Turn 8's 2.4x was 3/3 under perplexity-weighted
# pooling (the value baked into cross_validation_test.py was a stale 2/3).
# 6.4x had no V-only prior run.
V_ONLY_PRIOR_SCORES: dict[int, dict[float, int | None]] = {
    8: {1.6: 3, 2.4: 3, 3.2: 3, 4.8: 2, 6.4: None},
    18: {1.6: 3, 2.4: 3, 3.2: 3, 4.8: 2, 6.4: None},
    24: {1.6: 3, 2.4: 3, 3.2: 3, 4.8: 2, 6.4: None},
}


# ------------------------ turn definitions ---------------------------

TURNS_TO_TEST = [
    {
        "turn_index_1based": 8,
        "content_type": "project conventions doc (CLAUDE.md)",
        "question": (
            "What library does the project use for password hashing? "
            "What is its Node.js test runner? "
            "Does the markdown converter use a library or regex?"
        ),
        "expected": (
            "bcryptjs for password hashing; Node's built-in test runner "
            "(node:test); regex-based markdown converter (no external "
            "libraries)."
        ),
        "score_fn_name": "score_turn8",
    },
    {
        "turn_index_1based": 18,
        "content_type": "JWT auth middleware (JS code)",
        "question": (
            "Which npm library does this auth middleware use? "
            "What HTTP status code does it return for missing or invalid tokens? "
            "What is the default JWT secret string used when JWT_SECRET is not set?"
        ),
        "expected": "jsonwebtoken; HTTP 401; default secret is 'dev-secret'.",
        "score_fn_name": "score_turn18",
    },
    {
        "turn_index_1based": 24,
        "content_type": "reports.js (weekly-report generator, JS code)",
        "question": (
            "How many days back does the weekly report cover? "
            "What are the three task status categories the report groups tasks into? "
            "What is the title format used for each generated report?"
        ),
        "expected": (
            "Last 7 days; pending / in_progress / done; "
            "title is \"Weekly Report - {date}\" (YYYY-MM-DD)."
        ),
        "score_fn_name": "score_turn24",
    },
]


# ----------------------------- helpers -------------------------------

def vram_str() -> str:
    if not torch.cuda.is_available():
        return "VRAM n/a"
    a = torch.cuda.memory_allocated() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    return f"VRAM allocated={a:.2f} GB, peak={p:.2f} GB"


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(
                    "[tool_use {name}] {input}".format(
                        name=block.get("name", ""),
                        input=json.dumps(block.get("input", {}), ensure_ascii=False),
                    )
                )
            elif btype == "tool_result":
                parts.append(
                    "[tool_result {tid}] {content}".format(
                        tid=block.get("tool_use_id", ""),
                        content=json.dumps(block.get("content", ""), ensure_ascii=False),
                    )
                )
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def turn_marker(turn_idx_1based: int, role: str, content_str: str) -> str:
    return f"=== TURN {turn_idx_1based} ({role}) ===\n{content_str}\n"


def to_legacy_kv(pkv) -> tuple:
    if pkv is None:
        return ()
    if isinstance(pkv, tuple):
        return pkv
    if hasattr(pkv, "to_legacy_cache"):
        try:
            return tuple(pkv.to_legacy_cache())
        except Exception:
            pass
    layers = getattr(pkv, "layers", None)
    if isinstance(layers, (list, tuple)) and len(layers) > 0:
        out = []
        for layer in layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                out.append((layer.keys, layer.values))
            elif hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
                out.append((layer.key_cache, layer.value_cache))
            else:
                raise RuntimeError(
                    f"Unknown layer type: {type(layer).__name__}, "
                    f"attrs={[a for a in dir(layer) if not a.startswith('_')]}"
                )
        return tuple(out)
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return tuple(zip(pkv.key_cache, pkv.value_cache))
    raise RuntimeError(
        f"Unknown past_key_values type: {type(pkv).__name__}, "
        f"attrs={[a for a in dir(pkv) if not a.startswith('_')]}"
    )


def wrap_legacy_kv(legacy: tuple, config) -> object:
    if not legacy:
        return None
    from transformers.cache_utils import DynamicCache
    return DynamicCache(legacy, config=config)


def detach_kv(legacy: tuple) -> tuple:
    return tuple((k.detach(), v.detach()) for k, v in legacy)


def kv_seq_len(legacy: tuple) -> int:
    if not legacy:
        return 0
    return int(legacy[0][0].shape[-2])


def mean_pool_chunks(x: torch.Tensor, n_chunks: int) -> torch.Tensor:
    if x.dim() != 2:
        raise ValueError(f"expected (T, D), got {tuple(x.shape)}")
    t = x.shape[0]
    if n_chunks > t:
        raise ValueError(f"n_chunks={n_chunks} > T={t}")
    boundaries = torch.linspace(0, t, n_chunks + 1).long().tolist()
    rows = []
    for i in range(n_chunks):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            e = s + 1
        rows.append(x[s:e, :].mean(dim=0, keepdim=True))
    return torch.cat(rows, dim=0)


def attention_weighted_pool(
    v: torch.Tensor, importance: torch.Tensor, n_chunks: int
) -> torch.Tensor:
    """Pool (B, H, S, Dh) along S into n_chunks groups, softmaxing
    importance scores within each chunk.
    """
    if v.shape[2] != importance.shape[0]:
        raise ValueError(
            f"v.shape[2]={v.shape[2]} but importance.shape[0]={importance.shape[0]}"
        )
    s = v.shape[2]
    if n_chunks > s:
        raise ValueError(f"n_chunks={n_chunks} > S={s}")
    boundaries = torch.linspace(0, s, n_chunks + 1).long().tolist()
    chunks = []
    for i in range(n_chunks):
        a, b = boundaries[i], boundaries[i + 1]
        if b <= a:
            b = a + 1
        chunk_v = v[:, :, a:b, :]
        chunk_w = importance[a:b].to(chunk_v.device)
        weights = F.softmax(chunk_w.float(), dim=0).to(chunk_v.dtype)
        pooled = (chunk_v * weights.view(1, 1, -1, 1)).sum(dim=2, keepdim=True)
        chunks.append(pooled)
    return torch.cat(chunks, dim=2)


def extract_new_kv_grad_safe(cache, slice_offset: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pull (K, V) per layer from a model-returned cache, sliced to
    positions starting at slice_offset. Does NOT detach/clone/contiguous
    so the autograd graph through the K/V projections is preserved.
    """
    if isinstance(cache, tuple):
        return [(k[:, :, slice_offset:, :], v[:, :, slice_offset:, :]) for (k, v) in cache]
    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple)) and len(layers) > 0:
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers:
            k = getattr(layer, "keys", None)
            if k is None:
                k = getattr(layer, "key_cache", None)
            v = getattr(layer, "values", None)
            if v is None:
                v = getattr(layer, "value_cache", None)
            if k is None or v is None:
                raise RuntimeError(
                    f"layer missing K/V tensors: {type(layer).__name__}, "
                    f"attrs={[a for a in dir(layer) if not a.startswith('_')]}"
                )
            out.append((k[:, :, slice_offset:, :], v[:, :, slice_offset:, :]))
        return out
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if kc is not None and vc is not None:
        return [
            (kc[i][:, :, slice_offset:, :], vc[i][:, :, slice_offset:, :])
            for i in range(len(kc))
        ]
    raise RuntimeError(
        f"Unknown cache type for grad-safe extract: {type(cache).__name__}, "
        f"attrs={[a for a in dir(cache) if not a.startswith('_')]}"
    )


# --------------------------- RoPE machinery --------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Local copy of Qwen2's rotate_half: split the last dim in half and
    rotate. rotate_half(x) = cat(-x2, x1) where x1, x2 are the halves.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def find_rotary_emb(model):
    """Locate the rotary embedding module. Standard path is
    model.model.rotary_emb; fall back to a module scan.
    """
    rot = getattr(getattr(model, "model", None), "rotary_emb", None)
    if rot is not None:
        return rot
    for name, mod in model.named_modules():
        if name.endswith("rotary_emb") or "RotaryEmbedding" in type(mod).__name__:
            return mod
    raise RuntimeError("Could not locate rotary embedding module on the model.")


def get_rotary_cos_sin(model, positions: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos, sin) for the given positions using the model's own
    rotary embedding module, so the frequencies exactly match the
    model's forward pass. positions: 1-D LongTensor or (1, n).

    Returns cos, sin each of shape (1, n, head_dim) in fp32. A fp32 dummy
    x is passed so the module returns fp32 cos/sin (its forward casts to
    x.dtype). The module's forward is @torch.no_grad, but cos/sin are
    pure functions of position -- no gradient is needed through them;
    gradient flows through the K tensor they multiply.
    """
    rot = find_rotary_emb(model)
    pos = positions.to(device=device, dtype=torch.long)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    dummy = torch.zeros(1, pos.shape[1], 1, dtype=torch.float32, device=device)
    cos, sin = rot(dummy, pos)
    return cos.float(), sin.float()


def inverse_rotary_pos_emb(
    k_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Apply the inverse of RoPE to k_states, recovering pre-RoPE K.

    Forward RoPE: x_embed = x*cos + rotate_half(x)*sin.
    Inverse (rotation by the negated angle, cos even / sin odd):
        x = x_embed*cos - rotate_half(x_embed)*sin.

    k_states: (B, H, n, head_dim). cos/sin: (B, n, head_dim) -> unsqueezed
    on `unsqueeze_dim` (=1) to broadcast over heads. Fully differentiable
    (multiply + add only); do NOT detach inside.
    """
    cos = cos.unsqueeze(unsqueeze_dim).to(k_states.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(k_states.dtype)
    return (k_states * cos) - (rotate_half(k_states) * sin)


def forward_rotary_pos_emb(
    k_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> torch.Tensor:
    """Forward RoPE (for round-trip verification only)."""
    cos = cos.unsqueeze(unsqueeze_dim).to(k_states.dtype)
    sin = sin.unsqueeze(unsqueeze_dim).to(k_states.dtype)
    return (k_states * cos) + (rotate_half(k_states) * sin)


def verify_inverse_rope(model, prefix_kv: tuple, test_ids: torch.Tensor, prefix_len: int) -> dict:
    """Round-trip check: K_post -> inverse -> forward-RoPE should recover
    K_post. The 8-bit model's cache is fp16 and trig is computed in fp32,
    so we assert closeness within tolerance rather than bitwise equality.
    """
    device = model.device
    n = int(test_ids.shape[1])
    positions = torch.arange(prefix_len, prefix_len + n, device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(
            input_ids=test_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=positions.unsqueeze(0),
            use_cache=True,
        )
        live = extract_new_kv_grad_safe(out.past_key_values, prefix_len)
        cos, sin = get_rotary_cos_sin(model, positions, device)
        # Use layer 0's K as the probe.
        k_post = live[0][0].float()
        k_pre = inverse_rotary_pos_emb(k_post, cos, sin)
        k_roundtrip = forward_rotary_pos_emb(k_pre, cos, sin)
        abs_diff = (k_roundtrip - k_post).abs()
        k_scale = k_post.abs().mean().clamp(min=1e-8)
        max_abs_diff = float(abs_diff.max().item())
        rel_diff = float((abs_diff.max() / k_scale).item())
    del out
    torch.cuda.empty_cache()
    # fp16 cache + fp32 trig: round-trip should be tight. Tolerate small
    # absolute slack relative to K's typical magnitude.
    verified = max_abs_diff < 1e-2 and rel_diff < 1e-2
    return {
        "max_abs_diff": max_abs_diff,
        "k_abs_mean": float(k_scale.item()),
        "rel_diff": rel_diff,
        "verified": verified,
    }


def verify_rope_for_qwen() -> tuple[bool, str]:
    try:
        from transformers.models.qwen2 import modeling_qwen2
        src = inspect.getsource(modeling_qwen2.Qwen2Attention.forward)
    except Exception as e:
        return False, f"could not inspect Qwen2Attention.forward: {e}"
    matched = [ln.strip() for ln in src.splitlines() if "apply_rotary_pos_emb" in ln]
    if not matched:
        return False, "apply_rotary_pos_emb not found in Qwen2Attention.forward"
    assign = next((ln for ln in matched if "= apply_rotary_pos_emb" in ln), None)
    if assign is None:
        return False, "apply_rotary_pos_emb mentioned but not assigned to: " + " | ".join(matched)
    lhs = assign.split("=", 1)[0].lower()
    has_q = "query" in lhs
    has_k = "key" in lhs
    has_v = "value" in lhs
    if has_q and has_k and not has_v:
        return True, assign
    return False, f"unexpected RoPE assignment shape (q={has_q} k={has_k} v={has_v}): {assign}"


def interpolate_positions(start: int, end: int, n: int, device=None) -> torch.Tensor:
    floats = torch.linspace(float(start), float(end), n)
    positions = floats.round().long()
    unique_count = int(torch.unique(positions).numel())
    if unique_count != n:
        print(
            f"  [interpolate_positions] WARNING: only {unique_count}/{n} "
            f"unique positions for [{start}, {end}], n={n}."
        )
    if device is not None:
        positions = positions.to(device)
    return positions.unsqueeze(0)


def verify_position_ids_take_effect(
    model, prefix_kv: tuple, prefix_len: int, model_dtype
) -> dict:
    """Confirm explicit position_ids take effect when past_key_values is
    present (rather than the model auto-deriving from cache length).
    """
    device = model.device
    n = 5
    hidden = model.config.hidden_size
    rand_emb = torch.randn(1, n, hidden, dtype=model_dtype, device=device)
    pos_a = torch.arange(100, 100 + n, device=device, dtype=torch.long).unsqueeze(0)
    pos_b = torch.arange(300, 300 + n, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out_a = model(
            inputs_embeds=rand_emb,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=pos_a,
            use_cache=True,
        )
        out_b = model(
            inputs_embeds=rand_emb,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=pos_b,
            use_cache=True,
        )
    live_a = extract_new_kv_grad_safe(out_a.past_key_values, prefix_len)
    live_b = extract_new_kv_grad_safe(out_b.past_key_values, prefix_len)
    k_diff_max = float((live_a[0][0].float() - live_b[0][0].float()).abs().max().item())
    v_diff_max = float((live_a[0][1].float() - live_b[0][1].float()).abs().max().item())
    took_effect = (k_diff_max > 1e-3) and (v_diff_max < 1e-1)
    return {"k_diff_max": k_diff_max, "v_diff_max": v_diff_max, "took_effect": took_effect}


def compute_perplexity_weights(
    logits: torch.Tensor, turn_ids: torch.Tensor
) -> torch.Tensor:
    if logits.shape[1] != turn_ids.shape[1]:
        raise ValueError(
            f"logits seq {logits.shape[1]} != turn_ids seq {turn_ids.shape[1]}"
        )
    device = logits.device
    target_ids = turn_ids[0, 1:].to(device).long()
    pred_logits = logits[0, :-1, :].float()
    nll = F.cross_entropy(pred_logits, target_ids, reduction="none")
    zero = torch.zeros(1, device=device, dtype=nll.dtype)
    return torch.cat([zero, nll], dim=0).detach().clone()


def get_stop_ids(tokenizer) -> set[int]:
    stops: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stops.add(int(tokenizer.eos_token_id))
    for tok in EXTRA_STOP_TOKEN_STRINGS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        for i in ids:
            stops.add(int(i))
    return stops


@torch.no_grad()
def manual_greedy_with_positions(
    model, tokenizer, prompt_text: str, past_kv_legacy: tuple,
    start_pos: int, max_new_tokens: int, stop_ids: set[int],
) -> str:
    device = model.device
    q_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_q = int(q_ids.shape[1])
    cache = wrap_legacy_kv(past_kv_legacy, model.config)
    pos_ids = torch.arange(
        start_pos, start_pos + n_q, device=device, dtype=torch.long,
    ).unsqueeze(0)
    out = model(
        input_ids=q_ids, past_key_values=cache, position_ids=pos_ids, use_cache=True,
    )
    cache = out.past_key_values
    next_pos = start_pos + n_q
    next_id = int(out.logits[0, -1, :].argmax().item())
    generated = [next_id]
    if next_id in stop_ids:
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    for _ in range(max_new_tokens - 1):
        next_inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        pos_ids = torch.tensor([[next_pos]], dtype=torch.long, device=device)
        out = model(
            input_ids=next_inp, past_key_values=cache,
            position_ids=pos_ids, use_cache=True,
        )
        cache = out.past_key_values
        next_pos += 1
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated.append(next_id)
        if next_id in stop_ids:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def manual_greedy_auto(
    model, tokenizer, prompt_text: str, past_kv_legacy: tuple,
    max_new_tokens: int, stop_ids: set[int],
) -> str:
    device = model.device
    q_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    cache = wrap_legacy_kv(past_kv_legacy, model.config)
    out = model(input_ids=q_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    next_id = int(out.logits[0, -1, :].argmax().item())
    generated = [next_id]
    if next_id in stop_ids:
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    for _ in range(max_new_tokens - 1):
        next_inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(input_ids=next_inp, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated.append(next_id)
        if next_id in stop_ids:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_kv(
    model, tokenizer, prompt_text: str, past_kv_legacy: tuple,
    max_new_tokens: int, stop_ids: set[int],
) -> str:
    device = model.device
    q_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    try:
        out = model.generate(
            input_ids=q_ids,
            past_key_values=wrap_legacy_kv(past_kv_legacy, model.config),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=list(stop_ids),
        )
        n_input = int(q_ids.shape[1])
        new_ids = out[0, n_input:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    except Exception:
        return manual_greedy_auto(
            model, tokenizer, prompt_text, past_kv_legacy, max_new_tokens, stop_ids,
        )


# --------------------------- score functions -------------------------

def score_turn8(ans: str) -> tuple[int, dict[str, bool]]:
    """Turn 8: project conventions doc (CLAUDE.md).

    Correct facts:
      - Password hashing: 'bcryptjs' (not bare 'bcrypt')
      - Test runner: Node's built-in (node:test)
      - Markdown converter: regex-based, no external library
    """
    a = ans.lower()
    fact_bcryptjs = "bcryptjs" in a
    fact_node_test = (
        "node:test" in a
        or "node's built-in test" in a
        or "built-in test runner" in a
    )
    has_regex = "regex" in a or "regexp" in a or "regular expression" in a
    has_md_lib = "marked" in a or "showdown" in a or "markdown-it" in a
    fact_regex = has_regex and not has_md_lib
    details = {
        "bcryptjs": fact_bcryptjs,
        "node_test": fact_node_test,
        "regex_only": fact_regex,
    }
    return int(fact_bcryptjs) + int(fact_node_test) + int(fact_regex), details


def score_turn18(ans: str) -> tuple[int, dict[str, bool]]:
    """Turn 18: JWT auth middleware (auth.js)."""
    a = ans.lower()
    fact_lib = "jsonwebtoken" in a  # require the package name, not bare 'jwt'
    fact_401 = "401" in a
    fact_secret = "dev-secret" in a or "dev secret" in a
    details = {
        "jsonwebtoken": fact_lib,
        "status_401": fact_401,
        "dev_secret": fact_secret,
    }
    return int(fact_lib) + int(fact_401) + int(fact_secret), details


def score_turn24(ans: str) -> tuple[int, dict[str, bool]]:
    """Turn 24: weekly report generator (reports.js)."""
    a = ans.lower()
    fact_days = (
        "7 days" in a
        or "seven days" in a
        or "last 7" in a
        or "-7 days" in a
    )
    has_pending = "pending" in a
    has_in_progress = "in_progress" in a or "in progress" in a
    has_done = "done" in a
    fact_groups = has_pending and has_in_progress and has_done
    fact_title = "weekly report" in a
    details = {
        "seven_days": fact_days,
        "groups_all_three": fact_groups,
        "weekly_report_title": fact_title,
        "_pending": has_pending,
        "_in_progress": has_in_progress,
        "_done": has_done,
    }
    return int(fact_days) + int(fact_groups) + int(fact_title), details


SCORE_FNS: dict[str, callable] = {
    "score_turn8": score_turn8,
    "score_turn18": score_turn18,
    "score_turn24": score_turn24,
}


# ------------------------ optimization driver ------------------------

def steps_for(n_tokens: int) -> int:
    return max(round(BASE_STEPS * n_tokens / BASE_TOKENS), MIN_STEPS)


def num_virtual_for(n_tokens: int, ratio: float) -> int:
    return max(round(n_tokens / ratio), 1)


def optimize(
    model, model_dtype, prefix_kv, prefix_len,
    init_virtual: torch.Tensor,
    k_target_list: list[torch.Tensor] | None,
    k_mag2_list: list[torch.Tensor] | None,
    v_target_list: list[torch.Tensor],
    v_mag2_list: list[torch.Tensor],
    virtual_pos: torch.Tensor,
    cos_v: torch.Tensor, sin_v: torch.Tensor,
    num_steps: int, label: str,
    use_k: bool, run_k_diag: bool,
) -> dict:
    """Optimize virtual token embeddings.

    use_k=True  -> loss = K loss (pre-RoPE) + V loss   (kv_optimized)
    use_k=False -> loss = V loss only                  (v_only baseline)

    The V path is identical regardless of use_k. K loss is always
    *computed* (for logging) but only added to the backward'd loss when
    use_k is True.
    """
    device = model.device
    virtual = init_virtual.detach().clone().to(dtype=torch.float32, device=device)
    virtual.requires_grad_(True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    optimizer = torch.optim.Adam([virtual], lr=LR)
    total_curve: list[float] = []
    k_curve: list[float] = []
    v_curve: list[float] = []
    k_diag: dict | None = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

    def compute_losses(live):
        k_loss = torch.zeros((), dtype=torch.float32, device=device)
        v_loss = torch.zeros((), dtype=torch.float32, device=device)
        for li, (k_live, v_live) in enumerate(live):
            v_l = F.mse_loss(v_live.float(), v_target_list[li].float())
            v_loss = v_loss + v_l / v_mag2_list[li].clamp(min=1e-8)
            if k_target_list is not None:
                k_pre = inverse_rotary_pos_emb(k_live, cos_v, sin_v)
                k_l = F.mse_loss(k_pre.float(), k_target_list[li].float())
                k_loss = k_loss + k_l / k_mag2_list[li].clamp(min=1e-8)
        return k_loss, v_loss

    try:
        for step in range(num_steps):
            inputs_embeds = virtual.to(dtype=model_dtype).unsqueeze(0)
            out = model(
                inputs_embeds=inputs_embeds,
                past_key_values=wrap_legacy_kv(prefix_kv, model.config),
                position_ids=virtual_pos,
                output_hidden_states=False,
                use_cache=True,
            )
            live = extract_new_kv_grad_safe(out.past_key_values, prefix_len)
            k_loss, v_loss = compute_losses(live)
            loss = (k_loss + v_loss) if use_k else v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0 and virtual.grad is None:
                raise AssertionError(
                    f"[{label}] Gradient did not flow through KV extraction."
                )
            combined_grad_norm = (
                float(virtual.grad.norm().item()) if virtual.grad is not None else 0.0
            )
            optimizer.step()

            total_curve.append(float(loss.detach().item()))
            k_curve.append(float(k_loss.detach().item()))
            v_curve.append(float(v_loss.detach().item()))

            # One-time K-only gradient diagnostic (first turn, first
            # ratio, kv_optimized). A dedicated forward isolates the K
            # path: if inverse_rotary_pos_emb broke the graph, V grads
            # alone would still pass the step-0 assertion above.
            if run_k_diag and step == 0 and k_target_list is not None:
                ie2 = virtual.to(dtype=model_dtype).unsqueeze(0)
                out2 = model(
                    inputs_embeds=ie2,
                    past_key_values=wrap_legacy_kv(prefix_kv, model.config),
                    position_ids=virtual_pos,
                    output_hidden_states=False,
                    use_cache=True,
                )
                live2 = extract_new_kv_grad_safe(out2.past_key_values, prefix_len)
                k_only = torch.zeros((), dtype=torch.float32, device=device)
                for li, (k_live2, _v2) in enumerate(live2):
                    k_pre2 = inverse_rotary_pos_emb(k_live2, cos_v, sin_v)
                    k_l2 = F.mse_loss(k_pre2.float(), k_target_list[li].float())
                    k_only = k_only + k_l2 / k_mag2_list[li].clamp(min=1e-8)
                optimizer.zero_grad(set_to_none=True)
                k_only.backward()
                k_only_grad_norm = (
                    float(virtual.grad.norm().item()) if virtual.grad is not None else 0.0
                )
                k_flows = virtual.grad is not None and k_only_grad_norm > 0.0
                if not k_flows:
                    print(
                        "  WARNING: Gradients do not flow through inverse RoPE "
                        "path -- K loss is not contributing to optimization"
                    )
                else:
                    print(
                        f"  [k-diag] k_only_grad_norm={k_only_grad_norm:.4f}  "
                        f"combined_grad_norm={combined_grad_norm:.4f}"
                    )
                k_diag = {
                    "k_gradient_flows": bool(k_flows),
                    "k_only_grad_norm": k_only_grad_norm,
                    "combined_grad_norm": combined_grad_norm,
                    "k_v_loss_ratio_step0": (
                        float(k_curve[0] / v_curve[0]) if v_curve[0] > 0 else None
                    ),
                }
                optimizer.zero_grad(set_to_none=True)
                del out2, live2, k_only, ie2

            if step in (0, num_steps // 2, num_steps - 1):
                print(
                    f"    [{label}] step {step:>4}/{num_steps}: "
                    f"total={total_curve[-1]:.4f} "
                    f"(k={k_curve[-1]:.4f} v={v_curve[-1]:.4f})  {vram_str()}"
                )
            elif (step + 1) % LOG_EVERY == 0:
                print(
                    f"    [{label}] step {step:>4}/{num_steps}: "
                    f"total={total_curve[-1]:.4f} (k={k_curve[-1]:.4f} v={v_curve[-1]:.4f})"
                )
            del out, live, loss, k_loss, v_loss, inputs_embeds
    finally:
        model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    opt_time = time.perf_counter() - t_start
    return {
        "virtual": virtual.detach(),
        "loss_curve": total_curve,
        "initial_total_loss": total_curve[0],
        "final_total_loss": total_curve[-1],
        "initial_k_loss": k_curve[0],
        "final_k_loss": k_curve[-1],
        "initial_v_loss": v_curve[0],
        "final_v_loss": v_curve[-1],
        "opt_time_s": opt_time,
        "k_diag": k_diag,
    }


# ------------------------ per-turn validation ------------------------

def run_turn_validation(
    model, embed, model_dtype, tokenizer,
    prefix_kv: tuple, prefix_len: int,
    turn_record: dict, question_text_inner: str, expected_str: str,
    score_fn, label: str, stop_ids: set[int],
    do_k_diag: bool,
) -> dict:
    device = model.device
    turn_ids = turn_record["ids"].to(device)
    n_test = int(turn_record["n_tokens"])
    virtual_pos_start = prefix_len
    virtual_pos_end = prefix_len + n_test - 1
    suffix_start_pos = prefix_len + n_test
    question_text = f"\n\nQuestion: {question_text_inner}\nAnswer:"

    print()
    print("=" * 78)
    print(f"  [{label}] turn={turn_record['turn_index_1based']} "
          f"n_test={n_test} prefix_len={prefix_len}")
    print(f"  virtual range=[{virtual_pos_start}, {virtual_pos_end}]  "
          f"suffix_start={suffix_start_pos}")
    print(f"  question: {question_text_inner}")
    print(f"  expected: {expected_str}")
    print("=" * 78)

    # ---- 1. Target forward: pre-RoPE K, V, perplexity (one no-grad pass) ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    target_positions = torch.arange(
        virtual_pos_start, virtual_pos_start + n_test, device=device, dtype=torch.long,
    )
    with torch.no_grad():
        kv_out = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=target_positions.unsqueeze(0),
            output_hidden_states=False,
            use_cache=True,
        )
        perplexity_weights = compute_perplexity_weights(kv_out.logits, turn_ids)
        cos_t, sin_t = get_rotary_cos_sin(model, target_positions, device)
        full_kv = to_legacy_kv(kv_out.past_key_values)
        k_pre_full: list[torch.Tensor] = []
        v_full: list[torch.Tensor] = []
        for k, v in full_kv:
            k_post = k[:, :, prefix_len:, :]
            k_pre = inverse_rotary_pos_emb(k_post.float(), cos_t, sin_t)
            k_pre_full.append(k_pre.detach().clone())
            v_full.append(v[:, :, prefix_len:, :].detach().clone())
            del k, v
    del kv_out, full_kv
    torch.cuda.empty_cache()
    print(f"  [{label}] K_pre/V_full: {len(v_full)} layers x "
          f"K{tuple(k_pre_full[0].shape)} V{tuple(v_full[0].shape)}  "
          f"perplexity[min,max]=[{perplexity_weights.min().item():.3f}, "
          f"{perplexity_weights.max().item():.3f}]  {vram_str()}")

    with torch.no_grad():
        turn_emb = embed(turn_ids).squeeze(0)  # (n_test, D)

    answers: dict[str, str] = {}
    gen_times: dict[str, float] = {}
    optim_meta: dict[str, dict] = {}
    k_diag_result: dict | None = None

    # ---- 2. Reference conditions ----
    @torch.no_grad()
    def kv_with_full_turn():
        out = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out.past_key_values))

    @torch.no_grad()
    def kv_with_raw_embeddings():
        out = model(
            inputs_embeds=embed(turn_ids),
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out.past_key_values))

    @torch.no_grad()
    def kv_with_virtual(v: torch.Tensor, virtual_pos: torch.Tensor) -> tuple:
        ie = v.to(dtype=model_dtype).unsqueeze(0)
        out = model(
            inputs_embeds=ie,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=virtual_pos,
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out.past_key_values))

    def gen_full(kv: tuple) -> str:
        return generate_with_kv(model, tokenizer, question_text, kv, MAX_NEW_TOKENS, stop_ids)

    def gen_posinterp(kv: tuple) -> str:
        return manual_greedy_with_positions(
            model, tokenizer, question_text, kv,
            start_pos=suffix_start_pos, max_new_tokens=MAX_NEW_TOKENS, stop_ids=stop_ids,
        )

    print(f"  [{label}] full_turn ...")
    full_kv_inf = kv_with_full_turn()
    t0 = time.perf_counter()
    answers["full_turn"] = gen_full(full_kv_inf)
    gen_times["full_turn"] = time.perf_counter() - t0
    print(f"    full_turn {gen_times['full_turn']:.1f}s: {answers['full_turn'][:120]!r}")
    del full_kv_inf
    torch.cuda.empty_cache()

    print(f"  [{label}] raw_embeddings ...")
    raw_kv_inf = kv_with_raw_embeddings()
    t0 = time.perf_counter()
    answers["raw_embeddings"] = gen_full(raw_kv_inf)
    gen_times["raw_embeddings"] = time.perf_counter() - t0
    print(f"    raw_embeddings {gen_times['raw_embeddings']:.1f}s: {answers['raw_embeddings'][:120]!r}")
    del raw_kv_inf
    torch.cuda.empty_cache()

    print(f"  [{label}] no_turn ...")
    t0 = time.perf_counter()
    answers["no_turn"] = gen_full(prefix_kv)
    gen_times["no_turn"] = time.perf_counter() - t0
    print(f"    no_turn {gen_times['no_turn']:.1f}s: {answers['no_turn'][:120]!r}")

    num_steps = steps_for(n_test)

    def build_targets(num_virtual: int):
        k_tgt, v_tgt, k_mag2, v_mag2 = [], [], [], []
        for li in range(len(v_full)):
            kp = attention_weighted_pool(
                k_pre_full[li], perplexity_weights, num_virtual,
            ).detach().clone()
            vp = attention_weighted_pool(
                v_full[li], perplexity_weights, num_virtual,
            ).detach().clone()
            k_tgt.append(kp)
            v_tgt.append(vp)
            k_mag2.append(kp.float().pow(2).mean().detach())
            v_mag2.append(vp.float().pow(2).mean().detach())
        return k_tgt, v_tgt, k_mag2, v_mag2

    def run_condition(cond_label: str, num_virtual: int, virtual_pos, cos_v, sin_v,
                      k_tgt, k_mag2, v_tgt, v_mag2, use_k: bool, run_diag: bool):
        nonlocal k_diag_result
        print()
        print(f"  [{label}/{cond_label}] V={num_virtual}, steps={num_steps}, "
              f"use_k={use_k} ...")
        init = mean_pool_chunks(turn_emb, num_virtual).detach().clone()
        res = optimize(
            model, model_dtype, prefix_kv, prefix_len,
            init, k_tgt if use_k else None, k_mag2 if use_k else None,
            v_tgt, v_mag2, virtual_pos, cos_v, sin_v,
            num_steps, cond_label, use_k, run_diag,
        )
        if res.get("k_diag") is not None:
            k_diag_result = res["k_diag"]
        print(f"  [{label}/{cond_label}] {num_steps} steps in {res['opt_time_s']:.1f}s, "
              f"total {res['initial_total_loss']:.4f} -> {res['final_total_loss']:.4f} "
              f"(k {res['initial_k_loss']:.4f}->{res['final_k_loss']:.4f}, "
              f"v {res['initial_v_loss']:.4f}->{res['final_v_loss']:.4f})")

        inf_kv = kv_with_virtual(res["virtual"], virtual_pos)
        t0 = time.perf_counter()
        ans = gen_posinterp(inf_kv)
        dt = time.perf_counter() - t0
        answers[cond_label] = ans
        gen_times[cond_label] = dt
        print(f"    {cond_label} {dt:.1f}s: {ans[:120]!r}")
        optim_meta[cond_label] = {
            "compression_ratio": round(n_test / num_virtual, 4),
            "num_virtual": num_virtual,
            "num_steps": num_steps,
            "use_k": use_k,
            "initial_k_loss": res["initial_k_loss"],
            "final_k_loss": res["final_k_loss"],
            "initial_v_loss": res["initial_v_loss"],
            "final_v_loss": res["final_v_loss"],
            "initial_total_loss": res["initial_total_loss"],
            "final_total_loss": res["final_total_loss"],
            "loss_curve": res["loss_curve"],
            "opt_time_s": res["opt_time_s"],
        }
        del init, inf_kv, res
        torch.cuda.empty_cache()

    # ---- 3. Sweep compression ratios ----
    for ratio in COMPRESSION_RATIOS:
        num_virtual = num_virtual_for(n_test, ratio)
        virtual_pos = interpolate_positions(
            virtual_pos_start, virtual_pos_end, num_virtual, device=device,
        )
        cos_v, sin_v = get_rotary_cos_sin(model, virtual_pos.squeeze(0), device)
        k_tgt, v_tgt, k_mag2, v_mag2 = build_targets(num_virtual)

        run_diag = do_k_diag and abs(ratio - COMPRESSION_RATIOS[0]) < 1e-9
        run_condition(
            f"kv_optimized_{ratio:.1f}x", num_virtual, virtual_pos, cos_v, sin_v,
            k_tgt, k_mag2, v_tgt, v_mag2, use_k=True, run_diag=run_diag,
        )

        # V-only baseline at the designated ratio (same targets, K dropped).
        if abs(ratio - V_ONLY_RATIO) < 1e-9:
            run_condition(
                "v_only_4.8x", num_virtual, virtual_pos, cos_v, sin_v,
                k_tgt, k_mag2, v_tgt, v_mag2, use_k=False, run_diag=False,
            )

        del k_tgt, v_tgt, k_mag2, v_mag2, virtual_pos, cos_v, sin_v
        torch.cuda.empty_cache()

    # Free the large per-turn target tensors before scoring. Reassign
    # rather than `del` so the names stay bound -- they are closure
    # variables of build_targets/run_condition above, and deleting the
    # binding trips pyflakes F821 (and would unbind the closure cells).
    k_pre_full.clear()
    v_full.clear()
    perplexity_weights = turn_emb = cos_t = sin_t = None  # noqa: F841
    torch.cuda.empty_cache()

    # ---- 4. Score ----
    cond_order = (
        ["full_turn", "raw_embeddings"]
        + [f"kv_optimized_{r:.1f}x" for r in COMPRESSION_RATIOS]
        + ["v_only_4.8x", "no_turn"]
    )
    conditions: dict[str, dict] = {}
    for cond in cond_order:
        ans = answers.get(cond, "")
        if ans and not ans.startswith("<error"):
            score_count, score_details = score_fn(ans)
        else:
            score_count, score_details = 0, {}
        entry: dict = {
            "answer": ans,
            "score": f"{score_count}/3",
            "score_count": score_count,
            "score_details": score_details,
            "gen_time_s": gen_times.get(cond, float("nan")),
        }
        if cond in optim_meta:
            entry.update(optim_meta[cond])
        conditions[cond] = entry

    return {"conditions": conditions, "cond_order": cond_order, "k_diag": k_diag_result}


# -------------------------------- main -------------------------------

def main():
    print("=" * 78)
    print("  Pre-RoPE K + V compression sweep")
    print("=" * 78)

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    device = model.device
    embed = model.get_input_embeddings()
    model_dtype = embed.weight.dtype
    print(f"Model loaded. dtype={model_dtype}. {vram_str()}")

    stop_ids = get_stop_ids(tokenizer)
    print(f"Stop token ids: {sorted(stop_ids)}")

    raw_lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    turns = [json.loads(line) for line in raw_lines if line.strip()]
    print(f"Loaded {len(turns)} turns from {TRANSCRIPT_PATH.name}")

    # ---- Prefix KV ----
    prefix_turn = turns[PREFIX_TURN_INDEX]
    prefix_role = prefix_turn.get("role", "?")
    prefix_content = _stringify_content(prefix_turn.get("content", ""))
    prefix_text = turn_marker(PREFIX_TURN_INDEX + 1, prefix_role, prefix_content)
    prefix_ids = tokenizer(
        prefix_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_prefix = int(prefix_ids.shape[1])
    print(f"\nBuilding prefix KV (turn {PREFIX_TURN_INDEX + 1}, {n_prefix} tokens) ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
    prefix_kv = detach_kv(to_legacy_kv(prefix_out.past_key_values))
    del prefix_out
    torch.cuda.empty_cache()
    prefix_len = kv_seq_len(prefix_kv)
    print(f"  prefix_kv kv_seq_len={prefix_len}. {vram_str()}")

    # ---- RoPE source check ----
    rope_verified, rope_finding = verify_rope_for_qwen()
    if rope_verified:
        print(f"[rope_check] CONFIRMED -- RoPE on Q,K only. line: {rope_finding}")
    else:
        print(f"[rope_check] WARNING -- could not confirm: {rope_finding}")

    # ---- Rotary module introspection ----
    rot = find_rotary_emb(model)
    print(f"[rotary] module type: {type(rot).__name__}")
    rot_attrs = [a for a in dir(rot) if not a.startswith("_")]
    print(f"[rotary] attrs: {rot_attrs}")
    inv_freq = getattr(rot, "inv_freq", None)
    if inv_freq is not None:
        print(f"[rotary] inv_freq shape={tuple(inv_freq.shape)} dtype={inv_freq.dtype}")

    # ---- Position-id runtime check (hard halt on failure) ----
    print("\nVerifying explicit position_ids take effect ...")
    posid_check = verify_position_ids_take_effect(model, prefix_kv, prefix_len, model_dtype)
    print(f"  k_diff_max={posid_check['k_diff_max']:.4f} "
          f"v_diff_max={posid_check['v_diff_max']:.4f} "
          f"took_effect={posid_check['took_effect']}")
    if not posid_check["took_effect"]:
        raise AssertionError(
            f"position_ids ignored: {posid_check} -- experiment cannot proceed."
        )

    # ---- Inverse-RoPE round-trip check (uses turn 8 as probe) ----
    probe_turn = turns[TURNS_TO_TEST[0]["turn_index_1based"] - 1]
    probe_text = turn_marker(
        TURNS_TO_TEST[0]["turn_index_1based"],
        probe_turn.get("role", "?"),
        _stringify_content(probe_turn.get("content", "")),
    )
    probe_ids = tokenizer(
        probe_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    print("\nVerifying inverse RoPE (round-trip K_post -> inverse -> forward) ...")
    inv_check = verify_inverse_rope(model, prefix_kv, probe_ids, prefix_len)
    print(f"  max_abs_diff={inv_check['max_abs_diff']:.3e} "
          f"k_abs_mean={inv_check['k_abs_mean']:.3e} "
          f"rel_diff={inv_check['rel_diff']:.3e} verified={inv_check['verified']}")
    if not inv_check["verified"]:
        print("!" * 78)
        print("  WARNING: inverse RoPE round-trip exceeded tolerance. K targets")
        print("  may be miscomputed. Continuing so V-only paths still run.")
        print("!" * 78)

    # ---- Pre-tokenize turns ----
    selected: list[dict] = []
    for spec in TURNS_TO_TEST:
        ti = spec["turn_index_1based"]
        turn = turns[ti - 1]
        role = turn.get("role", "?")
        content = _stringify_content(turn.get("content", ""))
        text = turn_marker(ti, role, content)
        ids_list = tokenizer(text, add_special_tokens=False).input_ids
        record = {
            "turn_index_1based": ti,
            "role": role,
            "text": text,
            "preview": text[:200],
            "n_tokens": len(ids_list),
            "ids": torch.tensor([ids_list], dtype=torch.long),
        }
        selected.append({"spec": spec, "record": record})
        print(f"\nSelected: T{ti} ({role}, {len(ids_list)} tokens, {spec['content_type']})")

    # ---- Run validation per turn ----
    turns_results: list[dict] = []
    k_diag_global: dict | None = None
    for idx, entry in enumerate(selected):
        spec = entry["spec"]
        record = entry["record"]
        ti = record["turn_index_1based"]
        score_fn = SCORE_FNS[spec["score_fn_name"]]
        do_k_diag = (idx == 0)  # first turn, first ratio inside run_turn_validation
        try:
            res = run_turn_validation(
                model, embed, model_dtype, tokenizer,
                prefix_kv, prefix_len,
                record, spec["question"], spec["expected"],
                score_fn, f"T{ti}", stop_ids, do_k_diag,
            )
            if res.get("k_diag") is not None:
                k_diag_global = res["k_diag"]
        except Exception as e:
            print(f"[ERROR] T{ti} validation crashed: {e}")
            traceback.print_exc()
            res = {"error": f"{type(e).__name__}: {e}", "conditions": {}, "cond_order": []}

        turns_results.append({
            "turn_index": ti,
            "role": record["role"],
            "token_count": record["n_tokens"],
            "content_type": spec["content_type"],
            "question": spec["question"],
            "expected": spec["expected"],
            "preview": record["preview"],
            "conditions": res.get("conditions", {}),
            "_cond_order": res.get("cond_order", []),
        })
        torch.cuda.empty_cache()

    # ---- Save JSON ----
    kd = k_diag_global or {}
    results = {
        "config": {
            "model_id": MODEL_ID,
            "target": "pre-RoPE K + V, perplexity-weighted pooling, per-layer normalized",
            "inverse_rope_verified": inv_check["verified"],
            "inverse_rope_check": inv_check,
            "k_gradient_flows": kd.get("k_gradient_flows"),
            "k_v_loss_ratio_step0": kd.get("k_v_loss_ratio_step0"),
            "k_only_grad_norm": kd.get("k_only_grad_norm"),
            "combined_grad_norm": kd.get("combined_grad_norm"),
            "position_interpolation": True,
            "compression_ratios": COMPRESSION_RATIOS,
            "v_only_baseline_ratio": V_ONLY_RATIO,
            "lr": LR,
            "steps_formula": "max(round(300 * turn_tokens / 250), 100)",
            "rope_verified": rope_verified,
            "rope_finding": rope_finding,
            "position_ids_take_effect": posid_check,
            "prefix_turn_index": PREFIX_TURN_INDEX + 1,
            "prefix_tokens": n_prefix,
            "stop_token_ids": sorted(stop_ids),
            "v_only_prior_scores": {
                str(t): {f"{r:.1f}x": s for r, s in d.items()}
                for t, d in V_ONLY_PRIOR_SCORES.items()
            },
        },
        "turns_tested": [
            {k: v for k, v in t.items() if not k.startswith("_")}
            for t in turns_results
        ],
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {RESULTS_PATH}")

    # ---- Console summary: per-turn answers ----
    print()
    print("=" * 78)
    print("  PER-TURN RESULTS")
    print("=" * 78)
    for t in turns_results:
        ti = t["turn_index"]
        print()
        print(f"=== TURN {ti} ({t['token_count']} tokens, {t['content_type']}) ===")
        print(f"Question: {t['question']}")
        print(f"Expected: {t['expected']}")
        print()
        for cond in t.get("_cond_order", []):
            e = t["conditions"].get(cond, {})
            score = e.get("score", "?/3")
            line = (e.get("answer", "") or "").replace("\n", " ")
            if len(line) > 150:
                line = line[:150] + "..."
            extra = ""
            if "final_total_loss" in e:
                extra = (
                    f"  [total {e['initial_total_loss']:.2f}->{e['final_total_loss']:.2f}"
                    f" k {e['initial_k_loss']:.2f}->{e['final_k_loss']:.2f}"
                    f" v {e['initial_v_loss']:.2f}->{e['final_v_loss']:.2f}]"
                )
            print(f"  [{cond:<20}] [{score}] {line}{extra}")

    # ---- Console summary: V-only vs K+V comparison ----
    print()
    print("=" * 78)
    print("  V-ONLY (prior) vs K+V (new) -- correct facts out of 3")
    print("=" * 78)
    for t in turns_results:
        ti = t["turn_index"]
        priors = V_ONLY_PRIOR_SCORES.get(ti, {})
        print()
        print(f"  Turn {ti} ({t['token_count']} tok, {t['content_type']})")
        print(f"  {'Ratio':>6} | {'V-only (prior)':>14} | {'K+V (new)':>9} | {'Change':>6}")
        print("  " + "-" * 48)
        for ratio in COMPRESSION_RATIOS:
            prior = priors.get(ratio)
            prior_str = f"{prior}/3" if prior is not None else "—"
            e = t["conditions"].get(f"kv_optimized_{ratio:.1f}x", {})
            new = e.get("score_count")
            new_str = f"{new}/3" if new is not None else "?/3"
            if prior is not None and new is not None:
                d = new - prior
                change = f"+{d}" if d > 0 else (str(d) if d < 0 else "=")
            else:
                change = ""
            marker = "  <- key" if abs(ratio - V_ONLY_RATIO) < 1e-9 else ""
            print(f"  {ratio:>5.1f}x | {prior_str:>14} | {new_str:>9} | {change:>6}{marker}")
        # v_only baseline computed this run (at V_ONLY_RATIO) for reference.
        vb = t["conditions"].get("v_only_4.8x", {})
        if "score_count" in vb:
            print(f"  (this run's v_only_4.8x baseline: {vb['score_count']}/3)")

    # ---- Sanity: raw_embeddings vs full_turn ----
    diverged = []
    for t in turns_results:
        f_a = (t["conditions"].get("full_turn", {}).get("answer") or "").strip()
        r_a = (t["conditions"].get("raw_embeddings", {}).get("answer") or "").strip()
        if not f_a or not r_a or f_a.startswith("<error") or r_a.startswith("<error"):
            continue
        if f_a != r_a:
            diverged.append(t["turn_index"])
    print()
    if diverged:
        print("!" * 78)
        print(f"  WARNING: raw_embeddings DIVERGED from full_turn on turns: {diverged}")
        print("!" * 78)
    else:
        print("[sanity] raw_embeddings matches full_turn on all tested turns.")

    # ---- K-gradient diagnostic summary ----
    if k_diag_global is not None:
        print()
        print("K-only gradient diagnostic (turn 8, 1.6x kv_optimized, step 0):")
        print(f"  k_gradient_flows   = {k_diag_global['k_gradient_flows']}")
        print(f"  k_only_grad_norm   = {k_diag_global['k_only_grad_norm']:.4f}")
        print(f"  combined_grad_norm = {k_diag_global['combined_grad_norm']:.4f}")
        print(f"  k_v_loss_ratio_0   = {k_diag_global['k_v_loss_ratio_step0']}")


if __name__ == "__main__":
    main()
