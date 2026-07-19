"""Token-level exemption: keep top-15% by V-uniqueness verbatim, compress
the rest with our standard recipe.

Idea: compression failures (e.g. "bcryptjs" → "bcrypt") happen because
specific string-literal tokens get diluted during pooling. Instead of
trying to compress those tokens, EXEMPT them — pass their KV entries
through untouched (exact values, exact RoPE rotations at original
positions). The remaining tokens get compressed via the standard
V-only / perplexity-weighted / position-interpolated recipe.

The final per-turn KV cache is a mix:
  - prefix entries (unchanged)
  - exempt KV entries (exact, at original positions)
  - virtual KV entries (optimized, at interpolated positions across
    the compressible range)

Conditions per turn (3 turns: 8, 18, 24):
  1. full_turn (ceiling)
  2. raw_embeddings (sanity)
  3. exempt_15pct_3.2x  (15% exempt, 3.2x overall — should be 3/3)
  4. exempt_15pct_4.0x  (15% exempt, 4.0x overall — main test point)
  5. exempt_15pct_4.8x  (15% exempt, 4.8x overall — direct vs uniform_4.8x)
  6. uniform_4.8x       (no exemption, cross_validation recipe — baseline)
  7. no_turn (floor)

Standalone — no imports from project files.
"""

from __future__ import annotations

import inspect
import json
import math
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------- constants -----------------------------

from config import MODEL_ID, strip_thinking
TRANSCRIPT_PATH = Path(__file__).parent / "transcript.jsonl"
RESULTS_PATH = Path(__file__).parent / "results_token_exemption.json"

PREFIX_TURN_INDEX = 0  # 0-indexed; turn 1 (system prompt)

EXEMPTION_RATE = 0.15
EXEMPTION_K = 5
PRIMARY_K = EXEMPTION_K
COMPRESSION_RATIOS = [3.2, 4.0, 4.8]
UNIFORM_BASELINE_RATIO = 4.8

LR = 0.003
BASE_STEPS = 300
BASE_TOKENS = 250
MIN_STEPS = 100
LOG_EVERY = 50
MAX_NEW_TOKENS = 256

EXTRA_STOP_TOKEN_STRINGS = ["<|im_end|>", "<|endoftext|>"]


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
            "bcryptjs for password hashing; "
            "Node's built-in test runner (node:test); "
            "regex-based markdown converter (no external libraries)."
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
        "expected": (
            "jsonwebtoken; HTTP 401; default secret is 'dev-secret'."
        ),
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


# --- KV manipulation helpers (new) ---

def slice_kv(legacy: tuple, start: int, end: int | None = None) -> tuple:
    """Slice every layer's K and V along the seq dim (dim=2)."""
    if end is None:
        return tuple((k[:, :, start:, :], v[:, :, start:, :]) for k, v in legacy)
    return tuple((k[:, :, start:end, :], v[:, :, start:end, :]) for k, v in legacy)


def gather_kv(legacy: tuple, indices: torch.Tensor) -> tuple:
    """index_select each layer's K and V along the seq dim (dim=2)."""
    return tuple(
        (k.index_select(2, indices), v.index_select(2, indices))
        for k, v in legacy
    )


def concat_kv(legacy_a: tuple, legacy_b: tuple) -> tuple:
    """Concatenate two legacy KV tuples along the seq dim, layer-wise."""
    if not legacy_a:
        return legacy_b
    if not legacy_b:
        return legacy_a
    if len(legacy_a) != len(legacy_b):
        raise ValueError(f"layer count mismatch: {len(legacy_a)} vs {len(legacy_b)}")
    return tuple(
        (torch.cat([k_a, k_b], dim=2), torch.cat([v_a, v_b], dim=2))
        for (k_a, v_a), (k_b, v_b) in zip(legacy_a, legacy_b)
    )


def select_exempt_indices(uniqueness: torch.Tensor, rate: float) -> torch.Tensor:
    """Top ceil(N * rate) indices by uniqueness, returned sorted ascending
    (so they're in position order — useful for downstream gather and
    diagnostics).
    """
    n = int(uniqueness.shape[0])
    k = max(1, math.ceil(n * rate))
    top_indices = torch.topk(uniqueness, k).indices
    return torch.sort(top_indices).values  # ascending


# --- pooling helpers ---

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
    """Uniform-size chunks with within-chunk softmaxed importance."""
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
        f"Unknown cache type for grad-safe extract: {type(cache).__name__}"
    )


# --- V-uniqueness (from token_importance_diagnostic.py) ---

@torch.no_grad()
def compute_v_uniqueness(v_full: list[torch.Tensor], k: int) -> torch.Tensor:
    """Per-token V-uniqueness, averaged over heads and layers, returned
    as (S,) fp32 on CPU. Vectorized via cumsum.
    """
    if not v_full:
        raise ValueError("v_full is empty")
    s = int(v_full[0].shape[2])
    device = v_full[0].device

    arange_s = torch.arange(s, device=device)
    los = (arange_s - k).clamp(min=0)
    his = (arange_s + k + 1).clamp(max=s)
    num_neighbors = (his - los - 1).clamp(min=1)

    eps = 1e-8
    per_layer_uniq: list[torch.Tensor] = []
    for v_layer in v_full:
        v = v_layer[0].float()
        n_kv, _, dh = v.shape
        zeros = torch.zeros(n_kv, 1, dh, dtype=v.dtype, device=device)
        cumsum_pad = torch.cat([zeros, v.cumsum(dim=1)], dim=1)
        window_sum = cumsum_pad[:, his, :] - cumsum_pad[:, los, :]
        neighbor_sum = window_sum - v
        neighbor_mean = neighbor_sum / num_neighbors.view(1, -1, 1)
        diff_norm = (v - neighbor_mean).norm(dim=-1)
        mean_norm = neighbor_mean.norm(dim=-1)
        head_uniq = diff_norm / (mean_norm + eps)
        per_layer_uniq.append(head_uniq.mean(dim=0))
        del cumsum_pad, window_sum, neighbor_sum, neighbor_mean, diff_norm, mean_norm
    stacked = torch.stack(per_layer_uniq, dim=0)
    return stacked.mean(dim=0).detach().cpu()


# --- RoPE / position id checks ---

def verify_rope_for_qwen() -> tuple[bool, str]:
    try:
        from transformers.models.qwen3 import modeling_qwen3
        src = inspect.getsource(modeling_qwen3.Qwen3Attention.forward)
    except Exception as e:
        return False, f"could not inspect Qwen3Attention.forward: {e}"
    matched = [ln.strip() for ln in src.splitlines() if "apply_rotary_pos_emb" in ln]
    if not matched:
        return False, "apply_rotary_pos_emb not found in Qwen3Attention.forward"
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
        return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))
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
    return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))


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
        return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))
    for _ in range(max_new_tokens - 1):
        next_inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(input_ids=next_inp, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated.append(next_id)
        if next_id in stop_ids:
            break
    return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))


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
        return strip_thinking(tokenizer.decode(new_ids, skip_special_tokens=True))
    except Exception:
        return manual_greedy_auto(
            model, tokenizer, prompt_text, past_kv_legacy, max_new_tokens, stop_ids,
        )


# --------------------------- score functions -------------------------

def score_turn8(ans: str) -> tuple[int, dict[str, bool]]:
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
    return int(fact_bcryptjs) + int(fact_node_test) + int(fact_regex), {
        "bcryptjs": fact_bcryptjs,
        "node_test": fact_node_test,
        "regex_only": fact_regex,
    }


def score_turn18(ans: str) -> tuple[int, dict[str, bool]]:
    a = ans.lower()
    fact_lib = "jsonwebtoken" in a
    fact_401 = "401" in a
    fact_secret = "dev-secret" in a or "dev secret" in a
    return int(fact_lib) + int(fact_401) + int(fact_secret), {
        "jsonwebtoken": fact_lib,
        "status_401": fact_401,
        "dev_secret": fact_secret,
    }


def score_turn24(ans: str) -> tuple[int, dict[str, bool]]:
    a = ans.lower()
    fact_days = (
        "7 days" in a or "seven days" in a or "last 7" in a or "-7 days" in a
    )
    has_pending = "pending" in a
    has_in_progress = "in_progress" in a or "in progress" in a
    has_done = "done" in a
    fact_groups = has_pending and has_in_progress and has_done
    fact_title = "weekly report" in a
    return int(fact_days) + int(fact_groups) + int(fact_title), {
        "seven_days": fact_days,
        "groups_all_three": fact_groups,
        "weekly_report_title": fact_title,
        "_pending": has_pending,
        "_in_progress": has_in_progress,
        "_done": has_done,
    }


SCORE_FNS: dict[str, Callable[[str], tuple[int, dict[str, bool]]]] = {
    "score_turn8": score_turn8,
    "score_turn18": score_turn18,
    "score_turn24": score_turn24,
}


# --------------------------- step formulas ---------------------------

def steps_for(n_tokens: int) -> int:
    return max(round(BASE_STEPS * n_tokens / BASE_TOKENS), MIN_STEPS)


# ------------------------ shared optimizer ---------------------------

def run_optimize_loop(
    model, model_dtype, past_kv_legacy: tuple, slice_offset: int,
    init: torch.Tensor, virtual_pos: torch.Tensor,
    target_v_list: list[torch.Tensor], v_mag2_list: list[torch.Tensor],
    num_steps: int, label: str,
) -> tuple[torch.Tensor, list[float]]:
    """Adam over fp32 virtual tokens; V-only normalized loss; returns
    (virtual.detach(), losses).

    Caller passes:
      - past_kv_legacy: detached prefix [+ exempt] KV
      - slice_offset:   physical length of past_kv_legacy
      - init:           (V, D) fp32 starting embeddings
      - virtual_pos:    (1, V) LongTensor of position_ids
      - target_v_list:  per-layer (1, n_kv, V, head_dim) target V values
      - v_mag2_list:    per-layer fp32 scalar mag² for normalization
    """
    device = model.device
    virtual = init.detach().clone().to(dtype=torch.float32, device=device)
    virtual.requires_grad_(True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    optimizer = torch.optim.Adam([virtual], lr=LR)
    losses: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cache_diag: dict = {}
    try:
        for step in range(num_steps):
            inputs_embeds = virtual.to(dtype=model_dtype).unsqueeze(0)
            out = model(
                inputs_embeds=inputs_embeds,
                past_key_values=wrap_legacy_kv(past_kv_legacy, model.config),
                position_ids=virtual_pos,
                output_hidden_states=False,
                use_cache=True,
            )
            if step == 0:
                cache_diag["cache_type"] = type(out.past_key_values).__name__
            live = extract_new_kv_grad_safe(out.past_key_values, slice_offset)
            if step == 0:
                v0 = live[0][1]
                cache_diag["v0_requires_grad"] = bool(v0.requires_grad)
                cache_diag["v0_grad_fn"] = (
                    type(v0.grad_fn).__name__ if v0.grad_fn is not None else "None"
                )
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for (_k_live, v_live), v_tgt, v_mag2 in zip(
                live, target_v_list, v_mag2_list,
            ):
                layer_loss = F.mse_loss(v_live.float(), v_tgt.float())
                loss = loss + layer_loss / v_mag2.clamp(min=1e-8)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0 and virtual.grad is None:
                print(f"  [{label}] GRADIENT FAILURE: virtual.grad is None.")
                for k_, v_ in cache_diag.items():
                    print(f"    {k_}: {v_}")
                raise AssertionError("Gradient did not flow through KV extraction.")
            optimizer.step()

            loss_val = float(loss.detach().item())
            losses.append(loss_val)
            if step in (0, num_steps // 2, num_steps - 1):
                print(f"    [{label}] step {step:>4}/{num_steps}: "
                      f"loss={loss_val:.4f}  {vram_str()}")
            elif (step + 1) % LOG_EVERY == 0:
                print(f"    [{label}] step {step:>4}/{num_steps}: loss={loss_val:.4f}")
            del out, loss, inputs_embeds
    finally:
        model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return virtual.detach(), losses


# ------------------- per-condition runners ---------------------------

def run_uniform_condition(
    model, embed, model_dtype, tokenizer,
    prefix_kv: tuple, prefix_len: int,
    turn_ids: torch.Tensor, n_test: int,
    perplexity: torch.Tensor, v_full: list[torch.Tensor], turn_emb: torch.Tensor,
    ratio: float, num_steps: int, label: str, stop_ids: set[int], suffix_start_pos: int,
    question_text: str,
) -> dict:
    """cross_validation-style: full-turn V target with uniform chunks +
    perplexity-weighted softmax, position interpolation across full
    turn range, no exemption. Used as the apples-to-apples baseline at
    UNIFORM_BASELINE_RATIO.
    """
    device = model.device
    num_virtual = max(round(n_test / ratio), 1)
    print()
    print(f"  [{label}] uniform: V={num_virtual}, ratio={ratio}, steps={num_steps}")

    # Full-turn V target
    target_v_list: list[torch.Tensor] = []
    v_mag2_list: list[torch.Tensor] = []
    for v_layer in v_full:
        v_pooled = attention_weighted_pool(v_layer, perplexity, num_virtual).detach().clone()
        target_v_list.append(v_pooled)
        v_mag2_list.append(v_pooled.float().pow(2).mean().detach())

    init = mean_pool_chunks(turn_emb, num_virtual).detach().clone()
    virtual_pos = interpolate_positions(
        prefix_len, prefix_len + n_test - 1, num_virtual, device=device,
    )

    t_start = time.perf_counter()
    virtual, losses = run_optimize_loop(
        model, model_dtype, prefix_kv, slice_offset=prefix_len,
        init=init, virtual_pos=virtual_pos,
        target_v_list=target_v_list, v_mag2_list=v_mag2_list,
        num_steps=num_steps, label=label,
    )
    opt_time = time.perf_counter() - t_start

    mean_pool_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (mean_pool_loss - final_loss) / mean_pool_loss if mean_pool_loss > 0 else 0.0
    print(f"  [{label}] {num_steps} steps in {opt_time:.1f}s, "
          f"mean_pool={mean_pool_loss:.4f} → final={final_loss:.4f} "
          f"({loss_reduction*100:.1f}% reduction)")

    # Build inference KV: prefix + virtual at interpolated positions
    with torch.no_grad():
        ie = virtual.to(dtype=model_dtype).unsqueeze(0)
        out = model(
            inputs_embeds=ie,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=virtual_pos,
            use_cache=True,
        )
        inf_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out, ie

    t0 = time.perf_counter()
    ans = manual_greedy_with_positions(
        model, tokenizer, question_text, inf_kv,
        start_pos=suffix_start_pos, max_new_tokens=MAX_NEW_TOKENS, stop_ids=stop_ids,
    )
    gen_time = time.perf_counter() - t0
    print(f"  [{label}] gen {gen_time:.1f}s: {ans[:120]!r}")

    del target_v_list, v_mag2_list, init, virtual, virtual_pos, inf_kv
    torch.cuda.empty_cache()

    return {
        "answer": ans,
        "compression_ratio": ratio,
        "num_exempt": 0,
        "num_virtual": num_virtual,
        "total_positions": num_virtual,
        "effective_compress_ratio": n_test / num_virtual,
        "num_steps": num_steps,
        "mean_pool_loss": mean_pool_loss,
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "loss_curve": losses,
        "opt_time_s": opt_time,
        "gen_time_s": gen_time,
    }


def run_exempt_condition(
    model, embed, model_dtype, tokenizer,
    prefix_plus_exempt_kv: tuple, pe_phys_len: int,
    prefix_len: int, n_test: int,
    perplexity: torch.Tensor, v_full: list[torch.Tensor], turn_emb: torch.Tensor,
    exempt_idx_dev: torch.Tensor, compressible_idx_dev: torch.Tensor,
    ratio: float, num_steps: int, label: str, stop_ids: set[int],
    suffix_start_pos: int, question_text: str,
) -> dict:
    """15%-exemption flow.

    Caller has already built prefix_plus_exempt_kv (prefix + exempt entries).
    This function:
      - computes V target on COMPRESSIBLE-only subset
      - runs Adam on virtual tokens at interpolated positions across the
        compressible position range
      - builds the final inference KV (prefix + exempt + virtual)
      - generates the answer
    """
    device = model.device
    num_exempt = int(exempt_idx_dev.shape[0])
    num_compressible = int(compressible_idx_dev.shape[0])
    total_budget = max(round(n_test / ratio), 1)
    num_virtual = max(total_budget - num_exempt, 1)
    effective_ratio = num_compressible / num_virtual

    # Compressible-only V target
    ppl_compressible = perplexity.index_select(0, compressible_idx_dev)
    target_v_list: list[torch.Tensor] = []
    v_mag2_list: list[torch.Tensor] = []
    for v_layer in v_full:
        v_compressible = v_layer.index_select(2, compressible_idx_dev)  # (1, n_kv, num_compressible, Dh)
        v_pooled = attention_weighted_pool(
            v_compressible, ppl_compressible, num_virtual,
        ).detach().clone()
        target_v_list.append(v_pooled)
        v_mag2_list.append(v_pooled.float().pow(2).mean().detach())
        del v_compressible

    # Compressible-only init
    turn_emb_compressible = turn_emb.index_select(0, compressible_idx_dev)  # (num_compressible, D)
    init = mean_pool_chunks(turn_emb_compressible, num_virtual).detach().clone()
    del turn_emb_compressible

    # Position range across compressible only (logical positions in full sequence)
    compressible_pos_min = int(prefix_len + compressible_idx_dev[0].item())
    compressible_pos_max = int(prefix_len + compressible_idx_dev[-1].item())
    virtual_pos = interpolate_positions(
        compressible_pos_min, compressible_pos_max, num_virtual, device=device,
    )
    print(f"  [{label}] V_exempt={num_exempt}, V_virtual={num_virtual}, "
          f"total={num_exempt + num_virtual} (target {total_budget}), "
          f"effective={effective_ratio:.2f}x on compressible "
          f"({num_compressible} tokens), pos∈[{compressible_pos_min},{compressible_pos_max}]")

    t_start = time.perf_counter()
    virtual, losses = run_optimize_loop(
        model, model_dtype, prefix_plus_exempt_kv, slice_offset=pe_phys_len,
        init=init, virtual_pos=virtual_pos,
        target_v_list=target_v_list, v_mag2_list=v_mag2_list,
        num_steps=num_steps, label=label,
    )
    opt_time = time.perf_counter() - t_start

    mean_pool_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (mean_pool_loss - final_loss) / mean_pool_loss if mean_pool_loss > 0 else 0.0
    print(f"  [{label}] {num_steps} steps in {opt_time:.1f}s, "
          f"mean_pool={mean_pool_loss:.4f} → final={final_loss:.4f} "
          f"({loss_reduction*100:.1f}% reduction)")

    # Build inference KV: prefix + exempt + virtual
    with torch.no_grad():
        ie = virtual.to(dtype=model_dtype).unsqueeze(0)
        out = model(
            inputs_embeds=ie,
            past_key_values=wrap_legacy_kv(prefix_plus_exempt_kv, model.config),
            position_ids=virtual_pos,
            use_cache=True,
        )
        inf_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out, ie

    t0 = time.perf_counter()
    ans = manual_greedy_with_positions(
        model, tokenizer, question_text, inf_kv,
        start_pos=suffix_start_pos, max_new_tokens=MAX_NEW_TOKENS, stop_ids=stop_ids,
    )
    gen_time = time.perf_counter() - t0
    print(f"  [{label}] gen {gen_time:.1f}s: {ans[:120]!r}")

    del target_v_list, v_mag2_list, init, virtual, virtual_pos, inf_kv, ppl_compressible
    torch.cuda.empty_cache()

    return {
        "answer": ans,
        "compression_ratio": ratio,
        "num_exempt": num_exempt,
        "num_virtual": num_virtual,
        "total_positions": num_exempt + num_virtual,
        "compressible_pos_min": compressible_pos_min,
        "compressible_pos_max": compressible_pos_max,
        "effective_compress_ratio": effective_ratio,
        "num_steps": num_steps,
        "mean_pool_loss": mean_pool_loss,
        "initial_loss": losses[0],
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "loss_curve": losses,
        "opt_time_s": opt_time,
        "gen_time_s": gen_time,
    }


# ------------------------- per-turn driver ---------------------------

def run_turn_validation(
    model, embed, model_dtype, tokenizer,
    prefix_kv: tuple, prefix_len: int,
    turn_record: dict, question_text_inner: str, expected_str: str,
    score_fn: Callable[[str], tuple[int, dict[str, bool]]],
    label: str, stop_ids: set[int],
) -> dict:
    device = model.device
    turn_ids = turn_record["ids"].to(device)
    n_test = int(turn_record["n_tokens"])
    suffix_start_pos = prefix_len + n_test
    question_text = f"\n\nQuestion: {question_text_inner}\nAnswer:"

    print()
    print("=" * 78)
    print(f"  [{label}] turn={turn_record['turn_index_1based']} "
          f"n_test={n_test} prefix_len={prefix_len}")
    print(f"  question: {question_text_inner}")
    print(f"  expected: {expected_str}")
    print("=" * 78)

    # ---- 1. Single forward of the full turn ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=torch.arange(
                prefix_len, prefix_len + n_test,
                device=device, dtype=torch.long,
            ).unsqueeze(0),
            output_hidden_states=False,
            use_cache=True,
        )
    perplexity = compute_perplexity_weights(out.logits, turn_ids)
    full_turn_kv_full = detach_kv(to_legacy_kv(out.past_key_values))  # prefix + turn
    del out
    torch.cuda.empty_cache()

    # Slice off prefix → turn-only legacy KV
    turn_only_kv = slice_kv(full_turn_kv_full, start=prefix_len)
    # Per-layer V (turn portion) for V-uniqueness and target construction
    v_full: list[torch.Tensor] = [v for (_k, v) in turn_only_kv]
    print(f"  [{label}] V_full: {len(v_full)} layers x {tuple(v_full[0].shape)} "
          f"perplexity[min,max]=[{perplexity.min().item():.3f}, "
          f"{perplexity.max().item():.3f}]  {vram_str()}")

    with torch.no_grad():
        turn_emb = embed(turn_ids).squeeze(0)  # (n_test, D)

    # ---- 2. V-uniqueness + exempt selection ----
    t_uniq = time.perf_counter()
    v_uniqueness = compute_v_uniqueness(v_full, k=EXEMPTION_K)  # (N,) on CPU
    print(f"  [{label}] v_uniqueness k={EXEMPTION_K} computed in "
          f"{time.perf_counter() - t_uniq:.1f}s, "
          f"range [{v_uniqueness.min().item():.3f}, {v_uniqueness.max().item():.3f}]")
    exempt_idx_cpu = select_exempt_indices(v_uniqueness, EXEMPTION_RATE)  # ascending
    num_exempt = int(exempt_idx_cpu.shape[0])
    compressible_idx_cpu = torch.tensor(
        [i for i in range(n_test) if i not in set(exempt_idx_cpu.tolist())],
        dtype=torch.long,
    )
    num_compressible = int(compressible_idx_cpu.shape[0])
    exempt_idx_dev = exempt_idx_cpu.to(device)
    compressible_idx_dev = compressible_idx_cpu.to(device)

    # Decode tokens for diagnostics
    tokens: list[str] = [
        tokenizer.decode([int(turn_ids[0, i].item())], skip_special_tokens=False)
        for i in range(n_test)
    ]
    exempt_tokens = [tokens[i] for i in exempt_idx_cpu.tolist()]
    exempt_uniq_scores = [
        float(v_uniqueness[i].item()) for i in exempt_idx_cpu.tolist()
    ]
    print(f"  [{label}] num_exempt={num_exempt} of {n_test} ({EXEMPTION_RATE*100:.0f}%)")
    print(f"  [{label}] first 30 exempt tokens (by position):")
    for j, (i, tok, sc) in enumerate(
        zip(exempt_idx_cpu.tolist()[:30], exempt_tokens[:30], exempt_uniq_scores[:30])
    ):
        print(f"    {j+1:>3}. pos {i:>4}  uniq={sc:.3f}  {tok!r}")

    # ---- 3. Build prefix_plus_exempt_kv (shared across all 3 exempt conditions) ----
    exempt_kv = gather_kv(turn_only_kv, exempt_idx_dev)  # (1, n_kv, num_exempt, Dh)
    prefix_plus_exempt_kv = detach_kv(concat_kv(prefix_kv, exempt_kv))
    pe_phys_len = kv_seq_len(prefix_plus_exempt_kv)
    print(f"  [{label}] prefix_plus_exempt_kv: kv_seq_len={pe_phys_len} "
          f"(= prefix_len {prefix_len} + num_exempt {num_exempt})")

    # ---- 4. Print budget table for exempt ratios ----
    print()
    print(f"  [{label}] Budget table:")
    print(f"    {'Ratio':>6} | {'Exempt':>6} | {'Virtual':>7} | "
          f"{'Total':>5} | {'Effective compress ratio':>26}")
    for ratio in COMPRESSION_RATIOS:
        total_budget = max(round(n_test / ratio), 1)
        nv = max(total_budget - num_exempt, 1)
        eff = num_compressible / nv
        print(f"    {ratio:>4.1f}x | {num_exempt:>6} | {nv:>7} | "
              f"{num_exempt + nv:>5} | {eff:>26.2f}x")

    answers: dict[str, str] = {}
    gen_times: dict[str, float] = {}
    optim_meta: dict[str, dict] = {}

    @torch.no_grad()
    def kv_with_full_turn():
        out_local = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out_local.past_key_values))

    @torch.no_grad()
    def kv_with_raw_embeddings():
        out_local = model(
            inputs_embeds=embed(turn_ids),
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out_local.past_key_values))

    def gen_full(kv: tuple) -> str:
        return generate_with_kv(model, tokenizer, question_text, kv, MAX_NEW_TOKENS, stop_ids)

    # ---- 5. Reference conditions ----
    print()
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
    print(f"    raw_embeddings {gen_times['raw_embeddings']:.1f}s: "
          f"{answers['raw_embeddings'][:120]!r}")
    del raw_kv_inf
    torch.cuda.empty_cache()

    print(f"  [{label}] no_turn ...")
    t0 = time.perf_counter()
    answers["no_turn"] = gen_full(prefix_kv)
    gen_times["no_turn"] = time.perf_counter() - t0
    print(f"    no_turn {gen_times['no_turn']:.1f}s: {answers['no_turn'][:120]!r}")

    # ---- 6. Exempt conditions ----
    num_steps = steps_for(n_test)
    for ratio in COMPRESSION_RATIOS:
        cond_label = f"exempt_15pct_{ratio:.1f}x"
        meta = run_exempt_condition(
            model, embed, model_dtype, tokenizer,
            prefix_plus_exempt_kv, pe_phys_len,
            prefix_len, n_test,
            perplexity, v_full, turn_emb,
            exempt_idx_dev, compressible_idx_dev,
            ratio, num_steps, f"{label}/{cond_label}", stop_ids,
            suffix_start_pos, question_text,
        )
        answers[cond_label] = meta["answer"]
        gen_times[cond_label] = meta["gen_time_s"]
        optim_meta[cond_label] = meta

    # ---- 7. uniform_4.8x baseline ----
    cond_label = f"uniform_{UNIFORM_BASELINE_RATIO:.1f}x"
    meta = run_uniform_condition(
        model, embed, model_dtype, tokenizer,
        prefix_kv, prefix_len, turn_ids, n_test,
        perplexity, v_full, turn_emb,
        UNIFORM_BASELINE_RATIO, num_steps, f"{label}/{cond_label}", stop_ids,
        suffix_start_pos, question_text,
    )
    answers[cond_label] = meta["answer"]
    gen_times[cond_label] = meta["gen_time_s"]
    optim_meta[cond_label] = meta

    # ---- 8. Free per-turn allocations ----
    del v_full, turn_only_kv, exempt_kv, prefix_plus_exempt_kv, full_turn_kv_full
    del perplexity, turn_emb
    torch.cuda.empty_cache()

    # ---- 9. Score every answer ----
    cond_order = (
        ["full_turn", "raw_embeddings"]
        + [f"exempt_15pct_{r:.1f}x" for r in COMPRESSION_RATIOS]
        + [f"uniform_{UNIFORM_BASELINE_RATIO:.1f}x", "no_turn"]
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

    return {
        "num_exempt": num_exempt,
        "num_compressible": num_compressible,
        "exempt_positions_local": exempt_idx_cpu.tolist(),
        "exempt_positions_logical": [int(prefix_len + i) for i in exempt_idx_cpu.tolist()],
        "exempt_tokens": exempt_tokens,
        "exempt_uniqueness_scores": exempt_uniq_scores,
        "v_uniqueness_range": [
            float(v_uniqueness.min().item()),
            float(v_uniqueness.max().item()),
        ],
        "conditions": conditions,
        "_cond_order": cond_order,
    }


# -------------------------------- main -------------------------------

def main():
    print("=" * 78)
    print("  Token-level exemption: top-15% by V-uniqueness kept verbatim")
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

    # ---- RoPE + posid checks ----
    rope_verified, rope_finding = verify_rope_for_qwen()
    if rope_verified:
        print(f"[rope_check] CONFIRMED — RoPE on Q,K only. line: {rope_finding}")
    else:
        print(f"[rope_check] WARNING — could not confirm: {rope_finding}")

    print()
    print("Verifying explicit position_ids take effect ...")
    posid_check = verify_position_ids_take_effect(model, prefix_kv, prefix_len, model_dtype)
    print(f"  k_diff_max={posid_check['k_diff_max']:.4f} "
          f"v_diff_max={posid_check['v_diff_max']:.4f} "
          f"took_effect={posid_check['took_effect']}")
    if not posid_check["took_effect"]:
        raise AssertionError(
            f"position_ids ignored: {posid_check} — experiment cannot proceed."
        )

    # ---- Pre-tokenize selected turns ----
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
        print(f"\nSelected: T{ti} ({role}, {len(ids_list)} tokens, "
              f"{spec['content_type']})")

    # ---- Run validation per turn ----
    turns_results: list[dict] = []
    for entry in selected:
        spec = entry["spec"]
        record = entry["record"]
        ti = record["turn_index_1based"]
        score_fn = SCORE_FNS[spec["score_fn_name"]]
        label = f"T{ti}"

        try:
            res = run_turn_validation(
                model, embed, model_dtype, tokenizer,
                prefix_kv, prefix_len,
                record, spec["question"], spec["expected"],
                score_fn, label, stop_ids,
            )
        except Exception as e:
            print(f"[ERROR] T{ti} validation crashed: {e}")
            traceback.print_exc()
            res = {"error": f"{type(e).__name__}: {e}", "conditions": {}, "_cond_order": []}

        turns_results.append({
            "turn_index": ti,
            "role": record["role"],
            "token_count": record["n_tokens"],
            "content_type": spec["content_type"],
            "question": spec["question"],
            "expected": spec["expected"],
            "preview": record["preview"],
            **{k: v for k, v in res.items() if k != "_cond_order"},
            "_cond_order": res.get("_cond_order", []),
        })
        torch.cuda.empty_cache()

    # ---- Save JSON ----
    results = {
        "config": {
            "model_id": MODEL_ID,
            "exemption_rate": EXEMPTION_RATE,
            "exemption_signal": f"v_uniqueness_k{EXEMPTION_K}",
            "compression_ratios": COMPRESSION_RATIOS,
            "uniform_baseline_ratio": UNIFORM_BASELINE_RATIO,
            "recipe": (
                "V-only, perplexity-weighted pooling on compressible tokens, "
                "position interpolation"
            ),
            "lr": LR,
            "steps_formula": "max(round(300 * turn_tokens / 250), 100)",
            "rope_verified": rope_verified,
            "rope_finding": rope_finding,
            "position_ids_take_effect": posid_check,
            "prefix_turn_index": PREFIX_TURN_INDEX + 1,
            "prefix_tokens": n_prefix,
            "stop_token_ids": sorted(stop_ids),
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

    # ---- Console summary: per-turn ----
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
        if "num_exempt" in t:
            print(f"num_exempt={t['num_exempt']} of {t['token_count']}")
        print()
        cond_order = t.get("_cond_order", [])
        for cond in cond_order:
            entry = t.get("conditions", {}).get(cond, {})
            score = entry.get("score", "?/3")
            ans = entry.get("answer", "")
            line = ans.replace("\n", " ")
            if len(line) > 160:
                line = line[:160] + "…"
            extra = ""
            if "mean_pool_loss" in entry:
                extra = (
                    f"  [V={entry.get('num_virtual', '?')}, "
                    f"eff={entry.get('effective_compress_ratio', float('nan')):.2f}x, "
                    f"mp {entry['mean_pool_loss']:.2f} → fin {entry['final_loss']:.2f}, "
                    f"{entry['loss_reduction']*100:.0f}%]"
                )
            print(f"  [{cond:<22}] [{score}] {line}{extra}")

    # ---- Console summary: cross-turn comparison table ----
    print()
    print("=" * 78)
    print("  CROSS-TURN COMPARISON (correct facts out of 3)")
    print("=" * 78)
    print(f"  {'Turn':>4} | "
          f"{'uniform_4.8x':>13} | "
          f"{'exempt_3.2x':>11} | "
          f"{'exempt_4.0x':>11} | "
          f"{'exempt_4.8x':>11} | "
          f"{'Δ vs uniform_4.8x':>18}")
    print("  " + "-" * 90)
    for t in turns_results:
        ti = t["turn_index"]
        conds = t.get("conditions", {})
        u_score = conds.get(f"uniform_{UNIFORM_BASELINE_RATIO:.1f}x", {}).get("score_count")
        e32 = conds.get("exempt_15pct_3.2x", {}).get("score_count")
        e40 = conds.get("exempt_15pct_4.0x", {}).get("score_count")
        e48 = conds.get("exempt_15pct_4.8x", {}).get("score_count")

        def fmt(x):
            return "?/3" if x is None else f"{x}/3"

        if u_score is not None and e48 is not None:
            if e48 > u_score:
                arrow = f"↑ (+{e48 - u_score})"
            elif e48 < u_score:
                arrow = f"↓ ({e48 - u_score})"
            else:
                arrow = "= (0)"
        else:
            arrow = "?"
        print(f"  {ti:>4} | "
              f"{fmt(u_score):>13} | "
              f"{fmt(e32):>11} | "
              f"{fmt(e40):>11} | "
              f"{fmt(e48):>11} | "
              f"{arrow:>18}")

    # ---- Sanity check: raw_embeddings vs full_turn ----
    diverged: list[int] = []
    for t in turns_results:
        f_a = (t.get("conditions", {}).get("full_turn", {}).get("answer") or "").strip()
        r_a = (t.get("conditions", {}).get("raw_embeddings", {}).get("answer") or "").strip()
        if not f_a or not r_a or f_a.startswith("<error") or r_a.startswith("<error"):
            continue
        if f_a != r_a:
            diverged.append(t["turn_index"])
    if diverged:
        print()
        print("!" * 78)
        print(f"  WARNING: raw_embeddings DIVERGED from full_turn on turns: {diverged}")
        print("!" * 78)
    else:
        print()
        print("[sanity] raw_embeddings matches full_turn on all tested turns.")


if __name__ == "__main__":
    main()
