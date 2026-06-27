"""Compression-eviction split grid: at a fixed overall ratio, is there an
optimal split between how much you COMPRESS and how much you EVICT?

Our standing approach (results_eviction_frontier.json) always compresses
2x then evicts to the target. But many splits hit the same overall ratio:
gentler compression leaves richer virtual tokens that may tolerate more
aggressive eviction, while harder compression leaves fewer-but-cruder
tokens with less eviction headroom. This script sweeps the split.

Grid (overall in {3x,4x,5x}; total_middle=3914, prefix protected):
  For each overall ratio, compress at {1.5x,2x,2.5x,3x,3.5x,4x} then
  evict the remainder to the target, plus pure compression at the overall
  ratio and pure eviction at the overall ratio. A "compress Nx -> evict"
  cell whose compressed size already meets the target (compress ratio >=
  overall ratio) has nothing to evict and reuses the no-eviction (pure
  compression) generation.

Sanity checks (no eviction): compressed at {1.5x,2x,2.5x,3x,3.5x,4x} to see where
pure compression starts to degrade, with full_context (ceiling) and
no_middle_truncated (floor).

Error bars: the whole grid runs over 3 generation seeds [42,123,456].
Compression is deterministic given its input, so to vary across seeds we
add small gaussian noise (N(0, 0.01), seeded by seed+turn) to the virtual
token initialization, giving the optimizer different starting points.
Conditions with no compression (pure_eviction, full_context, truncated)
are deterministic under greedy generation; they are still run once per
seed (expect std=0).

Eviction signal: pre-state attention (our best signal), computed ONCE on
the raw uncompressed cache and reused everywhere. For compressed caches
the raw importance is aggregated raw->virtual via the exact pooling chunk
boundaries; for pure eviction the raw scores are ranked directly. The
ranking is seed-independent (chunk boundaries depend only on the
compression ratio); only the cache contents being evicted differ by seed.

SCOPE: 6 compression ratios x 3 seeds = 18 sequential optimization runs.
Pure compression is reported for overall 3x and 4x (those caches are
built anyway); the 5x pure-compression cell stays N/A (no 5x compression
run). Plan on a multi-hour wall-clock budget.

Eight questions; keyword scorers calibrated full_context=8/8,
truncated=0/8 (Q3 negation-aware). Standalone -- no imports from other
project files. No K loss. No uv.
"""

from __future__ import annotations

import inspect
import json
import statistics
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------- constants -----------------------------

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
TRANSCRIPT_PATH = Path(__file__).parent / "transcript.jsonl"
RESULTS_PATH = Path(__file__).parent / "results_compression_eviction_grid.json"

TOKEN_BUDGET = 4096
NUM_PREFIX = 1
NUM_SUFFIX = 1

COMPRESS_RATIOS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # compression arms (also the sanity checks)
OVERALL_RATIOS = [3.0, 4.0, 5.0]         # overall target ratios
SEEDS = [42, 123, 456]
INIT_NOISE_STD = 0.01                    # gaussian noise on virtual-token init, per seed
# Pure compression at an overall ratio is populated only when that ratio
# is also a compression arm (we already build the cache for it): 3x and
# 4x. 5x pure compression would need a 5x compression run, which is not
# in COMPRESS_RATIOS, so the 5x pure-compression cell stays N/A.
PURE_COMPRESSION_RATIOS = {3.0, 4.0}

LR = 0.003
BASE_STEPS = 300
BASE_TOKENS = 250
MIN_STEPS = 100
LOG_EVERY = 50
MAX_NEW_TOKENS = 256

QUESTIONS = [
    "What type of application is being built in this project?",
    "What database tables were created, and what are their key columns?",
    "What does the markdown-to-HTML converter handle? List the supported elements.",
    "How does the authentication system work? What library is used for password hashing?",
    "What validation rules does the task validation middleware enforce?",
    "Which modules depend on the database module, and how do they use it?",
    "What was the last thing the agent worked on or verified?",
    "List all the source files that were created during this session.",
]

EXTRA_STOP_TOKEN_STRINGS = ["<|im_end|>", "<|endoftext|>"]


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


def gather_kv(legacy: tuple, indices: torch.Tensor) -> tuple:
    return tuple(
        (k.index_select(2, indices), v.index_select(2, indices))
        for k, v in legacy
    )


def evict_kv(legacy: tuple, keep_indices: torch.Tensor) -> tuple:
    return detach_kv(gather_kv(legacy, keep_indices))


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


def chunk_boundaries(s: int, n_chunks: int) -> list[tuple[int, int]]:
    """Replicate attention_weighted_pool's chunk spans exactly, so
    pre-state importance can be aggregated per virtual token."""
    b = torch.linspace(0, s, n_chunks + 1).long().tolist()
    spans = []
    for i in range(n_chunks):
        a, e = b[i], b[i + 1]
        if e <= a:
            e = a + 1
        spans.append((a, e))
    return spans


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


def verify_position_ids_take_effect(model, prefix_kv, prefix_len, model_dtype) -> dict:
    device = model.device
    n = 5
    hidden = model.config.hidden_size
    rand_emb = torch.randn(1, n, hidden, dtype=model_dtype, device=device)
    pos_a = torch.arange(100, 100 + n, device=device, dtype=torch.long).unsqueeze(0)
    pos_b = torch.arange(300, 300 + n, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out_a = model(inputs_embeds=rand_emb, past_key_values=wrap_legacy_kv(prefix_kv, model.config),
                      position_ids=pos_a, use_cache=True)
        out_b = model(inputs_embeds=rand_emb, past_key_values=wrap_legacy_kv(prefix_kv, model.config),
                      position_ids=pos_b, use_cache=True)
    live_a = extract_new_kv_grad_safe(out_a.past_key_values, prefix_len)
    live_b = extract_new_kv_grad_safe(out_b.past_key_values, prefix_len)
    k_diff_max = float((live_a[0][0].float() - live_b[0][0].float()).abs().max().item())
    v_diff_max = float((live_a[0][1].float() - live_b[0][1].float()).abs().max().item())
    return {"k_diff_max": k_diff_max, "v_diff_max": v_diff_max,
            "took_effect": (k_diff_max > 1e-3) and (v_diff_max < 1e-1)}


def compute_perplexity_weights(logits: torch.Tensor, turn_ids: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] != turn_ids.shape[1]:
        raise ValueError(f"logits seq {logits.shape[1]} != turn_ids seq {turn_ids.shape[1]}")
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
        for i in tokenizer.encode(tok, add_special_tokens=False):
            stops.add(int(i))
    return stops


# ---------------------- attention implementation toggle --------------

def set_attn_impl(model, impl: str) -> str:
    prev = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = impl
    for module in model.modules():
        cfg = getattr(module, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = impl
    return prev


@torch.no_grad()
def compute_attention_importance(
    model, cache_legacy: tuple, suffix_ids: torch.Tensor,
    suffix_start_pos: int, cache_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Suffix-attention importance per cache position. Eager required for
    output_attentions; restored afterward. Per layer: sum / max over
    heads and suffix queries onto each cache column, L1-normalize per
    layer, accumulate. Returns (importance_sum, importance_max) on CPU.
    """
    device = model.device
    prev_impl = set_attn_impl(model, "eager")
    try:
        n_q = int(suffix_ids.shape[1])
        pos = torch.arange(suffix_start_pos, suffix_start_pos + n_q, device=device, dtype=torch.long).unsqueeze(0)
        out = model(
            input_ids=suffix_ids.to(device),
            past_key_values=wrap_legacy_kv(cache_legacy, model.config),
            position_ids=pos, use_cache=True, output_attentions=True,
        )
        atts = out.attentions
        if atts is None or atts[0] is None:
            raise RuntimeError("output_attentions returned None even under eager attention.")
        imp_sum = torch.zeros(cache_len, dtype=torch.float32, device=device)
        imp_max = torch.zeros(cache_len, dtype=torch.float32, device=device)
        for layer_att in atts:
            a = layer_att[0, :, :, :cache_len].float()  # (heads, q, cache_len)
            contrib = a.sum(dim=(0, 1))
            imp_sum += contrib / contrib.sum().clamp(min=1e-8)
            mx = a.amax(dim=(0, 1))
            imp_max += mx / mx.sum().clamp(min=1e-8)
        del out, atts
    finally:
        set_attn_impl(model, prev_impl)
    torch.cuda.empty_cache()
    return imp_sum.detach().cpu(), imp_max.detach().cpu()


# ----------------------------- generation ----------------------------

@torch.no_grad()
def manual_greedy_with_positions(model, tokenizer, prompt_text, past_kv_legacy, start_pos, max_new_tokens, stop_ids):
    device = model.device
    q_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    n_q = int(q_ids.shape[1])
    cache = wrap_legacy_kv(past_kv_legacy, model.config)
    pos_ids = torch.arange(start_pos, start_pos + n_q, device=device, dtype=torch.long).unsqueeze(0)
    out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos_ids, use_cache=True)
    cache = out.past_key_values
    next_pos = start_pos + n_q
    next_id = int(out.logits[0, -1, :].argmax().item())
    generated = [next_id]
    if next_id in stop_ids:
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    for _ in range(max_new_tokens - 1):
        next_inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        pos_ids = torch.tensor([[next_pos]], dtype=torch.long, device=device)
        out = model(input_ids=next_inp, past_key_values=cache, position_ids=pos_ids, use_cache=True)
        cache = out.past_key_values
        next_pos += 1
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated.append(next_id)
        if next_id in stop_ids:
            break
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def manual_greedy_auto(model, tokenizer, prompt_text, past_kv_legacy, max_new_tokens, stop_ids):
    device = model.device
    q_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
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
def generate_with_kv(model, tokenizer, prompt_text, past_kv_legacy, max_new_tokens, stop_ids):
    device = model.device
    q_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    try:
        out = model.generate(
            input_ids=q_ids, past_key_values=wrap_legacy_kv(past_kv_legacy, model.config),
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id, eos_token_id=list(stop_ids),
        )
        return tokenizer.decode(out[0, int(q_ids.shape[1]):], skip_special_tokens=True).strip()
    except Exception:
        return manual_greedy_auto(model, tokenizer, prompt_text, past_kv_legacy, max_new_tokens, stop_ids)


def run_questions_full(model, tokenizer, kv_legacy, label, stop_ids):
    answers, total = {}, 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        t0 = time.perf_counter()
        try:
            ans = generate_with_kv(model, tokenizer, f"\n\nQuestion: {q}\nAnswer:", kv_legacy, MAX_NEW_TOKENS, stop_ids)
        except Exception as e:
            ans = f"<error: {type(e).__name__}: {e}>"
            traceback.print_exc()
        total += time.perf_counter() - t0
        answers[f"Q{i}"] = ans
        print(f"  [{label}] Q{i}: {ans[:110]!r}")
    return answers, total


def run_questions_truncated(model, tokenizer, kv_legacy, suffix_text, label, stop_ids):
    answers, total = {}, 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        t0 = time.perf_counter()
        try:
            ans = generate_with_kv(model, tokenizer, suffix_text + f"\n\nQuestion: {q}\nAnswer:",
                                   kv_legacy, MAX_NEW_TOKENS, stop_ids)
        except Exception as e:
            ans = f"<error: {type(e).__name__}: {e}>"
            traceback.print_exc()
        total += time.perf_counter() - t0
        answers[f"Q{i}"] = ans
        print(f"  [{label}] Q{i}: {ans[:110]!r}")
    return answers, total


def run_questions_with_positions(model, tokenizer, kv_legacy, suffix_text, suffix_start_pos, label, stop_ids):
    answers, total = {}, 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        t0 = time.perf_counter()
        try:
            ans = manual_greedy_with_positions(
                model, tokenizer, suffix_text + f"\n\nQuestion: {q}\nAnswer:", kv_legacy,
                start_pos=suffix_start_pos, max_new_tokens=MAX_NEW_TOKENS, stop_ids=stop_ids)
        except Exception as e:
            ans = f"<error: {type(e).__name__}: {e}>"
            traceback.print_exc()
        total += time.perf_counter() - t0
        answers[f"Q{i}"] = ans
        print(f"  [{label}] Q{i}: {ans[:110]!r}")
    return answers, total


# --------------------------- score functions -------------------------
# Calibrated: full_context=8/8, truncated=0/8. Q3 is negation-aware.

def score_q1(ans: str) -> bool:
    a = ans.lower()
    return ("express" in a) and ("sqlite" in a) and ("task" in a)


def score_q2(ans: str) -> bool:
    a = ans.lower()
    return ("users" in a) and ("tasks" in a) and ("reports" in a)


def score_q3(ans: str) -> bool:
    a = ans.lower()
    has_core = ("bold" in a) and ("italic" in a) and ("code" in a)
    mentions_extra = ("image" in a) or ("link" in a)
    negated = (
        "does not" in a or "doesn't" in a or "do not" in a or "not handle" in a
        or "no images" in a or "no image" in a or "no links" in a or "without" in a
        or "does not support" in a or "doesn't support" in a or "not support" in a
    )
    return has_core and not (mentions_extra and not negated)


def score_q4(ans: str) -> bool:
    a = ans.lower()
    return ("jwt" in a) and ("bcryptjs" in a)


def score_q5(ans: str) -> bool:
    a = ans.lower()
    return ("title" in a and "pending" in a
            and ("in_progress" in a or "in progress" in a) and "done" in a)


def score_q6(ans: str) -> bool:
    a = ans.lower()
    return ("auth" in a) and ("task" in a) and ("report" in a)


def score_q7(ans: str) -> bool:
    return "db.js" in ans.lower()


def score_q8(ans: str) -> bool:
    a = ans.lower()
    return ("db.js" in a and "markdown.js" in a and "validate" in a
            and "auth.js" in a and "reports.js" in a)


SCORE_FNS: list[Callable[[str], bool]] = [
    score_q1, score_q2, score_q3, score_q4, score_q5, score_q6, score_q7, score_q8,
]


def score_answers(answers: dict[str, str]) -> tuple[dict[str, int], int]:
    scores, total = {}, 0
    for i, fn in enumerate(SCORE_FNS, start=1):
        ans = answers.get(f"Q{i}", "") or ""
        ok = 0
        if ans and not ans.startswith("<error"):
            try:
                ok = int(bool(fn(ans)))
            except Exception:
                ok = 0
        scores[f"Q{i}"] = ok
        total += ok
    return scores, total


# ----------------------- transcript loading --------------------------

def load_and_split(path: Path, tokenizer, budget: int) -> dict:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    raw_turns = [json.loads(line) for line in raw_lines if line.strip()]
    per_turn: list[dict] = []
    for i, turn in enumerate(raw_turns, start=1):
        role = turn.get("role", "?")
        content = _stringify_content(turn.get("content", ""))
        text = turn_marker(i, role, content)
        ids_list = tokenizer(text, add_special_tokens=False).input_ids
        per_turn.append({
            "turn_index_1based": i, "role": role, "text": text,
            "ids_list": ids_list, "n_tokens": len(ids_list), "preview": text[:200],
        })
    included, running = [], 0
    for rec in per_turn:
        remaining = budget - running
        if remaining <= 0:
            break
        if rec["n_tokens"] <= remaining:
            included.append(rec)
            running += rec["n_tokens"]
        else:
            tids = rec["ids_list"][:remaining]
            ttext = tokenizer.decode(tids, skip_special_tokens=False)
            included.append({
                "turn_index_1based": rec["turn_index_1based"], "role": rec["role"],
                "text": ttext, "ids_list": tids, "n_tokens": len(tids),
                "preview": ttext[:200] + " [...truncated]",
            })
            running += len(tids)
            break
    n_inc = len(included)
    if n_inc < NUM_PREFIX + NUM_SUFFIX + 1:
        raise ValueError(f"Budget {budget} only fits {n_inc} turns.")
    prefix_records = included[:NUM_PREFIX]
    suffix_records = included[-NUM_SUFFIX:]
    middle_records = included[NUM_PREFIX:n_inc - NUM_SUFFIX]
    if len(middle_records) == 0:
        raise ValueError("No middle turns left after prefix/suffix split.")
    prefix_n = sum(r["n_tokens"] for r in prefix_records)
    pos = prefix_n
    for r in middle_records:
        r["start_pos"] = pos
        r["end_pos"] = pos + r["n_tokens"] - 1
        pos += r["n_tokens"]
    suffix_start_pos = pos
    sp = suffix_start_pos
    for r in suffix_records:
        r["start_pos"] = sp
        r["end_pos"] = sp + r["n_tokens"] - 1
        sp += r["n_tokens"]
    for r in prefix_records + middle_records + suffix_records:
        r["ids"] = torch.tensor([r["ids_list"]], dtype=torch.long)
    return {
        "prefix_records": prefix_records, "middle_records": middle_records,
        "suffix_records": suffix_records, "prefix_len": prefix_n,
        "suffix_len": sum(r["n_tokens"] for r in suffix_records),
        "total_middle_tokens": sum(r["n_tokens"] for r in middle_records),
        "suffix_start_pos": suffix_start_pos,
    }


def print_dry_run(split: dict) -> None:
    print("\n" + "=" * 78 + "\n  Context split\n" + "=" * 78)
    print(f"PREFIX: {split['prefix_len']} tok  MIDDLE: {len(split['middle_records'])} turns, "
          f"{split['total_middle_tokens']} tok  SUFFIX: {split['suffix_len']} tok  "
          f"suffix_start_pos={split['suffix_start_pos']}")
    for r in split["middle_records"]:
        print(f"  turn {r['turn_index_1based']:>3} ({r['role']:<10}): {r['n_tokens']:>4} tok  "
              f"pos=[{r['start_pos']}, {r['end_pos']}]")


@torch.no_grad()
def build_full_context_kv(model, prefix_records, middle_records, suffix_records) -> tuple:
    device = model.device
    all_ids = torch.cat([r["ids"].to(device) for r in [*prefix_records, *middle_records, *suffix_records]], dim=1)
    out = model(input_ids=all_ids, use_cache=True)
    return detach_kv(to_legacy_kv(out.past_key_values))


@torch.no_grad()
def build_raw_middle_kv(model, prefix_records, middle_records) -> tuple:
    """Prefix + all middle tokens verbatim (no suffix) at original
    positions -> the uncompressed pre-state cache."""
    device = model.device
    all_ids = torch.cat([r["ids"].to(device) for r in [*prefix_records, *middle_records]], dim=1)
    out = model(input_ids=all_ids, use_cache=True)
    return detach_kv(to_legacy_kv(out.past_key_values))


# ----------------------- sequential compression ----------------------

def steps_for(n_tokens: int) -> int:
    return max(round(BASE_STEPS * n_tokens / BASE_TOKENS), MIN_STEPS)


def run_optimize_loop(model, model_dtype, past_kv_legacy, slice_offset, init, virtual_pos,
                      target_v_list, v_mag2_list, num_steps, label):
    device = model.device
    virtual = init.detach().clone().to(dtype=torch.float32, device=device)
    virtual.requires_grad_(True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    optimizer = torch.optim.Adam([virtual], lr=LR)
    losses: list[float] = []
    try:
        for step in range(num_steps):
            inputs_embeds = virtual.to(dtype=model_dtype).unsqueeze(0)
            out = model(inputs_embeds=inputs_embeds, past_key_values=wrap_legacy_kv(past_kv_legacy, model.config),
                        position_ids=virtual_pos, output_hidden_states=False, use_cache=True)
            live = extract_new_kv_grad_safe(out.past_key_values, slice_offset)
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for (_k, v_live), v_tgt, v_mag2 in zip(live, target_v_list, v_mag2_list):
                loss = loss + F.mse_loss(v_live.float(), v_tgt.float()) / v_mag2.clamp(min=1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0 and virtual.grad is None:
                raise AssertionError(f"[{label}] gradient did not flow through KV extraction.")
            optimizer.step()
            losses.append(float(loss.detach().item()))
            if step in (0, num_steps // 2, num_steps - 1):
                print(f"    [{label}] step {step:>4}/{num_steps}: loss={losses[-1]:.4f}  {vram_str()}")
            elif (step + 1) % LOG_EVERY == 0:
                print(f"    [{label}] step {step:>4}/{num_steps}: loss={losses[-1]:.4f}")
            del out, loss, inputs_embeds
    finally:
        model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return virtual.detach(), losses


def seeded_init(turn_emb: torch.Tensor, num_virtual: int, seed: int, turn_index: int) -> torch.Tensor:
    """Mean-pool init plus seed+turn dependent gaussian noise so the
    optimizer starts from a different point each seed. The noise is drawn
    on CPU from a deterministically seeded generator (device-independent)
    and added in the init's dtype. seed=None / std<=0 leaves init
    unperturbed (the original deterministic behaviour)."""
    init = mean_pool_chunks(turn_emb, num_virtual).detach().clone()
    if seed is None or INIT_NOISE_STD <= 0:
        return init
    g = torch.Generator().manual_seed(int(seed) * 100003 + int(turn_index))
    noise = INIT_NOISE_STD * torch.randn(init.shape, generator=g, dtype=torch.float32)
    return (init.float() + noise.to(init.device)).to(init.dtype)


def compress_turn_uniform(model, embed, model_dtype, running_kv, turn_record, ratio, seed, label):
    device = model.device
    turn_ids = turn_record["ids"].to(device)
    n_tokens = int(turn_record["n_tokens"])
    turn_start_pos = int(turn_record["start_pos"])
    turn_end_pos = int(turn_record["end_pos"])
    turn_index = int(turn_record["turn_index_1based"])
    phys_off = kv_seq_len(running_kv)
    t_start = time.perf_counter()
    target_pos = torch.arange(turn_start_pos, turn_start_pos + n_tokens, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        kv_out = model(input_ids=turn_ids, past_key_values=wrap_legacy_kv(running_kv, model.config),
                       position_ids=target_pos, output_hidden_states=False, use_cache=True)
    perplexity = compute_perplexity_weights(kv_out.logits, turn_ids)
    full_kv = detach_kv(to_legacy_kv(kv_out.past_key_values))
    del kv_out
    v_full = [v[:, :, phys_off:, :] for (_k, v) in full_kv]
    num_virtual = max(round(n_tokens / ratio), 1)
    num_virtual = min(num_virtual, n_tokens)
    target_v_list, v_mag2_list = [], []
    for v_layer in v_full:
        v_pooled = attention_weighted_pool(v_layer, perplexity, num_virtual).detach().clone()
        target_v_list.append(v_pooled)
        v_mag2_list.append(v_pooled.float().pow(2).mean().detach())
    with torch.no_grad():
        turn_emb = embed(turn_ids).squeeze(0)
    init = seeded_init(turn_emb, num_virtual, seed, turn_index)
    virtual_pos = interpolate_positions(turn_start_pos, turn_end_pos, num_virtual, device=device)
    num_steps = steps_for(n_tokens)
    print(f"  [{label}] turn {turn_index}: N={n_tokens} -> V={num_virtual}, steps={num_steps}, seed={seed}")
    virtual, losses = run_optimize_loop(model, model_dtype, running_kv, slice_offset=phys_off, init=init,
                                        virtual_pos=virtual_pos, target_v_list=target_v_list,
                                        v_mag2_list=v_mag2_list, num_steps=num_steps, label=label)
    with torch.no_grad():
        ie = virtual.to(dtype=model_dtype).unsqueeze(0)
        out = model(inputs_embeds=ie, past_key_values=wrap_legacy_kv(running_kv, model.config),
                    position_ids=virtual_pos, use_cache=True)
        new_running_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out, ie
    meta = {"turn_index": turn_index, "role": turn_record["role"],
            "n_tokens": n_tokens, "num_virtual": num_virtual,
            "initial_loss": losses[0], "final_loss": losses[-1], "time_s": time.perf_counter() - t_start}
    del target_v_list, v_mag2_list, init, virtual_pos, virtual, perplexity, v_full, full_kv, turn_emb
    torch.cuda.empty_cache()
    return new_running_kv, meta


def run_sequential(model, embed, model_dtype, prefix_kv, middle_records, ratio, seed, label):
    """Returns (running_kv, per_turn_meta, virtual_blocks, raw_blocks)."""
    print("\n" + "=" * 78 + f"\n  Sequential compression: {label} (ratio={ratio}, seed={seed})\n" + "=" * 78)
    running_kv = prefix_kv
    per_turn_meta, virtual_blocks, raw_blocks = [], [], []
    v_off, r_off = 0, 0
    for r in middle_records:
        running_kv, meta = compress_turn_uniform(model, embed, model_dtype, running_kv, r, ratio, seed,
                                                 label=f"{label}/turn{r['turn_index_1based']}")
        nv, nt = meta["num_virtual"], meta["n_tokens"]
        virtual_blocks.append((meta["turn_index"], v_off, v_off + nv))
        raw_blocks.append((meta["turn_index"], r_off, r_off + nt))
        meta["v_local_start"], meta["v_local_end"] = v_off, v_off + nv
        meta["r_local_start"], meta["r_local_end"] = r_off, r_off + nt
        v_off += nv
        r_off += nt
        per_turn_meta.append(meta)
    return running_kv, per_turn_meta, virtual_blocks, raw_blocks


# ----------------------- eviction selection --------------------------

def select_topk_keep(importance_middle: torch.Tensor, n_keep: int) -> list[int]:
    n = int(importance_middle.shape[0])
    n_keep = max(1, min(n_keep, n))
    idx = torch.topk(importance_middle, n_keep).indices
    return sorted(int(i) for i in idx.tolist())


def per_turn_survival(blocks_local, keep_set: set[int]) -> list[dict]:
    rows = []
    for (ti, a, b) in blocks_local:
        total = b - a
        kept = sum(1 for j in range(a, b) if j in keep_set)
        rows.append({"turn_index": ti, "positions_total": total,
                     "positions_kept": kept, "positions_evicted": total - kept})
    return rows


def aggregate_raw_to_virtual(raw_imp_middle: torch.Tensor, per_turn_meta: list[dict]) -> torch.Tensor:
    total_virtual = sum(m["num_virtual"] for m in per_turn_meta)
    virt = torch.zeros(total_virtual, dtype=torch.float32)
    for m in per_turn_meta:
        s, nv = m["n_tokens"], m["num_virtual"]
        r0, v0 = m["r_local_start"], m["v_local_start"]
        for i, (a, b) in enumerate(chunk_boundaries(s, nv)):
            virt[v0 + i] = raw_imp_middle[r0 + a: r0 + b].sum()
    return virt


def dist_stats(t: torch.Tensor) -> dict:
    tf = t.float()
    return {"min": float(tf.min().item()), "max": float(tf.max().item()),
            "mean": float(tf.mean().item()), "median": float(tf.median().item())}


def ratio_key(r: float) -> str:
    return f"{int(r)}x" if r == int(r) else f"{r:.1f}x"


def target_keep(total_middle: int, overall_ratio: float) -> int:
    # round-half-up so the exact .5 case matches the planned budget.
    return max(1, int(total_middle / overall_ratio + 0.5))


def mean_std(totals: list[float]) -> tuple[float, float]:
    if not totals:
        return float("nan"), float("nan")
    m = sum(totals) / len(totals)
    s = statistics.stdev(totals) if len(totals) > 1 else 0.0
    return m, s


# -------------------------------- main -------------------------------

def main():
    print("=" * 78 + "\n  Compression-eviction split grid (3 seeds)\n" + "=" * 78)

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    model.eval()
    device = model.device
    embed = model.get_input_embeddings()
    model_dtype = embed.weight.dtype
    print(f"Model loaded. dtype={model_dtype}. {vram_str()}")

    stop_ids = get_stop_ids(tokenizer)
    split = load_and_split(TRANSCRIPT_PATH, tokenizer, TOKEN_BUDGET)
    print_dry_run(split)

    prefix_records = split["prefix_records"]
    middle_records = split["middle_records"]
    suffix_records = split["suffix_records"]
    prefix_len = split["prefix_len"]
    suffix_start_pos = split["suffix_start_pos"]
    total_middle = split["total_middle_tokens"]
    suffix_text = "".join(r["text"] for r in suffix_records)
    suffix_ids = suffix_records[0]["ids"]

    with torch.no_grad():
        out = model(input_ids=prefix_records[0]["ids"].to(device), use_cache=True)
    prefix_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out
    torch.cuda.empty_cache()

    rope_verified, rope_finding = verify_rope_for_qwen()
    print(f"[rope_check] verified={rope_verified}: {rope_finding}")
    posid = verify_position_ids_take_effect(model, prefix_kv, prefix_len, model_dtype)
    print(f"[posid_check] {posid}")
    if not posid["took_effect"]:
        raise AssertionError(f"position_ids ignored: {posid}")

    targets = {ratio_key(r): target_keep(total_middle, r) for r in OVERALL_RATIOS}
    print("\nTargets (middle entries kept) per overall ratio: "
          + ", ".join(f"{k}->{v}" for k, v in targets.items()))

    def gen_pos(label, kv):
        return run_questions_with_positions(model, tokenizer, kv, suffix_text, suffix_start_pos, label, stop_ids)

    def score_pos(label, kv):
        ans, _t = gen_pos(label, kv)
        sc, tot = score_answers(ans)
        return {"answers": ans, "scores": sc, "total": tot}

    def evict_and_score(label, source_kv, keep_local, source_size):
        keep_global = list(range(prefix_len)) + [prefix_len + j for j in keep_local]
        keep_t = torch.tensor(sorted(keep_global), dtype=torch.long, device=device)
        ev = evict_kv(source_kv, keep_t)
        res = score_pos(label, ev)
        del ev
        torch.cuda.empty_cache()
        return res

    # ---- full_context cache (ceiling) + raw pre-state cache (once) ----
    print("\n=== full_context cache (once) ===")
    full_kv = build_full_context_kv(model, prefix_records, middle_records, suffix_records)

    print("\n=== raw pre-state cache + attention importance (once) ===")
    raw_kv = build_raw_middle_kv(model, prefix_records, middle_records)
    raw_cache_len = kv_seq_len(raw_kv)
    print(f"  raw cache len={raw_cache_len} (prefix {prefix_len} + middle {total_middle}). {vram_str()}")
    raw_imp_sum, _raw_imp_max = compute_attention_importance(
        model, raw_kv, suffix_ids, suffix_start_pos, raw_cache_len)
    raw_mid = raw_imp_sum[prefix_len:raw_cache_len].clone()   # (total_middle,)
    print(f"  raw middle importance: {dist_stats(raw_mid)}")

    # ---- result containers ----
    grid_cells = [f"compress_{ratio_key(r)}_evict" for r in COMPRESS_RATIOS] + ["pure_compression", "pure_eviction"]
    grid: dict[str, dict[str, dict]] = {
        ratio_key(o): {cell: {"seeds": {}} for cell in grid_cells} for o in OVERALL_RATIOS
    }
    sanity_keys = [f"compressed_{ratio_key(r)}" for r in COMPRESS_RATIOS] + ["full_context", "truncated"]
    sanity: dict[str, dict] = {k: {"seeds": {}} for k in sanity_keys}
    compression_meta: dict[str, dict] = {}  # f"{seed}:{ratio_key}" -> per-turn meta summary

    # ---- per-seed sweep ----
    for seed in SEEDS:
        s = str(seed)
        print("\n" + "#" * 78 + f"\n  SEED {seed}\n" + "#" * 78)

        # deterministic conditions (run per seed for empirical determinism check)
        print(f"\n=== [seed {seed}] full_context ===")
        fc_ans, _ = run_questions_full(model, tokenizer, full_kv, f"full_context[seed={seed}]", stop_ids)
        fc_sc, fc_tot = score_answers(fc_ans)
        sanity["full_context"]["seeds"][s] = {"answers": fc_ans, "scores": fc_sc, "total": fc_tot}

        print(f"\n=== [seed {seed}] no_middle_truncated ===")
        tr_ans, _ = run_questions_truncated(model, tokenizer, prefix_kv, suffix_text, f"truncated[seed={seed}]", stop_ids)
        tr_sc, tr_tot = score_answers(tr_ans)
        sanity["truncated"]["seeds"][s] = {"answers": tr_ans, "scores": tr_sc, "total": tr_tot}

        # pure eviction at each overall ratio (raw cache, raw signal; seed-independent)
        for o in OVERALL_RATIOS:
            ok = ratio_key(o)
            tgt = targets[ok]
            print(f"\n=== [seed {seed}] pure_eviction {ok} (keep {tgt}/{total_middle}) ===")
            keep_raw = select_topk_keep(raw_mid, tgt)
            res = evict_and_score(f"pure_evict_{ok}[seed={seed}]", raw_kv, keep_raw, total_middle)
            res["middle_kept"] = len(keep_raw)
            res["middle_evicted"] = total_middle - len(keep_raw)
            grid[ok]["pure_eviction"]["seeds"][s] = res

        # compression arms
        for r in COMPRESS_RATIOS:
            rk = ratio_key(r)
            kv, meta, vblocks, _rblocks = run_sequential(
                model, embed, model_dtype, prefix_kv, middle_records, r, seed, f"compress_{rk}")
            mc = sum(m["num_virtual"] for m in meta)
            print(f"  compressed_{rk}: middle={mc}, cache={kv_seq_len(kv)}")
            compression_meta[f"{seed}:{rk}"] = {
                "compressed_entries": mc,
                "per_turn": [{"turn_index": m["turn_index"], "n_tokens": m["n_tokens"],
                              "num_virtual": m["num_virtual"], "final_loss": m["final_loss"]} for m in meta],
            }
            vblocks_local = [(ti, a, b) for (ti, a, b) in vblocks]

            # sanity: pure compression (no eviction)
            noevict = score_pos(f"compressed_{rk}[seed={seed}]", kv)
            noevict["compressed_entries"] = mc
            sanity[f"compressed_{rk}"]["seeds"][s] = noevict
            if r in PURE_COMPRESSION_RATIOS:
                grid[rk]["pure_compression"]["seeds"][s] = dict(noevict)

            # pre-state importance mapped to virtual tokens (seed-independent for this ratio)
            prestate_imp = aggregate_raw_to_virtual(raw_mid, meta)

            # grid eviction cells: this ratio as a compress arm at each overall ratio
            cell = f"compress_{rk}_evict"
            for o in OVERALL_RATIOS:
                ok = ratio_key(o)
                tgt = targets[ok]
                if mc > tgt:
                    keep = select_topk_keep(prestate_imp, tgt)
                    res = evict_and_score(f"{cell}@{ok}[seed={seed}]", kv, keep, mc)
                    res["middle_kept"] = len(keep)
                    res["middle_evicted"] = mc - len(keep)
                    res["per_turn_survival"] = per_turn_survival(vblocks_local, set(keep))
                    grid[ok][cell]["seeds"][s] = res
                else:
                    # mc <= tgt: compression alone already meets the overall
                    # ratio, so there is nothing to evict -> this cell is
                    # pure compression at the compress ratio.
                    res = dict(noevict)
                    res["middle_kept"] = mc
                    res["middle_evicted"] = 0
                    res["degenerate"] = (
                        f"no eviction: compressed {mc} <= target {tgt} "
                        f"(achieved ratio {total_middle / mc:.3f}x)")
                    grid[ok][cell]["seeds"][s] = res

            del kv
            torch.cuda.empty_cache()

    del full_kv, raw_kv
    torch.cuda.empty_cache()

    # ---- pure compression 4x/5x intentionally out of scope ----
    for o in OVERALL_RATIOS:
        if o not in PURE_COMPRESSION_RATIOS:
            grid[ratio_key(o)]["pure_compression"] = {
                "status": "N/A",
                "reason": f"pure compression at {ratio_key(o)} not run (scope: only {sorted(PURE_COMPRESSION_RATIOS)})",
            }

    # ---- aggregate mean/std over seeds ----
    def totals_of(seedmap: dict) -> list[float]:
        out = []
        for sd in (str(x) for x in SEEDS):
            entry = seedmap.get(sd)
            if entry and "total" in entry:
                out.append(float(entry["total"]))
        return out

    for ok in grid:
        for cell, payload in grid[ok].items():
            if "seeds" not in payload:
                continue
            tots = totals_of(payload["seeds"])
            m, sd = mean_std(tots)
            payload["mean"] = m
            payload["std"] = sd
            payload["per_seed_total"] = {k: payload["seeds"][k].get("total") for k in payload["seeds"]}
    for k, payload in sanity.items():
        tots = totals_of(payload["seeds"])
        m, sd = mean_std(tots)
        payload["mean"] = m
        payload["std"] = sd
        payload["per_seed_total"] = {sk: payload["seeds"][sk].get("total") for sk in payload["seeds"]}

    # ---- write JSON ----
    results = {
        "config": {
            "model_id": MODEL_ID,
            "compress_ratios": COMPRESS_RATIOS,
            "overall_ratios": OVERALL_RATIOS,
            "pure_compression_ratios": sorted(PURE_COMPRESSION_RATIOS),
            "seeds": SEEDS,
            "init_noise_std": INIT_NOISE_STD,
            "importance_signal": "prestate_attention",
            "ratio_convention": "overall = total_middle_tokens / middle_positions_kept (prefix excluded)",
            "prefix_protected": prefix_len,
            "total_middle_tokens": total_middle,
            "raw_cache_size": raw_cache_len,
            "targets_kept": targets,
            "suffix_start_pos": suffix_start_pos,
            "lr": LR,
            "rope_verified": rope_verified,
            "position_ids_take_effect": posid,
            "scoring": "keyword, calibrated full=8/8 trunc=0/8 (Q3 negation-aware)",
            "scope_note": "pure compression reported for 3x/4x; 5x N/A; deterministic conds run 3x (expect std=0)",
        },
        "attention_scores": {"raw_middle": dist_stats(raw_mid)},
        "compression_meta": compression_meta,
        "grid": grid,
        "sanity_checks": sanity,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")

    # ---- console: per-overall split tables ----
    def fmt_seed_cell(payload: dict, sd: str) -> str:
        entry = payload.get("seeds", {}).get(sd)
        if not entry or "total" not in entry:
            return "  -  "
        return f"{entry['total']}/8 "

    def fmt_meanstd(payload: dict) -> str:
        if "mean" not in payload or payload.get("mean") != payload.get("mean"):  # nan check
            return "   N/A   "
        return f"{payload['mean']:.1f} ± {payload['std']:.1f}"

    row_labels = {}
    for r in COMPRESS_RATIOS:
        row_labels[f"compress_{ratio_key(r)}_evict"] = f"compress {ratio_key(r)} → evict"
    row_labels["pure_compression"] = "pure compression"
    row_labels["pure_eviction"] = "pure eviction"

    seed_strs = [str(x) for x in SEEDS]
    print("\n" + "=" * 78 + "\n  COMPRESSION-EVICTION SPLIT GRID (correct out of 8)\n" + "=" * 78)
    for o in OVERALL_RATIOS:
        ok = ratio_key(o)
        print(f"\nOverall {ok}  (keep {targets[ok]}/{total_middle}):")
        header = f"  {'Split':<24} | " + " | ".join(f"Seed {sd:>3}" for sd in seed_strs) + " | Mean ± Std"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for cell in grid_cells:
            payload = grid[ok][cell]
            label = row_labels[cell]
            if "seeds" not in payload:  # N/A pure_compression
                cells = " | ".join(f"{'N/A':>7}" for _ in seed_strs)
                print(f"  {label:<24} | {cells} |   N/A")
                continue
            cells = " | ".join(f"{fmt_seed_cell(payload, sd):>7}" for sd in seed_strs)
            print(f"  {label:<24} | {cells} | {fmt_meanstd(payload)}")

    # ---- console: sanity table ----
    print("\n" + "=" * 78 + "\n  SANITY CHECKS (correct out of 8)\n" + "=" * 78)
    header = f"  {'Condition':<20} | " + " | ".join(f"Seed {sd:>3}" for sd in seed_strs) + " | Mean ± Std"
    print(header)
    print("  " + "-" * (len(header) - 2))
    sanity_order = [f"compressed_{ratio_key(r)}" for r in COMPRESS_RATIOS] + ["full_context", "truncated"]
    for k in sanity_order:
        payload = sanity[k]
        cells = " | ".join(f"{fmt_seed_cell(payload, sd):>7}" for sd in seed_strs)
        print(f"  {k:<20} | {cells} | {fmt_meanstd(payload)}")

    # ---- calibration ----
    fc_mean = sanity["full_context"]["mean"]
    tr_mean = sanity["truncated"]["mean"]
    print()
    if fc_mean == 8 and tr_mean == 0:
        print("[calibration] OK -- full_context=8/8, truncated=0/8 (all seeds).")
    else:
        print("!" * 78 + f"\n  CALIBRATION WARNING: full_context mean={fc_mean} (expect 8), "
              f"truncated mean={tr_mean} (expect 0).\n" + "!" * 78)

    # ---- readout: best split per overall ratio ----
    print("\n" + "=" * 78 + "\n  BEST SPLIT PER OVERALL RATIO (by mean)\n" + "=" * 78)
    for o in OVERALL_RATIOS:
        ok = ratio_key(o)
        scored = [(cell, grid[ok][cell].get("mean")) for cell in grid_cells
                  if "seeds" in grid[ok][cell] and grid[ok][cell].get("mean") == grid[ok][cell].get("mean")]
        if not scored:
            continue
        best = max(scored, key=lambda kv: kv[1])
        ranked = ", ".join(f"{row_labels[c]}={m:.1f}" for c, m in sorted(scored, key=lambda kv: kv[1], reverse=True))
        print(f"  {ok}: best = {row_labels[best[0]]} ({best[1]:.1f}/8)")
        print(f"       {ranked}")


if __name__ == "__main__":
    main()
