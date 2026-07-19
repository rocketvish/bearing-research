"""Cross-turn validation of the V-only / perplexity-pooled / position-
interpolated compression recipe.

Single-turn feasibility (results_feasibility_pplpool.json) ran the
sweep on turn 8 (CLAUDE.md, 242 tokens, project-conventions doc) and
hit 3/3 down to ~3.2x compression. This script repeats the same sweep
on two additional turns with different content types and sizes:

  - Turn 18: JWT auth middleware (240 tokens, JS code) — same SIZE as
    turn 8 but very different CONTENT TYPE
  - Turn 24: reports.js (1231 tokens, large JS file) — ~5x the size of
    turn 8, tests scaling

Each turn gets its own 3-fact question (only-answerable-from-this-turn)
and its own keyword-based scorer.

Also records `mean_pool_loss` per condition (= step-0 loss, since the
optimizer starts from the mean-pool init). This is a diagnostic for
whether mean-pool loss (cheap, no optimization) predicts compression
quality — a potential routing signal for hybrid compression.

Standalone — does not import from compressor.py / context_builder.py /
feasibility_test.py.
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

from config import MODEL_ID, strip_thinking
TRANSCRIPT_PATH = Path(__file__).parent / "transcript.jsonl"
RESULTS_PATH = Path(__file__).parent / "results_cross_validation.json"

PREFIX_TURN_INDEX = 0  # 0-indexed; turn 1 (system prompt)
COMPRESSION_RATIOS = [1.6, 2.4, 3.2, 4.8]

LR = 0.003
BASE_STEPS = 300
BASE_TOKENS = 250
MIN_STEPS = 100
LOG_EVERY = 50
MAX_NEW_TOKENS = 256

EXTRA_STOP_TOKEN_STRINGS = ["<|im_end|>", "<|endoftext|>"]

# Hardcoded turn-8 baselines from results_feasibility_pplpool.json so
# the cross-turn comparison table prints all three turns together.
# Score values are per the user's spec for this run; loss values come
# directly from the prior run's initial/final fields (initial_loss is
# the step-0 loss = mean-pool loss, since Adam starts from mean-pool).
TURN8_SCORES: dict[float, int] = {1.6: 3, 2.4: 2, 3.2: 3, 4.8: 2}
TURN8_LOSSES: dict[float, dict[str, float]] = {
    1.6: {"mean_pool_loss": 16.4915, "final_loss": 3.0866},
    2.4: {"mean_pool_loss": 28.6411, "final_loss": 4.8513},
    3.2: {"mean_pool_loss": 38.5014, "final_loss": 5.8813},
    4.8: {"mean_pool_loss": 43.5659, "final_loss": 7.9193},
}


# ------------------------ turn definitions ---------------------------

TURNS_TO_TEST = [
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

def score_turn18(ans: str) -> tuple[int, dict[str, bool]]:
    """Turn 18: JWT auth middleware (auth.js).

    Correct facts:
      - Library: 'jsonwebtoken' (not just 'jwt')
      - Status: 401 for missing/invalid tokens
      - Default secret string: 'dev-secret'
    """
    a = ans.lower()
    fact_lib = "jsonwebtoken" in a  # require the package name, not abbreviated 'jwt'
    fact_401 = "401" in a
    fact_secret = "dev-secret" in a or "dev secret" in a
    details = {
        "jsonwebtoken": fact_lib,
        "status_401": fact_401,
        "dev_secret": fact_secret,
    }
    return int(fact_lib) + int(fact_401) + int(fact_secret), details


def score_turn24(ans: str) -> tuple[int, dict[str, bool]]:
    """Turn 24: weekly report generator (reports.js).

    Correct facts:
      - Time window: 7 days (last 7 days, -7 days SQL)
      - Status categories: pending, in_progress, done (all three present)
      - Title: 'Weekly Report - {date}' (literal 'Weekly Report' prefix)
    """
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
    "score_turn18": score_turn18,
    "score_turn24": score_turn24,
}


# ------------------------ per-turn validation ------------------------

def steps_for(n_tokens: int) -> int:
    return max(round(BASE_STEPS * n_tokens / BASE_TOKENS), MIN_STEPS)


def num_virtual_for(n_tokens: int, ratio: float) -> int:
    return max(round(n_tokens / ratio), 1)


def run_turn_validation(
    model, embed, model_dtype, tokenizer,
    prefix_kv: tuple, prefix_len: int,
    turn_record: dict, question_text_inner: str, expected_str: str,
    score_fn, label: str, stop_ids: set[int],
) -> dict:
    """Run the full sweep on one turn. Returns a dict in the JSON
    schema (without the surrounding turn metadata, which the caller
    fills in).
    """
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

    # ---- 1. V_full + perplexity (one no-grad forward) ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        kv_out = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=torch.arange(
                virtual_pos_start, virtual_pos_start + n_test,
                device=device, dtype=torch.long,
            ).unsqueeze(0),
            output_hidden_states=False,
            use_cache=True,
        )
    perplexity_weights = compute_perplexity_weights(kv_out.logits, turn_ids)
    full_kv = to_legacy_kv(kv_out.past_key_values)
    v_full: list[torch.Tensor] = []
    for k, v in full_kv:
        v_new = v[:, :, prefix_len:, :].detach().clone()
        v_full.append(v_new)
        del k, v
    del kv_out, full_kv
    torch.cuda.empty_cache()
    print(f"  [{label}] V_full: {len(v_full)} layers x {tuple(v_full[0].shape)} "
          f"perplexity[min,max]=[{perplexity_weights.min().item():.3f}, "
          f"{perplexity_weights.max().item():.3f}]  {vram_str()}")

    # Pre-embed turn for init
    with torch.no_grad():
        turn_emb = embed(turn_ids).squeeze(0)  # (n_test, D)
    print(f"  [{label}] turn_emb {tuple(turn_emb.shape)} dtype={turn_emb.dtype}")

    answers: dict[str, str] = {}
    gen_times: dict[str, float] = {}
    optim_meta: dict[str, dict] = {}

    # ---- 2. Reference: full_turn ----
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

    print(f"  [{label}] full_turn: building cache + generating ...")
    full_kv_inf = kv_with_full_turn()
    t0 = time.perf_counter()
    answers["full_turn"] = gen_full(full_kv_inf)
    gen_times["full_turn"] = time.perf_counter() - t0
    print(f"    full_turn {gen_times['full_turn']:.1f}s: {answers['full_turn'][:120]!r}")
    del full_kv_inf
    torch.cuda.empty_cache()

    print(f"  [{label}] raw_embeddings: building cache + generating ...")
    raw_kv_inf = kv_with_raw_embeddings()
    t0 = time.perf_counter()
    answers["raw_embeddings"] = gen_full(raw_kv_inf)
    gen_times["raw_embeddings"] = time.perf_counter() - t0
    print(f"    raw_embeddings {gen_times['raw_embeddings']:.1f}s: {answers['raw_embeddings'][:120]!r}")
    del raw_kv_inf
    torch.cuda.empty_cache()

    print(f"  [{label}] no_turn: generating ...")
    t0 = time.perf_counter()
    answers["no_turn"] = gen_full(prefix_kv)
    gen_times["no_turn"] = time.perf_counter() - t0
    print(f"    no_turn {gen_times['no_turn']:.1f}s: {answers['no_turn'][:120]!r}")

    # ---- 3. Optimized conditions across compression ratios ----
    for ratio in COMPRESSION_RATIOS:
        cond_label = f"optimized_{ratio:.1f}x"
        num_virtual = num_virtual_for(n_test, ratio)
        num_steps = steps_for(n_test)
        est_min = num_steps * 0.5 / 60.0  # rough estimate, ~0.5s per step
        print()
        print(f"  [{label}/{cond_label}] V={num_virtual}, steps={num_steps}, "
              f"est ~{est_min:.1f}min ...")

        # Build target_v_list and v_mag2 (per layer) using perplexity-weighted pool
        target_v_list: list[torch.Tensor] = []
        v_mag2_list: list[torch.Tensor] = []
        for v_layer in v_full:
            v_pooled = attention_weighted_pool(
                v_layer, perplexity_weights, num_virtual,
            ).detach().clone()
            target_v_list.append(v_pooled)
            v_mag2_list.append(v_pooled.float().pow(2).mean().detach())

        # Init virtual from mean-pooled embeddings
        init = mean_pool_chunks(turn_emb, num_virtual).detach().clone()
        virtual = init.detach().clone().to(dtype=torch.float32, device=device)
        virtual.requires_grad_(True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        virtual_pos = interpolate_positions(
            virtual_pos_start, virtual_pos_end, num_virtual, device=device,
        )

        optimizer = torch.optim.Adam([virtual], lr=LR)
        losses: list[float] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        cache_diag: dict = {}
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
                if step == 0:
                    cache_diag["cache_type"] = type(out.past_key_values).__name__
                live = extract_new_kv_grad_safe(out.past_key_values, prefix_len)
                if step == 0:
                    v0 = live[0][1]
                    cache_diag["v0_requires_grad"] = bool(v0.requires_grad)
                    cache_diag["v0_grad_fn"] = (
                        type(v0.grad_fn).__name__ if v0.grad_fn is not None else "None"
                    )
                # V-only normalized loss; K intentionally unused, not deleted.
                loss = torch.zeros((), dtype=torch.float32, device=device)
                for (_k_live, v_live), v_tgt, v_mag2 in zip(
                    live, target_v_list, v_mag2_list,
                ):
                    layer_loss = F.mse_loss(v_live.float(), v_tgt.float())
                    loss = loss + layer_loss / v_mag2.clamp(min=1e-8)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if step == 0 and virtual.grad is None:
                    print(f"  [{label}/{cond_label}] GRADIENT FAILURE: virtual.grad is None.")
                    for k_, v_ in cache_diag.items():
                        print(f"    {k_}: {v_}")
                    raise AssertionError("Gradient did not flow through KV extraction.")
                optimizer.step()

                loss_val = float(loss.detach().item())
                losses.append(loss_val)
                if step in (0, num_steps // 2, num_steps - 1):
                    print(f"    [{cond_label}] step {step:>4}/{num_steps}: "
                          f"loss={loss_val:.4f}  {vram_str()}")
                elif (step + 1) % LOG_EVERY == 0:
                    print(f"    [{cond_label}] step {step:>4}/{num_steps}: loss={loss_val:.4f}")

                del out, loss, inputs_embeds
        finally:
            model.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        opt_time = time.perf_counter() - t_start

        # mean_pool_loss = step-0 loss (Adam starts from mean-pool init,
        # so the loss BEFORE any optimizer step is exactly the mean-pool
        # loss). Recorded explicitly so downstream analysis can use it
        # as a routing signal candidate without needing to rerun.
        mean_pool_loss = losses[0]
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = (
            (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0.0
        )
        print(f"  [{label}/{cond_label}] {num_steps} steps in {opt_time:.1f}s, "
              f"mean_pool_loss={mean_pool_loss:.4f} → final_loss={final_loss:.4f} "
              f"({loss_reduction*100:.1f}% reduction)")

        # Build inference KV with optimized virtual + position interpolation
        inf_kv = kv_with_virtual(virtual.detach(), virtual_pos)

        # Generate via manual_greedy_with_positions (compressed conditions)
        t0 = time.perf_counter()
        ans = gen_posinterp(inf_kv)
        dt = time.perf_counter() - t0
        answers[cond_label] = ans
        gen_times[cond_label] = dt

        print(f"    {cond_label} {dt:.1f}s: {ans[:120]!r}")

        optim_meta[cond_label] = {
            "compression_ratio": ratio,
            "num_virtual": num_virtual,
            "num_steps": num_steps,
            "mean_pool_loss": mean_pool_loss,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_reduction": loss_reduction,
            "loss_curve": losses,
            "opt_time_s": opt_time,
        }

        # Free per-ratio allocations
        del target_v_list, v_mag2_list, init, virtual, virtual_pos, optimizer, inf_kv
        torch.cuda.empty_cache()
        print(f"  [{label}/{cond_label}] freed; {vram_str()}")

    # Free per-turn allocations
    del v_full, perplexity_weights, turn_emb
    torch.cuda.empty_cache()

    # ---- 4. Score every answer ----
    conditions: dict[str, dict] = {}
    cond_order = ["full_turn", "raw_embeddings"] + [
        f"optimized_{r:.1f}x" for r in COMPRESSION_RATIOS
    ] + ["no_turn"]
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

    return {"conditions": conditions, "cond_order": cond_order}


# -------------------------------- main -------------------------------

def print_all_turns(turns: list[dict], tokenizer) -> None:
    print()
    print("=" * 78)
    print("  Transcript dump (turn index, role, token count, preview)")
    print("=" * 78)
    for i, turn in enumerate(turns, start=1):
        role = turn.get("role", "?")
        content = _stringify_content(turn.get("content", ""))
        marker = turn_marker(i, role, content)
        n_tok = len(tokenizer(marker, add_special_tokens=False).input_ids)
        preview = marker[:150].replace("\n", " ")
        print(f"  T{i:>2} {role:<10} tok={n_tok:>5} | {preview!r}")


def main():
    print("=" * 78)
    print("  Cross-turn validation of compression recipe")
    print("=" * 78)

    # ---- Load model + tokenizer ----
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

    # ---- Load + dump transcript ----
    raw_lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    turns = [json.loads(line) for line in raw_lines if line.strip()]
    print(f"Loaded {len(turns)} turns from {TRANSCRIPT_PATH.name}")
    print_all_turns(turns, tokenizer)

    # ---- Tokenize prefix and build prefix_kv ----
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
        print(f"[rope_check] CONFIRMED — RoPE on Q,K only. line: {rope_finding}")
    else:
        print(f"[rope_check] WARNING — could not confirm: {rope_finding}")

    # ---- Position-id runtime check (hard halt on failure) ----
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

    # ---- Pre-tokenize selected turns and build records ----
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
        print(f"  question: {spec['question']}")
        print(f"  expected: {spec['expected']}")

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

    # ---- Save results JSON ----
    turn8_baseline_json = {
        "scores": {f"{r:.1f}x": f"{TURN8_SCORES[r]}/3" for r in COMPRESSION_RATIOS},
        "losses": {
            f"{r:.1f}x": {
                "mean_pool_loss": TURN8_LOSSES[r]["mean_pool_loss"],
                "final_loss": TURN8_LOSSES[r]["final_loss"],
                "loss_reduction": (
                    (TURN8_LOSSES[r]["mean_pool_loss"] - TURN8_LOSSES[r]["final_loss"])
                    / TURN8_LOSSES[r]["mean_pool_loss"]
                ),
            }
            for r in COMPRESSION_RATIOS
        },
        "token_count": 242,
        "content_type": "project conventions doc (CLAUDE.md)",
    }

    results = {
        "config": {
            "model_id": MODEL_ID,
            "compression_ratios": COMPRESSION_RATIOS,
            "lr": LR,
            "steps_formula": "max(round(300 * turn_tokens / 250), 100)",
            "target": "V-only, perplexity-weighted pool, per-layer normalized",
            "position_interpolation": True,
            "rope_verified": rope_verified,
            "rope_finding": rope_finding,
            "position_ids_take_effect": posid_check,
            "prefix_turn_index": PREFIX_TURN_INDEX + 1,
            "prefix_tokens": n_prefix,
            "stop_token_ids": sorted(stop_ids),
        },
        "turn8_baseline": turn8_baseline_json,
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
        print()
        cond_order = t.get("_cond_order", [])
        for cond in cond_order:
            entry = t["conditions"].get(cond, {})
            score = entry.get("score", "?/3")
            ans = entry.get("answer", "")
            line = ans.replace("\n", " ")
            if len(line) > 160:
                line = line[:160] + "…"
            extra = ""
            if "mean_pool_loss" in entry:
                extra = (
                    f"  [mean_pool {entry['mean_pool_loss']:.2f} → "
                    f"final {entry['final_loss']:.2f}, "
                    f"{entry['loss_reduction']*100:.0f}%]"
                )
            print(f"  [{cond:<18}] [{score}] {line}{extra}")

    # ---- Console summary: cross-turn comparison table ----
    print()
    print("=" * 78)
    print("  CROSS-TURN COMPARISON (correct facts out of 3)")
    print("=" * 78)
    header_cols = [f"Turn {t['turn_index']} ({t['token_count']} tok)" for t in turns_results]
    header = (
        f"  {'Ratio':>6} | {'Turn 8 (242 tok)':>18} | "
        + " | ".join(f"{c:>20}" for c in header_cols)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for ratio in COMPRESSION_RATIOS:
        cells = [f"{TURN8_SCORES[ratio]}/3"]
        for t in turns_results:
            entry = t["conditions"].get(f"optimized_{ratio:.1f}x", {})
            sc = entry.get("score", "?/3")
            cells.append(sc)
        cells_str = " | ".join(f"{c:>20}" for c in cells[1:])
        print(f"  {ratio:>4.1f}x | {cells[0]:>18} | {cells_str}")

    # ---- Console summary: mean-pool loss vs answer quality ----
    print()
    print("=" * 78)
    print("  MEAN-POOL LOSS vs ANSWER QUALITY (correlation diagnostic)")
    print("=" * 78)
    print(f"  {'Turn':>4} | {'Ratio':>5} | {'Mean-pool loss':>14} | "
          f"{'Final loss':>10} | {'Reduction':>9} | {'Score':>5}")
    print("  " + "-" * 70)
    # Turn 8 rows (baseline)
    for ratio in COMPRESSION_RATIOS:
        m = TURN8_LOSSES[ratio]["mean_pool_loss"]
        f_ = TURN8_LOSSES[ratio]["final_loss"]
        red = (m - f_) / m if m > 0 else 0.0
        sc = TURN8_SCORES[ratio]
        print(f"  {8:>4} | {ratio:>4.1f}x | {m:>14.4f} | {f_:>10.4f} | "
              f"{red*100:>7.1f}% | {sc:>3}/3   (baseline)")
    # New turns
    for t in turns_results:
        ti = t["turn_index"]
        for ratio in COMPRESSION_RATIOS:
            entry = t["conditions"].get(f"optimized_{ratio:.1f}x", {})
            m = entry.get("mean_pool_loss")
            f_ = entry.get("final_loss")
            red = entry.get("loss_reduction")
            sc = entry.get("score_count", "?")
            if m is None or f_ is None:
                print(f"  {ti:>4} | {ratio:>4.1f}x | {'—':>14} | {'—':>10} | "
                      f"{'—':>9} | {sc!s:>3}/3")
            else:
                print(f"  {ti:>4} | {ratio:>4.1f}x | {m:>14.4f} | {f_:>10.4f} | "
                      f"{(red or 0)*100:>7.1f}% | {sc!s:>3}/3")

    # ---- Sanity check: raw_embeddings vs full_turn ----
    diverged_turns: list[int] = []
    for t in turns_results:
        f_a = (t["conditions"].get("full_turn", {}).get("answer") or "").strip()
        r_a = (t["conditions"].get("raw_embeddings", {}).get("answer") or "").strip()
        if not f_a or not r_a or f_a.startswith("<error") or r_a.startswith("<error"):
            continue
        if f_a != r_a:
            diverged_turns.append(t["turn_index"])
    if diverged_turns:
        print()
        print("!" * 78)
        print(f"  WARNING: raw_embeddings DIVERGED from full_turn on turns: {diverged_turns}")
        print("  inputs_embeds path is not reproducing input_ids path. Compressed")
        print("  results on those turns may be unreliable.")
        print("!" * 78)
    else:
        print()
        print("[sanity] raw_embeddings matches full_turn on all tested turns.")


if __name__ == "__main__":
    main()
