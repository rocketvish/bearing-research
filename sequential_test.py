"""Sequential multi-turn V-only compression test.

Single-turn feasibility (results_feasibility_pplpool.json) showed that
V-only target + position interpolation + perplexity-weighted V pooling
hit 3/3 down to ~3.2x compression. This script tests whether the same
recipe survives across a full multi-turn conversation when middle turns
are compressed sequentially against a running KV cache.

Pipeline per middle turn (recipe from feasibility_test.py):
  1. V-only target (no K in loss) — avoids RoPE position-encoding mismatch
  2. Position interpolation — virtual tokens get explicit position_ids
     spread across the turn's original-position span
  3. Perplexity-weighted V pooling — pool targets weighted by per-token
     surprise (cross-entropy from the same forward used for V targets)
  4. Per-layer mag² normalization, lr=0.003, Adam fp32

We run the full sweep at COMPRESSION_RATIOS = [2.0, 3.0] in one
execution and compare against three reference conditions: full_context
(ceiling), no_middle_raw_embeddings (sanity — must match full_context),
and truncated (floor). Eight questions per condition.

Standalone — copies any helpers it needs from feasibility_test.py.
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
RESULTS_PATH = Path(__file__).parent / "results_sequential_v2.json"

TOKEN_BUDGET = 4096
NUM_PREFIX = 1
NUM_SUFFIX = 1

COMPRESSION_RATIOS = [1.5, 2.0]
HYBRID_TARGET_RATIO = 2.0
# 8.0 (vs the obvious 5.0): on this transcript the two largest turns
# (~1500 tokens each) dominate the budget, so a 5.0 cap forces the
# greedy algorithm to reject ALL verbatim candidates and degenerate to
# uniform 2x. 8.0 raises the implicit-rest-of-budget enough that at
# least one large high-surprise turn can be kept verbatim, which is
# the actual hybrid hypothesis we want to test. If it still degenerates
# (depending on which turn happens to have highest surprise), the run
# logs that and continues.
HYBRID_MAX_COMPRESS_RATIO = 8.0
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

# Qwen-specific: <|im_end|> = 151645 (default eos), <|endoftext|> = 151643.
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


def pool_kv_seq(t: torch.Tensor, n_chunks: int) -> torch.Tensor:
    s = t.shape[2]
    if n_chunks > s:
        raise ValueError(f"n_chunks={n_chunks} > S={s}")
    boundaries = torch.linspace(0, s, n_chunks + 1).long().tolist()
    chunks = []
    for i in range(n_chunks):
        a, b = boundaries[i], boundaries[i + 1]
        if b <= a:
            b = a + 1
        chunks.append(t[:, :, a:b, :].mean(dim=2, keepdim=True))
    return torch.cat(chunks, dim=2)


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
    positions starting at ``slice_offset``. Does NOT detach/clone/
    contiguous — preserves the autograd graph through the K/V
    projections of the new tokens.

    For sequential compression, ``slice_offset`` is the cache's
    PHYSICAL length BEFORE the current forward (kv_seq_len(running_kv)),
    which differs from "original position" once any compression has
    happened.
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
    """Empirically verify explicit position_ids take effect when
    past_key_values is present.

    Uses prefix_kv as past_key_values because the failure mode is the
    model ignoring explicit position_ids when a cache is present and
    auto-deriving from cache length instead. Without a cache present,
    position_ids are always honored — we'd never observe the bug.
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
    """Per-token surprise (cross-entropy) for each position in turn_ids.

    logits[0, i, :] predicts turn_ids[0, i+1]. Position 0 has no
    preceding logit in this forward, so prepend 0.0 per spec.
    """
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
    model,
    tokenizer,
    prompt_text: str,
    past_kv_legacy: tuple,
    start_pos: int,
    max_new_tokens: int,
    stop_ids: set[int],
) -> str:
    """Greedy decode with explicit per-step position_ids.

    NOTE: callers for compressed and raw_embeddings conditions pass the
    SUFFIX text as part of ``prompt_text`` and ``start_pos =
    suffix_start_pos`` — the suffix is intentionally NOT in
    past_kv_legacy. See the build sites' comments on why this is
    equivalent to including the suffix in the cache for a causal model
    but easier to keep position bookkeeping clear.
    """
    device = model.device
    q_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_q = int(q_ids.shape[1])

    cache = wrap_legacy_kv(past_kv_legacy, model.config)
    pos_ids = torch.arange(
        start_pos, start_pos + n_q, device=device, dtype=torch.long
    ).unsqueeze(0)
    out = model(
        input_ids=q_ids,
        past_key_values=cache,
        position_ids=pos_ids,
        use_cache=True,
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
            input_ids=next_inp,
            past_key_values=cache,
            position_ids=pos_ids,
            use_cache=True,
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
    """Manual greedy fallback that uses cache-derived positions (no explicit position_ids)."""
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
    """Try model.generate(past_key_values=...); fall back to manual_greedy_auto."""
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
            model, tokenizer, prompt_text, past_kv_legacy, max_new_tokens, stop_ids
        )


# ----------------------- transcript loading --------------------------

def load_and_split(path: Path, tokenizer, budget: int) -> dict:
    """Tokenize each turn, greedy-fit into budget, split into 1+middle+1.

    Returns a dict with 'prefix', 'middle' (list), 'suffix' records and
    bookkeeping. Each record carries ids (LongTensor (1, n)), n_tokens,
    text, preview, role, and original-position metadata.
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    raw_turns = [json.loads(line) for line in raw_lines if line.strip()]

    # Tokenize each turn with the standard marker.
    per_turn: list[dict] = []
    for i, turn in enumerate(raw_turns, start=1):
        role = turn.get("role", "?")
        content = _stringify_content(turn.get("content", ""))
        text = turn_marker(i, role, content)
        ids_list = tokenizer(text, add_special_tokens=False).input_ids
        per_turn.append({
            "turn_index_1based": i,
            "role": role,
            "text": text,
            "ids_list": ids_list,
            "n_tokens": len(ids_list),
            "preview": text[:200],
        })

    # Greedy fit.
    included: list[dict] = []
    running = 0
    for rec in per_turn:
        remaining = budget - running
        if remaining <= 0:
            break
        if rec["n_tokens"] <= remaining:
            included.append(rec)
            running += rec["n_tokens"]
        else:
            truncated_ids = rec["ids_list"][:remaining]
            truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False)
            included.append({
                "turn_index_1based": rec["turn_index_1based"],
                "role": rec["role"],
                "text": truncated_text,
                "ids_list": truncated_ids,
                "n_tokens": len(truncated_ids),
                "preview": truncated_text[:200] + " [...truncated]",
            })
            running += len(truncated_ids)
            break

    n_inc = len(included)
    if n_inc < NUM_PREFIX + NUM_SUFFIX + 1:
        raise ValueError(
            f"Budget {budget} only fits {n_inc} turns; "
            f"need at least {NUM_PREFIX + NUM_SUFFIX + 1}."
        )

    prefix_records = included[:NUM_PREFIX]
    suffix_records = included[-NUM_SUFFIX:]
    middle_records = included[NUM_PREFIX : n_inc - NUM_SUFFIX]
    if len(middle_records) == 0:
        raise ValueError("No middle turns left after prefix/suffix split.")

    # Original-position bookkeeping. Prefix at [0, prefix_len-1]; each
    # middle turn occupies the next n_tokens positions; suffix follows.
    prefix_n = sum(r["n_tokens"] for r in prefix_records)
    pos = prefix_n
    for r in middle_records:
        r["start_pos"] = pos
        r["end_pos"] = pos + r["n_tokens"] - 1
        pos += r["n_tokens"]
    suffix_start_pos = pos
    suffix_pos = suffix_start_pos
    for r in suffix_records:
        r["start_pos"] = suffix_pos
        r["end_pos"] = suffix_pos + r["n_tokens"] - 1
        suffix_pos += r["n_tokens"]

    # Tensorise ids for convenience.
    for r in prefix_records + middle_records + suffix_records:
        r["ids"] = torch.tensor([r["ids_list"]], dtype=torch.long)

    return {
        "prefix_records": prefix_records,
        "middle_records": middle_records,
        "suffix_records": suffix_records,
        "prefix_len": prefix_n,
        "suffix_len": sum(r["n_tokens"] for r in suffix_records),
        "total_middle_tokens": sum(r["n_tokens"] for r in middle_records),
        "suffix_start_pos": suffix_start_pos,
        "n_total_in_budget": running,
        "n_total_turns_loaded": len(per_turn),
    }


def print_dry_run(split: dict) -> None:
    print()
    print("=" * 78)
    print("  Dry-run context split")
    print("=" * 78)
    p = split["prefix_records"][0]
    print(f"PREFIX  (turn {p['turn_index_1based']}, role={p['role']}): "
          f"{p['n_tokens']} tokens  pos=[{0}, {split['prefix_len'] - 1}]")
    print(f"  preview: {p['text'][:200]!r}")
    print()
    print(f"MIDDLE  ({len(split['middle_records'])} turns, "
          f"{split['total_middle_tokens']} tokens):")
    for r in split["middle_records"]:
        print(f"  turn {r['turn_index_1based']:>3} ({r['role']:<10}): "
              f"{r['n_tokens']:>4} tok  "
              f"pos=[{r['start_pos']}, {r['end_pos']}]  "
              f"{r['text'][:100]!r}")
    print()
    s = split["suffix_records"][0]
    print(f"SUFFIX  (turn {s['turn_index_1based']}, role={s['role']}): "
          f"{s['n_tokens']} tokens  pos=[{s['start_pos']}, {s['end_pos']}]")
    print(f"  preview: {s['text'][:200]!r}")
    print()
    print(f"Totals: {split['n_total_turns_loaded']} turns loaded, "
          f"{split['n_total_in_budget']} tokens in budget; "
          f"prefix={split['prefix_len']}, "
          f"middle={split['total_middle_tokens']}, "
          f"suffix={split['suffix_len']}, "
          f"suffix_start_pos={split['suffix_start_pos']}.")


# ------------------------- single-turn compress ----------------------

def steps_for(n_tokens: int) -> int:
    return max(round(BASE_STEPS * n_tokens / BASE_TOKENS), MIN_STEPS)


def num_virtual_for(n_tokens: int, ratio: float) -> int:
    return max(round(n_tokens / ratio), 1)


def compress_one_turn(
    model,
    embed,
    model_dtype,
    running_kv: tuple,
    turn_record: dict,
    num_virtual: int,
    num_steps: int,
    label: str,
) -> tuple[tuple, dict]:
    """Compress one middle turn into ``num_virtual`` optimized virtual
    tokens, extending ``running_kv`` with their K/V at interpolated
    positions.

    Returns (new_running_kv, meta_dict).
    """
    device = model.device
    turn_ids = turn_record["ids"].to(device)
    n_tokens = int(turn_record["n_tokens"])
    turn_start_pos = int(turn_record["start_pos"])
    turn_end_pos = int(turn_record["end_pos"])
    cache_phys_offset = kv_seq_len(running_kv)  # physical cache length BEFORE this turn

    t_start = time.perf_counter()

    # ---- (a) Combined V target + perplexity forward (no_grad) ----
    target_pos = torch.arange(
        turn_start_pos, turn_start_pos + n_tokens, device=device, dtype=torch.long
    ).unsqueeze(0)
    with torch.no_grad():
        kv_out = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(running_kv, model.config),
            position_ids=target_pos,
            output_hidden_states=False,
            use_cache=True,
        )
    perplexity_weights = compute_perplexity_weights(kv_out.logits, turn_ids)
    full_kv = to_legacy_kv(kv_out.past_key_values)
    v_full: list[torch.Tensor] = []
    for k, v in full_kv:
        v_new = v[:, :, cache_phys_offset:, :].detach().clone()  # (1, n_kv, n_tokens, Dh)
        v_full.append(v_new)
        del k, v
    del kv_out, full_kv

    target_v_list: list[torch.Tensor] = []
    v_mag2_list: list[torch.Tensor] = []
    for v_layer in v_full:
        v_pooled = attention_weighted_pool(
            v_layer, perplexity_weights, num_virtual
        ).detach().clone()
        target_v_list.append(v_pooled)
        v_mag2_list.append(v_pooled.float().pow(2).mean().detach())
    del v_full
    torch.cuda.empty_cache()

    # ---- (b) Init virtual from mean-pooled input embeddings ----
    with torch.no_grad():
        turn_emb = embed(turn_ids).squeeze(0)  # (n_tokens, D)
    init = mean_pool_chunks(turn_emb, num_virtual).detach().clone()
    del turn_emb

    # Edge case: num_virtual >= n_tokens — optimization has nothing to
    # gain, virtual = real embeddings already. Skip the loop, use init
    # directly. (E.g. a 2-token turn at 2.0x → num_virtual=1 still
    # benefits from optimization, but a 3-token turn at 2.0x →
    # num_virtual=2 with init exactly matching the 2 mean-pooled real
    # embeddings; loss should already be tiny.)
    skip_optimization = num_virtual >= n_tokens
    if skip_optimization:
        print(f"  [{label}] num_virtual({num_virtual}) >= n_tokens({n_tokens}); "
              f"skipping optimization (using mean-pool init).")

    # ---- (c) Optimization loop ----
    virtual = init.detach().clone().to(dtype=torch.float32, device=device)
    if not skip_optimization:
        virtual.requires_grad_(True)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        virtual_pos = interpolate_positions(turn_start_pos, turn_end_pos, num_virtual, device=device)

        optimizer = torch.optim.Adam([virtual], lr=LR)
        losses: list[float] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            for step in range(num_steps):
                inputs_embeds = virtual.to(dtype=model_dtype).unsqueeze(0)
                out = model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=wrap_legacy_kv(running_kv, model.config),
                    position_ids=virtual_pos,
                    output_hidden_states=False,
                    use_cache=True,
                )
                live = extract_new_kv_grad_safe(out.past_key_values, cache_phys_offset)
                # V-only loss; K (live[L][0]) intentionally unused and not deleted.
                loss = torch.zeros((), dtype=torch.float32, device=device)
                for (_k_live, v_live), v_tgt, v_mag2 in zip(live, target_v_list, v_mag2_list):
                    layer_loss = F.mse_loss(v_live.float(), v_tgt.float())
                    loss = loss + layer_loss / v_mag2.clamp(min=1e-8)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if step == 0 and virtual.grad is None:
                    raise AssertionError(
                        f"[{label}] gradient did not flow through KV extraction."
                    )
                optimizer.step()

                loss_val = float(loss.detach().item())
                losses.append(loss_val)
                if step in (0, num_steps // 2, num_steps - 1):
                    print(f"    [{label}] step {step:>3}/{num_steps}: loss={loss_val:.4f}  {vram_str()}")
                elif (step + 1) % LOG_EVERY == 0:
                    print(f"    [{label}] step {step:>3}/{num_steps}: loss={loss_val:.4f}")

                del out, loss, inputs_embeds
        finally:
            model.eval()

        del optimizer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    else:
        # No optimization: synthesize a one-step "loss curve" from the
        # current init's distance to the target so meta is still well-formed.
        with torch.no_grad():
            inputs_embeds = virtual.to(dtype=model_dtype).unsqueeze(0)
            virtual_pos = interpolate_positions(
                turn_start_pos, turn_end_pos, num_virtual, device=device
            )
            out = model(
                inputs_embeds=inputs_embeds,
                past_key_values=wrap_legacy_kv(running_kv, model.config),
                position_ids=virtual_pos,
                output_hidden_states=False,
                use_cache=True,
            )
            live = extract_new_kv_grad_safe(out.past_key_values, cache_phys_offset)
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for (_k_live, v_live), v_tgt, v_mag2 in zip(live, target_v_list, v_mag2_list):
                layer_loss = F.mse_loss(v_live.float(), v_tgt.float())
                loss = loss + layer_loss / v_mag2.clamp(min=1e-8)
            losses = [float(loss.item())]
            del out, loss, inputs_embeds

    # ---- (d) Build extended running_kv with the optimized virtual tokens ----
    with torch.no_grad():
        ie = virtual.detach().to(dtype=model_dtype).unsqueeze(0)
        out = model(
            inputs_embeds=ie,
            past_key_values=wrap_legacy_kv(running_kv, model.config),
            position_ids=virtual_pos,
            use_cache=True,
        )
        new_running_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out, ie

    dt = time.perf_counter() - t_start
    initial_loss = losses[0]
    final_loss = losses[-1]
    new_phys = kv_seq_len(new_running_kv)

    print(f"  [{label}] turn {turn_record['turn_index_1based']} "
          f"({turn_record['role']}): {n_tokens} → {num_virtual} "
          f"({n_tokens / max(num_virtual, 1):.2f}x), "
          f"{0 if skip_optimization else num_steps} steps, "
          f"loss {initial_loss:.4f} → {final_loss:.4f}, "
          f"{dt:.1f}s; cache_phys={new_phys}; {vram_str()}")

    meta = {
        "turn_index": turn_record["turn_index_1based"],
        "role": turn_record["role"],
        "original_tokens": n_tokens,
        "virtual_tokens": num_virtual,
        "steps": 0 if skip_optimization else num_steps,
        "turn_start_pos": turn_start_pos,
        "turn_end_pos": turn_end_pos,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_curve": losses,
        "time_s": dt,
        "skipped_optimization": skip_optimization,
    }

    # ---- (e) Free per-turn intermediates ----
    del target_v_list, v_mag2_list, perplexity_weights, init, virtual_pos, virtual
    torch.cuda.empty_cache()

    return new_running_kv, meta


def run_sequential_compression(
    model, embed, model_dtype, prefix_kv: tuple,
    middle_records: list[dict], ratio: float, ratio_label: str,
) -> tuple[tuple, list[dict]]:
    print()
    print("=" * 78)
    print(f"  Sequential compression at {ratio_label} (ratio={ratio})")
    print("=" * 78)
    running_kv = prefix_kv
    per_turn_meta: list[dict] = []
    for r in middle_records:
        n_tok = r["n_tokens"]
        nv = num_virtual_for(n_tok, ratio)
        ns = steps_for(n_tok)
        label = f"{ratio_label}/turn{r['turn_index_1based']}"
        running_kv, meta = compress_one_turn(
            model, embed, model_dtype, running_kv, r,
            num_virtual=nv, num_steps=ns, label=label,
        )
        per_turn_meta.append(meta)
    return running_kv, per_turn_meta


# ----------------------- hybrid (perplexity-based) -------------------

def score_perplexity_for_turns(
    model, prefix_kv: tuple, middle_records: list[dict],
) -> list[dict]:
    """Sequentially forward each middle turn against an accumulating
    UNCOMPRESSED running KV cache and compute per-turn mean surprise.

    The accumulating cache is built from real text (input_ids), one
    turn at a time, with explicit position_ids. This is critical: the
    surprise of token i in turn T must reflect a clean preceding
    context — if we ran scoring against an already-compressed cache
    we'd be measuring "what does the model find surprising AFTER its
    earlier context has been distorted by compression artifacts",
    which is not the property we want to use to decide what to keep
    verbatim.

    Returns a list of dicts (in turn order) with keys
    ``turn_index, role, n_tokens, mean_surprise``.
    """
    print()
    print("=" * 78)
    print("  Scoring per-turn surprise (raw KV accumulation, no compression)")
    print("=" * 78)
    device = model.device
    running_kv = prefix_kv
    scores: list[dict] = []
    for r in middle_records:
        turn_ids = r["ids"].to(device)
        pos = torch.arange(
            r["start_pos"], r["start_pos"] + r["n_tokens"],
            device=device, dtype=torch.long,
        ).unsqueeze(0)
        with torch.no_grad():
            out = model(
                input_ids=turn_ids,
                past_key_values=wrap_legacy_kv(running_kv, model.config),
                position_ids=pos,
                output_hidden_states=False,
                use_cache=True,
            )
        ppl = compute_perplexity_weights(out.logits, turn_ids)
        mean_surprise = float(ppl.mean().item())
        scores.append({
            "turn_index": r["turn_index_1based"],
            "role": r["role"],
            "n_tokens": r["n_tokens"],
            "mean_surprise": mean_surprise,
        })
        running_kv = detach_kv(to_legacy_kv(out.past_key_values))
        del out, ppl
        torch.cuda.empty_cache()
        print(f"  turn {r['turn_index_1based']:>3} ({r['role']:<10}): "
              f"{r['n_tokens']:>4} tok  mean_surprise={mean_surprise:.4f}")
    # Discard the accumulated raw cache; it served only the scoring pass.
    del running_kv
    torch.cuda.empty_cache()
    return scores


def decide_hybrid_modes(
    middle_records: list[dict],
    scores: list[dict],
    target_ratio: float,
    max_compress_ratio: float,
) -> tuple[dict, dict]:
    """Greedy hybrid selection: keep highest-surprise turns verbatim
    while still meeting overall target_ratio assuming all remaining
    turns are compressed at most at max_compress_ratio. The first turn
    that breaks this constraint, plus all remaining, are compressed at
    exactly the ratio that meets the target (capped at
    max_compress_ratio).

    Returns:
      decisions: dict turn_index -> {mode, ratio, virtual, mean_surprise}
      meta: dict with target/actual virtual counts, threshold, compress_ratio,
            degenerated flag.
    """
    total_middle_tokens = sum(r["n_tokens"] for r in middle_records)
    target_virtual = total_middle_tokens / target_ratio

    # Sort turns by surprise descending. Keep the (record, score) pairs.
    score_by_idx = {s["turn_index"]: s for s in scores}
    ranked = sorted(
        middle_records,
        key=lambda r: score_by_idx[r["turn_index_1based"]]["mean_surprise"],
        reverse=True,
    )

    decisions: dict[int, dict] = {}
    verbatim_token_sum = 0
    remaining_token_sum = total_middle_tokens
    threshold_surprise: float | None = None
    compress_ratio_used: float | None = None
    degenerated = False

    for r in ranked:
        ti = r["turn_index_1based"]
        n = r["n_tokens"]
        s = score_by_idx[ti]["mean_surprise"]

        new_verbatim = verbatim_token_sum + n
        new_remaining = remaining_token_sum - n
        # If we kept THIS verbatim and compressed all rest at the cap,
        # would the total still meet target_virtual?
        hypothetical = new_verbatim + (new_remaining / max_compress_ratio)

        if hypothetical <= target_virtual:
            decisions[ti] = {
                "mode": "verbatim",
                "ratio": 1.0,
                "virtual": n,
                "mean_surprise": s,
            }
            verbatim_token_sum = new_verbatim
            remaining_token_sum = new_remaining
        else:
            # Compress this turn and every remaining (so far undecided)
            # turn at the same ratio that hits target exactly. Capped.
            need_compressed_virtual = max(target_virtual - verbatim_token_sum, 1.0)
            ratio_needed = remaining_token_sum / need_compressed_virtual
            ratio_used = min(ratio_needed, max_compress_ratio)
            compress_ratio_used = ratio_used
            threshold_surprise = s

            for rr in ranked:
                tj = rr["turn_index_1based"]
                if tj in decisions:
                    continue
                ns = score_by_idx[tj]["mean_surprise"]
                nv = max(round(rr["n_tokens"] / ratio_used), 1)
                decisions[tj] = {
                    "mode": "compressed",
                    "ratio": ratio_used,
                    "virtual": nv,
                    "mean_surprise": ns,
                }
            break

    # Edge case: every turn got marked verbatim. That means
    # total_middle_tokens <= target_virtual, i.e. we can't compress at
    # all under the constraint. Mark as a (different) degeneration.
    if all(d["mode"] == "verbatim" for d in decisions.values()):
        # No compression happened. Surprise threshold N/A.
        compress_ratio_used = None

    # Detect "degenerated to uniform compression": no verbatim turns
    # were kept, every turn ended up compressed at the same ratio.
    n_verbatim = sum(1 for d in decisions.values() if d["mode"] == "verbatim")
    if n_verbatim == 0:
        degenerated = True

    predicted_total_virtual = sum(d["virtual"] for d in decisions.values())

    meta = {
        "total_middle_tokens": total_middle_tokens,
        "target_virtual": target_virtual,
        "predicted_total_virtual": predicted_total_virtual,
        "n_verbatim_turns": n_verbatim,
        "n_compressed_turns": len(decisions) - n_verbatim,
        "surprise_threshold": threshold_surprise,
        "compress_ratio_used": compress_ratio_used,
        "max_compress_ratio_cap": max_compress_ratio,
        "degenerated_to_uniform": degenerated,
    }
    return decisions, meta


def print_decision_table(
    middle_records: list[dict],
    scores: list[dict],
    decisions: dict,
    meta: dict,
) -> None:
    print()
    print("=" * 78)
    print("  Hybrid mode decisions")
    print("=" * 78)
    score_by_idx = {s["turn_index"]: s for s in scores}
    ranked = sorted(
        middle_records,
        key=lambda r: score_by_idx[r["turn_index_1based"]]["mean_surprise"],
        reverse=True,
    )
    print(f"  target_ratio={HYBRID_TARGET_RATIO}, max_compress_cap={HYBRID_MAX_COMPRESS_RATIO}")
    print(f"  total_middle_tokens={meta['total_middle_tokens']}, "
          f"target_virtual={meta['target_virtual']:.0f}, "
          f"predicted_virtual={meta['predicted_total_virtual']}")
    if meta["surprise_threshold"] is not None:
        print(f"  surprise_threshold={meta['surprise_threshold']:.4f}  "
              f"compress_ratio_used={meta['compress_ratio_used']:.3f}")
    else:
        print("  surprise_threshold=N/A  (no verbatim/compressed split — see below)")
    print(f"  n_verbatim={meta['n_verbatim_turns']}, "
          f"n_compressed={meta['n_compressed_turns']}, "
          f"degenerated_to_uniform={meta['degenerated_to_uniform']}")
    if meta["degenerated_to_uniform"]:
        print("  [hybrid] DEGENERATED — every turn marked compressed at the same "
              "ratio. Result will be ~equivalent to uniform compression.")
    print()
    print(f"  {'Rank':>4} | {'Turn':>4} | {'Role':<10} | {'Tokens':>6} | "
          f"{'Mean surprise':>13} | {'Mode':<10} | {'Virtual':>7} | {'Ratio':>6}")
    for rank, r in enumerate(ranked, start=1):
        ti = r["turn_index_1based"]
        d = decisions[ti]
        s = score_by_idx[ti]["mean_surprise"]
        ratio_str = f"{d['ratio']:.2f}x" if d["mode"] == "compressed" else "1.00x"
        print(f"  {rank:>4} | {ti:>4} | {r['role']:<10} | "
              f"{r['n_tokens']:>6} | {s:>13.4f} | "
              f"{d['mode']:<10} | {d['virtual']:>7} | {ratio_str:>6}")


@torch.no_grad()
def extend_kv_verbatim(
    model, running_kv: tuple, turn_record: dict,
) -> tuple[tuple, dict]:
    """Forward a turn's full token IDs into ``running_kv`` with explicit
    original-position position_ids and use_cache=True. Returns the
    extended (detached) running_kv and a verbatim meta dict whose
    fields mirror ``compress_one_turn``'s meta (so downstream code can
    treat verbatim and compressed turns uniformly).
    """
    device = model.device
    turn_ids = turn_record["ids"].to(device)
    n_tokens = int(turn_record["n_tokens"])
    turn_start_pos = int(turn_record["start_pos"])
    turn_end_pos = int(turn_record["end_pos"])
    pos = torch.arange(
        turn_start_pos, turn_start_pos + n_tokens, device=device, dtype=torch.long,
    ).unsqueeze(0)

    t_start = time.perf_counter()
    out = model(
        input_ids=turn_ids,
        past_key_values=wrap_legacy_kv(running_kv, model.config),
        position_ids=pos,
        use_cache=True,
    )
    new_running_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out
    torch.cuda.empty_cache()
    dt = time.perf_counter() - t_start

    meta = {
        "turn_index": turn_record["turn_index_1based"],
        "role": turn_record["role"],
        "original_tokens": n_tokens,
        "virtual_tokens": n_tokens,  # 1:1, no compression
        "steps": 0,
        "turn_start_pos": turn_start_pos,
        "turn_end_pos": turn_end_pos,
        "initial_loss": 0.0,
        "final_loss": 0.0,
        "loss_curve": [],
        "time_s": dt,
        "skipped_optimization": False,
        "mode": "verbatim",
        "compress_ratio": 1.0,
    }
    return new_running_kv, meta


def run_hybrid_compression(
    model, embed, model_dtype, prefix_kv: tuple,
    middle_records: list[dict],
    decisions: dict,
    scores: list[dict],
    label: str,
) -> tuple[tuple, list[dict]]:
    """Sequential compression with mixed verbatim/compressed mode."""
    print()
    print("=" * 78)
    print(f"  Hybrid sequential compression: {label}")
    print("=" * 78)
    score_by_idx = {s["turn_index"]: s for s in scores}
    running_kv = prefix_kv
    per_turn_meta: list[dict] = []
    for r in middle_records:
        ti = r["turn_index_1based"]
        d = decisions[ti]
        sub_label = f"{label}/turn{ti}/{d['mode']}"
        if d["mode"] == "verbatim":
            running_kv, meta = extend_kv_verbatim(model, running_kv, r)
            print(f"  [{sub_label}] turn {ti} ({r['role']}): "
                  f"{r['n_tokens']} tokens kept verbatim, "
                  f"{meta['time_s']:.1f}s; cache_phys={kv_seq_len(running_kv)}; "
                  f"{vram_str()}")
            meta["mean_surprise"] = score_by_idx[ti]["mean_surprise"]
        else:
            n_tok = r["n_tokens"]
            ns = steps_for(n_tok)
            running_kv, meta = compress_one_turn(
                model, embed, model_dtype, running_kv, r,
                num_virtual=d["virtual"], num_steps=ns, label=sub_label,
            )
            meta["mode"] = "compressed"
            meta["compress_ratio"] = d["ratio"]
            meta["mean_surprise"] = score_by_idx[ti]["mean_surprise"]
        per_turn_meta.append(meta)
    return running_kv, per_turn_meta


# ----------------------- inference cache builds ----------------------

def _concat_ids(records: list[dict], device) -> torch.Tensor:
    chunks = [r["ids"].to(device) for r in records]
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def build_full_context_kv(
    model, prefix_records: list[dict],
    middle_records: list[dict], suffix_records: list[dict],
) -> tuple:
    """Forward prefix + middle + suffix as one concatenated text
    sequence; detach the resulting cache.

    This is the ceiling condition. Note that the SUFFIX is included in
    this cache. For the compressed and raw_embeddings conditions the
    suffix is intentionally NOT in their caches and instead gets
    concatenated into the generation prompt (see the
    manual_greedy_with_positions call sites). For a causal model the
    two arrangements produce equivalent results — the suffix's K/V are
    the same whether computed now (here) or later (during generation),
    as long as the positions match. The asymmetry is intentional:
    auto-positioning handles full_context cleanly with no interpolation
    needed, while the compressed/raw_embeddings conditions rely on
    explicit position_ids during generation to keep position
    bookkeeping local to the generator. If the raw_embeddings sanity
    check ever fails on a question, this asymmetry is one of the first
    places to look.
    """
    device = model.device
    all_ids = _concat_ids([*prefix_records, *middle_records, *suffix_records], device)
    out = model(input_ids=all_ids, use_cache=True)
    return detach_kv(to_legacy_kv(out.past_key_values))


@torch.no_grad()
def build_raw_embeddings_kv(
    model, embed, prefix_kv: tuple, middle_records: list[dict],
) -> tuple:
    """Sequentially extend prefix_kv with each middle turn forwarded
    via inputs_embeds (input embeddings of the same token ids), with
    explicit position_ids matching the original 1:1 layout.

    The SUFFIX is intentionally NOT included here — it's passed as part
    of the generation prompt with start_pos=suffix_start_pos. See the
    matching comment in build_full_context_kv: this asymmetry is
    deliberate, the two paths are equivalent for a causal model, and
    differences in raw_embeddings vs full_context answers are the
    first signal that something in this asymmetry is broken.
    """
    device = model.device
    running = prefix_kv
    for r in middle_records:
        ids = r["ids"].to(device)
        emb = embed(ids)  # (1, n_tokens, D)
        pos = torch.arange(
            r["start_pos"], r["start_pos"] + r["n_tokens"],
            device=device, dtype=torch.long,
        ).unsqueeze(0)
        out = model(
            inputs_embeds=emb,
            past_key_values=wrap_legacy_kv(running, model.config),
            position_ids=pos,
            use_cache=True,
        )
        running = detach_kv(to_legacy_kv(out.past_key_values))
        del out, emb, ids
        torch.cuda.empty_cache()
    return running


# ----------------------------- generation ----------------------------

def run_questions_full(
    model, tokenizer, kv_legacy: tuple, label: str, stop_ids: set[int],
) -> tuple[dict, float]:
    """generate_with_kv (auto-positioning) for full_context."""
    answers: dict[str, str] = {}
    total = 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        prompt = f"\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        try:
            ans = generate_with_kv(
                model, tokenizer, prompt, kv_legacy, MAX_NEW_TOKENS, stop_ids
            )
        except Exception as e:
            ans = f"<error: {type(e).__name__}: {e}>"
            traceback.print_exc()
        dt = time.perf_counter() - t0
        total += dt
        answers[f"Q{i}"] = ans
        print(f"  [{label}] Q{i} {dt:.1f}s: {ans[:120]!r}")
    return answers, total


def run_questions_truncated(
    model, tokenizer, kv_legacy: tuple, suffix_text: str, label: str, stop_ids: set[int],
) -> tuple[dict, float]:
    """generate_with_kv with suffix in prompt; auto-positions place the
    suffix's first token at prefix_len (correct per spec for the
    truncated condition).
    """
    answers: dict[str, str] = {}
    total = 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        prompt = suffix_text + f"\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        try:
            ans = generate_with_kv(
                model, tokenizer, prompt, kv_legacy, MAX_NEW_TOKENS, stop_ids
            )
        except Exception as e:
            ans = f"<error: {type(e).__name__}: {e}>"
            traceback.print_exc()
        dt = time.perf_counter() - t0
        total += dt
        answers[f"Q{i}"] = ans
        print(f"  [{label}] Q{i} {dt:.1f}s: {ans[:120]!r}")
    return answers, total


def run_questions_with_positions(
    model, tokenizer, kv_legacy: tuple, suffix_text: str,
    suffix_start_pos: int, label: str, stop_ids: set[int],
) -> tuple[dict, float]:
    """For compressed and raw_embeddings: cache contains prefix +
    middle (without suffix). The suffix goes into the generation
    prompt and gets explicit position_ids starting at suffix_start_pos
    so it lives at its original-conversation positions regardless of
    cache physical length.
    """
    answers: dict[str, str] = {}
    total = 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        prompt = suffix_text + f"\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        try:
            ans = manual_greedy_with_positions(
                model, tokenizer, prompt, kv_legacy,
                start_pos=suffix_start_pos, max_new_tokens=MAX_NEW_TOKENS,
                stop_ids=stop_ids,
            )
        except Exception as e:
            ans = f"<error: {type(e).__name__}: {e}>"
            traceback.print_exc()
        dt = time.perf_counter() - t0
        total += dt
        answers[f"Q{i}"] = ans
        print(f"  [{label}] Q{i} {dt:.1f}s: {ans[:120]!r}")
    return answers, total


# -------------------------------- main -------------------------------

def main():
    print("=" * 78)
    print("  Sequential multi-turn V-only compression test")
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

    # ---- Load + split + dry-run printout ----
    split = load_and_split(TRANSCRIPT_PATH, tokenizer, TOKEN_BUDGET)
    print_dry_run(split)

    prefix_records = split["prefix_records"]
    middle_records = split["middle_records"]
    suffix_records = split["suffix_records"]
    prefix_len = split["prefix_len"]
    suffix_len = split["suffix_len"]
    suffix_start_pos = split["suffix_start_pos"]
    suffix_text = "".join(r["text"] for r in suffix_records)

    # ---- Compute prefix KV (no_grad, detached) ----
    print()
    print("Computing prefix KV ...")
    prefix_ids = prefix_records[0]["ids"].to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(input_ids=prefix_ids, use_cache=True)
    prefix_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out
    torch.cuda.empty_cache()
    print(f"  prefix_kv kv_seq_len={kv_seq_len(prefix_kv)} (= prefix_len={prefix_len}). {vram_str()}")

    # ---- RoPE source check ----
    rope_verified, rope_finding = verify_rope_for_qwen()
    if rope_verified:
        print(f"[rope_check] CONFIRMED — RoPE applies to Q and K only. line: {rope_finding}")
    else:
        print(f"[rope_check] WARNING — could not confirm: {rope_finding}")

    # ---- Position-id runtime check (hard halt on failure) ----
    print()
    print("Verifying explicit position_ids take effect with past_key_values present ...")
    posid_check = verify_position_ids_take_effect(model, prefix_kv, prefix_len, model_dtype)
    print(f"  k_diff_max={posid_check['k_diff_max']:.4f} "
          f"v_diff_max={posid_check['v_diff_max']:.4f} "
          f"took_effect={posid_check['took_effect']}")
    if not posid_check["took_effect"]:
        raise AssertionError(
            f"position_ids ignored: {posid_check} — sequential compression cannot work."
        )

    # ---- Storage for results ----
    answers_by_condition: dict[str, dict] = {}
    times_by_condition: dict[str, float] = {}
    compression_runs: dict[str, dict] = {}

    # ---- Condition: full_context ----
    print()
    print("=" * 78)
    print("  Condition: full_context")
    print("=" * 78)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    full_kv = build_full_context_kv(
        model, prefix_records, middle_records, suffix_records,
    )
    print(f"  full_kv kv_seq_len={kv_seq_len(full_kv)}. {vram_str()}")
    answers_by_condition["full_context"], times_by_condition["full_context"] = run_questions_full(
        model, tokenizer, full_kv, "full_context", stop_ids,
    )
    del full_kv
    torch.cuda.empty_cache()
    print(f"  after full_context — {vram_str()}")

    # ---- Condition: no_middle_raw_embeddings ----
    print()
    print("=" * 78)
    print("  Condition: no_middle_raw_embeddings (sanity check)")
    print("=" * 78)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    raw_kv = build_raw_embeddings_kv(model, embed, prefix_kv, middle_records)
    print(f"  raw_kv kv_seq_len={kv_seq_len(raw_kv)} "
          f"(= prefix_len + sum(middle_tokens) = {prefix_len + split['total_middle_tokens']}). "
          f"{vram_str()}")
    answers_by_condition["no_middle_raw_embeddings"], times_by_condition["no_middle_raw_embeddings"] = (
        run_questions_with_positions(
            model, tokenizer, raw_kv, suffix_text, suffix_start_pos,
            "raw_embeddings", stop_ids,
        )
    )
    del raw_kv
    torch.cuda.empty_cache()
    print(f"  after raw_embeddings — {vram_str()}")

    # ---- Conditions: uniform compression at each COMPRESSION_RATIOS ----
    for ratio in COMPRESSION_RATIOS:
        ratio_key = f"{int(ratio)}x" if ratio == int(ratio) else f"{ratio:.1f}x"
        cond_label = f"compressed_{ratio_key}"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        running_kv, per_turn_meta = run_sequential_compression(
            model, embed, model_dtype, prefix_kv,
            middle_records, ratio, ratio_key,
        )
        total_virtual = sum(m["virtual_tokens"] for m in per_turn_meta)
        total_original = sum(m["original_tokens"] for m in per_turn_meta)
        overall_ratio = total_original / max(total_virtual, 1)
        total_time = sum(m["time_s"] for m in per_turn_meta)
        compression_runs[ratio_key] = {
            "ratio": ratio,
            "total_original_tokens": total_original,
            "total_virtual_tokens": total_virtual,
            "overall_ratio": overall_ratio,
            "total_time_s": total_time,
            "per_turn": per_turn_meta,
        }
        print(f"  [{ratio_key}] sweep done: "
              f"{total_original} → {total_virtual} virtual ({overall_ratio:.2f}x), "
              f"{total_time:.1f}s. cache_phys={kv_seq_len(running_kv)}. {vram_str()}")

        answers_by_condition[cond_label], times_by_condition[cond_label] = (
            run_questions_with_positions(
                model, tokenizer, running_kv, suffix_text, suffix_start_pos,
                cond_label, stop_ids,
            )
        )
        del running_kv
        torch.cuda.empty_cache()
        print(f"  after {cond_label} — {vram_str()}")

    # ---- Condition: hybrid_2x (perplexity-driven verbatim/compress mix) ----
    surprise_scores = score_perplexity_for_turns(model, prefix_kv, middle_records)
    decisions, decision_meta = decide_hybrid_modes(
        middle_records, surprise_scores,
        target_ratio=HYBRID_TARGET_RATIO,
        max_compress_ratio=HYBRID_MAX_COMPRESS_RATIO,
    )
    print_decision_table(middle_records, surprise_scores, decisions, decision_meta)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    hybrid_running_kv, hybrid_per_turn = run_hybrid_compression(
        model, embed, model_dtype, prefix_kv,
        middle_records, decisions, surprise_scores, label="hybrid_2x",
    )
    h_total_virtual = sum(m["virtual_tokens"] for m in hybrid_per_turn)
    h_total_original = sum(m["original_tokens"] for m in hybrid_per_turn)
    h_overall_ratio = h_total_original / max(h_total_virtual, 1)
    h_total_time = sum(m["time_s"] for m in hybrid_per_turn)
    compression_runs["hybrid_2x"] = {
        "target_overall_ratio": HYBRID_TARGET_RATIO,
        "actual_overall_ratio": h_overall_ratio,
        "total_original_tokens": h_total_original,
        "total_virtual_tokens": h_total_virtual,
        "surprise_threshold": decision_meta["surprise_threshold"],
        "compress_ratio_used": decision_meta["compress_ratio_used"],
        "max_compress_ratio_cap": HYBRID_MAX_COMPRESS_RATIO,
        "n_verbatim_turns": decision_meta["n_verbatim_turns"],
        "n_compressed_turns": decision_meta["n_compressed_turns"],
        "degenerated_to_uniform": decision_meta["degenerated_to_uniform"],
        "total_time_s": h_total_time,
        "per_turn": hybrid_per_turn,
    }
    print(f"  [hybrid_2x] done: {h_total_original} → {h_total_virtual} "
          f"({h_overall_ratio:.2f}x actual vs {HYBRID_TARGET_RATIO:.2f}x target), "
          f"{h_total_time:.1f}s. cache_phys={kv_seq_len(hybrid_running_kv)}. {vram_str()}")
    answers_by_condition["hybrid_2x"], times_by_condition["hybrid_2x"] = (
        run_questions_with_positions(
            model, tokenizer, hybrid_running_kv, suffix_text, suffix_start_pos,
            "hybrid_2x", stop_ids,
        )
    )
    del hybrid_running_kv
    torch.cuda.empty_cache()
    print(f"  after hybrid_2x — {vram_str()}")

    # ---- Condition: truncated ----
    print()
    print("=" * 78)
    print("  Condition: truncated")
    print("=" * 78)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    answers_by_condition["truncated"], times_by_condition["truncated"] = run_questions_truncated(
        model, tokenizer, prefix_kv, suffix_text, "truncated", stop_ids,
    )
    print(f"  after truncated — {vram_str()}")

    # ---- Save results JSON ----
    context_split_json = {
        "prefix": {
            "turn_index": prefix_records[0]["turn_index_1based"],
            "role": prefix_records[0]["role"],
            "tokens": prefix_records[0]["n_tokens"],
            "preview": prefix_records[0]["preview"],
        },
        "middle": [
            {
                "turn_index": r["turn_index_1based"],
                "role": r["role"],
                "tokens": r["n_tokens"],
                "start_pos": r["start_pos"],
                "end_pos": r["end_pos"],
                "preview": r["text"][:200],
            }
            for r in middle_records
        ],
        "suffix": {
            "turn_index": suffix_records[0]["turn_index_1based"],
            "role": suffix_records[0]["role"],
            "tokens": suffix_records[0]["n_tokens"],
            "preview": suffix_records[0]["preview"],
        },
    }

    results = {
        "config": {
            "model_id": MODEL_ID,
            "compression_ratios": COMPRESSION_RATIOS,
            "hybrid_target_ratio": HYBRID_TARGET_RATIO,
            "hybrid_max_compress_ratio": HYBRID_MAX_COMPRESS_RATIO,
            "steps_formula": "max(round(300 * turn_tokens / 250), 100)",
            "learning_rate": LR,
            "target": "V-only, perplexity-weighted pool, per-layer normalized",
            "position_interpolation": True,
            "rope_verified": rope_verified,
            "rope_finding": rope_finding,
            "position_ids_take_effect": posid_check,
            "prefix_turns": NUM_PREFIX,
            "suffix_turns": NUM_SUFFIX,
            "middle_turns": len(middle_records),
            "prefix_tokens": prefix_len,
            "suffix_tokens": suffix_len,
            "total_middle_tokens": split["total_middle_tokens"],
            "suffix_start_pos": suffix_start_pos,
            "stop_token_ids": sorted(stop_ids),
        },
        "context_split": context_split_json,
        "surprise_scores": surprise_scores,
        "compression_runs": compression_runs,
        "conditions": {
            cond: {
                "answers": answers_by_condition.get(cond, {}),
                "total_gen_time_s": times_by_condition.get(cond, float("nan")),
            }
            for cond in [
                "full_context",
                "no_middle_raw_embeddings",
                "compressed_1.5x",
                "compressed_2x",
                "hybrid_2x",
                "truncated",
            ]
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")

    # ---- Console summary: uniform compression tables ----
    for ratio in COMPRESSION_RATIOS:
        ratio_key = f"{int(ratio)}x" if ratio == int(ratio) else f"{ratio:.1f}x"
        run = compression_runs.get(ratio_key)
        if run is None:
            continue
        print()
        print(f"=== {ratio} COMPRESSION ===")
        print(f"  {'Turn':>5} | {'Role':<10} | {'Tokens':>6} | {'Virtual':>7} | "
              f"{'Steps':>5} | {'Loss start → end':>22} | {'Time':>6}")
        for m in run["per_turn"]:
            print(f"  {m['turn_index']:>5} | {m['role']:<10} | "
                  f"{m['original_tokens']:>6} | {m['virtual_tokens']:>7} | "
                  f"{m['steps']:>5} | "
                  f"{m['initial_loss']:>9.4f} → {m['final_loss']:<9.4f} | "
                  f"{m['time_s']:>5.1f}s")
        print(f"  Total: {run['total_original_tokens']} middle tokens → "
              f"{run['total_virtual_tokens']} virtual "
              f"({run['overall_ratio']:.2f}x overall)  "
              f"in {run['total_time_s']:.1f}s")

    # ---- Console summary: hybrid compression table ----
    h_run = compression_runs.get("hybrid_2x")
    if h_run is not None:
        print()
        print("=== HYBRID_2x COMPRESSION (perplexity-driven verbatim/compress) ===")
        print(f"  target={h_run['target_overall_ratio']:.2f}x, "
              f"actual={h_run['actual_overall_ratio']:.2f}x, "
              f"verbatim={h_run['n_verbatim_turns']}, "
              f"compressed={h_run['n_compressed_turns']}, "
              f"degenerated={h_run['degenerated_to_uniform']}")
        thr = h_run.get("surprise_threshold")
        cru = h_run.get("compress_ratio_used")
        print(f"  surprise_threshold={'N/A' if thr is None else f'{thr:.4f}'}, "
              f"compress_ratio_used={'N/A' if cru is None else f'{cru:.3f}x'} "
              f"(cap={h_run['max_compress_ratio_cap']:.1f}x)")
        print(f"  {'Turn':>5} | {'Role':<10} | {'Mode':<10} | {'Tokens':>6} | "
              f"{'Virtual':>7} | {'Steps':>5} | {'Loss start → end':>22} | {'Time':>6}")
        for m in h_run["per_turn"]:
            mode = m.get("mode", "compressed")
            i_loss = m["initial_loss"]
            f_loss = m["final_loss"]
            print(f"  {m['turn_index']:>5} | {m['role']:<10} | {mode:<10} | "
                  f"{m['original_tokens']:>6} | {m['virtual_tokens']:>7} | "
                  f"{m['steps']:>5} | "
                  f"{i_loss:>9.4f} → {f_loss:<9.4f} | "
                  f"{m['time_s']:>5.1f}s")
        print(f"  Total: {h_run['total_original_tokens']} middle tokens → "
              f"{h_run['total_virtual_tokens']} virtual "
              f"({h_run['actual_overall_ratio']:.2f}x overall)  "
              f"in {h_run['total_time_s']:.1f}s")

    # ---- Console summary: per-question table ----
    print()
    print("=" * 78)
    print("  ANSWERS PER QUESTION")
    print("=" * 78)
    cond_labels = [
        "full_context",
        "no_middle_raw_embeddings",
        "compressed_1.5x",
        "compressed_2x",
        "hybrid_2x",
        "truncated",
    ]
    for i, q in enumerate(QUESTIONS, start=1):
        print()
        print(f"Q{i}: {q}")
        for cond in cond_labels:
            ans = answers_by_condition.get(cond, {}).get(f"Q{i}", "")
            line = ans.replace("\n", " ")
            if len(line) > 160:
                line = line[:160] + "…"
            print(f"  [{cond:<26}] {line}")

    # ---- Sanity check: raw_embeddings vs full_context ----
    full = answers_by_condition.get("full_context", {})
    raw = answers_by_condition.get("no_middle_raw_embeddings", {})
    diverged: list[str] = []
    for i in range(1, len(QUESTIONS) + 1):
        key = f"Q{i}"
        f_a = (full.get(key) or "").strip()
        r_a = (raw.get(key) or "").strip()
        if not f_a or not r_a:
            continue
        if f_a.startswith("<error") or r_a.startswith("<error"):
            continue
        if f_a != r_a:
            diverged.append(key)
    if diverged:
        print()
        print("!" * 78)
        print(f"  WARNING: raw_embeddings DIVERGED from full_context on: {diverged}")
        print("  inputs_embeds path is not reproducing the input_ids path on these.")
        print("  Compressed-condition results may be unreliable.")
        print("  See comments in build_full_context_kv / build_raw_embeddings_kv —")
        print("  the suffix is in the cache for full_context but in the prompt for")
        print("  raw_embeddings; that asymmetry is the most likely source.")
        print("!" * 78)
    else:
        print()
        print("[sanity] raw_embeddings matches full_context on all questions — pipeline OK.")


if __name__ == "__main__":
    main()
