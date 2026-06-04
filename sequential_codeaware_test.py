"""Sequential multi-turn compression with code-aware pattern exemption.

V-uniqueness (results_sequential_exemption.json) picked exempt tokens by
a statistical signal and kept noisy fragments (" ===", "FO", "CUS")
alongside real code. It also missed "js" (suffix of "bcryptjs") because
that subword is statistically unremarkable — yet obviously part of a
package name to anyone reading code.

This script swaps the statistical signal for domain knowledge. The
matcher is deliberately TARGETED at the patterns whose tokens actually
cause compression failures:
  - file paths (any span with a "/" separator)
  - package / module names (the require(...) / import-from target)
  - short single-quoted string literals (<=30 chars: 'dev-secret',
    'pending', not JSON-wrapped file bodies)
  - span expansion +-1 token (so e.g. "bcrypt" pulls in the adjacent
    "js")
camelCase/snake_case identifiers, SQL keywords, and HTTP methods/status
codes are intentionally NOT matched: they cover most tokens in any
source-code turn (boilerplate that compresses fine) and would inflate
exemption so much that nothing gets compressed. We keep matched tokens
verbatim (exact KV at original positions) and compress the rest with the
standard V-only / perplexity-weighted / position-interpolated recipe.

Exemption is CONTENT-DRIVEN, not a fixed rate: a code-heavy turn may
exempt 25%, a short acknowledgement 5%. That is a feature — it matches
exemption to code density. The per-turn rate is printed.

Conditions (one model load):
  1. full_context           — all turns verbatim (ceiling)
  2. codeaware_exempt_2x     — code-pattern exemption, 2.0x target
  3. codeaware_exempt_2.5x   — 2.5x target
  4. codeaware_exempt_3x     — 3.0x target (stretch; headroom for future
                               KV-eviction experiments)
  5. sequential_uniform_2x   — no exemption baseline (run live)
  6. no_middle_truncated     — prefix + suffix only (floor)

Eight questions, copied verbatim from sequential_test.py. Keyword scorers
copied from sequential_exemption_test.py, calibrated so full_context=8/8
and truncated=0/8.

Standalone — no imports from other project files. No V-uniqueness. No K
loss. No uv.
"""

from __future__ import annotations

import inspect
import json
import re
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
RESULTS_PATH = Path(__file__).parent / "results_sequential_codeaware.json"

TOKEN_BUDGET = 4096
NUM_PREFIX = 1
NUM_SUFFIX = 1

TARGET_RATIOS = [2.0, 2.5, 3.0]
UNIFORM_RATIO = 2.0
SPAN_EXPANSION = 1  # expand each exempt region by this many tokens per side

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


# --------------------- code-aware pattern matcher --------------------
# Each category maps a name -> (compiled regex, capture_group_index).
# capture_group 0 = whole match; >0 = a specific group (used to exempt
# just the package/import target rather than the surrounding call).

# Targeted matcher: only the patterns whose tokens actually cause
# compression failures. Deliberately NOT included: camelCase/PascalCase,
# snake_case, SQL keywords, HTTP methods/status codes — those match most
# tokens in any source-code turn (boilerplate that compresses fine) and
# inflate exemption to the point that nothing gets compressed.
MAX_STRING_LITERAL_CHARS = 30  # cap single-quoted literal span so JSON-wrapped
                               # tool-result file bodies aren't mistaken for one
                               # giant string literal.

# (name, pattern, capture_group, max_span_chars-or-None)
CODE_PATTERNS: list[tuple[str, str, int, int | None]] = [
    # File paths: any span containing a "/" separator.
    ("file_paths", r"[\w.\-]+/[\w./\-]+", 0, None),
    # Package / module names: the require(...) / import-from target.
    ("package_names", r"require\(\s*'([^'\n]+)'\s*\)", 1, None),
    ("package_names", r"require\(\s*\"([^\"\n]+)\"\s*\)", 1, None),
    ("package_names", r"\bfrom\s+'([^'\n]+)'", 1, None),
    ("package_names", r"\bfrom\s+\"([^\"\n]+)\"", 1, None),
    # Short single-quoted string literals ('dev-secret', 'pending', ...).
    # Single-quoted only (double quotes are the JSON tool-result wrapper)
    # and length-capped so file bodies are never matched as a literal.
    ("string_literals", r"'[^'\n]*'", 0, MAX_STRING_LITERAL_CHARS),
]

_COMPILED_PATTERNS = [
    (name, re.compile(pat), grp, cap) for name, pat, grp, cap in CODE_PATTERNS
]


def select_code_exempt_indices(
    text: str, token_ids: torch.Tensor, tokenizer,
) -> tuple[torch.Tensor, dict[str, list[str]]]:
    """Return (exempt_token_indices ascending, matched_patterns).

    Content-driven and cache-independent: identify code-relevant character
    spans via regex over the decoded turn text, map them back to token
    indices using the fast tokenizer's offset mapping, expand each exempt
    region by SPAN_EXPANSION tokens per side (so e.g. "bcrypt" pulls in
    the adjacent "js"), and dedupe.

    matched_patterns maps each category to the unique matched substrings
    (sorted) for diagnostics.
    """
    n = int(token_ids.shape[0]) if token_ids.dim() == 1 else int(token_ids.shape[1])
    empty = (torch.empty(0, dtype=torch.long), {})

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if len(offsets) != n:
        # Boundary mismatch (e.g. the budget-truncated final middle turn).
        # Skip exemption for this turn rather than risk misaligned spans.
        print(f"  [codeaware] offset/token length mismatch "
              f"({len(offsets)} vs {n}); skipping exemption for this turn.")
        return empty

    def span_to_token_indices(cstart: int, cend: int) -> list[int]:
        return [
            i for i, (a, b) in enumerate(offsets)
            if b > a and a < cend and b > cstart  # overlap, skip empty offsets
        ]

    exempt: set[int] = set()
    matched: dict[str, set[str]] = {}
    for name, rgx, grp, cap in _COMPILED_PATTERNS:
        for m in rgx.finditer(text):
            try:
                cstart, cend = m.span(grp)
            except (IndexError, re.error):
                continue
            if cend <= cstart:
                continue
            if cap is not None and (cend - cstart) > cap:
                continue  # skip over-long quoted spans (JSON-wrapped file bodies)
            toks = span_to_token_indices(cstart, cend)
            if not toks:
                continue
            exempt.update(toks)
            matched.setdefault(name, set()).add(m.group(grp))

    # Span expansion +-SPAN_EXPANSION tokens per side.
    if exempt and SPAN_EXPANSION > 0:
        expanded: set[int] = set()
        for i in exempt:
            for j in range(i - SPAN_EXPANSION, i + SPAN_EXPANSION + 1):
                if 0 <= j < n:
                    expanded.add(j)
        exempt = expanded

    indices = torch.tensor(sorted(exempt), dtype=torch.long)
    matched_lists = {k: sorted(v) for k, v in matched.items()}
    return indices, matched_lists


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


def slice_kv(legacy: tuple, start: int, end: int | None = None) -> tuple:
    if end is None:
        return tuple((k[:, :, start:, :], v[:, :, start:, :]) for k, v in legacy)
    return tuple((k[:, :, start:end, :], v[:, :, start:end, :]) for k, v in legacy)


def gather_kv(legacy: tuple, indices: torch.Tensor) -> tuple:
    return tuple(
        (k.index_select(2, indices), v.index_select(2, indices))
        for k, v in legacy
    )


def concat_kv(legacy_a: tuple, legacy_b: tuple) -> tuple:
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


# ----------------------------- generation ----------------------------

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


def run_questions_full(
    model, tokenizer, kv_legacy: tuple, label: str, stop_ids: set[int],
) -> tuple[dict, float]:
    answers: dict[str, str] = {}
    total = 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        prompt = f"\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        try:
            ans = generate_with_kv(model, tokenizer, prompt, kv_legacy, MAX_NEW_TOKENS, stop_ids)
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
    answers: dict[str, str] = {}
    total = 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        prompt = suffix_text + f"\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        try:
            ans = generate_with_kv(model, tokenizer, prompt, kv_legacy, MAX_NEW_TOKENS, stop_ids)
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
    answers: dict[str, str] = {}
    total = 0.0
    for i, q in enumerate(QUESTIONS, start=1):
        prompt = suffix_text + f"\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        try:
            ans = manual_greedy_with_positions(
                model, tokenizer, prompt, kv_legacy,
                start_pos=suffix_start_pos, max_new_tokens=MAX_NEW_TOKENS, stop_ids=stop_ids,
            )
        except Exception as e:
            ans = f"<error: {type(e).__name__}: {e}>"
            traceback.print_exc()
        dt = time.perf_counter() - t0
        total += dt
        answers[f"Q{i}"] = ans
        print(f"  [{label}] Q{i} {dt:.1f}s: {ans[:120]!r}")
    return answers, total


# --------------------------- score functions -------------------------
# Keyword-based, grounded in the full_context answers from
# results_sequential_v2.json. Calibrated: full_context=8/8, truncated=0/8.

def score_q1(ans: str) -> bool:
    a = ans.lower()
    return ("express" in a) and ("sqlite" in a) and ("task" in a)


def score_q2(ans: str) -> bool:
    a = ans.lower()
    return ("users" in a) and ("tasks" in a) and ("reports" in a)


def score_q3(ans: str) -> bool:
    a = ans.lower()
    has_core = ("bold" in a) and ("italic" in a) and ("code" in a)
    hallucinated = ("link" in a) or ("image" in a)
    return has_core and not hallucinated


def score_q4(ans: str) -> bool:
    a = ans.lower()
    return ("jwt" in a) and ("bcryptjs" in a)


def score_q5(ans: str) -> bool:
    a = ans.lower()
    return (
        "title" in a
        and "pending" in a
        and ("in_progress" in a or "in progress" in a)
        and "done" in a
    )


def score_q6(ans: str) -> bool:
    a = ans.lower()
    return ("auth" in a) and ("task" in a) and ("report" in a)


def score_q7(ans: str) -> bool:
    a = ans.lower()
    return "db.js" in a


def score_q8(ans: str) -> bool:
    a = ans.lower()
    return (
        "db.js" in a
        and "markdown.js" in a
        and "validate" in a
        and "auth.js" in a
        and "reports.js" in a
    )


SCORE_FNS: list[Callable[[str], bool]] = [
    score_q1, score_q2, score_q3, score_q4, score_q5, score_q6, score_q7, score_q8,
]


def score_answers(answers: dict[str, str]) -> tuple[dict[str, int], int]:
    scores: dict[str, int] = {}
    total = 0
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
            "turn_index_1based": i,
            "role": role,
            "text": text,
            "ids_list": ids_list,
            "n_tokens": len(ids_list),
            "preview": text[:200],
        })

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
          f"{p['n_tokens']} tokens  pos=[0, {split['prefix_len'] - 1}]")
    print(f"MIDDLE  ({len(split['middle_records'])} turns, "
          f"{split['total_middle_tokens']} tokens):")
    for r in split["middle_records"]:
        print(f"  turn {r['turn_index_1based']:>3} ({r['role']:<10}): "
              f"{r['n_tokens']:>4} tok  pos=[{r['start_pos']}, {r['end_pos']}]  "
              f"{r['text'][:80]!r}")
    s = split["suffix_records"][0]
    print(f"SUFFIX  (turn {s['turn_index_1based']}, role={s['role']}): "
          f"{s['n_tokens']} tokens  pos=[{s['start_pos']}, {s['end_pos']}]")
    print(f"Totals: prefix={split['prefix_len']}, "
          f"middle={split['total_middle_tokens']}, suffix={split['suffix_len']}, "
          f"suffix_start_pos={split['suffix_start_pos']}.")


# ----------------------- inference cache builds ----------------------

def _concat_ids(records: list[dict], device) -> torch.Tensor:
    chunks = [r["ids"].to(device) for r in records]
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def build_full_context_kv(
    model, prefix_records: list[dict],
    middle_records: list[dict], suffix_records: list[dict],
) -> tuple:
    device = model.device
    all_ids = _concat_ids([*prefix_records, *middle_records, *suffix_records], device)
    out = model(input_ids=all_ids, use_cache=True)
    return detach_kv(to_legacy_kv(out.past_key_values))


# ----------------------- single-turn compression ---------------------

def steps_for(n_tokens: int) -> int:
    return max(round(BASE_STEPS * n_tokens / BASE_TOKENS), MIN_STEPS)


def run_optimize_loop(
    model, model_dtype, past_kv_legacy: tuple, slice_offset: int,
    init: torch.Tensor, virtual_pos: torch.Tensor,
    target_v_list: list[torch.Tensor], v_mag2_list: list[torch.Tensor],
    num_steps: int, label: str,
) -> tuple[torch.Tensor, list[float]]:
    """Adam over fp32 virtual tokens; V-only normalized loss."""
    device = model.device
    virtual = init.detach().clone().to(dtype=torch.float32, device=device)
    virtual.requires_grad_(True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    optimizer = torch.optim.Adam([virtual], lr=LR)
    losses: list[float] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
            live = extract_new_kv_grad_safe(out.past_key_values, slice_offset)
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
                print(f"    [{label}] step {step:>4}/{num_steps}: loss={loss_val:.4f}  {vram_str()}")
            elif (step + 1) % LOG_EVERY == 0:
                print(f"    [{label}] step {step:>4}/{num_steps}: loss={loss_val:.4f}")
            del out, loss, inputs_embeds
    finally:
        model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return virtual.detach(), losses


def compress_turn_codeaware(
    model, embed, model_dtype, tokenizer, running_kv: tuple,
    turn_record: dict, ratio: float, use_exemption: bool,
    exempt_idx_cpu: torch.Tensor, matched_patterns: dict, label: str,
) -> tuple[tuple, dict]:
    """Compress one middle turn against the running cache.

    use_exemption=True : keep the precomputed code-pattern exempt tokens
                         verbatim (exact KV at original positions),
                         compress the rest.
    use_exemption=False: compress ALL tokens (uniform baseline).

    exempt_idx_cpu / matched_patterns are precomputed by
    select_code_exempt_indices (ratio-independent); ignored when
    use_exemption is False.
    """
    device = model.device
    turn_ids = turn_record["ids"].to(device)
    n_tokens = int(turn_record["n_tokens"])
    turn_start_pos = int(turn_record["start_pos"])
    phys_off = kv_seq_len(running_kv)

    t_start = time.perf_counter()

    # ---- (a) Target forward: perplexity + new-turn KV (no grad) ----
    target_pos = torch.arange(
        turn_start_pos, turn_start_pos + n_tokens, device=device, dtype=torch.long,
    ).unsqueeze(0)
    with torch.no_grad():
        kv_out = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(running_kv, model.config),
            position_ids=target_pos,
            output_hidden_states=False,
            use_cache=True,
        )
    perplexity = compute_perplexity_weights(kv_out.logits, turn_ids)
    full_kv = detach_kv(to_legacy_kv(kv_out.past_key_values))
    del kv_out
    new_turn_kv = slice_kv(full_kv, start=phys_off)
    v_full: list[torch.Tensor] = [v for (_k, v) in new_turn_kv]

    # ---- (b) Exempt / compressible split ----
    if use_exemption:
        exempt_local = exempt_idx_cpu
    else:
        exempt_local = torch.empty(0, dtype=torch.long)
        matched_patterns = {}
    exempt_set = set(exempt_local.tolist())
    compressible_idx_cpu = torch.tensor(
        [i for i in range(n_tokens) if i not in exempt_set], dtype=torch.long,
    )
    num_exempt = int(exempt_local.shape[0])
    num_compressible = int(compressible_idx_cpu.shape[0])
    exempt_rate = num_exempt / max(n_tokens, 1)
    exempt_idx_dev = exempt_local.to(device)
    compressible_idx_dev = compressible_idx_cpu.to(device)

    # ---- (c) running_plus_exempt cache ----
    if num_exempt > 0:
        exempt_kv = gather_kv(new_turn_kv, exempt_idx_dev)
        running_plus_exempt = detach_kv(concat_kv(running_kv, exempt_kv))
    else:
        exempt_kv = None
        running_plus_exempt = running_kv
    pe_len = kv_seq_len(running_plus_exempt)

    # ---- (d) Budget ----
    total_budget = max(round(n_tokens / ratio), 1)
    if num_compressible == 0:
        # Entire turn exempted (rare, tiny code-only turns). Nothing to
        # compress; the running cache already holds the exempt entries.
        dt = time.perf_counter() - t_start
        new_phys = kv_seq_len(running_plus_exempt)
        print(f"  [{label}] turn {turn_record['turn_index_1based']}: "
              f"N={n_tokens}, exempt={num_exempt} (100%), virtual=0; "
              f"all-verbatim. cache_phys {phys_off}→{new_phys}; {dt:.1f}s")
        meta = {
            "turn_index": turn_record["turn_index_1based"],
            "role": turn_record["role"], "n_tokens": n_tokens,
            "num_exempt": num_exempt, "exempt_rate_pct": round(exempt_rate * 100, 2),
            "num_virtual": 0, "num_compressible": 0, "total_budget": total_budget,
            "effective_ratio": 1.0, "num_steps": 0,
            "initial_loss": 0.0, "final_loss": 0.0, "loss_curve": [],
            "time_s": dt,
            "exempt_tokens_preview": _decode_preview(tokenizer, turn_ids, exempt_local, 20),
            "matched_patterns": matched_patterns,
            "cache_phys_before": phys_off, "cache_phys_after": new_phys,
        }
        del perplexity, v_full, new_turn_kv, full_kv
        if exempt_kv is not None:
            del exempt_kv
        torch.cuda.empty_cache()
        return running_plus_exempt, meta

    num_virtual = max(total_budget - num_exempt, 1)
    num_virtual = min(num_virtual, num_compressible)
    effective_ratio = num_compressible / num_virtual

    # ---- (e) Compressible-only V targets ----
    ppl_compressible = perplexity.index_select(0, compressible_idx_dev)
    target_v_list: list[torch.Tensor] = []
    v_mag2_list: list[torch.Tensor] = []
    for v_layer in v_full:
        v_comp = v_layer.index_select(2, compressible_idx_dev)
        v_pooled = attention_weighted_pool(v_comp, ppl_compressible, num_virtual).detach().clone()
        target_v_list.append(v_pooled)
        v_mag2_list.append(v_pooled.float().pow(2).mean().detach())
        del v_comp

    # ---- (f) Init + interpolated positions over compressible range ----
    with torch.no_grad():
        turn_emb = embed(turn_ids).squeeze(0)
    turn_emb_comp = turn_emb.index_select(0, compressible_idx_dev)
    init = mean_pool_chunks(turn_emb_comp, num_virtual).detach().clone()
    del turn_emb, turn_emb_comp

    comp_pos_min = turn_start_pos + int(compressible_idx_cpu[0].item())
    comp_pos_max = turn_start_pos + int(compressible_idx_cpu[-1].item())
    virtual_pos = interpolate_positions(comp_pos_min, comp_pos_max, num_virtual, device=device)

    num_steps = steps_for(n_tokens)
    print(f"  [{label}] turn {turn_record['turn_index_1based']}: N={n_tokens}, "
          f"exempt={num_exempt} ({exempt_rate * 100:.1f}%), virtual={num_virtual}, "
          f"eff={effective_ratio:.2f}x on {num_compressible} compressible, steps={num_steps}")

    # ---- (g) Optimize ----
    virtual, losses = run_optimize_loop(
        model, model_dtype, running_plus_exempt, slice_offset=pe_len,
        init=init, virtual_pos=virtual_pos,
        target_v_list=target_v_list, v_mag2_list=v_mag2_list,
        num_steps=num_steps, label=label,
    )

    # ---- (h) Extend running cache: running + exempt + virtual ----
    with torch.no_grad():
        ie = virtual.to(dtype=model_dtype).unsqueeze(0)
        out = model(
            inputs_embeds=ie,
            past_key_values=wrap_legacy_kv(running_plus_exempt, model.config),
            position_ids=virtual_pos,
            use_cache=True,
        )
        new_running_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out, ie

    dt = time.perf_counter() - t_start
    initial_loss = losses[0]
    final_loss = losses[-1]
    new_phys = kv_seq_len(new_running_kv)
    print(f"  [{label}] done: N={n_tokens} → {num_exempt} exempt + {num_virtual} virtual "
          f"= {num_exempt + num_virtual} (target {total_budget}), "
          f"loss {initial_loss:.4f} → {final_loss:.4f}, {dt:.1f}s; "
          f"cache_phys {phys_off}→{new_phys}; {vram_str()}")

    meta = {
        "turn_index": turn_record["turn_index_1based"],
        "role": turn_record["role"],
        "n_tokens": n_tokens,
        "num_exempt": num_exempt,
        "exempt_rate_pct": round(exempt_rate * 100, 2),
        "num_virtual": num_virtual,
        "num_compressible": num_compressible,
        "total_budget": total_budget,
        "effective_ratio": round(effective_ratio, 4),
        "num_steps": num_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_curve": losses,
        "time_s": dt,
        "exempt_tokens_preview": _decode_preview(tokenizer, turn_ids, exempt_local, 20),
        "matched_patterns": matched_patterns,
        "cache_phys_before": phys_off,
        "cache_phys_after": new_phys,
    }

    del target_v_list, v_mag2_list, init, virtual_pos, virtual
    del perplexity, ppl_compressible, v_full, new_turn_kv, full_kv
    if exempt_kv is not None:
        del exempt_kv
    torch.cuda.empty_cache()
    return new_running_kv, meta


def _decode_preview(tokenizer, turn_ids: torch.Tensor, idx_cpu: torch.Tensor, n: int) -> list[str]:
    return [
        tokenizer.decode([int(turn_ids[0, i].item())], skip_special_tokens=False)
        for i in idx_cpu.tolist()[:n]
    ]


def run_sequential(
    model, embed, model_dtype, tokenizer, prefix_kv: tuple,
    middle_records: list[dict], ratio: float, use_exemption: bool,
    exempt_by_turn: dict[int, tuple[torch.Tensor, dict]], label: str,
) -> tuple[tuple, list[dict]]:
    print()
    print("=" * 78)
    print(f"  Sequential compression: {label} "
          f"(ratio={ratio}, code-exemption={'on' if use_exemption else 'off'})")
    print("=" * 78)
    running_kv = prefix_kv
    per_turn_meta: list[dict] = []
    for r in middle_records:
        ti = r["turn_index_1based"]
        exempt_idx, patterns = exempt_by_turn.get(ti, (torch.empty(0, dtype=torch.long), {}))
        running_kv, meta = compress_turn_codeaware(
            model, embed, model_dtype, tokenizer, running_kv, r, ratio,
            use_exemption, exempt_idx, patterns,
            label=f"{label}/turn{ti}",
        )
        per_turn_meta.append(meta)
    return running_kv, per_turn_meta


# -------------------------------- main -------------------------------

def main():
    print("=" * 78)
    print("  Sequential multi-turn compression with code-aware exemption")
    print("=" * 78)

    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if not tokenizer.is_fast:
        raise RuntimeError("Code-aware exemption needs a fast tokenizer (offset mapping).")
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

    split = load_and_split(TRANSCRIPT_PATH, tokenizer, TOKEN_BUDGET)
    print_dry_run(split)

    prefix_records = split["prefix_records"]
    middle_records = split["middle_records"]
    suffix_records = split["suffix_records"]
    prefix_len = split["prefix_len"]
    suffix_start_pos = split["suffix_start_pos"]
    total_middle_tokens = split["total_middle_tokens"]
    suffix_text = "".join(r["text"] for r in suffix_records)

    # ---- Prefix KV ----
    print("\nComputing prefix KV ...")
    prefix_ids = prefix_records[0]["ids"].to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(input_ids=prefix_ids, use_cache=True)
    prefix_kv = detach_kv(to_legacy_kv(out.past_key_values))
    del out
    torch.cuda.empty_cache()
    print(f"  prefix_kv kv_seq_len={kv_seq_len(prefix_kv)} (= prefix_len {prefix_len}). {vram_str()}")

    # ---- RoPE + posid checks ----
    rope_verified, rope_finding = verify_rope_for_qwen()
    if rope_verified:
        print(f"[rope_check] CONFIRMED — RoPE on Q,K only. line: {rope_finding}")
    else:
        print(f"[rope_check] WARNING — could not confirm: {rope_finding}")

    print("\nVerifying explicit position_ids take effect ...")
    posid_check = verify_position_ids_take_effect(model, prefix_kv, prefix_len, model_dtype)
    print(f"  k_diff_max={posid_check['k_diff_max']:.4f} "
          f"v_diff_max={posid_check['v_diff_max']:.4f} "
          f"took_effect={posid_check['took_effect']}")
    if not posid_check["took_effect"]:
        raise AssertionError(
            f"position_ids ignored: {posid_check} — experiment cannot proceed."
        )

    # ---- Precompute code-aware exempt indices per turn (ratio-independent) ----
    print()
    print("=" * 78)
    print("  Code-aware pattern matching (per middle turn)")
    print("=" * 78)
    exempt_by_turn: dict[int, tuple[torch.Tensor, dict]] = {}
    for r in middle_records:
        ti = r["turn_index_1based"]
        idx, patterns = select_code_exempt_indices(r["text"], r["ids"][0], tokenizer)
        exempt_by_turn[ti] = (idx, patterns)
        rate = int(idx.shape[0]) / max(r["n_tokens"], 1) * 100
        cats = ", ".join(f"{k}={len(v)}" for k, v in sorted(patterns.items()))
        print(f"  turn {ti:>3} ({r['role']:<10}): {r['n_tokens']:>4} tok  "
              f"exempt={int(idx.shape[0]):>4} ({rate:4.1f}%)  [{cats}]")

    answers_by_condition: dict[str, dict] = {}
    times_by_condition: dict[str, float] = {}
    per_ratio_summary: dict[str, dict] = {}
    uniform_per_turn: list[dict] = []

    # ---- Condition: full_context (ceiling) ----
    print()
    print("=" * 78)
    print("  Condition: full_context")
    print("=" * 78)
    full_kv = build_full_context_kv(model, prefix_records, middle_records, suffix_records)
    print(f"  full_kv kv_seq_len={kv_seq_len(full_kv)}. {vram_str()}")
    answers_by_condition["full_context"], times_by_condition["full_context"] = run_questions_full(
        model, tokenizer, full_kv, "full_context", stop_ids,
    )
    del full_kv
    torch.cuda.empty_cache()

    # ---- Conditions: codeaware_exempt at each target ratio ----
    for ratio in TARGET_RATIOS:
        ratio_key = f"{ratio:.1f}x"
        cond_label = f"codeaware_exempt_{int(ratio)}x" if ratio == int(ratio) else f"codeaware_exempt_{ratio:.1f}x"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        running_kv, per_turn = run_sequential(
            model, embed, model_dtype, tokenizer, prefix_kv, middle_records,
            ratio, use_exemption=True, exempt_by_turn=exempt_by_turn,
            label=cond_label,
        )
        total_exempt = sum(m["num_exempt"] for m in per_turn)
        total_virtual = sum(m["num_virtual"] for m in per_turn)
        total_positions = total_exempt + total_virtual
        overall_eff = total_middle_tokens / max(total_positions, 1)
        per_ratio_summary[ratio_key] = {
            "total_exempt": total_exempt,
            "total_virtual": total_virtual,
            "total_positions": total_positions,
            "overall_effective_ratio": round(overall_eff, 4),
            "per_turn": per_turn,
        }
        print(f"  [{cond_label}] {total_middle_tokens} tok → {total_exempt} exempt + "
              f"{total_virtual} virtual = {total_positions} positions ({overall_eff:.2f}x overall). "
              f"cache_phys={kv_seq_len(running_kv)}")
        answers_by_condition[cond_label], times_by_condition[cond_label] = (
            run_questions_with_positions(
                model, tokenizer, running_kv, suffix_text, suffix_start_pos,
                cond_label, stop_ids,
            )
        )
        del running_kv
        torch.cuda.empty_cache()

    # ---- Condition: sequential_uniform_2x (baseline) ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    uniform_running_kv, uniform_per_turn = run_sequential(
        model, embed, model_dtype, tokenizer, prefix_kv, middle_records,
        UNIFORM_RATIO, use_exemption=False, exempt_by_turn=exempt_by_turn,
        label="uniform_2x",
    )
    u_virtual = sum(m["num_virtual"] for m in uniform_per_turn)
    u_overall = total_middle_tokens / max(u_virtual, 1)
    print(f"  [uniform_2x] {total_middle_tokens} tok → {u_virtual} virtual "
          f"({u_overall:.2f}x overall). cache_phys={kv_seq_len(uniform_running_kv)}")
    answers_by_condition["sequential_uniform_2x"], times_by_condition["sequential_uniform_2x"] = (
        run_questions_with_positions(
            model, tokenizer, uniform_running_kv, suffix_text, suffix_start_pos,
            "uniform_2x", stop_ids,
        )
    )
    del uniform_running_kv
    torch.cuda.empty_cache()

    # ---- Condition: no_middle_truncated (floor) ----
    print()
    print("=" * 78)
    print("  Condition: no_middle_truncated")
    print("=" * 78)
    answers_by_condition["no_middle_truncated"], times_by_condition["no_middle_truncated"] = (
        run_questions_truncated(model, tokenizer, prefix_kv, suffix_text, "truncated", stop_ids)
    )

    # ---- Score ----
    scores_by_condition: dict[str, dict] = {}
    totals_by_condition: dict[str, int] = {}
    for cond, ans in answers_by_condition.items():
        sc, total = score_answers(ans)
        scores_by_condition[cond] = sc
        totals_by_condition[cond] = total

    # ---- Build JSON ----
    cond_order = [
        "full_context",
        "codeaware_exempt_2x",
        "codeaware_exempt_2.5x",
        "codeaware_exempt_3x",
        "sequential_uniform_2x",
        "no_middle_truncated",
    ]
    results = {
        "config": {
            "model_id": MODEL_ID,
            "exemption_signal": "code_pattern_matching",
            "patterns": [
                "file_paths",
                "package_names",
                "string_literals_single_quoted_max30",
                "span_expansion",
            ],
            "dropped_patterns": [
                "camelCase", "snake_case", "sql_keywords", "http_methods",
                "status_codes",
            ],
            "max_string_literal_chars": MAX_STRING_LITERAL_CHARS,
            "span_expansion": SPAN_EXPANSION,
            "target_ratios": TARGET_RATIOS,
            "uniform_baseline_ratio": UNIFORM_RATIO,
            "middle_turns": len(middle_records),
            "total_middle_tokens": total_middle_tokens,
            "prefix_tokens": prefix_len,
            "suffix_tokens": split["suffix_len"],
            "suffix_start_pos": suffix_start_pos,
            "lr": LR,
            "steps_formula": "max(round(300 * turn_tokens / 250), 100)",
            "target": "V-only, perplexity-weighted pool, per-layer normalized",
            "position_interpolation": True,
            "rope_verified": rope_verified,
            "rope_finding": rope_finding,
            "position_ids_take_effect": posid_check,
            "stop_token_ids": sorted(stop_ids),
            "scoring": "keyword-based, calibrated full_context=8/8 truncated=0/8",
        },
        "exempt_summary": {"per_ratio": {
            rk: {
                "total_exempt": s["total_exempt"],
                "total_virtual": s["total_virtual"],
                "total_positions": s["total_positions"],
                "overall_effective_ratio": s["overall_effective_ratio"],
                "per_turn": [
                    {
                        "turn_index": m["turn_index"],
                        "n_tokens": m["n_tokens"],
                        "num_exempt": m["num_exempt"],
                        "exempt_rate_pct": m["exempt_rate_pct"],
                        "num_virtual": m["num_virtual"],
                        "effective_ratio": m["effective_ratio"],
                        "exempt_tokens_preview": m["exempt_tokens_preview"],
                        "matched_patterns": m["matched_patterns"],
                        "final_loss": m["final_loss"],
                    }
                    for m in s["per_turn"]
                ],
            }
            for rk, s in per_ratio_summary.items()
        }},
        "uniform_summary": {
            "total_virtual_tokens": u_virtual,
            "overall_ratio": round(u_overall, 4),
            "per_turn": [
                {
                    "turn_index": m["turn_index"],
                    "n_tokens": m["n_tokens"],
                    "num_virtual": m["num_virtual"],
                    "final_loss": m["final_loss"],
                }
                for m in uniform_per_turn
            ],
        },
        "conditions": {
            cond: {
                "answers": answers_by_condition.get(cond, {}),
                "scores": scores_by_condition.get(cond, {}),
                "total_correct": totals_by_condition.get(cond, 0),
                "total_gen_time_s": times_by_condition.get(cond, float("nan")),
            }
            for cond in cond_order
        },
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")

    # ---- Console: per-ratio exempt details ----
    for ratio in TARGET_RATIOS:
        rk = f"{ratio:.1f}x"
        s = per_ratio_summary.get(rk)
        if s is None:
            continue
        print()
        print(f"=== {rk} ===")
        for m in s["per_turn"]:
            print(f"  Turn {m['turn_index']:>3}: {m['n_tokens']:>4} tok, "
                  f"exempt={m['num_exempt']:>4} ({m['exempt_rate_pct']:4.1f}%), "
                  f"virtual={m['num_virtual']:>4}, eff={m['effective_ratio']:.2f}x")
            mp = m["matched_patterns"]
            if mp:
                bits = []
                for cat in ("file_paths", "package_names", "string_literals"):
                    if mp.get(cat):
                        sample = ", ".join(mp[cat][:4])
                        bits.append(f"{cat}=[{sample}]")
                if bits:
                    print(f"    Patterns: {'  '.join(bits)}")
        print(f"  TOTAL: {total_middle_tokens} tok → {s['total_exempt']} exempt + "
              f"{s['total_virtual']} virtual = {s['total_positions']} "
              f"({s['overall_effective_ratio']:.2f}x overall)")

    # ---- Console: per-question answers ----
    print()
    print("=" * 78)
    print("  ANSWERS PER QUESTION")
    print("=" * 78)
    sym = {1: "✓", 0: "✗"}
    for i, q in enumerate(QUESTIONS, start=1):
        print()
        print(f"Q{i}: {q}")
        for cond in cond_order:
            ans = answers_by_condition.get(cond, {}).get(f"Q{i}", "")
            mark = sym.get(scores_by_condition.get(cond, {}).get(f"Q{i}", 0), "?")
            line = ans.replace("\n", " ")
            if len(line) > 150:
                line = line[:150] + "…"
            print(f"  Q{i} [{cond:<24}] [{mark}] {line}")

    # ---- Console: final comparison ----
    print()
    print("=" * 78)
    print("  FINAL COMPARISON")
    print("=" * 78)
    print(f"  {'Condition':<24} | {'Score':>5} | " + " ".join(f"Q{i}" for i in range(1, 9)))
    print("  " + "-" * 60)
    for cond in cond_order:
        sc = scores_by_condition.get(cond, {})
        cells = " ".join(f" {sym.get(sc.get(f'Q{i}', 0), '?')}" for i in range(1, 9))
        print(f"  {cond:<24} | {str(totals_by_condition.get(cond, 0)) + '/8':>5} | {cells}")

    # ---- Calibration sanity ----
    fc = totals_by_condition.get("full_context", 0)
    tr = totals_by_condition.get("no_middle_truncated", 0)
    print()
    if fc == 8 and tr == 0:
        print("[calibration] OK — full_context=8/8, truncated=0/8.")
    else:
        print("!" * 78)
        print(f"  CALIBRATION WARNING: full_context={fc}/8 (expect 8), "
              f"truncated={tr}/8 (expect 0). Scorers or generation may be off.")
        print("!" * 78)


if __name__ == "__main__":
    main()
