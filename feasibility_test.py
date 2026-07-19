"""V-only compression sweep with RoPE position interpolation and
perplexity-weighted V pooling.

Attention-weighted pooling regressed results because the importance
weights were question-specific. This run swaps to perplexity-based
weighting: tokens the model finds surprising (high cross-entropy on
next-token prediction) get more weight in the V pool. "bcryptjs" is
surprising; "the" is not. The weighting is question-independent.

Per-token surprise is computed from the SAME forward pass we already
do to capture V targets — we just also extract its logits and compute
cross-entropy against the next token. No extra forward pass.

Unlike attention weights (one vector per layer), perplexity gives a
single (n_test,) vector reused at every layer. Per-layer mag²
normalization values are recomputed from the new perplexity-weighted
V targets so the loss scale reflects the actual targets in use.

This run keeps everything else from the position-interpolation run:
  - V-only loss
  - Explicit position_ids spreading virtual tokens across the original
    middle's [prefix_len, prefix_len + n_test - 1] span
  - Suffix tokens forwarded at positions starting from prefix_len + n_test
  - Per-layer mag² normalization
  - lr=0.003, 300 steps, Adam fp32

Sweep: V in [150, 100, 75, 50, 25].

Standalone — does not import from compressor.py or context_builder.py.
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
RESULTS_PATH = Path(__file__).parent / "results_feasibility_pplpool.json"

TEST_TURN_INDEX = 7
PREFIX_TURN_INDEX = 0

COMPRESSION_VS = [150, 100, 75, 50, 25]
NUM_STEPS = 300
LR = 0.003
LOG_EVERY = 50
MAX_NEW_TOKENS = 256

QUESTION = (
    "What library does the project use for password hashing? "
    "What is its Node.js test runner? "
    "Does the markdown converter use a library or regex?"
)
EXPECTED_ANSWER = (
    "bcryptjs for password hashing; Node's built-in test runner (node:test); "
    "regex-based markdown converter (no external libraries)."
)


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
    """Uniform-mean pool of (B, H, S, Dh) along S into n_chunks groups.

    Retained as a reference; the V-target builder now uses
    attention_weighted_pool instead. Embedding-init pooling still uses
    mean_pool_chunks.
    """
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
    """Pool a V tensor of shape (B, H, S, Dh) along S into n_chunks
    groups, weighting each chunk by its softmaxed importance scores.

    importance shape: (S,) — typically per-token attention from the
    question's positions to the turn's positions, averaged across heads
    and across query positions for one layer.

    Within a chunk [a, b]:
      weights = softmax(importance[a:b])         shape (b-a,)
      pooled  = sum(weights * v[:, :, a:b, :])    shape (B, H, 1, Dh)
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
        chunk_v = v[:, :, a:b, :]                # (B, H, b-a, Dh)
        chunk_w = importance[a:b].to(chunk_v.device)  # (b-a,)
        weights = F.softmax(chunk_w.float(), dim=0).to(chunk_v.dtype)
        pooled = (chunk_v * weights.view(1, 1, -1, 1)).sum(dim=2, keepdim=True)
        chunks.append(pooled)
    return torch.cat(chunks, dim=2)


def extract_new_kv_grad_safe(cache, prefix_len: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pull (K, V) per layer, sliced to new positions, no detach/clone/contiguous."""
    if isinstance(cache, tuple):
        return [(k[:, :, prefix_len:, :], v[:, :, prefix_len:, :]) for (k, v) in cache]
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
            out.append((k[:, :, prefix_len:, :], v[:, :, prefix_len:, :]))
        return out
    kc = getattr(cache, "key_cache", None)
    vc = getattr(cache, "value_cache", None)
    if kc is not None and vc is not None:
        return [
            (kc[i][:, :, prefix_len:, :], vc[i][:, :, prefix_len:, :])
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
    """Spread n positions across [start, end] inclusive. Returns (1, n) LongTensor."""
    floats = torch.linspace(float(start), float(end), n)
    positions = floats.round().long()
    unique_count = int(torch.unique(positions).numel())
    if unique_count != n:
        print(
            f"  [interpolate_positions] WARNING: only {unique_count}/{n} "
            f"unique positions for [{start}, {end}], n={n} — rounding collisions."
        )
    if device is not None:
        positions = positions.to(device)
    return positions.unsqueeze(0)


def verify_position_ids_take_effect(
    model, prefix_kv: tuple, prefix_len: int, model_dtype
) -> dict:
    """Empirically verify that explicit position_ids take effect when
    past_key_values is present.

    The test deliberately uses prefix_kv as past_key_values because the
    failure mode we're detecting is specifically the model ignoring
    explicit position_ids when past_key_values is present and auto-
    computing positions from the cache length instead. Without a cache
    we'd never observe the bug — position_ids are always honored when
    starting fresh. With a cache present, some implementations or
    versions silently override the user's position_ids; that's exactly
    what would invalidate this whole experiment, so we test the cached
    path directly.
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

    k_a = live_a[0][0].float()
    k_b = live_b[0][0].float()
    v_a = live_a[0][1].float()
    v_b = live_b[0][1].float()

    k_diff_max = float((k_a - k_b).abs().max().item())
    v_diff_max = float((v_a - v_b).abs().max().item())

    took_effect = (k_diff_max > 1e-3) and (v_diff_max < 1e-1)
    return {
        "k_diff_max": k_diff_max,
        "v_diff_max": v_diff_max,
        "took_effect": took_effect,
    }


def compute_perplexity_weights(
    logits: torch.Tensor, test_ids: torch.Tensor
) -> torch.Tensor:
    """Per-token surprise (cross-entropy) for each turn position.

    logits shape: (1, n_test, vocab) — output of the test-turn forward
    with prefix KV. logits[0, i, :] is the prediction for the token at
    position i+1 in test_ids; cross-entropy against test_ids[0, i+1]
    gives the surprise of that token given everything before it.

    Position 0 has no preceding logit in this forward (its prediction
    came from the prefix forward, whose logits we did not retain), so
    we prepend 0.0 per spec.

    Uses F.cross_entropy(reduction='none') instead of materializing
    log_softmax — saves a 152k-vocab × 241-position fp32 intermediate.

    Returns shape (n_test,) fp32 on the same device as logits.
    """
    if logits.shape[1] != test_ids.shape[1]:
        raise ValueError(
            f"logits seq {logits.shape[1]} != test_ids seq {test_ids.shape[1]}"
        )
    device = logits.device
    target_ids = test_ids[0, 1:].to(device).long()       # (n_test - 1,)
    pred_logits = logits[0, :-1, :].float()              # (n_test - 1, vocab)
    nll = F.cross_entropy(pred_logits, target_ids, reduction="none")  # (n_test - 1,)
    zero = torch.zeros(1, device=device, dtype=nll.dtype)
    surprise = torch.cat([zero, nll], dim=0).detach().clone()
    return surprise


def score_answer(ans: str) -> tuple[int, dict[str, bool]]:
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


@torch.no_grad()
def manual_greedy(
    model,
    tokenizer,
    question_text: str,
    past_kv_legacy: tuple,
    max_new_tokens: int,
) -> str:
    eos_id = tokenizer.eos_token_id
    device = model.device
    q_ids = tokenizer(
        question_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    cache = wrap_legacy_kv(past_kv_legacy, model.config)
    out = model(input_ids=q_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    next_id = int(out.logits[0, -1, :].argmax().item())
    generated = [next_id]
    if eos_id is not None and next_id == eos_id:
        return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))

    for _ in range(max_new_tokens - 1):
        next_inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(input_ids=next_inp, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))


@torch.no_grad()
def manual_greedy_with_positions(
    model,
    tokenizer,
    question_text: str,
    past_kv_legacy: tuple,
    start_pos: int,
    max_new_tokens: int,
) -> str:
    """Greedy decode with explicit per-step position_ids (for compressed conditions)."""
    eos_id = tokenizer.eos_token_id
    device = model.device
    q_ids = tokenizer(
        question_text, return_tensors="pt", add_special_tokens=False
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
    if eos_id is not None and next_id == eos_id:
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
        if eos_id is not None and next_id == eos_id:
            break

    return strip_thinking(tokenizer.decode(generated, skip_special_tokens=True))


@torch.no_grad()
def generate_with_kv(
    model, tokenizer, question_text: str, past_kv_legacy: tuple
) -> str:
    device = model.device
    q_ids = tokenizer(
        question_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    try:
        out = model.generate(
            input_ids=q_ids,
            past_key_values=wrap_legacy_kv(past_kv_legacy, model.config),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        n_input = int(q_ids.shape[1])
        new_ids = out[0, n_input:]
        return strip_thinking(tokenizer.decode(new_ids, skip_special_tokens=True))
    except Exception:
        return manual_greedy(
            model, tokenizer, question_text, past_kv_legacy, MAX_NEW_TOKENS
        )


# -------------------------------- main -------------------------------

def main():
    print("=" * 78)
    print("  V-only sweep with perplexity-weighted pooling + position interpolation")
    print("=" * 78)

    # -------------------- 1. Load model + transcript ----------------
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

    raw_lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    turns = [json.loads(line) for line in raw_lines if line.strip()]
    print(f"Loaded transcript: {len(turns)} turns from {TRANSCRIPT_PATH.name}")

    # -------------------- 2. Select test turn -----------------------
    print()
    print(f"Selected test turn: index {TEST_TURN_INDEX} (1-indexed: {TEST_TURN_INDEX + 1})")
    test_turn = turns[TEST_TURN_INDEX]
    test_role = test_turn.get("role", "?")
    test_content = _stringify_content(test_turn.get("content", ""))
    test_text = turn_marker(TEST_TURN_INDEX + 1, test_role, test_content)
    test_ids = tokenizer(
        test_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_test = int(test_ids.shape[1])
    print(f"  role: {test_role}, token count: {n_test}")
    print(f"  text (first 200 chars): {test_text[:200]!r}")

    # -------------------- 3. Prefix KV cache ------------------------
    print()
    prefix_turn = turns[PREFIX_TURN_INDEX]
    prefix_role = prefix_turn.get("role", "?")
    prefix_content = _stringify_content(prefix_turn.get("content", ""))
    prefix_text = turn_marker(PREFIX_TURN_INDEX + 1, prefix_role, prefix_content)
    prefix_ids = tokenizer(
        prefix_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_prefix = int(prefix_ids.shape[1])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
    raw_pkv = prefix_out.past_key_values
    print(f"  past_key_values type: {type(raw_pkv).__name__}")
    prefix_kv = detach_kv(to_legacy_kv(raw_pkv))
    del prefix_out, raw_pkv
    torch.cuda.empty_cache()
    prefix_len = kv_seq_len(prefix_kv)
    print(f"Prefix KV: {n_prefix} tokens, kv_seq_len={prefix_len}, "
          f"layers={len(prefix_kv)}. {vram_str()}")

    # -------------------- 4. Embed test_ids once --------------------
    with torch.no_grad():
        turn_emb = embed(test_ids).squeeze(0)
    print(f"\nturn_emb: {tuple(turn_emb.shape)} dtype={turn_emb.dtype}")

    # -------------------- 5. RoPE source check ----------------------
    print()
    rope_verified, rope_finding = verify_rope_for_qwen()
    if rope_verified:
        print("[rope_check] CONFIRMED: RoPE applies to Q and K only (V is position-independent).")
        print(f"[rope_check] Source line: {rope_finding}")
    else:
        print("[rope_check] WARNING: could not confirm RoPE shape.")
        print(f"[rope_check] Finding: {rope_finding}")

    # -------------------- 6. Position-id runtime check --------------
    print()
    print("Verifying that explicit position_ids take effect with past_key_values present ...")
    posid_check = verify_position_ids_take_effect(model, prefix_kv, prefix_len, model_dtype)
    print(f"  k_diff_max = {posid_check['k_diff_max']:.4f} (expect > 1e-3 — RoPE on K differs)")
    print(f"  v_diff_max = {posid_check['v_diff_max']:.4f} (expect ~0 — V is position-independent)")
    print(f"  took_effect = {posid_check['took_effect']}")
    if not posid_check["took_effect"]:
        print()
        print("!" * 78)
        print("  FATAL: position_ids did NOT take effect with past_key_values present.")
        print("  The whole experiment depends on this. Aborting.")
        print("!" * 78)
        raise AssertionError(
            f"position_ids ignored: k_diff_max={posid_check['k_diff_max']:.4g}, "
            f"v_diff_max={posid_check['v_diff_max']:.4g}"
        )
    print("  [posid_check] OK — position_ids honored with past_key_values present.")

    # -------------------- 7. Position layout ------------------------
    virtual_pos_start = prefix_len
    virtual_pos_end = prefix_len + n_test - 1
    suffix_start_pos = prefix_len + n_test
    print()
    print("Position layout:")
    print(f"  prefix:           [0, {prefix_len - 1}]")
    print(f"  virtual range:    [{virtual_pos_start}, {virtual_pos_end}]  (interpolated for V<{n_test})")
    print(f"  suffix start:     {suffix_start_pos}")

    virtual_pos_by_v: dict[int, torch.Tensor] = {}
    for v_count in COMPRESSION_VS:
        vp = interpolate_positions(virtual_pos_start, virtual_pos_end, v_count, device=device)
        virtual_pos_by_v[v_count] = vp
        first3 = vp[0, :3].tolist()
        last3 = vp[0, -3:].tolist()
        print(f"  V={v_count:>3}: positions = {first3} … {last3}  shape={tuple(vp.shape)}")

    # -------------------- 8. V_full target + perplexity weights -----
    print()
    print("Computing V_full + perplexity weights (one forward) ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        kv_out = model(
            input_ids=test_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            output_hidden_states=False,
            use_cache=True,
        )
    # Compute perplexity weights from the same forward's logits before
    # we drop kv_out — no extra forward pass.
    perplexity_weights = compute_perplexity_weights(kv_out.logits, test_ids)
    print(f"  perplexity_weights shape={tuple(perplexity_weights.shape)}, "
          f"min={perplexity_weights.min().item():.4f}, "
          f"max={perplexity_weights.max().item():.4f}, "
          f"mean={perplexity_weights.mean().item():.4f}")

    full_kv = to_legacy_kv(kv_out.past_key_values)
    n_layers = len(full_kv)
    v_full: list[torch.Tensor] = []
    v_full_bytes = 0
    for k, v in full_kv:
        v_new = v[:, :, prefix_len:, :].detach().clone()
        v_full.append(v_new)
        v_full_bytes += v_new.numel() * v_new.element_size()
        del k, v
    del kv_out, full_kv
    torch.cuda.empty_cache()
    v_shape = tuple(v_full[0].shape)
    print(f"  V_full: {n_layers} layers × {v_shape} ({v_full_bytes / 1024**2:.2f} MB)")
    print(f"  After V_full + perplexity — {vram_str()}")

    # Establish question_text for downstream reference conditions.
    question_text = f"\n\nQuestion: {QUESTION}\nAnswer:"

    # -------------------- 9. Perplexity diagnostic ------------------
    surprise_cpu = perplexity_weights.detach().cpu()
    token_strs: list[str] = []
    for i in range(n_test):
        tok = tokenizer.decode([int(test_ids[0, i].item())], skip_special_tokens=False)
        token_strs.append(tok)

    top_k = 10
    bot_k = 10
    top_idx = torch.topk(surprise_cpu, top_k).indices.tolist()
    bot_idx = torch.topk(-surprise_cpu, bot_k).indices.tolist()

    print()
    print(f"Top {top_k} tokens by surprise (cross-entropy, NLL):")
    for i in top_idx:
        print(f"  pos {i:>3}: {token_strs[i]!r:<20}  nll={surprise_cpu[i].item():.4f}")

    print(f"Bottom {bot_k} tokens by surprise:")
    for i in bot_idx:
        print(f"  pos {i:>3}: {token_strs[i]!r:<20}  nll={surprise_cpu[i].item():.4f}")

    top20 = torch.topk(surprise_cpu, 20).indices.tolist()
    top20_strs = [token_strs[i].lower() for i in top20]
    has_js_top20 = any("js" in s for s in top20_strs)
    has_bcrypt_top20 = any("bcrypt" in s for s in top20_strs)
    has_bcryptjs_top20 = any("bcryptjs" in s for s in top20_strs)
    print()
    print("Load-bearing token presence in top 20 (by surprise):")
    print(f"  'js'        : {has_js_top20}")
    print(f"  'bcrypt'    : {has_bcrypt_top20}")
    print(f"  'bcryptjs'  : {has_bcryptjs_top20}")
    if not has_js_top20 and not has_bcryptjs_top20:
        print("  NOTE: neither 'js' nor 'bcryptjs' appear in top 20 — perplexity "
              "weighting may not have the signal we need to fix the bcrypt vs "
              "bcryptjs distinction. Continuing anyway.")

    top_tokens_for_json = [
        {"position": int(i), "token": token_strs[i], "surprise": float(surprise_cpu[i].item())}
        for i in top_idx
    ]

    # -------------------- 10. Build perplexity-weighted V targets ---
    # Same (n_test,) surprise vector reused at every layer (unlike
    # attention which gave a different vector per layer).
    print()
    print("Building perplexity-weighted V targets ...")
    targets_by_v: dict[int, list[torch.Tensor]] = {}
    v_mag2_by_v: dict[int, list[torch.Tensor]] = {}
    for v_count in COMPRESSION_VS:
        pooled = []
        mag2 = []
        for v_layer in v_full:
            v_pooled = attention_weighted_pool(
                v_layer, perplexity_weights, v_count
            ).detach().clone()
            pooled.append(v_pooled)
            # mag² recomputed from the perplexity-weighted target,
            # so the loss scale reflects the actual targets in use.
            mag2.append(v_pooled.float().pow(2).mean().detach())
        targets_by_v[v_count] = pooled
        v_mag2_by_v[v_count] = mag2
        size_mb = sum(t.numel() * t.element_size() for t in pooled) / 1024**2
        print(f"  V={v_count:>3}  ({n_test/v_count:.2f}x)  pooled V {tuple(pooled[0].shape)}  "
              f"{size_mb:.2f} MB  mag2[0]={mag2[0].item():.4g}")
    print(f"After targets — {vram_str()}")

    # -------------------- 11. Reference conditions ------------------
    print()
    print("Reference conditions (full_turn / raw_embeddings / no_turn) ...")

    answers: dict[str, str] = {}
    gen_times: dict[str, float] = {}
    optim_meta: dict[str, dict] = {}

    @torch.no_grad()
    def kv_with_full_turn():
        out = model(
            input_ids=test_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out.past_key_values))

    @torch.no_grad()
    def kv_with_raw_embeddings():
        emb = embed(test_ids)
        out = model(
            inputs_embeds=emb,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out.past_key_values))

    @torch.no_grad()
    def kv_with_virtual_posinterp(v: torch.Tensor, virtual_pos: torch.Tensor) -> tuple:
        ie = v.to(dtype=model_dtype).unsqueeze(0)
        out = model(
            inputs_embeds=ie,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=virtual_pos,
            use_cache=True,
        )
        return detach_kv(to_legacy_kv(out.past_key_values))

    def run_ref(label: str, kv: tuple) -> None:
        try:
            t0 = time.perf_counter()
            ans = generate_with_kv(model, tokenizer, question_text, kv)
            dt = time.perf_counter() - t0
            answers[label] = ans
            gen_times[label] = dt
            print(f"  [{label}] {dt:.1f}s answer: {ans[:120]!r}")
        except Exception as e:
            print(f"  [{label}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            answers[label] = f"<error: {type(e).__name__}: {e}>"
            gen_times[label] = float("nan")

    full_kv_inf = kv_with_full_turn()
    print(f"  full_turn_kv seq_len={kv_seq_len(full_kv_inf)}")
    run_ref("full_turn", full_kv_inf)
    del full_kv_inf
    torch.cuda.empty_cache()

    raw_kv_inf = kv_with_raw_embeddings()
    print(f"  raw_embeddings_kv seq_len={kv_seq_len(raw_kv_inf)}")
    run_ref("raw_embeddings", raw_kv_inf)
    del raw_kv_inf
    torch.cuda.empty_cache()

    print(f"  no_turn_kv seq_len={kv_seq_len(prefix_kv)}")
    run_ref("no_turn", prefix_kv)

    print(f"After reference conditions — {vram_str()}")

    # -------------------- 12. Optimization driver -------------------

    def optimize_v_only_posinterp(
        init_virtual: torch.Tensor,
        target_v_list: list[torch.Tensor],
        v_mag2_list: list[torch.Tensor],
        virtual_pos: torch.Tensor,
        label: str,
    ) -> tuple[torch.Tensor, list[float]]:
        v_count = int(init_virtual.shape[0])
        print()
        print("=" * 78)
        print(f"  [{label}] V={v_count}, lr={LR}, target=V-only ppl-pooled, posinterp")
        print("=" * 78)

        virtual = init_virtual.detach().clone().to(dtype=torch.float32, device=device)
        virtual.requires_grad_(True)

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        optimizer = torch.optim.Adam([virtual], lr=LR)
        losses: list[float] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.perf_counter()
        cache_diag: dict = {}

        try:
            for step in range(NUM_STEPS):
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
                    layers_attr = getattr(out.past_key_values, "layers", None)
                    if isinstance(layers_attr, (list, tuple)) and len(layers_attr) > 0:
                        cache_diag["layer0_type"] = type(layers_attr[0]).__name__
                    else:
                        cache_diag["layer0_type"] = "N/A"

                live = extract_new_kv_grad_safe(out.past_key_values, prefix_len)
                if step == 0:
                    v0 = live[0][1]
                    cache_diag["v0_shape"] = tuple(v0.shape)
                    cache_diag["v0_requires_grad"] = bool(v0.requires_grad)
                    cache_diag["v0_grad_fn"] = (
                        type(v0.grad_fn).__name__ if v0.grad_fn is not None else "None"
                    )

                # V-only loss, per-layer normalized. K (live[L][0]) is
                # intentionally NOT referenced and NOT deleted here.
                loss = torch.zeros((), dtype=torch.float32, device=device)
                for (_k_live, v_live), v_tgt, v_mag2 in zip(live, target_v_list, v_mag2_list):
                    layer_loss = F.mse_loss(v_live.float(), v_tgt.float())
                    loss = loss + layer_loss / v_mag2.clamp(min=1e-8)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if step == 0:
                    if virtual.grad is None:
                        print(f"  [{label}] GRADIENT FAILURE: virtual.grad is None.")
                        for k_, v_ in cache_diag.items():
                            print(f"    {k_:<22} {v_}")
                        raise AssertionError(
                            "Gradient did not flow through KV extraction."
                        )
                    print(f"  [{label}] grad OK after step 0: "
                          f"virtual.grad.norm={virtual.grad.norm().item():.4f}, "
                          f"v0_grad_fn={cache_diag.get('v0_grad_fn')}")

                optimizer.step()

                loss_val = float(loss.detach().item())
                losses.append(loss_val)

                if step in (0, NUM_STEPS // 2, NUM_STEPS - 1):
                    print(f"  step {step:>3}: loss={loss_val:.4f}  {vram_str()}")
                elif (step + 1) % LOG_EVERY == 0:
                    print(f"  step {step:>3}: loss={loss_val:.4f}")

                del out, loss, inputs_embeds
        finally:
            model.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt_total = time.perf_counter() - t_start
        print(f"  [{label}] {NUM_STEPS} steps in {dt_total:.1f}s, "
              f"loss {losses[0]:.4f} → {losses[-1]:.4f}")
        return virtual.detach(), losses

    # -------------------- 13. Compression sweep ---------------------
    print()
    print("=" * 78)
    print(f"  Sweep V = {COMPRESSION_VS}  (perplexity-weighted V pool, posinterp)")
    print("=" * 78)

    for v_count in COMPRESSION_VS:
        label = f"optimized_{v_count}_v_pplpool"

        init = mean_pool_chunks(turn_emb, v_count).detach().clone()
        target_v = targets_by_v[v_count]
        v_mag2 = v_mag2_by_v[v_count]
        virtual_pos = virtual_pos_by_v[v_count]

        try:
            virtual, losses = optimize_v_only_posinterp(
                init, target_v, v_mag2, virtual_pos, label
            )
        except Exception as e:
            print(f"[ERROR] optimize_v_only_posinterp({v_count}) failed: {e}")
            traceback.print_exc()
            answers[label] = f"<error: {type(e).__name__}: {e}>"
            gen_times[label] = float("nan")
            optim_meta[label] = {
                "num_virtual": v_count,
                "compression_ratio": round(n_test / v_count, 4),
                "error": f"{type(e).__name__}: {e}",
            }
            del init
            torch.cuda.empty_cache()
            continue

        try:
            inf_kv = kv_with_virtual_posinterp(virtual, virtual_pos)
            print(f"  inference_kv seq_len={kv_seq_len(inf_kv)} "
                  f"(virtual covers positions {virtual_pos[0, 0].item()}–{virtual_pos[0, -1].item()})")
            t0 = time.perf_counter()
            ans = manual_greedy_with_positions(
                model, tokenizer, question_text, inf_kv,
                start_pos=suffix_start_pos, max_new_tokens=MAX_NEW_TOKENS,
            )
            dt = time.perf_counter() - t0
            answers[label] = ans
            gen_times[label] = dt
            print(f"  [{label}] {dt:.1f}s answer: {ans[:120]!r}")
            del inf_kv
        except Exception as e:
            print(f"[ERROR] inference for {label} failed: {e}")
            traceback.print_exc()
            answers[label] = f"<error: {type(e).__name__}: {e}>"
            gen_times[label] = float("nan")

        optim_meta[label] = {
            "num_virtual": v_count,
            "compression_ratio": round(n_test / v_count, 4),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_curve": losses,
        }

        del init, virtual, target_v, v_mag2
        del targets_by_v[v_count], v_mag2_by_v[v_count]
        torch.cuda.empty_cache()
        print(f"  after free — {vram_str()}")

    del v_full, turn_emb, perplexity_weights
    torch.cuda.empty_cache()

    # -------------------- 14. Save JSON -----------------------------
    condition_order = [
        "full_turn",
        "raw_embeddings",
        *[f"optimized_{v}_v_pplpool" for v in COMPRESSION_VS],
        "no_turn",
    ]
    results: dict = {
        "config": {
            "target": "V-only (RoPE applies to K only, V is position-independent)",
            "rope_verified": rope_verified,
            "rope_finding": rope_finding,
            "loss_normalization": "per-layer, divided by perplexity-weighted target V mean squared magnitude",
            "learning_rate": LR,
            "num_steps": NUM_STEPS,
            "optimizer": "Adam fp32",
            "position_interpolation": True,
            "virtual_position_range": [virtual_pos_start, virtual_pos_end],
            "suffix_start_position": suffix_start_pos,
            "position_ids_take_effect": posid_check,
            "pooling": "perplexity-weighted",
            "top_tokens": top_tokens_for_json,
            "load_bearing_in_top20": {
                "js": has_js_top20,
                "bcrypt": has_bcrypt_top20,
                "bcryptjs": has_bcryptjs_top20,
            },
        },
        "test_turn": {
            "index": TEST_TURN_INDEX,
            "role": test_role,
            "token_count": n_test,
        },
        "question": QUESTION,
        "expected": EXPECTED_ANSWER,
        "conditions": {},
    }
    for label in condition_order:
        cond: dict = {
            "answer": answers.get(label, ""),
            "gen_time_s": gen_times.get(label, float("nan")),
        }
        if label in optim_meta:
            cond.update(optim_meta[label])
        results["conditions"][label] = cond
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {RESULTS_PATH}")

    # -------------------- 15. Console summary -----------------------
    print()
    print("=" * 78)
    print("  PERPLEXITY-WEIGHTED-POOL SWEEP RESULTS")
    print("=" * 78)
    print(f"Test turn:   index {TEST_TURN_INDEX} (turn {TEST_TURN_INDEX + 1}, "
          f"role={test_role}), {n_test} tokens")
    print(f"Prefix:      turn {PREFIX_TURN_INDEX + 1} only, {n_prefix} tokens")
    print(f"Question:    {QUESTION}")
    print(f"Expected:    {EXPECTED_ANSWER}")
    print(f"posid_check: k_diff_max={posid_check['k_diff_max']:.4f}, "
          f"v_diff_max={posid_check['v_diff_max']:.4f}, "
          f"took_effect={posid_check['took_effect']}")
    print(f"top20 contains: js={has_js_top20}, bcrypt={has_bcrypt_top20}, "
          f"bcryptjs={has_bcryptjs_top20}")
    print()
    for label in condition_order:
        ans = answers.get(label, "")
        score, _ = score_answer(ans) if ans and not ans.startswith("<error") else (0, {})
        print(f"[{label:<32}] (score {score}/3) {ans[:160]}")

    print()
    print("Compression table:")
    header = (
        f"  {'Virtual tokens':>14} | {'Compression':>11} | "
        f"{'Start loss':>10} | {'Final loss':>10} | {'Correct':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    def row(v_label: str, v_count_or_dash, ratio_str: str, init_loss, final_loss, score):
        v_str = str(v_count_or_dash) if v_count_or_dash != "—" else "—"
        i_str = f"{init_loss:.4f}" if isinstance(init_loss, (int, float)) else "—"
        f_str = f"{final_loss:.4f}" if isinstance(final_loss, (int, float)) else "—"
        s_str = f"{score}/3"
        print(f"  {v_str:>14} | {ratio_str:>11} | {i_str:>10} | {f_str:>10} | "
              f"{s_str:>7}  {v_label}")

    full_score, _ = score_answer(answers.get("full_turn", ""))
    row("(full_turn)", n_test, "1.00x", "—", "—", full_score)
    for v_count in COMPRESSION_VS:
        label = f"optimized_{v_count}_v_pplpool"
        meta = optim_meta.get(label, {})
        if "error" in meta:
            row(f"({label} ERROR)", v_count, f"{n_test/v_count:.2f}x", "—", "—", 0)
            continue
        s, _ = score_answer(answers.get(label, ""))
        row(f"({label})", v_count, f"{n_test/v_count:.2f}x",
            meta.get("initial_loss"), meta.get("final_loss"), s)
    no_score, _ = score_answer(answers.get("no_turn", ""))
    row("(no_turn)", 0, "∞", "—", "—", no_score)

    full_ans = answers.get("full_turn", "")
    raw_ans = answers.get("raw_embeddings", "")
    if (
        full_ans and raw_ans
        and not full_ans.startswith("<error")
        and not raw_ans.startswith("<error")
    ):
        if full_ans.strip() != raw_ans.strip():
            print()
            print("!" * 78)
            print("  WARNING: raw_embeddings answer DIFFERS from full_turn.")
            print("  inputs_embeds path is not reproducing the input_ids path.")
            print("  Sweep results may be unreliable.")
            print("!" * 78)
        else:
            print()
            print("[sanity] raw_embeddings matches full_turn — inputs_embeds path OK.")


if __name__ == "__main__":
    main()
