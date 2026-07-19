"""Phase 0 feasibility check for Qwen2.5-Coder-7B-Instruct.

Tests what fits in VRAM and how fast it runs. Optimised for clarity
and detailed logging, not throughput. Each step is wrapped in try/except
so partial failure still produces a final summary report.
"""

from __future__ import annotations

import contextlib
import json
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_ID
TRANSCRIPT_PATH = Path(__file__).parent / "transcript.jsonl"
TARGET_SIZES = [2048, 4096]

RESULTS: dict = {
    "model_id": MODEL_ID,
    "gpu_name": None,
    "gpu_total_vram_gb": None,
    "model_vram_gb": None,
    "transcript_tokens": None,
    "max_seq_fits": None,
    "fwd_4096_seconds": None,
    "fwd_4096_vram_gb": None,
    "kv_4096_bytes": None,
    "hidden_4096_bytes": None,
    "pooled_kv_bytes": None,
    "pooled_kv_ratio": None,
    "n_turns": None,
    "grad_20_ok": None,
    "grad_20_seconds": None,
    "grad_20_peak_vram_gb": None,
    "grad_20_used_checkpointing": False,
    "grad_2200_ok": None,
    "grad_2200_seconds": None,
    "grad_2200_peak_vram_gb": None,
    "grad_2200_used_checkpointing": False,
}


# ----------------------------- helpers --------------------------------

def vram() -> tuple[float, float]:
    """Return (allocated_GB, peak_GB)."""
    if not torch.cuda.is_available():
        return (0.0, 0.0)
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    return (alloc, peak)


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def vram_str() -> str:
    a, p = vram()
    return f"VRAM allocated={a:.2f} GB, peak={p:.2f} GB"


def fmt_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == "GB":
            return f"{f:.2f} {u}"
        f /= 1024
    return f"{f:.2f} GB"


@contextlib.contextmanager
def step(name: str):
    print()
    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)
    reset_peak()
    t0 = time.perf_counter()
    try:
        yield
        dt = time.perf_counter() - t0
        print(f"[{name}] OK in {dt:.2f}s — {vram_str()}")
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"[{name}] FAILED after {dt:.2f}s: {type(e).__name__}: {e}")
        print(f"[{name}] {vram_str()}")
        traceback.print_exc()


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


# ---------------------------- step 1: load -----------------------------

def step1_load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        RESULTS["gpu_name"] = props.name
        RESULTS["gpu_total_vram_gb"] = props.total_memory / 1024**3
        print(f"GPU: {props.name}  total VRAM: {RESULTS['gpu_total_vram_gb']:.2f} GB")
    else:
        print("WARNING: CUDA not available")

    print(f"Loading tokenizer for {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading model {MODEL_ID} with 8-bit quantization ...")
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()

    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print()
    print(f"Total parameters:        {n_params:,}")
    print(f"num_hidden_layers:       {cfg.num_hidden_layers}")
    print(f"hidden_size:             {cfg.hidden_size}")
    print(f"num_attention_heads:     {cfg.num_attention_heads}")
    print(f"num_key_value_heads:     {getattr(cfg, 'num_key_value_heads', cfg.num_attention_heads)}")
    print(f"head_dim:                {cfg.hidden_size // cfg.num_attention_heads}")
    print(f"vocab_size:              {cfg.vocab_size}")
    print(f"torch_dtype (config):    {getattr(cfg, 'torch_dtype', None)}")

    a, _ = vram()
    RESULTS["model_vram_gb"] = a
    print(f"Model VRAM after load:   {a:.2f} GB")
    return tokenizer, model


# ---------------------- step 2: tokenize transcript --------------------

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


def step2_tokenize(tokenizer):
    """Build full token sequence + turn map.

    Returns:
        full_ids: list[int]
        turn_map: list[(turn_idx, role, start, end)] over full sequence
        per_size: dict[int, dict] with truncated ids and turn map for each size
    """
    raw_lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    turns = [json.loads(line) for line in raw_lines if line.strip()]
    print(f"Loaded {len(turns)} turns from {TRANSCRIPT_PATH.name}")

    full_ids: list[int] = []
    turn_map: list[tuple[int, str, int, int]] = []

    for i, turn in enumerate(turns, start=1):
        role = turn.get("role", "?")
        content = _stringify_content(turn.get("content", ""))
        marker = f"=== TURN {i} ({role}) ===\n{content}\n"
        ids = tokenizer(marker, add_special_tokens=False).input_ids
        start = len(full_ids)
        end = start + len(ids)
        full_ids.extend(ids)
        turn_map.append((i, role, start, end))

    total = len(full_ids)
    RESULTS["transcript_tokens"] = total
    RESULTS["n_turns"] = len(turns)
    print(f"Total transcript tokens: {total}")
    tokens_per_turn = [end - start for _, _, start, end in turn_map]
    print(
        f"Tokens per turn: min={min(tokens_per_turn)}, "
        f"max={max(tokens_per_turn)}, mean={sum(tokens_per_turn)/len(tokens_per_turn):.1f}"
    )

    sizes = TARGET_SIZES + [total]
    per_size: dict[int, dict] = {}
    for size in sizes:
        if size > total:
            size = total
        ids = full_ids[:size]
        sub_map = []
        for ti, role, s, e in turn_map:
            if s >= size:
                break
            sub_map.append((ti, role, s, min(e, size)))
        per_size[size] = {"ids": ids, "turn_map": sub_map}
        print(f"  size={size:>6}: {len(ids)} tokens, {len(sub_map)} turns included")

    print(f"After tokenization — {vram_str()}")
    return full_ids, turn_map, per_size


# ------------------ step 3: forward + hidden states --------------------

def step3_forward_hidden(model, ids: list[int]):
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    reset_peak()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    hs = out.hidden_states
    print(f"hidden_states tuple length: {len(hs)} (expect num_layers+1)")
    print(f"first hidden state shape:   {tuple(hs[0].shape)}")
    print(f"last hidden state shape:    {tuple(hs[-1].shape)}")
    total_bytes = sum(h.numel() * h.element_size() for h in hs)
    print(f"hidden states total bytes:  {fmt_bytes(total_bytes)}")
    print(f"forward wall time:          {dt:.3f}s")
    print(f"after forward — {vram_str()}")

    RESULTS["fwd_4096_seconds"] = dt
    _, peak = vram()
    RESULTS["fwd_4096_vram_gb"] = peak
    RESULTS["hidden_4096_bytes"] = total_bytes

    del out, hs, input_ids
    torch.cuda.empty_cache()


# ----------------------- step 4: forward + KV --------------------------

def _extract_kv(past_key_values):
    """Return list of (K, V) per layer, supporting legacy tuple or Cache class."""
    if past_key_values is None:
        return []
    if isinstance(past_key_values, tuple):
        return list(past_key_values)
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            return list(past_key_values.to_legacy_cache())
        except Exception:
            pass
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return list(zip(past_key_values.key_cache, past_key_values.value_cache))
    if hasattr(past_key_values, "layers"):
        out = []
        for layer in past_key_values.layers:
            # Don't use `a or b` here: these attrs are tensors and bool(tensor)
            # raises "Boolean value of Tensor ... is ambiguous". Newer
            # transformers (Qwen3 Cache layout) exposes layer.keys/.values.
            k = getattr(layer, "keys", None)
            if k is None:
                k = getattr(layer, "key_cache", None)
            v = getattr(layer, "values", None)
            if v is None:
                v = getattr(layer, "value_cache", None)
            out.append((k, v))
        return out
    raise RuntimeError(f"Unknown past_key_values type: {type(past_key_values)}")


def step4_forward_kv(model, ids: list[int]):
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    reset_peak()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True, output_hidden_states=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    pkv = _extract_kv(out.past_key_values)
    print(f"KV cache layers: {len(pkv)}")
    k0, v0 = pkv[0]
    print(f"K[0] shape: {tuple(k0.shape)}  dtype={k0.dtype}")
    print(f"V[0] shape: {tuple(v0.shape)}  dtype={v0.dtype}")
    total_bytes = 0
    for k, v in pkv:
        total_bytes += k.numel() * k.element_size()
        total_bytes += v.numel() * v.element_size()
    print(f"KV cache total bytes: {fmt_bytes(total_bytes)}")
    print(f"forward wall time:    {dt:.3f}s")
    print(f"after forward — {vram_str()}")

    RESULTS["kv_4096_bytes"] = total_bytes
    return pkv, input_ids


# --------------------- step 5: mean-pool by turn -----------------------

def step5_mean_pool(pkv, turn_map, seq_len: int):
    """K/V shape: (B, n_kv_heads, seq, head_dim). Pool seq dim per turn."""
    pooled_k = []
    pooled_v = []
    n_turns = len(turn_map)
    tpt = [end - start for _, _, start, end in turn_map]
    print(f"n_turns covered: {n_turns}")
    print(f"tokens-per-turn: min={min(tpt)} max={max(tpt)} mean={sum(tpt)/n_turns:.1f}")

    k0, v0 = pkv[0]
    print(f"original K shape: {tuple(k0.shape)}")
    print(f"original V shape: {tuple(v0.shape)}")

    total_bytes = 0
    for k, v in pkv:
        # k, v: (B, n_kv_heads, seq, head_dim)
        per_turn_k = []
        per_turn_v = []
        for _, _, start, end in turn_map:
            end = min(end, seq_len)
            if end <= start:
                continue
            per_turn_k.append(k[:, :, start:end, :].mean(dim=2, keepdim=True))
            per_turn_v.append(v[:, :, start:end, :].mean(dim=2, keepdim=True))
        kp = torch.cat(per_turn_k, dim=2)
        vp = torch.cat(per_turn_v, dim=2)
        pooled_k.append(kp)
        pooled_v.append(vp)
        total_bytes += kp.numel() * kp.element_size()
        total_bytes += vp.numel() * vp.element_size()

    print(f"pooled K shape (layer 0): {tuple(pooled_k[0].shape)}")
    print(f"pooled V shape (layer 0): {tuple(pooled_v[0].shape)}")
    print(f"pooled KV total bytes: {fmt_bytes(total_bytes)}")
    ratio = seq_len / max(n_turns, 1)
    print(f"compression ratio: {seq_len} tokens -> {n_turns} turns ({ratio:.1f}:1)")
    print(f"after pooling — {vram_str()}")

    RESULTS["pooled_kv_bytes"] = total_bytes
    RESULTS["pooled_kv_ratio"] = ratio


# ---------------------- step 6: input embeddings -----------------------

def step6_embeddings(model, ids: list[int]):
    embed = model.get_input_embeddings()
    print(f"embedding weight shape: {tuple(embed.weight.shape)}")
    print(f"embedding dtype:        {embed.weight.dtype}")
    embed_dim = embed.weight.shape[1]
    print(f"embedding_dim:          {embed_dim}")

    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.no_grad():
        embs = embed(input_ids)
    print(f"input embeddings shape: {tuple(embs.shape)}  dtype={embs.dtype}")

    rand = torch.randn(1, 200, embed_dim, dtype=embs.dtype, device=model.device)
    print(f"random embed tensor:    {tuple(rand.shape)} dtype={rand.dtype}")
    with torch.no_grad():
        out = model(inputs_embeds=rand, use_cache=False)
    print(f"forward(inputs_embeds=rand) logits shape: {tuple(out.logits.shape)}")
    print(f"after embed test — {vram_str()}")

    del embs, rand, out, input_ids
    torch.cuda.empty_cache()
    return embed_dim


# --------------------- step 7: gradient feasibility --------------------

def _grad_test(model, embed_dim: int, n_virtual: int, n_real: int, real_ids):
    """Run forward+backward. Returns (success, seconds, peak_gb, used_checkpoint)."""
    embed = model.get_input_embeddings()

    def run(checkpoint: bool) -> tuple[bool, float, float]:
        if checkpoint:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        else:
            model.gradient_checkpointing_disable()

        torch.cuda.empty_cache()
        reset_peak()
        dtype = embed.weight.dtype
        virt = torch.randn(
            1, n_virtual, embed_dim,
            dtype=dtype, device=model.device, requires_grad=True,
        )

        if n_real > 0:
            real_tensor = torch.tensor([real_ids[:n_real]], dtype=torch.long, device=model.device)
            with torch.no_grad():
                real_emb = embed(real_tensor)
            inputs_embeds = torch.cat([virt, real_emb], dim=1)
        else:
            inputs_embeds = virt

        t0 = time.perf_counter()
        out = model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
        )
        loss = out.hidden_states[-1].sum()
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        if virt.grad is None:
            raise RuntimeError("virt.grad is None after backward()")
        print(f"  grad shape: {tuple(virt.grad.shape)}  dtype={virt.grad.dtype}")

        _, peak = vram()
        del virt, out, loss, inputs_embeds
        if n_real > 0:
            del real_emb, real_tensor
        torch.cuda.empty_cache()
        return True, dt, peak

    try:
        ok, dt, peak = run(checkpoint=False)
        return ok, dt, peak, False
    except Exception as e:
        if not is_oom(e):
            raise
        print(f"  OOM without checkpointing: {e}")
        print(f"  {vram_str()}")
        torch.cuda.empty_cache()
        print("  retrying with gradient_checkpointing_enable() ...")
        try:
            ok, dt, peak = run(checkpoint=True)
            return ok, dt, peak, True
        except Exception as e2:
            print(f"  OOM even with checkpointing: {e2}")
            return False, 0.0, 0.0, True
        finally:
            model.gradient_checkpointing_disable()


def step7_grad_small(model, embed_dim: int):
    print("Test 7a: 20 virtual tokens, requires_grad=True, no real prefix")
    ok, dt, peak, ckpt = _grad_test(model, embed_dim, n_virtual=20, n_real=0, real_ids=[])
    RESULTS["grad_20_ok"] = ok
    RESULTS["grad_20_seconds"] = dt
    RESULTS["grad_20_peak_vram_gb"] = peak
    RESULTS["grad_20_used_checkpointing"] = ckpt
    print(f"  ok={ok}  time={dt:.3f}s  peak={peak:.2f} GB  checkpoint={ckpt}")
    return ok


def step7b_grad_large(model, embed_dim: int, real_ids):
    print("Test 7b: 200 virtual tokens (grad) + 2000 real token embeddings (no grad)")
    ok, dt, peak, ckpt = _grad_test(
        model, embed_dim, n_virtual=200, n_real=2000, real_ids=real_ids
    )
    RESULTS["grad_2200_ok"] = ok
    RESULTS["grad_2200_seconds"] = dt
    RESULTS["grad_2200_peak_vram_gb"] = peak
    RESULTS["grad_2200_used_checkpointing"] = ckpt
    print(f"  ok={ok}  time={dt:.3f}s  peak={peak:.2f} GB  checkpoint={ckpt}")
    return ok


# ----------------------- step 8: scale test ----------------------------

def _try_forward(model, ids: list[int]) -> tuple[bool, float, float]:
    torch.cuda.empty_cache()
    reset_peak()
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        _, peak = vram()
        del out, input_ids
        torch.cuda.empty_cache()
        return True, dt, peak
    except Exception as e:
        if not is_oom(e):
            raise
        torch.cuda.empty_cache()
        return False, time.perf_counter() - t0, vram()[1]


def step8_scale(model, full_ids: list[int]):
    total = len(full_ids)
    print(f"Trying full transcript: {total} tokens with hidden states ...")
    ok, dt, peak = _try_forward(model, full_ids)
    if ok:
        print(f"  full transcript fits! {dt:.2f}s peak={peak:.2f} GB")
        RESULTS["max_seq_fits"] = total
        return
    print(f"  full transcript OOMed (peak={peak:.2f} GB) — binary search")

    lo, hi = 4096, total
    best = 4096
    step_size = 256
    while hi - lo > step_size:
        mid = ((lo + hi) // 2 // step_size) * step_size
        if mid <= lo:
            mid = lo + step_size
        print(f"  trying {mid} ...")
        ok, dt, peak = _try_forward(model, full_ids[:mid])
        if ok:
            print(f"    fits ({dt:.2f}s peak={peak:.2f} GB)")
            best = mid
            lo = mid
        else:
            print(f"    OOM (peak={peak:.2f} GB)")
            hi = mid
    print(f"max sequence that fits: {best} tokens")
    RESULTS["max_seq_fits"] = best


# --------------------------- step 9: summary ---------------------------

def step9_summary():
    r = RESULTS
    print()
    print("=" * 78)
    print("  PHASE 0 SUMMARY")
    print("=" * 78)

    def gb(x):
        return f"{x:.2f} GB" if isinstance(x, (int, float)) else "n/a"

    def s(x):
        return f"{x:.2f}s" if isinstance(x, (int, float)) else "n/a"

    def by(x):
        return fmt_bytes(x) if isinstance(x, int) else "n/a"

    print(f"Model: {MODEL_ID} (8-bit)")
    print(f"GPU: {r['gpu_name']} ({gb(r['gpu_total_vram_gb'])})")
    print(f"Model VRAM: {gb(r['model_vram_gb'])}")
    print(f"Transcript tokens: {r['transcript_tokens']}")
    print(f"Max sequence that fits: {r['max_seq_fits']} tokens")
    print()
    print(f"Forward pass (4096 tokens): {s(r['fwd_4096_seconds'])}, {gb(r['fwd_4096_vram_gb'])} VRAM peak")
    print(f"KV cache (4096 tokens): {by(r['kv_4096_bytes'])}")
    print(f"Hidden states (4096 tokens): {by(r['hidden_4096_bytes'])}")
    ratio = r["pooled_kv_ratio"]
    ratio_str = f"{ratio:.1f}:1" if isinstance(ratio, (int, float)) else "n/a"
    print(f"Mean-pooled KV ({r['n_turns']} turns): {by(r['pooled_kv_bytes'])} (compression ratio: {ratio_str})")
    print()
    grad_ok = r["grad_20_ok"]
    print(f"Gradient feasibility: {'YES' if grad_ok else 'NO'}")
    print(f"  Forward+backward (20 virtual tokens): {s(r['grad_20_seconds'])}, {gb(r['grad_20_peak_vram_gb'])} peak VRAM")
    print(f"  Gradient checkpointing (20 vt): {'YES' if r['grad_20_used_checkpointing'] else 'NO'}")
    print(f"  Forward+backward (200 vt + 2000 real): {s(r['grad_2200_seconds'])}, {gb(r['grad_2200_peak_vram_gb'])} peak VRAM, ok={r['grad_2200_ok']}")
    print(f"  Gradient checkpointing (200+2000): {'YES' if r['grad_2200_used_checkpointing'] else 'NO'}")
    print()

    kv_inject_ok = (r["kv_4096_bytes"] is not None) and (r["pooled_kv_bytes"] is not None)
    embed_opt_ok = bool(grad_ok)

    print("Recommendation:")
    print(f"  - KV injection (Approach D): {'FEASIBLE' if kv_inject_ok else 'NOT FEASIBLE'}")
    print(f"  - Input embedding optimization (Option 4): {'FEASIBLE' if embed_opt_ok else 'NOT FEASIBLE'}")

    if embed_opt_ok and r["grad_2200_ok"]:
        rec = "Input embedding optimization — full 2200-token grad path works"
    elif kv_inject_ok and not embed_opt_ok:
        rec = "KV injection — gradient path didn't fit; use cache-based approach"
    elif embed_opt_ok and not r["grad_2200_ok"]:
        rec = "Input embedding optimization for short contexts only; KV injection for full transcript"
    else:
        rec = "Neither approach validated end-to-end at this scale"
    print(f"  - Recommended approach: {rec}")


# ------------------------------- main ---------------------------------

def main():
    tokenizer = None
    model = None
    full_ids: list[int] = []
    per_size: dict = {}

    with step("Step 1: Load model"):
        tokenizer, model = step1_load_model()

    if model is None:
        print("Model failed to load — aborting.")
        step9_summary()
        return

    with step("Step 2: Tokenize transcript"):
        full_ids, _turn_map, per_size = step2_tokenize(tokenizer)

    ids_4096 = per_size.get(4096, {}).get("ids", full_ids[:4096])
    map_4096 = per_size.get(4096, {}).get("turn_map", [])

    with step("Step 3: Forward pass with hidden_states (4096)"):
        step3_forward_hidden(model, ids_4096)

    pkv = None
    with step("Step 4: Forward pass with KV cache (4096)"):
        pkv, _ = step4_forward_kv(model, ids_4096)

    if pkv is not None:
        with step("Step 5: Mean-pool KV cache by turn (4096)"):
            step5_mean_pool(pkv, map_4096, seq_len=len(ids_4096))
        del pkv
        torch.cuda.empty_cache()

    embed_dim = None
    with step("Step 6: Input embedding access"):
        embed_dim = step6_embeddings(model, ids_4096)

    grad_small_ok = False
    if embed_dim is not None:
        with step("Step 7: Gradient feasibility (20 virtual tokens)"):
            grad_small_ok = step7_grad_small(model, embed_dim)

    if grad_small_ok and embed_dim is not None:
        with step("Step 7b: Gradient feasibility (200 virtual + 2000 real)"):
            step7b_grad_large(model, embed_dim, full_ids)
    else:
        print("Skipping Step 7b (Step 7 did not pass).")

    with step("Step 8: Scale test (full transcript)"):
        step8_scale(model, full_ids)

    step9_summary()


if __name__ == "__main__":
    main()
