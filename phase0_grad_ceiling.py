"""Binary-search the on-GPU gradient-optimization ceiling for Qwen3-8B 8-bit.

phase0_check.py only probed two fixed grad points (20 and 2200 tokens) and,
crucially, ran them WITHOUT gradient checkpointing. The 2200-token point
"succeeded" only by spilling to host RAM (51.53 GB peak on a 31.84 GB card,
125 s instead of ~1 s). This script finds the exact ceiling under the SAME
conditions the real optimizer uses.

It mirrors compressor.optimize_compress:
  - virtual embeddings in fp32, fp32 Adam moments
  - model.gradient_checkpointing_enable() + enable_input_require_grads()
  - forward(inputs_embeds=..., output_hidden_states=True, use_cache=False)
  - loss = mse(last_hidden[:, anchor, :].float(), target.float())
  - cast virtual to the model embedding dtype only at concat time

We sweep the TOTAL sequence length (virtual + real) and binary-search for the
largest total that stays resident in VRAM. On Windows/WDDM an over-budget
allocation does NOT raise cuda OOM — it silently spills into shared host RAM
and runs ~100x slower. So "fits on-GPU" is detected two ways, and a trial is
counted as spilled if EITHER fires:
  1. peak reserved VRAM exceeds the physical card, or
  2. steady-state step time exceeds TIME_THRESHOLD_S (the 100x slowdown).

The reported ceiling is the number that should set the chunk-size limit for
clinical compression (currently hard-coded as 1500 in optimize_compress).
"""

from __future__ import annotations

import time
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_ID

# ----------------------------- config --------------------------------

NUM_VIRTUAL = 50          # fp32 grad tokens (production default)
TOTAL_MIN = 200           # smallest total seq (virtual + real) to test
TOTAL_MAX = 2200          # largest total seq to test
STEP = 50                 # search granularity, in tokens
STEPS_PER_TRIAL = 3       # optimizer steps per trial (>=2 to allocate Adam moments)
LR = 0.01

# A trial counts as spilled (does NOT fit on-GPU) if either of these trips.
TIME_THRESHOLD_S = 15.0   # steady-state step slower than this => RAM spill
SPILL_HEADROOM_GB = 0.0   # subtract from physical VRAM for the reserved-mem test

# Recommended safe limit leaves this much headroom below the measured ceiling
# (fragmentation, the warm-start mean-pool, per-condition caches, etc.).
SAFE_HEADROOM_TOKENS = 100


# ----------------------------- helpers -------------------------------

def gb(x: float) -> float:
    return x / 1024**3


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_alloc_gb() -> float:
    return gb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0


def peak_reserved_gb() -> float:
    return gb(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0.0


# ----------------------------- model ---------------------------------

def load_model():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = gb(props.total_memory)
        print(f"GPU: {props.name}  total VRAM: {total:.2f} GB")
    else:
        total = 0.0
        print("WARNING: CUDA not available")

    print(f"Loading {MODEL_ID} (8-bit) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    embed = model.get_input_embeddings()
    print(
        f"Model loaded. dtype={embed.weight.dtype}, "
        f"embed_dim={embed.weight.shape[1]}, "
        f"resident VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )
    return tokenizer, model, embed, total


# --------------------------- single trial ----------------------------

def run_trial(model, embed, total_tokens: int, total_vram_gb: float) -> dict:
    """Run STEPS_PER_TRIAL optimizer steps at the given total sequence length.

    Returns a dict of metrics. `spilled` is the on-GPU verdict.
    """
    device = model.device
    model_dtype = embed.weight.dtype
    embed_dim = embed.weight.shape[1]
    n_virtual = NUM_VIRTUAL
    n_real = total_tokens - n_virtual
    assert n_real >= 1, f"total {total_tokens} <= NUM_VIRTUAL {n_virtual}"

    # fp32 virtual embeddings + fp32 Adam (the conditioning the real optimizer
    # relies on). Warm-started with small noise; the exact values don't matter
    # for a memory/throughput probe.
    virtual = (0.01 * torch.randn(1, n_virtual, embed_dim, device=device)).to(
        dtype=torch.float32
    )
    virtual.requires_grad_(True)
    optimizer = torch.optim.Adam([virtual], lr=LR)

    # Real prefix/suffix embeddings (no grad), like build_compressed_input.
    real_ids = torch.randint(0, embed.weight.shape[0], (1, n_real), device=device)
    with torch.no_grad():
        real_emb = embed(real_ids)

    # Anchor = first token after the virtual block (start of the "suffix"),
    # mirroring optimize_compress's compressed_anchor = prefix_len + V.
    anchor = n_virtual
    target = torch.randn(1, embed_dim, device=device, dtype=torch.float32)

    torch.cuda.empty_cache()
    reset_peak()

    step_times: list[float] = []
    final_loss = float("nan")
    for _ in range(STEPS_PER_TRIAL):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Differentiable cast at concat time (CLAUDE.md invariant).
        virtual_cast = virtual.to(model_dtype)
        inputs_embeds = torch.cat([virtual_cast, real_emb], dim=1)
        out = model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states[-1][:, anchor, :]
        loss = F.mse_loss(hs.float(), target.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        final_loss = float(loss.detach().item())
        del out, hs, loss, inputs_embeds, virtual_cast

    p_alloc = peak_alloc_gb()
    p_reserved = peak_reserved_gb()
    # Steady-state = median step (drops first-step warmup / Adam-state alloc).
    steady = sorted(step_times)[len(step_times) // 2]

    spill_by_vram = p_reserved > (total_vram_gb - SPILL_HEADROOM_GB)
    spill_by_time = steady > TIME_THRESHOLD_S
    spilled = bool(spill_by_vram or spill_by_time)

    del virtual, optimizer, real_emb, real_ids, target
    torch.cuda.empty_cache()

    return {
        "total": total_tokens,
        "n_virtual": n_virtual,
        "n_real": n_real,
        "peak_alloc_gb": p_alloc,
        "peak_reserved_gb": p_reserved,
        "steady_step_s": steady,
        "all_step_s": step_times,
        "final_loss": final_loss,
        "spill_by_vram": spill_by_vram,
        "spill_by_time": spill_by_time,
        "spilled": spilled,
    }


def _fmt(r: dict) -> str:
    verdict = "SPILL" if r["spilled"] else "on-GPU"
    why = []
    if r["spill_by_vram"]:
        why.append("vram")
    if r["spill_by_time"]:
        why.append("time")
    why_s = f" ({'+'.join(why)})" if why else ""
    return (
        f"  total={r['total']:>5} (V={r['n_virtual']}, real={r['n_real']:>5})  "
        f"reserved={r['peak_reserved_gb']:6.2f} GB  "
        f"alloc={r['peak_alloc_gb']:6.2f} GB  "
        f"step={r['steady_step_s']:7.2f}s  -> {verdict}{why_s}"
    )


# ------------------------------- main --------------------------------

def main():
    print("=" * 78)
    print("  Gradient-optimization on-GPU ceiling — Qwen3-8B 8-bit, with checkpointing")
    print("=" * 78)
    tokenizer, model, embed, total_vram = load_model()

    # Enable the exact memory regime optimize_compress uses.
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print("gradient_checkpointing_enable() + enable_input_require_grads() ON")
    print(
        f"Spill test: reserved > {total_vram - SPILL_HEADROOM_GB:.2f} GB "
        f"OR steady step > {TIME_THRESHOLD_S:.0f}s\n"
    )

    results: list[dict] = []

    def trial(total: int) -> dict:
        try:
            r = run_trial(model, embed, total, total_vram)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            r = {
                "total": total, "n_virtual": NUM_VIRTUAL, "n_real": total - NUM_VIRTUAL,
                "peak_alloc_gb": peak_alloc_gb(), "peak_reserved_gb": peak_reserved_gb(),
                "steady_step_s": float("inf"), "all_step_s": [], "final_loss": float("nan"),
                "spill_by_vram": True, "spill_by_time": True, "spilled": True,
            }
            print(f"  total={total}: hard CUDA OOM: {e}")
        results.append(r)
        print(_fmt(r))
        return r

    ceiling = None
    try:
        print("Endpoint checks:")
        lo_r = trial(TOTAL_MIN)
        if lo_r["spilled"]:
            print(
                f"\nEven the minimum total={TOTAL_MIN} spills — ceiling is below the "
                f"search range. Lower TOTAL_MIN or NUM_VIRTUAL and re-run."
            )
            _summary(results, None, total_vram)
            return

        hi_r = trial(TOTAL_MAX)
        if not hi_r["spilled"]:
            print(
                f"\nThe maximum total={TOTAL_MAX} still fits on-GPU — ceiling is at "
                f"or above the search range. Raise TOTAL_MAX to find the true edge."
            )
            _summary(results, TOTAL_MAX, total_vram)
            return

        # Invariant: lo fits, hi spills. Binary-search the largest fitting total.
        lo, hi = TOTAL_MIN, TOTAL_MAX
        best = TOTAL_MIN
        print("\nBinary search:")
        while hi - lo > STEP:
            mid = ((lo + hi) // 2 // STEP) * STEP
            if mid <= lo:
                mid = lo + STEP
            r = trial(mid)
            if r["spilled"]:
                hi = mid
            else:
                best = mid
                lo = mid
        ceiling = best
    except Exception:
        traceback.print_exc()
    finally:
        model.gradient_checkpointing_disable()

    _summary(results, ceiling, total_vram)


def _summary(results: list[dict], ceiling, total_vram: float):
    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"Model: {MODEL_ID} (8-bit), GPU total VRAM: {total_vram:.2f} GB")
    print(f"NUM_VIRTUAL={NUM_VIRTUAL}, search step={STEP}, steps/trial={STEPS_PER_TRIAL}")
    print(f"Spill if reserved > {total_vram - SPILL_HEADROOM_GB:.2f} GB "
          f"or steady step > {TIME_THRESHOLD_S:.0f}s\n")

    for r in sorted(results, key=lambda x: x["total"]):
        print(_fmt(r))

    print()
    if ceiling is None:
        print("No clean ceiling determined — see notes above.")
        return
    safe = max(NUM_VIRTUAL + 1, ceiling - SAFE_HEADROOM_TOKENS)
    print(f"On-GPU gradient ceiling (largest total that fits): {ceiling} tokens")
    print(f"Recommended safe chunk-size limit (-{SAFE_HEADROOM_TOKENS} headroom): {safe} tokens")
    print()
    print("optimize_compress currently hard-codes a 1500-token cap (compressor.py:171).")
    if ceiling >= 1500:
        print(f"  -> Measured ceiling {ceiling} >= 1500: the 1500 cap is SAFE (conservative).")
    else:
        print(f"  -> Measured ceiling {ceiling} < 1500: the 1500 cap is TOO HIGH for 8B; "
              f"lower it to {safe}.")


if __name__ == "__main__":
    main()
