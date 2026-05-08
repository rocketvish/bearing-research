# Bearing Research

## What this project is
Research code for native context compression in LLMs. We're testing whether coding agent conversation context can be compressed into optimized input embedding vectors while maintaining the model's understanding.

## Key approach
Hybrid compression: keep important turns (prefix + suffix) as verbatim text tokens, compress middle turns into learned virtual embedding vectors using per-instance gradient optimization. No pre-training needed — the model is frozen, only the virtual token embeddings are optimized.

## Hardware constraints
- GPU: NVIDIA RTX 5090 (31.84 GB VRAM)
- Model: Qwen 2.5 Coder 7B-Instruct, 8-bit quantization (8.11 GB)
- Forward pass (4096 tokens): 1.3s, 13.21 GB peak
- Gradient computation spills to RAM above ~1500 total tokens (causes 100x slowdown)
- Keep optimization sequences under 1500 tokens total (prefix + virtual + suffix)
- Use gradient checkpointing during optimization loops

## Optimization-loop invariants (compressor.optimize_compress)
- The 1500-token ceiling is **hard-enforced**: `optimize_compress` raises
  `ValueError` if `prefix_len + V + suffix_len > 1500`. A warning was not
  enough — RAM spill silently turns a 2-minute run into hours.
- Virtual token tensor and Adam moments are kept in **fp32**. fp16 Adam
  produced NaNs in early tests. Cast to the model's embedding dtype
  (fp16/bf16) only at concat time inside `build_compressed_input`; the
  cast is differentiable so gradient flow is preserved.
- Loss is computed in fp32 (`mse_loss(hs.float(), target.float())`).
- `gradient_checkpointing_enable()` and `enable_input_require_grads()`
  must both be called before the loop. Checkpointing alone won't
  propagate gradients to `inputs_embeds`. Both are disabled in `finally`.
- Default config: `NUM_VIRTUAL=50`, `OPT_STEPS=300`, `OPT_LR=0.01` —
  yields total_seq ≈ 1250 for the current transcript.

## Key files
- transcript.jsonl: Real coding agent session transcript (57 turns, 18454 tokens)
- phase0_check.py: Hardware feasibility verification (completed)
- context_builder.py: Loads transcript, splits into prefix/middle/suffix
- compressor.py: Mean-pool (v0) and gradient optimization (v1) compression
- evaluate.py: Runs 5 conditions, asks 8 questions, compares answers
- feasibility_test.py: Single-turn KV-cache compression test (one prefix
  turn + one ~250-token test turn → NUM_VIRTUAL optimized embeddings).
  Validates the mechanism end-to-end before running the full pipeline.

## KV-cache handling (feasibility_test.py and any KV-injection code)
- **Always pass past_key_values as a legacy tuple-of-tuples**
  `((k, v), (k, v), ...)`, not as a `Cache` object. transformers wraps
  the legacy tuple in a fresh `DynamicCache.from_legacy_cache()` per
  forward call, so the original tuple's tensors are not mutated. Passing
  a `Cache` directly lets `update()` extend it in place across calls,
  which silently corrupts the prefix KV across optimization steps.
- **Detach the prefix KV** before reusing it for gradient work:
  `tuple((k.detach(), v.detach()) for k, v in legacy)`. This blocks
  gradient flow through the prefix half of `K_full = concat(K_prefix,
  K_new)`. Only the new-token half receives gradient, which is what
  makes KV-anchored optimization cheap (forward over only the virtual
  tokens, attention over the full prefix).
- **`use_cache` semantics**: pass `past_key_values` with `use_cache=False`
  during optimization (the model uses the cache for attention but does
  not return a new one). Use `use_cache=True` only when you need the
  extended cache back, e.g. when building the per-condition inference
  caches. `output_hidden_states=True` works in both modes.
- **Generation with a frozen KV**: prefer `model.generate(input_ids=q,
  past_key_values=kv, ...)` and fall back silently to a manual greedy
  loop (forward → argmax → append → forward 1 token with extended
  cache) if `generate` rejects the cache argument.
- **Gradient flow check**: after the first optimization step, the loss
  should change. If it stays exactly constant, the most likely cause is
  that gradients are not reaching `virtual` — typically because the KV
  was passed as a Cache (not a tuple), or `enable_input_require_grads()`
  was not called alongside `gradient_checkpointing_enable()`.

## Environment
- Python 3.12, plain venv (not uv)
- PyTorch nightly with CUDA 12.8 (required for RTX 5090 sm_120)
- transformers, accelerate, bitsandbytes
- Activate: .venv\Scripts\Activate.ps1
- Run scripts with: python <script.py>

## Conventions
- Use torch.no_grad() for all inference except optimization loops
- Print VRAM usage at key points
- Handle errors with try/except so partial failures don't block other conditions
- Windows environment: use semicolons not && in shell commands