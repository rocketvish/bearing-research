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

## Key files
- transcript.jsonl: Real coding agent session transcript (57 turns, 18454 tokens)
- phase0_check.py: Hardware feasibility verification (completed)
- context_builder.py: Loads transcript, splits into prefix/middle/suffix
- compressor.py: Mean-pool (v0) and gradient optimization (v1) compression
- evaluate.py: Runs 5 conditions, asks 8 questions, compares answers

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