"""Central configuration for the Bearing research scripts.

Single source of truth for the model under test and shared generation
post-processing helpers. Import MODEL_ID from here rather than hardcoding it.
"""

from __future__ import annotations

import re

# The model all scripts load. Switched from Qwen2.5-Coder-7B-Instruct to the
# Qwen3-8B base instruct model for the mechanical Bearing adaptation.
MODEL_ID = "Qwen/Qwen3-8B"

# Whether Qwen3's chain-of-thought "thinking" mode is enabled. Qwen3 is a
# hybrid-thinking model whose only native switch is the chat template's
# `enable_thinking` argument. The compression scripts do NOT use the chat
# template — they generate from raw turn-marker text, inputs_embeds, and
# injected KV, where thinking mode does not meaningfully apply — so this is a
# forward-looking flag, not wired into the generate() call sites today.
# Default False for the compression experiments; flip to True (and route
# prompts through apply_chat_template(enable_thinking=ENABLE_THINKING)) to run
# a thinking-enabled ablation later. As a safety net, any <think>...</think>
# that does appear in generated text is stripped via strip_thinking().
ENABLE_THINKING = False

# Matches a complete <think>...</think> chain-of-thought block (non-greedy,
# spanning newlines).
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> reasoning traces from generated output.

    Qwen3 emits chain-of-thought wrapped in <think>...</think> during
    generation. We let the model reason while generating but strip the traces
    before the text is scored or compared downstream. Also drops a dangling,
    unterminated <think> block (generation cut off by max_new_tokens before
    </think> was produced) and tidies surrounding whitespace.

    This applies to *generated* text only. Forward passes (surprisal /
    V-target capture) are not generation and must not be passed through here.
    """
    cleaned = _THINK_RE.sub("", text)
    # An unclosed <think> means generation stopped mid-reasoning with no
    # answer after it; discard everything from the tag onward.
    open_idx = cleaned.find("<think>")
    if open_idx != -1:
        cleaned = cleaned[:open_idx]
    return cleaned.strip()
