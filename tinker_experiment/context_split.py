"""Port of ContextBuilder's split procedure without the torch dependency.

Same procedure as context_builder.py (Parts 4-5): stringify each turn with
a `=== TURN i (role) ===` marker, greedily include turns until a token
budget (truncating the overflowing turn), then split into
prefix / middle / suffix by turn counts.
"""

from __future__ import annotations

import json
from pathlib import Path


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


class SplitContext:
    def __init__(self, prefix_text: str, middle_text: str, suffix_text: str, tokenizer):
        self.prefix_text = prefix_text
        self.middle_text = middle_text
        self.suffix_text = suffix_text
        self._tok = tokenizer

    def n_tokens(self, text: str) -> int:
        return len(self._tok(text, add_special_tokens=False).input_ids)

    @property
    def middle_len(self) -> int:
        return self.n_tokens(self.middle_text)

    @property
    def full_text(self) -> str:
        return self.prefix_text + self.middle_text + self.suffix_text


def split_transcript(
    transcript_path: str | Path,
    tokenizer,
    token_budget: int = 4096,
    num_prefix: int = 2,
    num_suffix: int = 3,
) -> SplitContext:
    raw_lines = Path(transcript_path).read_text(encoding="utf-8").splitlines()
    all_turns = [json.loads(line) for line in raw_lines if line.strip()]

    included_text: list[str] = []
    running = 0
    for i, turn in enumerate(all_turns, start=1):
        role = turn.get("role", "?")
        content = _stringify_content(turn.get("content", ""))
        marker = f"=== TURN {i} ({role}) ===\n{content}\n"
        ids = tokenizer(marker, add_special_tokens=False).input_ids
        remaining = token_budget - running
        if remaining <= 0:
            break
        if len(ids) <= remaining:
            included_text.append(marker)
            running += len(ids)
        else:
            truncated = tokenizer.decode(ids[:remaining], skip_special_tokens=False)
            included_text.append(truncated)
            running += remaining
            break

    n = len(included_text)
    if n < num_prefix + num_suffix + 1:
        raise ValueError(f"budget {token_budget} fits only {n} turns")

    ctx = SplitContext(
        prefix_text="".join(included_text[:num_prefix]),
        middle_text="".join(included_text[num_prefix : n - num_suffix]),
        suffix_text="".join(included_text[n - num_suffix :]),
        tokenizer=tokenizer,
    )
    print(
        f"[split] {len(all_turns)} turns total, {n} included; "
        f"prefix={ctx.n_tokens(ctx.prefix_text)} tok, "
        f"middle={ctx.middle_len} tok ({n - num_prefix - num_suffix} turns), "
        f"suffix={ctx.n_tokens(ctx.suffix_text)} tok"
    )
    return ctx
