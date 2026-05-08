"""Context window construction for the compression experiments.

Parses transcript.jsonl into per-turn token sequences, picks a window
of turns that fits a token budget, and splits it into verbatim
prefix / compressible middle / verbatim suffix segments.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _stringify_content(content) -> str:
    """Flatten a Claude-style message ``content`` field into a string.

    Mirrors the logic used in phase0_check.py so token counts are
    comparable across phases.
    """
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


class ContextBuilder:
    """Load a transcript and slice it into prefix/middle/suffix segments."""

    def __init__(
        self,
        transcript_path: str | Path,
        tokenizer,
        token_budget: int = 4096,
        num_prefix: int = 2,
        num_suffix: int = 3,
    ):
        self.transcript_path = Path(transcript_path)
        self.tokenizer = tokenizer
        self.token_budget = token_budget
        self.num_prefix = num_prefix
        self.num_suffix = num_suffix

        # Load and tokenize all turns.
        raw_lines = self.transcript_path.read_text(encoding="utf-8").splitlines()
        all_turns = [json.loads(line) for line in raw_lines if line.strip()]

        per_turn_text: list[str] = []
        per_turn_ids: list[list[int]] = []
        per_turn_role: list[str] = []
        for i, turn in enumerate(all_turns, start=1):
            role = turn.get("role", "?")
            content = _stringify_content(turn.get("content", ""))
            marker = f"=== TURN {i} ({role}) ===\n{content}\n"
            ids = tokenizer(marker, add_special_tokens=False).input_ids
            per_turn_text.append(marker)
            per_turn_ids.append(ids)
            per_turn_role.append(role)

        # Greedy fit: include turns until the next would overflow; if it
        # would overflow, include it truncated so we land near the budget.
        included_text: list[str] = []
        included_ids: list[list[int]] = []
        included_role: list[str] = []
        running = 0
        for text, ids, role in zip(per_turn_text, per_turn_ids, per_turn_role):
            remaining = token_budget - running
            if remaining <= 0:
                break
            if len(ids) <= remaining:
                included_text.append(text)
                included_ids.append(ids)
                included_role.append(role)
                running += len(ids)
            else:
                # Partially include the overflowing turn.
                truncated_ids = ids[:remaining]
                truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=False)
                included_text.append(truncated_text)
                included_ids.append(truncated_ids)
                included_role.append(role)
                running += len(truncated_ids)
                break

        n_included = len(included_ids)
        if n_included < num_prefix + num_suffix + 1:
            raise ValueError(
                f"Token budget {token_budget} only fits {n_included} turns; "
                f"need at least {num_prefix + num_suffix + 1} for prefix/middle/suffix split."
            )

        self.n_total_turns = len(all_turns)
        self.n_included_turns = n_included
        self.turn_texts = included_text
        self.turn_ids = included_ids
        self.turn_roles = included_role

        # Segment slices.
        prefix_slice = slice(0, num_prefix)
        middle_slice = slice(num_prefix, n_included - num_suffix)
        suffix_slice = slice(n_included - num_suffix, n_included)

        self._prefix_ids: list[int] = [t for ids in included_ids[prefix_slice] for t in ids]
        self._middle_ids: list[int] = [t for ids in included_ids[middle_slice] for t in ids]
        self._suffix_ids: list[int] = [t for ids in included_ids[suffix_slice] for t in ids]

        self._prefix_text = "".join(included_text[prefix_slice])
        self._middle_text = "".join(included_text[middle_slice])
        self._suffix_text = "".join(included_text[suffix_slice])

        self.n_middle_turns = max(0, n_included - num_prefix - num_suffix)
        self.full_ids: list[int] = (
            self._prefix_ids + self._middle_ids + self._suffix_ids
        )

        print(
            f"[ContextBuilder] {self.n_total_turns} transcript turns, "
            f"{n_included} included within {token_budget}-token budget"
        )
        print(
            f"[ContextBuilder] prefix={len(self._prefix_ids)} tok ({num_prefix} turns), "
            f"middle={len(self._middle_ids)} tok ({self.n_middle_turns} turns), "
            f"suffix={len(self._suffix_ids)} tok ({num_suffix} turns), "
            f"total={len(self.full_ids)} tok"
        )

    # -------------------------- accessors --------------------------

    @property
    def prefix_len(self) -> int:
        return len(self._prefix_ids)

    @property
    def middle_len(self) -> int:
        return len(self._middle_ids)

    @property
    def suffix_len(self) -> int:
        return len(self._suffix_ids)

    @property
    def prefix_text(self) -> str:
        return self._prefix_text

    @property
    def suffix_text(self) -> str:
        return self._suffix_text

    @property
    def full_text(self) -> str:
        return self._prefix_text + self._middle_text + self._suffix_text

    def get_full_context(self, device=None) -> torch.Tensor:
        t = torch.tensor([self.full_ids], dtype=torch.long)
        return t.to(device) if device is not None else t

    def get_verbatim_parts(self, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = torch.tensor([self._prefix_ids], dtype=torch.long)
        suffix = torch.tensor([self._suffix_ids], dtype=torch.long)
        if device is not None:
            prefix = prefix.to(device)
            suffix = suffix.to(device)
        return prefix, suffix

    def get_middle_text(self) -> str:
        return self._middle_text

    def get_middle_ids(self, device=None) -> torch.Tensor:
        t = torch.tensor([self._middle_ids], dtype=torch.long)
        return t.to(device) if device is not None else t

    def num_middle_tokens(self) -> int:
        return self.middle_len

    # ------------------ compressed-input assembly ------------------

    def build_compressed_input(self, virtual_embeds: torch.Tensor, model) -> torch.Tensor:
        """Concatenate embed(prefix) + virtual_embeds + embed(suffix).

        virtual_embeds may be float32 (so the optimizer stays in fp32);
        we cast it to the embedding dtype just before concatenation.
        Casting is differentiable, so the gradient path is preserved.
        """
        embed = model.get_input_embeddings()
        device = model.device
        prefix_t, suffix_t = self.get_verbatim_parts(device=device)

        with torch.no_grad():
            prefix_emb = embed(prefix_t)
            suffix_emb = embed(suffix_t)

        # virtual_embeds: (1, V, D) — cast dtype but keep gradient.
        virtual_cast = virtual_embeds.to(prefix_emb.dtype)
        if virtual_cast.device != prefix_emb.device:
            virtual_cast = virtual_cast.to(prefix_emb.device)

        return torch.cat([prefix_emb, virtual_cast, suffix_emb], dim=1)
