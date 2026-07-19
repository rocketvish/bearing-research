"""Context compression strategies.

ContextCompressor implements four ways to compress the "middle" turns of
a conversation context into a fixed-length sequence of embedding vectors:

  - mean_pool_compress: chunked-mean of input embeddings (v0, baseline)
  - optimize_compress:  gradient descent on virtual embeddings to match
                        the model's hidden state at the suffix anchor (v1)
  - text_compress:      LLM-generated text summary, then re-embedded
                        (text-level baseline)

compute_target_hidden_states() produces the supervision target for v1.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from config import strip_thinking


def _vram_str() -> str:
    if not torch.cuda.is_available():
        return "VRAM n/a"
    a = torch.cuda.memory_allocated() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    return f"VRAM allocated={a:.2f} GB, peak={p:.2f} GB"


class ContextCompressor:
    def __init__(self, model, tokenizer, context_builder):
        self.model = model
        self.tokenizer = tokenizer
        self.cb = context_builder
        self.embed_layer = model.get_input_embeddings()
        self._target: torch.Tensor | None = None
        self._target_anchor: int | None = None

    # ------------------------- target -------------------------------

    def compute_target_hidden_states(self) -> torch.Tensor:
        """Last-layer hidden state at the suffix-anchor position.

        Anchor position = prefix_len + middle_len, i.e. the first token
        of the suffix in the full uncompressed context. The hidden state
        at this position reflects everything the model has seen so far,
        so matching it is what we ask the optimized virtual tokens to do.
        """
        device = self.model.device
        full_ids = self.cb.get_full_context(device=device)
        anchor_pos = self.cb.prefix_len + self.cb.middle_len
        print(
            f"[target] full_len={full_ids.shape[1]}, anchor_pos={anchor_pos} "
            f"(start of suffix)"
        )

        was_training = self.model.training
        self.model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(
                input_ids=full_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        hs = out.hidden_states[-1]  # (1, L, D)
        target = hs[:, anchor_pos, :].detach().clone()
        print(
            f"[target] hidden_states[-1].shape={tuple(hs.shape)}, "
            f"target.shape={tuple(target.shape)}, "
            f"target_norm={target.norm().item():.3f}, "
            f"forward={dt:.2f}s, {_vram_str()}"
        )

        del out, hs, full_ids
        torch.cuda.empty_cache()
        if was_training:
            self.model.train()

        self._target = target
        self._target_anchor = anchor_pos
        return target

    # ------------------------- v0: mean pool ------------------------

    def mean_pool_compress(self, num_virtual_tokens: int = 50) -> torch.Tensor:
        """Chunked mean pool of the middle turns' input embeddings.

        Splits the middle into ``num_virtual_tokens`` roughly equal
        chunks (using linspace boundaries so non-divisible lengths
        still produce exactly V chunks) and averages each chunk.
        """
        device = self.model.device
        middle_ids = self.cb.get_middle_ids(device=device)
        m = middle_ids.shape[1]
        if m == 0:
            raise ValueError("Middle has 0 tokens — nothing to compress.")
        if num_virtual_tokens > m:
            raise ValueError(
                f"num_virtual_tokens={num_virtual_tokens} > middle_len={m}"
            )

        with torch.no_grad():
            middle_emb = self.embed_layer(middle_ids)  # (1, M, D)

        boundaries = torch.linspace(0, m, num_virtual_tokens + 1).long().tolist()
        chunks = []
        for i in range(num_virtual_tokens):
            s, e = boundaries[i], boundaries[i + 1]
            if e <= s:
                # Edge case for very small chunks: take a single token.
                e = s + 1
            chunks.append(middle_emb[:, s:e, :].mean(dim=1, keepdim=True))
        pooled = torch.cat(chunks, dim=1)  # (1, V, D)

        chunk_sizes = [boundaries[i + 1] - boundaries[i] for i in range(num_virtual_tokens)]
        print(
            f"[mean_pool] middle_len={m}, V={num_virtual_tokens}, "
            f"chunk min/max/mean = {min(chunk_sizes)}/{max(chunk_sizes)}/"
            f"{sum(chunk_sizes)/len(chunk_sizes):.1f}, "
            f"pooled.shape={tuple(pooled.shape)} dtype={pooled.dtype}"
        )

        del middle_emb, middle_ids
        torch.cuda.empty_cache()
        return pooled

    # ------------------------ v1: optimize --------------------------

    def optimize_compress(
        self,
        num_virtual_tokens: int = 50,
        num_steps: int = 300,
        lr: float = 0.01,
        log_every: int = 50,
    ) -> tuple[torch.Tensor, float]:
        """Gradient-optimize virtual embeddings to match the target hidden state.

        Virtual embeddings are kept in float32 so Adam moments stay
        well-conditioned; build_compressed_input casts them to the
        model's embedding dtype only at concat time.
        """
        if self._target is None:
            self.compute_target_hidden_states()
        target = self._target
        assert target is not None

        device = self.model.device
        # Warm start from mean-pool (cast to fp32 for the optimizer).
        warm = self.mean_pool_compress(num_virtual_tokens=num_virtual_tokens)
        virtual = warm.detach().clone().to(dtype=torch.float32, device=device)
        virtual.requires_grad_(True)

        prefix_len = self.cb.prefix_len
        suffix_len = self.cb.suffix_len
        total_seq = prefix_len + num_virtual_tokens + suffix_len
        compressed_anchor = prefix_len + num_virtual_tokens

        print(
            f"[optimize] V={num_virtual_tokens}, prefix={prefix_len}, "
            f"suffix={suffix_len}, total_seq={total_seq}, "
            f"compressed_anchor={compressed_anchor}"
        )
        if total_seq > 1300:
            # phase0_grad_ceiling.py measured this on Qwen3-8B 8-bit (RTX 5090,
            # checkpointing on): the on-GPU ceiling is 1400 tokens; above that
            # the fp32-Adam backward overflows to host RAM and causes a ~100x
            # slowdown (2.8s -> 39s/step at 1450, 211s at 2200). 1300 is the safe
            # limit (~2.5 GB headroom). Refuse to silently turn a 2-minute
            # optimization into a multi-hour one.
            raise ValueError(
                f"optimize_compress: total_seq={total_seq} (prefix={prefix_len} + "
                f"V={num_virtual_tokens} + suffix={suffix_len}) exceeds the 1300-"
                f"token gradient budget measured for Qwen3-8B in Phase 0. Reduce "
                f"num_virtual_tokens (currently {num_virtual_tokens}, max "
                f"{max(0, 1300 - prefix_len - suffix_len)} for this context) "
                f"or shrink the context window."
            )

        # Gradient checkpointing trades compute for VRAM; needed because
        # the 8-bit weights pin a lot of VRAM already.
        self.model.gradient_checkpointing_enable()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        was_training = self.model.training
        self.model.eval()

        optimizer = torch.optim.Adam([virtual], lr=lr)
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        final_loss = float("nan")
        t0 = time.perf_counter()
        try:
            for step in range(1, num_steps + 1):
                inputs_embeds = self.cb.build_compressed_input(virtual, self.model)
                out = self.model(
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=True,
                    use_cache=False,
                )
                hs = out.hidden_states[-1][:, compressed_anchor, :]  # (1, D)
                # Cast hs to fp32 to match the target/virtual fp32 path.
                loss = F.mse_loss(hs.float(), target.float())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                final_loss = float(loss.detach().item())
                if step == 1 or step % log_every == 0 or step == num_steps:
                    print(
                        f"[optimize] step {step:>4}/{num_steps}  "
                        f"loss={final_loss:.4f}  {_vram_str()}"
                    )

                del out, hs, inputs_embeds, loss
        finally:
            self.model.gradient_checkpointing_disable()
            if was_training:
                self.model.train()
            else:
                self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"[optimize] {num_steps} steps in {dt:.1f}s, final_loss={final_loss:.4f}")

        return virtual.detach(), final_loss

    # ------------------------- text baseline ------------------------

    def text_compress(self, max_summary_tokens: int = 200) -> tuple[str, torch.Tensor, int]:
        """Generate a text summary of the middle and re-embed it."""
        middle_text = self.cb.get_middle_text()
        prompt = (
            "Summarize the following coding agent conversation turns in 2-3 "
            "sentences, preserving: files created, key decisions made, errors "
            "encountered:\n\n"
            f"{middle_text}"
        )
        device = self.model.device
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        prompt_len = prompt_ids.shape[1]

        was_training = self.model.training
        self.model.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_summary_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        new_ids = generated[0, prompt_len:]
        summary_text = strip_thinking(self.tokenizer.decode(new_ids, skip_special_tokens=True))

        # Re-embed the summary text on its own (don't include the instruction).
        summary_ids = self.tokenizer(
            summary_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        with torch.no_grad():
            summary_embeds = self.embed_layer(summary_ids)
        summary_token_count = int(summary_ids.shape[1])

        print(
            f"[text_compress] generated {new_ids.shape[0]} tokens in {dt:.1f}s, "
            f"summary_tokens={summary_token_count}"
        )
        print(f"[text_compress] summary preview: {summary_text[:200]!r}")

        if was_training:
            self.model.train()
        else:
            self.model.eval()

        del prompt_ids, generated, summary_ids
        torch.cuda.empty_cache()
        return summary_text, summary_embeds, summary_token_count
