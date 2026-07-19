"""Swappable model-client interface for the eval harness.

The harness (question loading, scoring, summarization, results logging)
talks only to the ChatClient protocol below. Implementations:

- TinkerChatClient: Tinker sampling client + cookbook renderer (base model
  or a trained checkpoint's sampling client).
- OpenAICompatClient: any OpenAI-compatible /chat/completions endpoint
  (vLLM, Ollama, Together, OpenRouter, ...).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Message = dict  # {"role": ..., "content": ...}


@runtime_checkable
class ChatClient(Protocol):
    name: str  # identifies the model/endpoint in results logs

    async def chat(
        self, messages: list[Message], max_tokens: int, temperature: float = 0.0
    ) -> str: ...

    def count_tokens(self, text: str) -> int: ...


class TinkerChatClient:
    """ChatClient over a Tinker sampling client with a cookbook renderer."""

    def __init__(self, model: str | None = None, renderer_name: str | None = None):
        from common import MODEL, RENDERER_NAME, ensure_api_key

        ensure_api_key()
        import tinker
        from tinker_cookbook import renderers, tokenizer_utils

        self._tinker = tinker
        self.model = model or MODEL
        self.name = self.model
        self.service_client = tinker.ServiceClient()
        self.tokenizer = tokenizer_utils.get_tokenizer(self.model)
        self.renderer = renderers.get_renderer(renderer_name or RENDERER_NAME, self.tokenizer)
        self._sampling_client = None

    def use_sampling_client(self, sampling_client, name: str | None = None) -> None:
        """Point this client at an existing sampling client, e.g. a trained
        checkpoint from save_weights_and_get_sampling_client."""
        self._sampling_client = sampling_client
        if name:
            self.name = name

    async def _get_sampling_client(self):
        if self._sampling_client is None:
            self._sampling_client = await self.service_client.create_sampling_client_async(
                base_model=self.model
            )
        return self._sampling_client

    async def use_checkpoint(self, model_path: str, name: str | None = None) -> None:
        """Point this client at a saved checkpoint path (tinker://...)."""
        self._sampling_client = await self.service_client.create_sampling_client_async(
            model_path=model_path
        )
        self.name = name or model_path

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    async def chat(
        self, messages: list[Message], max_tokens: int, temperature: float = 0.0
    ) -> str:
        client = await self._get_sampling_client()
        prompt = self.renderer.build_generation_prompt(messages)
        params = self._tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
        )
        result = await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        message, _termination = self.renderer.parse_response(result.sequences[0].tokens)
        return (message.get("content") or "").strip()


class OpenAICompatClient:
    """ChatClient over any OpenAI-compatible /chat/completions endpoint.

    Token counting uses an HF tokenizer when given one, otherwise a
    chars/4 estimate (fine for logging; budget enforcement should use the
    target model's real tokenizer).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        tokenizer=None,
        extra_headers: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"{model} @ {self.base_url}"
        self.api_key = api_key
        self.tokenizer = tokenizer
        self.extra_headers = extra_headers or {}

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer(text, add_special_tokens=False).input_ids)
        return max(1, len(text) // 4)

    async def chat(
        self, messages: list[Message], max_tokens: int, temperature: float = 0.0
    ) -> str:
        import httpx

        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=120.0) as http:
            resp = await http.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
