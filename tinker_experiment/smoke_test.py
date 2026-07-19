"""Verify Tinker API access: one tiny sample from the chosen model."""

import asyncio

from common import MODEL, RENDERER_NAME, ensure_api_key

ensure_api_key()

import tinker  # noqa: E402
from tinker_cookbook import renderers, tokenizer_utils  # noqa: E402


async def main():
    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(base_model=MODEL)
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)

    prompt = renderer.build_generation_prompt(
        [{"role": "user", "content": "Reply with exactly one word: which language is Flask written in?"}]
    )
    params = tinker.SamplingParams(
        max_tokens=20, temperature=0.0, stop=renderer.get_stop_sequences()
    )
    result = await sampling_client.sample_async(
        prompt=prompt, num_samples=1, sampling_params=params
    )
    tokens = result.sequences[0].tokens
    message, termination = renderer.parse_response(tokens)
    print(f"model={MODEL} renderer={RENDERER_NAME}")
    print(f"answer={message['content']!r} termination={termination}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    asyncio.run(main())
