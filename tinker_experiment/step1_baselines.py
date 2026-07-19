"""Step 1 — Baselines (no training).

Split the original transcript as in Parts 4-5, build 2x and 3x compressed
middles via budget-enforced LLM summarization, then run the 8-question eval
at full / 2x / 3x with temperature 0. Saves summaries (reused in Step 4) and
per-question results.
"""

import asyncio
import json
import time

from clients import TinkerChatClient
from common import (
    DATA_DIR,
    MODEL,
    NUM_PREFIX,
    NUM_SUFFIX,
    RESULTS_DIR,
    TRANSCRIPT_PATH,
)
from context_split import split_transcript
from harness import build_condition_text, eval_condition, summarize_to_budget, write_results

TOKEN_BUDGET = 4096  # same window as Parts 4-5


async def main():
    t0 = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    client = TinkerChatClient()
    ctx = split_transcript(
        TRANSCRIPT_PATH, client.tokenizer, TOKEN_BUDGET, NUM_PREFIX, NUM_SUFFIX
    )

    middle_len = ctx.middle_len
    summaries = {}
    for ratio in (2, 3):
        budget = middle_len // ratio
        print(f"\n[summarize] {ratio}x: middle {middle_len} tok -> budget {budget} tok")
        text, n = await summarize_to_budget(client, ctx.middle_text, budget)
        summaries[str(ratio)] = {"text": text, "tokens": n, "budget": budget}

    (DATA_DIR / "original_summaries.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "summarizer": MODEL,
                "middle_tokens": middle_len,
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    conditions = {
        "full": build_condition_text(ctx.prefix_text, ctx.middle_text, ctx.suffix_text, False),
        "2x": build_condition_text(ctx.prefix_text, summaries["2"]["text"], ctx.suffix_text, True),
        "3x": build_condition_text(ctx.prefix_text, summaries["3"]["text"], ctx.suffix_text, True),
    }

    results = {"model": MODEL, "conditions": {}}
    for label, context_text in conditions.items():
        print(f"\n=== condition: {label} ({client.count_tokens(context_text)} context tokens) ===")
        results["conditions"][label] = await eval_condition(client, context_text, label)

    write_results(RESULTS_DIR / "step1_baselines.json", results)
    print(f"\nDone in {time.time() - t0:.0f}s\nSummary:")
    for label, r in results["conditions"].items():
        print(f"  {label:>5}: {r['total']}/8  (context {r['context_tokens']} tok)")


if __name__ == "__main__":
    asyncio.run(main())
