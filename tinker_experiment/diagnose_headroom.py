"""Find where compression headroom lives, since the Parts 4-5 window shows
no loss at 2x/3x on Qwen3.6-35B-A3B.

Conditions:
  A. window-4096, middle omitted entirely (floor for that split)
  B. full 57-turn transcript, full context (does the scorer still hold?)
  C. full transcript, middle omitted (floor for the full-transcript split)
"""

import asyncio

from clients import TinkerChatClient
from common import NUM_PREFIX, NUM_SUFFIX, RESULTS_DIR, TRANSCRIPT_PATH
from context_split import split_transcript
from harness import eval_condition, write_results

OMIT = "\n[... middle turns omitted ...]\n"


async def main():
    client = TinkerChatClient()

    win = split_transcript(TRANSCRIPT_PATH, client.tokenizer, 4096, NUM_PREFIX, NUM_SUFFIX)
    full = split_transcript(TRANSCRIPT_PATH, client.tokenizer, 10**9, NUM_PREFIX, NUM_SUFFIX)

    conditions = {
        "win4096_no_middle": win.prefix_text + OMIT + win.suffix_text,
        "fulltx_full": full.full_text,
        "fulltx_no_middle": full.prefix_text + OMIT + full.suffix_text,
    }

    results = {"model": client.name, "conditions": {}}
    for label, text in conditions.items():
        print(f"\n=== {label} ({client.count_tokens(text)} context tokens) ===")
        results["conditions"][label] = await eval_condition(client, text, label)

    write_results(RESULTS_DIR / "diagnose_headroom.json", results)
    for label, r in results["conditions"].items():
        print(f"  {label:>20}: {r['total']}/8")


if __name__ == "__main__":
    asyncio.run(main())
