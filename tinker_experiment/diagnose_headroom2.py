"""Second headroom diagnostic: same 4096-token window as Parts 4-5, but
prefix=1 turn / suffix=1 turn so nearly all facts sit in the compressible
middle. Measures the full-context sanity and the no-middle floor.
"""

import asyncio

from clients import TinkerChatClient
from common import RESULTS_DIR, TRANSCRIPT_PATH
from context_split import split_transcript
from harness import eval_condition, write_results

OMIT = "\n[... middle turns omitted ...]\n"


async def main():
    client = TinkerChatClient()
    ctx = split_transcript(TRANSCRIPT_PATH, client.tokenizer, 4096, num_prefix=1, num_suffix=1)

    conditions = {
        "p1s1_full": ctx.full_text,
        "p1s1_no_middle": ctx.prefix_text + OMIT + ctx.suffix_text,
    }

    results = {"model": client.name, "conditions": {}}
    for label, text in conditions.items():
        print(f"\n=== {label} ({client.count_tokens(text)} context tokens) ===")
        results["conditions"][label] = await eval_condition(client, text, label)

    write_results(RESULTS_DIR / "diagnose_headroom2.json", results)
    for label, r in results["conditions"].items():
        print(f"  {label:>16}: {r['total']}/8  {r['scores']}")


if __name__ == "__main__":
    asyncio.run(main())
