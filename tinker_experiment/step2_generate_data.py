"""Step 2 — Synthetic training data.

For each of N_TRANSCRIPTS fact sheets (scenarios.py):
  1. LLM writes a multi-turn coding-agent transcript embedding every fact,
     in the same `=== TURN i (role) ===` format as the eval transcript.
  2. Mechanical verification: every fact answer appears as a substring and
     no banned term appears. One repair retry, then drop missing facts if
     >= 6 remain, else discard the transcript.
  3. Split prefix(1 turn)/middle/suffix(1 turn), summarize middle at 2x with
     the identical procedure as Step 1.
  4. Build QA pairs. Training keeps only facts whose answer survives in the
     compressed context (training on deleted facts would teach fabrication);
     held-out transcripts keep all facts with a survival flag.

Outputs data/synthetic_transcripts.jsonl and data/train_qa.jsonl,
data/heldout_qa.jsonl with a schema note in data/SCHEMA.md.
"""

import argparse
import asyncio
import json
import random
import re
import time

from clients import TinkerChatClient
from common import DATA_DIR, MODEL
from harness import build_condition_text, summarize_to_budget
from scenarios import BANNED_TERMS, build_fact_sheet

N_TRANSCRIPTS = 50
N_HELDOUT = 5
SEED = 20260714
CONCURRENCY = 8
GEN_TEMPERATURE = 0.9
MIN_FACTS = 6

GEN_SYSTEM = (
    "You write realistic coding-agent conversation transcripts for research. "
    "Transcripts alternate between a user and an assistant (the coding agent) "
    "building a small project. The agent runs commands, writes code, reports "
    "results. Style: terse, technical, like a real session log."
)


def gen_prompt(sheet: dict) -> str:
    fact_lines = []
    for f in sheet["facts"]:
        line = f"- ({f['category']}) {f['question']} -> {f['answer']}"
        if f["is_trap"]:
            line += f"  [deliberately NOT the popular default '{f['popular_default']}' — the transcript should use {f['answer']}]"
        fact_lines.append(line)
    facts_block = "\n".join(fact_lines)
    return f"""Write a coding-agent conversation transcript for this project:

- Stack: {sheet['framework']} ({sheet['language']}), database: {sheet['database']}
- Project: a {sheet['domain']}

Format requirements (strict):
- 10 to 14 turns, each starting with a marker line: === TURN <i> (user) === or === TURN <i> (assistant) ===
- Turn 1 (user): the project brief.
- Last turn: a short recent exchange where the agent verifies something.
- Total length roughly 1500-2500 words of realistic content: commands, code
  snippets, file contents, test output.

Content requirements (strict):
- Every fact below must appear VERBATIM (exact spelling) at least once,
  embedded naturally in commands, code, or discussion:
{facts_block}
- Where a fact is marked as not-the-default, have the user or agent briefly
  note the choice (one clause is enough), then use it consistently.
- NEVER mention any of these terms: Express, SQLite, bcryptjs, node:test,
  task management.

Output only the transcript, starting with === TURN 1 (user) ===."""


def verify_facts(text: str, facts: list[dict]) -> list[dict]:
    low = text.lower()
    return [f for f in facts if f["answer"].lower() not in low]


def find_banned(text: str) -> list[str]:
    low = text.lower()
    return [b for b in BANNED_TERMS if b in low]


def split_turns(text: str) -> list[str]:
    """Split generated transcript into turns, keeping marker lines."""
    parts = re.split(r"(?m)^(?==== TURN )", text.strip())
    return [p for p in parts if p.strip()]


async def generate_transcript(client: TinkerChatClient, sheet: dict) -> dict | None:
    messages = [
        {"role": "system", "content": GEN_SYSTEM},
        {"role": "user", "content": gen_prompt(sheet)},
    ]
    text = await client.chat(messages, max_tokens=4096, temperature=GEN_TEMPERATURE)

    for attempt in range(2):
        missing = verify_facts(text, sheet["facts"])
        banned = find_banned(text)
        if not missing and not banned:
            break
        if attempt == 1:
            break
        problems = []
        if missing:
            problems.append(
                "these facts are missing and must appear verbatim: "
                + "; ".join(f"{f['question']} -> {f['answer']}" for f in missing)
            )
        if banned:
            problems.append(f"remove all mentions of banned terms: {banned}")
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Revise the transcript. " + ". ".join(problems)
             + " Keep the same format and everything else that already satisfies the requirements. Output only the full revised transcript."},
        ]
        text = await client.chat(messages, max_tokens=4096, temperature=GEN_TEMPERATURE)

    banned = find_banned(text)
    if banned:
        print(f"  [{sheet['id']}] DISCARDED: banned terms {banned}")
        return None
    missing = verify_facts(text, sheet["facts"])
    kept_facts = [f for f in sheet["facts"] if f not in missing]
    if len(kept_facts) < MIN_FACTS:
        print(f"  [{sheet['id']}] DISCARDED: only {len(kept_facts)} facts embedded")
        return None
    if missing:
        print(f"  [{sheet['id']}] dropped {len(missing)} unembedded fact(s): "
              f"{[f['answer'] for f in missing]}")

    turns = split_turns(text)
    if len(turns) < 4:
        print(f"  [{sheet['id']}] DISCARDED: only {len(turns)} turns parsed")
        return None

    prefix = turns[0]
    middle = "".join(turns[1:-1])
    suffix = turns[-1]
    middle_tokens = client.count_tokens(middle)
    budget = middle_tokens // 2
    summary, summary_tokens = await summarize_to_budget(client, middle, budget)

    compressed_context = build_condition_text(prefix, summary, suffix, compressed=True)
    low = compressed_context.lower()
    qa = []
    for f in kept_facts:
        qa.append({
            "question": f["question"],
            "answer": f["answer"],
            "category": f["category"],
            "is_trap": f["is_trap"],
            "popular_default": f["popular_default"],
            "answer_in_compressed": f["answer"].lower() in low,
        })

    return {
        **{k: sheet[k] for k in ("id", "framework", "language", "database", "domain")},
        "transcript_text": text,
        "prefix_text": prefix,
        "middle_text": middle,
        "suffix_text": suffix,
        "middle_tokens": middle_tokens,
        "summary_text": summary,
        "summary_tokens": summary_tokens,
        "summary_budget": budget,
        "qa": qa,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=N_TRANSCRIPTS)
    parser.add_argument("--append", action="store_true",
                        help="merge with existing synthetic_transcripts.jsonl")
    args = parser.parse_args()

    t0 = time.time()
    DATA_DIR.mkdir(exist_ok=True)
    client = TinkerChatClient()
    rng = random.Random(SEED + args.start)
    sheets = [build_fact_sheet(i, rng) for i in range(args.start, args.start + args.count)]

    sem = asyncio.Semaphore(CONCURRENCY)

    async def bounded(sheet):
        async with sem:
            try:
                r = await generate_transcript(client, sheet)
                if r:
                    n_surv = sum(q["answer_in_compressed"] for q in r["qa"])
                    print(f"  [{sheet['id']}] OK: {len(r['qa'])} facts, "
                          f"{n_surv} survive 2x, middle {r['middle_tokens']} tok")
                return r
            except Exception as e:
                print(f"  [{sheet['id']}] ERROR: {type(e).__name__}: {e}")
                return None

    results = await asyncio.gather(*[bounded(s) for s in sheets])
    transcripts = [r for r in results if r]
    print(f"\n{len(transcripts)}/{len(sheets)} transcripts generated OK")

    if args.append:
        existing = []
        with open(DATA_DIR / "synthetic_transcripts.jsonl", encoding="utf-8") as fh:
            existing = [json.loads(line) for line in fh]
        known = {r["id"] for r in existing}
        transcripts = [dict(r, split=None) for r in existing] + [
            r for r in transcripts if r["id"] not in known
        ]

    # Hold out the last N_HELDOUT by id order — never trained on.
    transcripts.sort(key=lambda r: r["id"])
    for i, r in enumerate(transcripts):
        r["split"] = "heldout" if i >= len(transcripts) - N_HELDOUT else "train"

    with open(DATA_DIR / "synthetic_transcripts.jsonl", "w", encoding="utf-8") as fh:
        for r in transcripts:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_train_qa = n_held_qa = n_filtered = 0
    with open(DATA_DIR / "train_qa.jsonl", "w", encoding="utf-8") as ftr, \
         open(DATA_DIR / "heldout_qa.jsonl", "w", encoding="utf-8") as fhe:
        for r in transcripts:
            compressed_context = build_condition_text(
                r["prefix_text"], r["summary_text"], r["suffix_text"], compressed=True
            )
            for q in r["qa"]:
                row = {
                    "transcript_id": r["id"],
                    "context": compressed_context,
                    **q,
                }
                if r["split"] == "train":
                    if q["answer_in_compressed"]:
                        ftr.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_train_qa += 1
                    else:
                        n_filtered += 1
                else:
                    fhe.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_held_qa += 1

    print(f"train QA: {n_train_qa} (filtered {n_filtered} whose answer did not survive 2x)")
    print(f"heldout QA: {n_held_qa} across {N_HELDOUT} transcripts")

    n_trap = sum(1 for r in transcripts for q in r["qa"] if q["is_trap"])
    n_all = sum(len(r["qa"]) for r in transcripts)
    print(f"trap facts: {n_trap}/{n_all} ({n_trap / n_all:.0%})")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
