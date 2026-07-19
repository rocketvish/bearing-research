"""Step 4 — Post-training evaluation.

Runs every condition against BOTH the base model and the fine-tuned
checkpoint under the identical protocol, so comparisons and the flip table
are protocol-controlled:

  1. Original transcript, 8 questions: full / 2x / 3x (summaries reused
     from Step 1 so the contexts are byte-identical).
  2. Synthetic held-out set at 2x compression (~40 questions).
  3. Full-context sanity check on both eval sets (forgetting check).
  4. Flip table for the original 8 questions per condition.

Held-out scoring: pass if the fact's answer (or its leading token) appears
case-insensitively in the model's answer.
"""

import argparse
import asyncio
import json
import time

from clients import TinkerChatClient
from common import (
    DATA_DIR,
    NUM_PREFIX,
    NUM_SUFFIX,
    RESULTS_DIR,
    TRANSCRIPT_PATH,
    ensure_api_key,
)
from context_split import split_transcript
from harness import answer_question, build_condition_text, eval_condition, write_results

ensure_api_key()

TOKEN_BUDGET = 4096


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def heldout_pass(fact_answer: str, model_answer: str) -> bool:
    a = model_answer.lower()
    full = fact_answer.lower()
    lead = full.split()[0]
    return (full in a) or (lead in a)


async def eval_heldout(client, rows, label: str, use_full_context: bool = False) -> dict:
    """rows: heldout_qa.jsonl rows (one per question, context included)."""
    transcripts = None
    if use_full_context:
        transcripts = {
            r["id"]: r for r in load_jsonl(DATA_DIR / "synthetic_transcripts.jsonl")
        }

    sem = asyncio.Semaphore(8)

    async def one(row):
        context = row["context"]
        if use_full_context:
            tr = transcripts[row["transcript_id"]]
            context = tr["prefix_text"] + tr["middle_text"] + tr["suffix_text"]
        async with sem:
            ans = await answer_question(client, context, row["question"])
        return {
            "transcript_id": row["transcript_id"],
            "question": row["question"],
            "expected": row["answer"],
            "is_trap": row["is_trap"],
            "answer_in_compressed": row["answer_in_compressed"],
            "model_answer": ans,
            "pass": heldout_pass(row["answer"], ans),
        }

    detail = await asyncio.gather(*[one(r) for r in rows])
    n_pass = sum(d["pass"] for d in detail)
    trap = [d for d in detail if d["is_trap"]]
    n_trap_pass = sum(d["pass"] for d in trap)
    surv = [d for d in detail if d["answer_in_compressed"]]
    n_surv_pass = sum(d["pass"] for d in surv)
    print(f"  [{label}] {n_pass}/{len(detail)} "
          f"(traps {n_trap_pass}/{len(trap)}; survived-in-context {n_surv_pass}/{len(surv)})")
    return {
        "model": client.name,
        "total": n_pass,
        "n": len(detail),
        "trap_pass": n_trap_pass,
        "trap_n": len(trap),
        "survived_pass": n_surv_pass,
        "survived_n": len(surv),
        "detail": detail,
    }


async def eval_original(client, tag: str) -> dict:
    ctx = split_transcript(TRANSCRIPT_PATH, client.tokenizer, TOKEN_BUDGET, NUM_PREFIX, NUM_SUFFIX)
    summaries = json.loads(
        (DATA_DIR / "original_summaries.json").read_text(encoding="utf-8")
    )["summaries"]
    conditions = {
        "full": build_condition_text(ctx.prefix_text, ctx.middle_text, ctx.suffix_text, False),
        "2x": build_condition_text(ctx.prefix_text, summaries["2"]["text"], ctx.suffix_text, True),
        "3x": build_condition_text(ctx.prefix_text, summaries["3"]["text"], ctx.suffix_text, True),
    }
    out = {}
    for label, text in conditions.items():
        print(f"=== original/{label} [{tag}] ===")
        out[label] = await eval_condition(client, text, f"{tag}:{label}", quiet=True)
    return out


def flip_table(base_scores: dict, ft_scores: dict) -> dict:
    flips = {}
    for cond in base_scores:
        per_q = {}
        for q in base_scores[cond]["scores"]:
            b = base_scores[cond]["scores"][q]
            f = ft_scores[cond]["scores"][q]
            per_q[q] = {(0, 1): "wrong->right", (1, 0): "right->wrong",
                        (1, 1): "right", (0, 0): "wrong"}[(b, f)]
        flips[cond] = per_q
    return flips


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="train_run")
    args = parser.parse_args()

    t0 = time.time()
    run_info = json.loads((RESULTS_DIR / f"{args.run_name}.json").read_text(encoding="utf-8"))
    checkpoint = run_info["checkpoint"]
    heldout_rows = load_jsonl(DATA_DIR / "heldout_qa.jsonl")
    print(f"checkpoint: {checkpoint}\nheldout questions: {len(heldout_rows)}")

    base = TinkerChatClient()
    ft = TinkerChatClient()
    await ft.use_checkpoint(checkpoint, name=f"finetuned:{checkpoint}")

    results = {"checkpoint": checkpoint, "original": {}, "heldout": {}}

    results["original"]["base"] = await eval_original(base, "base")
    results["original"]["finetuned"] = await eval_original(ft, "finetuned")

    print("\n=== heldout 2x ===")
    results["heldout"]["base_2x"] = await eval_heldout(base, heldout_rows, "base:2x")
    results["heldout"]["finetuned_2x"] = await eval_heldout(ft, heldout_rows, "ft:2x")
    print("\n=== heldout full-context sanity ===")
    results["heldout"]["base_full"] = await eval_heldout(
        base, heldout_rows, "base:full", use_full_context=True)
    results["heldout"]["finetuned_full"] = await eval_heldout(
        ft, heldout_rows, "ft:full", use_full_context=True)

    results["flip_table"] = flip_table(
        results["original"]["base"], results["original"]["finetuned"]
    )

    suffix = "" if args.run_name == "train_run" else f"_{args.run_name}"
    write_results(RESULTS_DIR / f"step4_eval{suffix}.json", results)

    print(f"\n===== SUMMARY ({time.time() - t0:.0f}s) =====")
    print("original transcript (8 questions):")
    for cond in ("full", "2x", "3x"):
        b = results["original"]["base"][cond]["total"]
        f = results["original"]["finetuned"][cond]["total"]
        print(f"  {cond:>5}: base {b}/8 -> finetuned {f}/8")
    print("heldout synthetic:")
    for key in ("base_2x", "finetuned_2x", "base_full", "finetuned_full"):
        r = results["heldout"][key]
        print(f"  {key:>14}: {r['total']}/{r['n']} (traps {r['trap_pass']}/{r['trap_n']})")
    print("flip table (original):")
    for cond, per_q in results["flip_table"].items():
        interesting = {q: v for q, v in per_q.items() if v in ("wrong->right", "right->wrong")}
        print(f"  {cond}: {interesting or 'no flips'}")


if __name__ == "__main__":
    asyncio.run(main())
