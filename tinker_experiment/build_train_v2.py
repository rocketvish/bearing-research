"""Rebuild training data for the retrain (spec's one permitted contingency).

Two changes vs train_qa.jsonl, both reported in results.md:
1. Contingency proper: ~20% of examples use the FULL (uncompressed) context,
   because the Step 4 full-context sanity check regressed (7/8 -> 0/8 on the
   original transcript).
2. Implementation-artifact fix: targets become sentence-form short factual
   answers instead of bare values. Run 1 showed bare-value targets collapse
   the answer register (1-3 word answers), which the multi-keyword scorer
   fails even when the fact is right.

Writes data/train_qa_v2.jsonl.
"""

import json
import random

from common import DATA_DIR
from harness import build_condition_text

SEED = 20260714
FULL_CONTEXT_FRACTION = 0.20

TARGET_SENTENCES = {
    "password_hashing": "Passwords are hashed with {a}.",
    "test_framework": "The project uses {a} as its test framework.",
    "package_manager": "Dependencies are installed with {a}.",
    "id_scheme": "Record IDs are generated using {a}.",
    "linter": "The project is linted with {a}.",
    "cache": "The caching layer is {a}.",
    "http_client": "Outbound HTTP requests are made with {a}.",
    "logging": "Logging is done with {a}.",
    "port": "The server listens on port {a}.",
    "main_file": "The main entry-point file is {a}.",
    "table": "The primary database table is {a}.",
    "env_var": "It is stored in the {a} environment variable.",
}


def main():
    rng = random.Random(SEED)
    transcripts = []
    with open(DATA_DIR / "synthetic_transcripts.jsonl", encoding="utf-8") as fh:
        transcripts = [json.loads(line) for line in fh]

    rows = []
    for r in transcripts:
        if r["split"] != "train":
            continue
        compressed = build_condition_text(
            r["prefix_text"], r["summary_text"], r["suffix_text"], compressed=True
        )
        full = r["prefix_text"] + r["middle_text"] + r["suffix_text"]
        for q in r["qa"]:
            if not q["answer_in_compressed"]:
                continue
            target = TARGET_SENTENCES[q["category"]].format(a=q["answer"])
            rows.append({
                "transcript_id": r["id"],
                "context": compressed,
                "context_kind": "2x",
                **{k: q[k] for k in ("question", "category", "is_trap", "popular_default")},
                "answer": target,
            })
            if rng.random() < FULL_CONTEXT_FRACTION:
                rows.append({
                    "transcript_id": r["id"],
                    "context": full,
                    "context_kind": "full",
                    **{k: q[k] for k in ("question", "category", "is_trap", "popular_default")},
                    "answer": target,
                })

    with open(DATA_DIR / "train_qa_v2.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_full = sum(1 for r in rows if r["context_kind"] == "full")
    print(f"train_qa_v2.jsonl: {len(rows)} rows ({n_full} full-context, "
          f"{n_full / len(rows):.0%})")


if __name__ == "__main__":
    main()
