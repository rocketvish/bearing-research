"""Shared config and helpers for the Tinker LoRA-recovery experiment.

Model choice: the spec's Qwen/Qwen3-30B-A3B-Instruct-2507 was retired from
the Tinker lineup on 2026-06-12. Qwen3.6-35B-A3B is the successor that
matches the spec's criteria (MoE ~3B active params, Qwen family). It is a
hybrid thinking model, so ALL rendering — training data, baseline evals,
post-training evals — goes through the qwen3_5_disable_thinking renderer
to keep chain-of-thought off and train/eval formatting identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

MODEL = "Qwen/Qwen3.6-35B-A3B"
RENDERER_NAME = "qwen3_5_disable_thinking"

EXPERIMENT_DIR = Path(__file__).parent
REPO_DIR = EXPERIMENT_DIR.parent
TRANSCRIPT_PATH = REPO_DIR / "transcript.jsonl"
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_DIR = EXPERIMENT_DIR / "results"

# Same 4096-token window as Parts 4-5, but narrower verbatim boundaries:
# with the original prefix=2/suffix=3 split, Qwen3.6-35B-A3B scores 8/8
# even with the middle deleted (see results/diagnose_headroom.json), so no
# compression ratio could show loss. prefix=1/suffix=1 moves the
# load-bearing facts into the compressible middle (floor drops to 2/8)
# while keeping the window text — and thus the calibrated scorer — intact.
NUM_PREFIX = 1
NUM_SUFFIX = 1

# Eval sampling: temperature 0, short factual answers.
MAX_ANSWER_TOKENS = 200


def ensure_api_key() -> None:
    """Read TINKER_API_KEY from the process env or the Windows user env."""
    if os.environ.get("TINKER_API_KEY"):
        return
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                val, _ = winreg.QueryValueEx(k, "TINKER_API_KEY")
            if val:
                os.environ["TINKER_API_KEY"] = val
                return
        except OSError:
            pass
    raise SystemExit("TINKER_API_KEY not found in environment or user registry")


# ------------------------- original 8-question eval -------------------------
# Copied verbatim from evaluate.py / compress_then_evict_test.py for
# behavioral continuity with Parts 4-5. Do not modify.

QUESTIONS = [
    "What type of application is being built in this project?",
    "What database tables were created, and what are their key columns?",
    "What does the markdown-to-HTML converter handle? List the supported elements.",
    "How does the authentication system work? What library is used for password hashing?",
    "What validation rules does the task validation middleware enforce?",
    "Which modules depend on the database module, and how do they use it?",
    "What was the last thing the agent worked on or verified?",
    "List all the source files that were created during this session.",
]


def score_q1(ans: str) -> bool:
    a = ans.lower()
    return ("express" in a) and ("sqlite" in a) and ("task" in a)


def score_q2(ans: str) -> bool:
    a = ans.lower()
    return ("users" in a) and ("tasks" in a) and ("reports" in a)


def score_q3(ans: str) -> bool:
    a = ans.lower()
    has_core = ("bold" in a) and ("italic" in a) and ("code" in a)
    mentions_extra = ("image" in a) or ("link" in a)
    # Only treat links/images as a hallucination when claimed as SUPPORTED.
    # A correct answer may say "does not handle images" -- don't penalize.
    negated = (
        "does not" in a or "doesn't" in a or "do not" in a or "not handle" in a
        or "no images" in a or "no image" in a or "no links" in a or "without" in a
        or "does not support" in a or "doesn't support" in a or "not support" in a
    )
    hallucinated = mentions_extra and not negated
    return has_core and not hallucinated


def score_q4(ans: str) -> bool:
    a = ans.lower()
    return ("jwt" in a) and ("bcryptjs" in a)


def score_q5(ans: str) -> bool:
    a = ans.lower()
    return (
        "title" in a and "pending" in a
        and ("in_progress" in a or "in progress" in a) and "done" in a
    )


def score_q6(ans: str) -> bool:
    a = ans.lower()
    return ("auth" in a) and ("task" in a) and ("report" in a)


def score_q7(ans: str) -> bool:
    return "db.js" in ans.lower()


def score_q8(ans: str) -> bool:
    a = ans.lower()
    return (
        "db.js" in a and "markdown.js" in a and "validate" in a
        and "auth.js" in a and "reports.js" in a
    )


SCORE_FNS = [score_q1, score_q2, score_q3, score_q4, score_q5, score_q6, score_q7, score_q8]

# Default-trap questions per the spec: parametric prior is wrong.
DEFAULT_TRAP_QUESTIONS = {4: "bcryptjs (not bcrypt)", 3: "regex (not marked)", 8: "custom files"}


def score_answers(answers: dict[str, str]) -> tuple[dict[str, int], int]:
    scores, total = {}, 0
    for i, fn in enumerate(SCORE_FNS, start=1):
        ans = answers.get(f"Q{i}", "") or ""
        ok = 0
        if ans and not ans.startswith("<error"):
            try:
                ok = int(bool(fn(ans)))
            except Exception:
                ok = 0
        scores[f"Q{i}"] = ok
        total += ok
    return scores, total
