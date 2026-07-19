"""Provider-agnostic eval harness.

Prompt construction, summarization, question answering, scoring, and
results logging — all against the ChatClient protocol in clients.py, so
the same harness runs against Tinker or any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from clients import ChatClient
from common import MAX_ANSWER_TOKENS, QUESTIONS, score_answers

PROMPT_PREFIX = "Given the following coding agent conversation:\n\n"

SUMMARIZE_SYSTEM = (
    "You are a precise technical summarizer. Summarize coding-agent "
    "conversation transcripts while preserving ALL technical specifics: "
    "library and package names (exact spelling), file paths and file names, "
    "function names, config values, database schema details, versions, "
    "and decisions made. Never substitute a similar-sounding library for "
    "the one actually used. Output only the summary, no preamble."
)


def prompt_suffix(question: str) -> str:
    return (
        f"\n\nQuestion: {question}\n"
        "Answer concisely and specifically based only on the conversation above.\n\n"
        "Answer:"
    )


def build_condition_text(
    prefix: str, middle_or_summary: str, suffix: str, compressed: bool
) -> str:
    if compressed:
        return prefix + f"\n[SUMMARY OF MIDDLE TURNS: {middle_or_summary}]\n" + suffix
    return prefix + middle_or_summary + suffix


def qa_messages(context_text: str, question: str) -> list[dict]:
    return [
        {"role": "user", "content": PROMPT_PREFIX + context_text + prompt_suffix(question)}
    ]


async def answer_question(client: ChatClient, context_text: str, question: str) -> str:
    try:
        return await client.chat(
            qa_messages(context_text, question), max_tokens=MAX_ANSWER_TOKENS, temperature=0.0
        )
    except Exception as e:  # keep partial failures from blocking other questions
        return f"<error: {type(e).__name__}: {e}>"


async def answer_all(
    client: ChatClient,
    context_text: str,
    questions: list[str],
    label: str,
    quiet: bool = False,
) -> dict[str, str]:
    results = await asyncio.gather(
        *[answer_question(client, context_text, q) for q in questions]
    )
    answers = {}
    for i, ans in enumerate(results, start=1):
        answers[f"Q{i}"] = ans
        if not quiet:
            print(f"  [{label}] Q{i}: {ans[:110]!r}")
    return answers


async def eval_condition(
    client: ChatClient,
    context_text: str,
    label: str,
    questions: list[str] = QUESTIONS,
    score_fn=score_answers,
    quiet: bool = False,
) -> dict:
    answers = await answer_all(client, context_text, questions, label, quiet=quiet)
    scores, total = score_fn(answers)
    print(f"  [{label}] score: {total}/{len(questions)}")
    return {
        "model": client.name,
        "context_tokens": client.count_tokens(context_text),
        "answers": answers,
        "scores": scores,
        "total": total,
    }


async def summarize_to_budget(
    client: ChatClient, middle_text: str, budget_tokens: int, max_retries: int = 2
) -> tuple[str, int]:
    """Summarize middle_text to <= budget_tokens; retry tighter, then truncate.

    Budget enforcement uses client.count_tokens — run this with a client
    whose tokenizer is the target model's.
    """
    ask = budget_tokens
    summary = ""
    for attempt in range(max_retries + 1):
        user = (
            f"Summarize the following coding-agent conversation excerpt in AT MOST "
            f"{ask} tokens (~{int(ask * 0.75)} words). Preserve every technical "
            "specific: exact library/package names, file names, schema/columns, "
            "validation rules, and what was done most recently.\n\n"
            f"---\n{middle_text}\n---"
        )
        summary = await client.chat(
            [
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=budget_tokens + 128,
            temperature=0.0,
        )
        n = client.count_tokens(summary)
        print(f"  [summarize] attempt {attempt + 1}: {n} tokens (budget {budget_tokens})")
        if n <= budget_tokens:
            return summary, n
        ask = max(32, int(ask * 0.8))
    if hasattr(client, "tokenizer") and client.tokenizer is not None:
        ids = client.tokenizer(summary, add_special_tokens=False).input_ids[:budget_tokens]
        summary = client.tokenizer.decode(ids, skip_special_tokens=True)
    else:
        summary = summary[: budget_tokens * 4]
    n = client.count_tokens(summary)
    print(f"  [summarize] hard-truncated to {n} tokens")
    return summary, n


def write_results(path: str | Path, results: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {path}")
