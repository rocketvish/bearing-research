"""Phase 1 end-to-end evaluation pipeline.

Loads the model once, builds a context window, produces three
compressed representations of the middle turns (mean-pooled, gradient-
optimized, text-summarized), then asks the same 8 questions under 5
conditions and writes the answers to results.json.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from compressor import ContextCompressor
from config import MODEL_ID, strip_thinking
from context_builder import ContextBuilder
TRANSCRIPT_PATH = Path(__file__).parent / "transcript.jsonl"
RESULTS_PATH = Path(__file__).parent / "results.json"

TOKEN_BUDGET = 4096
NUM_PREFIX = 2
NUM_SUFFIX = 3
NUM_VIRTUAL = 50
OPT_STEPS = 300
OPT_LR = 0.01
MAX_NEW_TOKENS = 200
SUMMARY_TOKENS = 200

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

PROMPT_PREFIX = "Given the following coding agent conversation:\n\n"


def prompt_suffix(question: str) -> str:
    return (
        f"\n\nQuestion: {question}\n"
        "Answer concisely and specifically based only on the conversation above.\n\n"
        "Answer:"
    )


def vram_str() -> str:
    if not torch.cuda.is_available():
        return "VRAM n/a"
    a = torch.cuda.memory_allocated() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    return f"VRAM allocated={a:.2f} GB, peak={p:.2f} GB"


# ----------------------------- model load ----------------------------

def load_model():
    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    print(f"Loaded. {vram_str()}")
    return tokenizer, model


# --------------------------- generation ------------------------------

@torch.no_grad()
def generate_from_text(model, tokenizer, text: str) -> tuple[str, int]:
    """Tokenize text, generate, return (answer, input_token_count)."""
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    n_input = int(input_ids.shape[1])
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, n_input:]
    answer = strip_thinking(tokenizer.decode(new_ids, skip_special_tokens=True))
    del input_ids, out
    return answer, n_input


@torch.no_grad()
def _manual_generate_from_embeds(
    model, tokenizer, inputs_embeds: torch.Tensor, max_new_tokens: int
) -> str:
    """Greedy fallback: forward, argmax, embed, repeat."""
    embed_layer = model.get_input_embeddings()
    eos_id = tokenizer.eos_token_id
    cur = inputs_embeds
    generated_ids: list[int] = []
    for _ in range(max_new_tokens):
        out = model(inputs_embeds=cur, use_cache=False)
        next_id = int(out.logits[0, -1, :].argmax().item())
        generated_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break
        next_emb = embed_layer(
            torch.tensor([[next_id]], dtype=torch.long, device=model.device)
        )
        cur = torch.cat([cur, next_emb], dim=1)
        del out
    return strip_thinking(tokenizer.decode(generated_ids, skip_special_tokens=True))


@torch.no_grad()
def generate_from_embeds(model, tokenizer, inputs_embeds: torch.Tensor) -> str:
    """Try model.generate(inputs_embeds=...); fall back to manual loop."""
    seq_len = int(inputs_embeds.shape[1])
    attn = torch.ones((1, seq_len), dtype=torch.long, device=inputs_embeds.device)
    try:
        out = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # generate(inputs_embeds=...) returns only the newly generated tokens.
        answer = strip_thinking(tokenizer.decode(out[0], skip_special_tokens=True))
        del out
        return answer
    except Exception:
        return _manual_generate_from_embeds(
            model, tokenizer, inputs_embeds, MAX_NEW_TOKENS
        )


# ----------------------- per-condition runners -----------------------

def run_full_context(model, tokenizer, cb: ContextBuilder) -> dict:
    answers: list[str] = []
    last_n_input = 0
    for i, q in enumerate(QUESTIONS):
        text = PROMPT_PREFIX + cb.full_text + prompt_suffix(q)
        t0 = time.perf_counter()
        ans, n_input = generate_from_text(model, tokenizer, text)
        dt = time.perf_counter() - t0
        last_n_input = n_input
        print(f"  [full_context] Q{i+1} {dt:.1f}s n_input={n_input}: {ans[:100]!r}")
        answers.append(ans)
    return {"answers": answers, "total_tokens": last_n_input}


def run_truncated(model, tokenizer, cb: ContextBuilder) -> dict:
    middle_marker = "\n[... middle turns omitted ...]\n"
    base_context = cb.prefix_text + middle_marker + cb.suffix_text
    answers: list[str] = []
    last_n_input = 0
    for i, q in enumerate(QUESTIONS):
        text = PROMPT_PREFIX + base_context + prompt_suffix(q)
        t0 = time.perf_counter()
        ans, n_input = generate_from_text(model, tokenizer, text)
        dt = time.perf_counter() - t0
        last_n_input = n_input
        print(f"  [truncated] Q{i+1} {dt:.1f}s n_input={n_input}: {ans[:100]!r}")
        answers.append(ans)
    return {"answers": answers, "total_tokens": last_n_input}


@torch.no_grad()
def _embed_text(model, tokenizer, text: str) -> torch.Tensor:
    embed = model.get_input_embeddings()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        model.device
    )
    return embed(ids)


def _run_virtual_condition(
    model, tokenizer, cb: ContextBuilder, virtual: torch.Tensor, label: str
) -> dict:
    """Shared logic for mean_pooled / optimized conditions."""
    embed = model.get_input_embeddings()
    device = model.device
    target_dtype = embed.weight.dtype

    # Pre-embed the static parts once.
    prompt_prefix_emb = _embed_text(model, tokenizer, PROMPT_PREFIX)
    prefix_t, suffix_t = cb.get_verbatim_parts(device=device)
    with torch.no_grad():
        prefix_emb = embed(prefix_t)
        suffix_emb = embed(suffix_t)
    virtual_cast = virtual.to(dtype=target_dtype, device=device)

    answers: list[str] = []
    total_seq_len = 0
    for i, q in enumerate(QUESTIONS):
        q_emb = _embed_text(model, tokenizer, prompt_suffix(q))
        inputs_embeds = torch.cat(
            [prompt_prefix_emb, prefix_emb, virtual_cast, suffix_emb, q_emb], dim=1
        )
        total_seq_len = int(inputs_embeds.shape[1])
        t0 = time.perf_counter()
        ans = generate_from_embeds(model, tokenizer, inputs_embeds)
        dt = time.perf_counter() - t0
        print(f"  [{label}] Q{i+1} {dt:.1f}s seq_len={total_seq_len}: {ans[:100]!r}")
        answers.append(ans)
        del q_emb, inputs_embeds

    del prompt_prefix_emb, prefix_emb, suffix_emb, virtual_cast
    torch.cuda.empty_cache()
    return {"answers": answers, "total_tokens": total_seq_len}


def run_mean_pooled(model, tokenizer, cb: ContextBuilder, virtual: torch.Tensor) -> dict:
    res = _run_virtual_condition(model, tokenizer, cb, virtual, "mean_pooled")
    res["num_virtual"] = int(virtual.shape[1])
    return res


def run_optimized(
    model,
    tokenizer,
    cb: ContextBuilder,
    virtual: torch.Tensor,
    final_loss: float,
) -> dict:
    res = _run_virtual_condition(model, tokenizer, cb, virtual, "optimized")
    res["num_virtual"] = int(virtual.shape[1])
    res["optimization_steps"] = OPT_STEPS
    res["final_loss"] = final_loss
    return res


def run_text_compressed(
    model,
    tokenizer,
    cb: ContextBuilder,
    summary_text: str,
    summary_token_count: int,
) -> dict:
    middle_replacement = (
        f"\n[SUMMARY OF MIDDLE TURNS: {summary_text}]\n"
    )
    base_context = cb.prefix_text + middle_replacement + cb.suffix_text
    answers: list[str] = []
    last_n_input = 0
    for i, q in enumerate(QUESTIONS):
        text = PROMPT_PREFIX + base_context + prompt_suffix(q)
        t0 = time.perf_counter()
        ans, n_input = generate_from_text(model, tokenizer, text)
        dt = time.perf_counter() - t0
        last_n_input = n_input
        print(f"  [text_compressed] Q{i+1} {dt:.1f}s n_input={n_input}: {ans[:100]!r}")
        answers.append(ans)
    return {
        "answers": answers,
        "total_tokens": last_n_input,
        "summary_tokens": summary_token_count,
    }


# ----------------------------- reporting -----------------------------

def print_comparison_table(results: dict) -> None:
    print()
    print("=" * 100)
    print("  COMPARISON TABLE")
    print("=" * 100)
    conditions = list(results["conditions"].keys())
    for i, q in enumerate(results["questions"]):
        print()
        print(f"Q{i+1}: {q}")
        print("-" * 100)
        for cond in conditions:
            data = results["conditions"][cond]
            if "error" in data:
                line = f"<error: {data['error']}>"
            else:
                ans = data.get("answers", [None] * len(results["questions"]))[i] or ""
                line = ans.replace("\n", " ")
                if len(line) > 200:
                    line = line[:200] + "…"
            print(f"  {cond:<18} {line}")


# -------------------------------- main -------------------------------

def main():
    tokenizer, model = load_model()

    print()
    print("=" * 78)
    print("  Building context window")
    print("=" * 78)
    cb = ContextBuilder(
        TRANSCRIPT_PATH,
        tokenizer,
        token_budget=TOKEN_BUDGET,
        num_prefix=NUM_PREFIX,
        num_suffix=NUM_SUFFIX,
    )

    print()
    print("=" * 78)
    print("  Building compressed representations")
    print("=" * 78)
    compressor = ContextCompressor(model, tokenizer, cb)

    # Compute target up front so optimize_compress can reuse it.
    target = None
    try:
        target = compressor.compute_target_hidden_states()
    except Exception as e:
        print(f"[FATAL] compute_target_hidden_states failed: {e}")
        traceback.print_exc()

    mean_pooled_virt = None
    try:
        mean_pooled_virt = compressor.mean_pool_compress(num_virtual_tokens=NUM_VIRTUAL)
    except Exception as e:
        print(f"[ERROR] mean_pool_compress failed: {e}")
        traceback.print_exc()

    optimized_virt = None
    final_loss = float("nan")
    if target is not None:
        try:
            optimized_virt, final_loss = compressor.optimize_compress(
                num_virtual_tokens=NUM_VIRTUAL,
                num_steps=OPT_STEPS,
                lr=OPT_LR,
            )
        except Exception as e:
            print(f"[ERROR] optimize_compress failed: {e}")
            traceback.print_exc()
    else:
        print("[skip] optimize_compress — no target")

    summary_text = ""
    summary_embeds = None
    summary_token_count = 0
    try:
        summary_text, summary_embeds, summary_token_count = compressor.text_compress(
            max_summary_tokens=SUMMARY_TOKENS
        )
    except Exception as e:
        print(f"[ERROR] text_compress failed: {e}")
        traceback.print_exc()

    print()
    print(f"After compression — {vram_str()}")

    # Run conditions.
    conditions: dict = {}

    def _bracket(label: str, idx: int):
        print()
        print("=" * 78)
        print(f"  Condition {idx}/5: {label}")
        print("=" * 78)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print(f"start — {vram_str()}")

    def _close():
        torch.cuda.empty_cache()
        print(f"done — {vram_str()}")

    _bracket("full_context", 1)
    try:
        conditions["full_context"] = run_full_context(model, tokenizer, cb)
    except Exception as e:
        print(f"[ERROR] full_context failed: {e}")
        traceback.print_exc()
        conditions["full_context"] = {"error": f"{type(e).__name__}: {e}"}
    _close()

    _bracket("truncated", 2)
    try:
        conditions["truncated"] = run_truncated(model, tokenizer, cb)
    except Exception as e:
        print(f"[ERROR] truncated failed: {e}")
        traceback.print_exc()
        conditions["truncated"] = {"error": f"{type(e).__name__}: {e}"}
    _close()

    _bracket("mean_pooled", 3)
    if mean_pooled_virt is not None:
        try:
            conditions["mean_pooled"] = run_mean_pooled(
                model, tokenizer, cb, mean_pooled_virt
            )
        except Exception as e:
            print(f"[ERROR] mean_pooled failed: {e}")
            traceback.print_exc()
            conditions["mean_pooled"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        conditions["mean_pooled"] = {"error": "mean_pool_compress did not produce embeddings"}
    _close()

    _bracket("optimized", 4)
    if optimized_virt is not None:
        try:
            conditions["optimized"] = run_optimized(
                model, tokenizer, cb, optimized_virt, final_loss
            )
        except Exception as e:
            print(f"[ERROR] optimized failed: {e}")
            traceback.print_exc()
            conditions["optimized"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        conditions["optimized"] = {"error": "optimize_compress did not produce embeddings"}
    _close()

    _bracket("text_compressed", 5)
    if summary_text:
        try:
            conditions["text_compressed"] = run_text_compressed(
                model, tokenizer, cb, summary_text, summary_token_count
            )
        except Exception as e:
            print(f"[ERROR] text_compressed failed: {e}")
            traceback.print_exc()
            conditions["text_compressed"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        conditions["text_compressed"] = {"error": "text_compress did not produce a summary"}
    _close()

    # Save and report.
    results = {
        "model_id": MODEL_ID,
        "token_budget": TOKEN_BUDGET,
        "num_prefix": NUM_PREFIX,
        "num_suffix": NUM_SUFFIX,
        "num_virtual": NUM_VIRTUAL,
        "questions": QUESTIONS,
        "conditions": conditions,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {RESULTS_PATH}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()
