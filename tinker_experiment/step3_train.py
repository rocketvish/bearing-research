"""Step 3 — LoRA SFT on Tinker.

Cookbook SFT pattern: conversation_to_datum with the same renderer as all
evals (qwen3_5_disable_thinking), forward_backward("cross_entropy") +
optim_step(Adam) per batch. Default LoRA rank (32, the cookbook default),
2 epochs, LR from tinker_cookbook.hyperparam_utils.get_lr when available.

Prints a token/cost estimate before launching; aborts if estimated cost
exceeds $25 unless --yes is passed (spec guardrail).
"""

import argparse
import asyncio
import json
import random
import time

from common import DATA_DIR, MODEL, RESULTS_DIR, ensure_api_key

ensure_api_key()

import tinker  # noqa: E402
from tinker_cookbook import renderers, tokenizer_utils  # noqa: E402
from tinker_cookbook.renderers import TrainOnWhat  # noqa: E402
from tinker_cookbook.supervised.data import conversation_to_datum  # noqa: E402

from harness import qa_messages  # noqa: E402
from common import RENDERER_NAME  # noqa: E402

EPOCHS = 2
BATCH_SIZE = 16
LORA_RANK = 32
MAX_LENGTH = 4096
TRAIN_PRICE_PER_M = 1.07  # USD, Qwen3.6-35B-A3B train price (docs, July 2026)
COST_CAP_USD = 25.0
SEED = 20260714


def _batch_loss(fb_result, batch) -> float:
    """Weighted mean NLL over the batch; tolerant of result-shape drift."""
    try:
        loss_sum = weight_sum = 0.0
        for out, datum in zip(fb_result.loss_fn_outputs, batch):
            logprobs = out["logprobs"].tolist()
            weights = datum.loss_fn_inputs["weights"].tolist()
            loss_sum += -sum(lp * w for lp, w in zip(logprobs, weights))
            weight_sum += sum(weights)
        return loss_sum / max(weight_sum, 1.0)
    except Exception as e:
        if not getattr(_batch_loss, "_warned", False):
            _batch_loss._warned = True
            print(f"[warn] per-token loss extraction failed ({type(e).__name__}: {e}); "
                  f"result attrs: {[a for a in dir(fb_result) if not a.startswith('_')]}")
        metrics = getattr(fb_result, "metrics", None) or {}
        for key in ("loss:mean", "loss", "train_mean_nll"):
            if key in metrics:
                return float(metrics[key])
        return float("nan")


def load_training_conversations(train_file: str) -> list[list[dict]]:
    rows = []
    with open(DATA_DIR / train_file, encoding="utf-8") as fh:
        for line in fh:
            rows.append(json.loads(line))
    conversations = []
    for r in rows:
        messages = qa_messages(r["context"], r["question"])
        messages.append({"role": "assistant", "content": r["answer"]})
        conversations.append(messages)
    return conversations


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip cost confirmation")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--train-file", default="train_qa.jsonl")
    parser.add_argument("--run-name", default="train_run",
                        help="basename for results/<run-name>.json and checkpoint name")
    args = parser.parse_args()

    t0 = time.time()
    conversations = load_training_conversations(args.train_file)
    print(f"{len(conversations)} training conversations")

    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)

    data = []
    for conv in conversations:
        datum = conversation_to_datum(
            conv, renderer, max_length=MAX_LENGTH,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        )
        data.append(datum)

    total_tokens_per_epoch = sum(len(d.model_input.to_ints()) for d in data)
    total_tokens = total_tokens_per_epoch * EPOCHS
    est_cost = total_tokens / 1e6 * TRAIN_PRICE_PER_M
    print(f"tokens/epoch={total_tokens_per_epoch:,}  epochs={EPOCHS}  "
          f"total={total_tokens:,} tokens  est. cost=${est_cost:.2f}")
    if total_tokens > 5_000_000:
        raise SystemExit("ABORT: exceeds the 5M training-token budget from the spec")
    if est_cost > COST_CAP_USD and not args.yes:
        raise SystemExit(f"ABORT: estimated cost ${est_cost:.2f} > ${COST_CAP_USD}; rerun with --yes")

    # Learning rate: cookbook recommendation if available.
    lr = args.lr
    if lr is None:
        try:
            from tinker_cookbook.hyperparam_utils import get_lr
            lr = get_lr(MODEL)
            print(f"lr from cookbook get_lr: {lr:.2e}")
        except Exception as e:
            lr = 1e-4
            print(f"cookbook get_lr unavailable ({e}); using default {lr:.0e}")

    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=MODEL, rank=LORA_RANK
    )

    rng = random.Random(SEED)
    losses = []
    step = 0
    for epoch in range(EPOCHS):
        rng.shuffle(data)
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i : i + BATCH_SIZE]
            fb_future = await training_client.forward_backward_async(batch, "cross_entropy")
            opt_future = await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=lr)
            )
            fb_result = await fb_future.result_async()
            await opt_future.result_async()

            loss = _batch_loss(fb_result, batch)
            losses.append(loss)
            step += 1
            print(f"epoch {epoch + 1} step {step}: loss={loss:.4f} "
                  f"({len(batch)} examples, {time.time() - t0:.0f}s elapsed)")

    # Persist a sampleable checkpoint reference for Step 4.
    sampling_path_future = await training_client.save_weights_for_sampler_async(name=args.run_name)
    sampling_path = (await sampling_path_future.result_async()).path
    print(f"checkpoint: {sampling_path}")

    RESULTS_DIR.mkdir(exist_ok=True)
    run_info = {
        "train_file": args.train_file,
        "model": MODEL,
        "renderer": RENDERER_NAME,
        "lora_rank": LORA_RANK,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": lr,
        "n_examples": len(data),
        "tokens_per_epoch": total_tokens_per_epoch,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(est_cost, 2),
        "losses": losses,
        "checkpoint": sampling_path,
        "wall_clock_s": round(time.time() - t0),
    }
    out_path = RESULTS_DIR / f"{args.run_name}.json"
    out_path.write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}  ({run_info['wall_clock_s']}s)")


if __name__ == "__main__":
    asyncio.run(main())
