"""Token importance diagnostic: perplexity vs V-uniqueness.

Diagnostic only — no optimization, no compression, no generation.
For each of three validated turns (8, 18, 24), compute two per-token
importance signals and compare which one better identifies the
"failure tokens" we've seen in compression results (bcryptjs / dev-secret
/ in_progress / etc).

Signals
-------
1. Perplexity: per-token cross-entropy from the V-target forward's
   logits (same as compute_perplexity_weights).

2. V-uniqueness at neighborhood size k: per-layer-per-head, compare
   each token's V vector against the mean of its 2k surrounding
   neighbors (excluding itself):

     uniq[i, L, h] = ||V[L, h, i, :] - neighbor_mean|| / (||neighbor_mean|| + eps)

   Then average over heads and layers → one scalar per token. Computed
   for k in {3, 5, 10}; k=5 is the primary.

Standalone — no imports from project files.
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ----------------------------- constants -----------------------------

from config import MODEL_ID
TRANSCRIPT_PATH = Path(__file__).parent / "transcript.jsonl"
RESULTS_PATH = Path(__file__).parent / "results_token_importance.json"

PREFIX_TURN_INDEX = 0  # 0-indexed; turn 1 (system prompt)

NEIGHBORHOOD_KS = [3, 5, 10]
PRIMARY_K = 5

TURNS_TO_ANALYZE = [
    {
        "turn_index_1based": 8,
        "content_type": "project conventions doc (CLAUDE.md)",
        "failure_tokens": ["bcryptjs", "js", "node", "test", "regex"],
    },
    {
        "turn_index_1based": 18,
        "content_type": "JWT auth middleware (JS code)",
        "failure_tokens": ["jsonwebtoken", "401", "dev", "secret"],
    },
    {
        "turn_index_1based": 24,
        "content_type": "reports.js (weekly-report generator, JS code)",
        "failure_tokens": ["in_progress", "done", "pending", "7"],
    },
]


# ----------------------------- helpers -------------------------------

def vram_str() -> str:
    if not torch.cuda.is_available():
        return "VRAM n/a"
    a = torch.cuda.memory_allocated() / 1024**3
    p = torch.cuda.max_memory_allocated() / 1024**3
    return f"VRAM allocated={a:.2f} GB, peak={p:.2f} GB"


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(
                    "[tool_use {name}] {input}".format(
                        name=block.get("name", ""),
                        input=json.dumps(block.get("input", {}), ensure_ascii=False),
                    )
                )
            elif btype == "tool_result":
                parts.append(
                    "[tool_result {tid}] {content}".format(
                        tid=block.get("tool_use_id", ""),
                        content=json.dumps(block.get("content", ""), ensure_ascii=False),
                    )
                )
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "\n".join(parts)
    return str(content)


def turn_marker(turn_idx_1based: int, role: str, content_str: str) -> str:
    return f"=== TURN {turn_idx_1based} ({role}) ===\n{content_str}\n"


def to_legacy_kv(pkv) -> tuple:
    if pkv is None:
        return ()
    if isinstance(pkv, tuple):
        return pkv
    if hasattr(pkv, "to_legacy_cache"):
        try:
            return tuple(pkv.to_legacy_cache())
        except Exception:
            pass
    layers = getattr(pkv, "layers", None)
    if isinstance(layers, (list, tuple)) and len(layers) > 0:
        out = []
        for layer in layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                out.append((layer.keys, layer.values))
            elif hasattr(layer, "key_cache") and hasattr(layer, "value_cache"):
                out.append((layer.key_cache, layer.value_cache))
            else:
                raise RuntimeError(
                    f"Unknown layer type: {type(layer).__name__}, "
                    f"attrs={[a for a in dir(layer) if not a.startswith('_')]}"
                )
        return tuple(out)
    if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
        return tuple(zip(pkv.key_cache, pkv.value_cache))
    raise RuntimeError(
        f"Unknown past_key_values type: {type(pkv).__name__}, "
        f"attrs={[a for a in dir(pkv) if not a.startswith('_')]}"
    )


def wrap_legacy_kv(legacy: tuple, config) -> object:
    if not legacy:
        return None
    from transformers.cache_utils import DynamicCache
    return DynamicCache(legacy, config=config)


def detach_kv(legacy: tuple) -> tuple:
    return tuple((k.detach(), v.detach()) for k, v in legacy)


def kv_seq_len(legacy: tuple) -> int:
    if not legacy:
        return 0
    return int(legacy[0][0].shape[-2])


def verify_rope_for_qwen() -> tuple[bool, str]:
    try:
        from transformers.models.qwen3 import modeling_qwen3
        src = inspect.getsource(modeling_qwen3.Qwen3Attention.forward)
    except Exception as e:
        return False, f"could not inspect Qwen3Attention.forward: {e}"
    matched = [ln.strip() for ln in src.splitlines() if "apply_rotary_pos_emb" in ln]
    if not matched:
        return False, "apply_rotary_pos_emb not found in Qwen3Attention.forward"
    assign = next((ln for ln in matched if "= apply_rotary_pos_emb" in ln), None)
    if assign is None:
        return False, "apply_rotary_pos_emb mentioned but not assigned to: " + " | ".join(matched)
    lhs = assign.split("=", 1)[0].lower()
    has_q = "query" in lhs
    has_k = "key" in lhs
    has_v = "value" in lhs
    if has_q and has_k and not has_v:
        return True, assign
    return False, f"unexpected RoPE assignment shape (q={has_q} k={has_k} v={has_v}): {assign}"


def compute_perplexity_weights(
    logits: torch.Tensor, turn_ids: torch.Tensor
) -> torch.Tensor:
    """Per-token cross-entropy from logits against the next token.
    Position 0 has no preceding logit in this forward, prepend 0.0.
    """
    if logits.shape[1] != turn_ids.shape[1]:
        raise ValueError(
            f"logits seq {logits.shape[1]} != turn_ids seq {turn_ids.shape[1]}"
        )
    device = logits.device
    target_ids = turn_ids[0, 1:].to(device).long()
    pred_logits = logits[0, :-1, :].float()
    nll = F.cross_entropy(pred_logits, target_ids, reduction="none")
    zero = torch.zeros(1, device=device, dtype=nll.dtype)
    return torch.cat([zero, nll], dim=0).detach().clone()


# ------------------- V-uniqueness (the new signal) -------------------

@torch.no_grad()
def compute_v_uniqueness(v_full: list[torch.Tensor], k: int) -> torch.Tensor:
    """V-uniqueness per token, averaged over heads and layers.

    For each layer's V tensor of shape (1, n_kv_heads, S, head_dim):
      window[i] = positions in [max(0, i-k), min(S, i+k+1))
      neighbors[i] = window[i] \\ {i}
      neighbor_mean[h, i, :] = mean over neighbors of V[h, j, :]
      uniq[h, i] = ||V[h, i, :] - neighbor_mean[h, i, :]||
                   / (||neighbor_mean[h, i, :]|| + 1e-8)

    Then average over heads → per-token-per-layer; average over layers
    → per-token. Returns shape (S,) on CPU as fp32.

    Vectorized via cumsum so we don't loop over S in Python.
    """
    if not v_full:
        raise ValueError("v_full is empty")
    s = int(v_full[0].shape[2])
    device = v_full[0].device

    # Per-token window bounds and neighbor counts (shared across layers).
    arange_s = torch.arange(s, device=device)
    los = (arange_s - k).clamp(min=0)              # (S,)
    his = (arange_s + k + 1).clamp(max=s)          # (S,)
    num_neighbors = (his - los - 1).clamp(min=1)   # (S,) — avoid div0

    eps = 1e-8
    per_layer_uniq: list[torch.Tensor] = []
    for v_layer in v_full:
        # v_layer: (1, n_kv, S, Dh)  →  v: (n_kv, S, Dh) in fp32
        v = v_layer[0].float()
        n_kv, _, dh = v.shape

        # cumsum padded with a leading zero so cumsum_pad[j] = sum(v[:, :j, :])
        zeros = torch.zeros(n_kv, 1, dh, dtype=v.dtype, device=device)
        cumsum_pad = torch.cat([zeros, v.cumsum(dim=1)], dim=1)  # (n_kv, S+1, Dh)

        # window_sum[h, i, :] = sum over j in [los[i], his[i]) of v[h, j, :]
        window_sum = cumsum_pad[:, his, :] - cumsum_pad[:, los, :]  # (n_kv, S, Dh)
        neighbor_sum = window_sum - v                                # exclude self
        neighbor_mean = neighbor_sum / num_neighbors.view(1, -1, 1)  # (n_kv, S, Dh)

        diff_norm = (v - neighbor_mean).norm(dim=-1)   # (n_kv, S)
        mean_norm = neighbor_mean.norm(dim=-1)         # (n_kv, S)
        head_uniq = diff_norm / (mean_norm + eps)      # (n_kv, S)
        per_token = head_uniq.mean(dim=0)              # (S,)
        per_layer_uniq.append(per_token)
        del cumsum_pad, window_sum, neighbor_sum, neighbor_mean, diff_norm, mean_norm

    stacked = torch.stack(per_layer_uniq, dim=0)  # (n_layers, S)
    return stacked.mean(dim=0).detach().cpu()


# --------------------------- rank helpers ----------------------------

def ranks_descending(scores: torch.Tensor) -> dict[int, int]:
    """position → rank (1-indexed, highest score = rank 1).

    Stable on ties via argsort's deterministic ordering on equal values
    (lower position wins on ties, since stable sort preserves order).
    """
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    return {int(pos): r + 1 for r, pos in enumerate(order)}


def find_token_positions(
    tokens: list[str], targets: list[str]
) -> dict[str, list[int]]:
    """For each target substring, return positions whose decoded token
    contains the target (case-insensitive).
    """
    out: dict[str, list[int]] = {}
    for target in targets:
        t = target.lower()
        out[target] = [i for i, s in enumerate(tokens) if t in s.lower()]
    return out


# ---------------------- per-turn analysis driver ---------------------

def analyze_turn(
    model, tokenizer, prefix_kv: tuple, prefix_len: int,
    turn_record: dict, failure_targets: list[str], label: str,
) -> dict:
    """Run one no-grad forward, compute both signals + analyses.
    Returns the per-turn dict for the JSON.
    """
    device = model.device
    turn_ids = turn_record["ids"].to(device)
    n_test = int(turn_record["n_tokens"])

    print()
    print("=" * 78)
    print(f"  [{label}] turn={turn_record['turn_index_1based']} "
          f"role={turn_record['role']} n_test={n_test}")
    print("=" * 78)

    # ---- 1. Forward to capture logits + V ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pos = torch.arange(
        prefix_len, prefix_len + n_test, device=device, dtype=torch.long,
    ).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=turn_ids,
            past_key_values=wrap_legacy_kv(prefix_kv, model.config),
            position_ids=pos,
            output_hidden_states=False,
            use_cache=True,
        )
    perplexity = compute_perplexity_weights(out.logits, turn_ids).cpu()
    full_kv = to_legacy_kv(out.past_key_values)
    v_full: list[torch.Tensor] = []
    for k_layer, v_layer in full_kv:
        v_new = v_layer[:, :, prefix_len:, :].detach().clone()
        v_full.append(v_new)
        del k_layer, v_layer
    del out, full_kv
    torch.cuda.empty_cache()
    print(f"  [{label}] V_full: {len(v_full)} layers x {tuple(v_full[0].shape)} "
          f"perplexity[min,max]=[{perplexity.min().item():.3f}, "
          f"{perplexity.max().item():.3f}]  {vram_str()}")

    # ---- 2. V-uniqueness for each k ----
    v_uniq_by_k: dict[int, torch.Tensor] = {}
    for k in NEIGHBORHOOD_KS:
        t0 = time.perf_counter()
        v_uniq_by_k[k] = compute_v_uniqueness(v_full, k)
        dt = time.perf_counter() - t0
        umin = v_uniq_by_k[k].min().item()
        umax = v_uniq_by_k[k].max().item()
        print(f"  [{label}] v_uniqueness k={k:>2}: range [{umin:.3f}, {umax:.3f}], "
              f"{dt:.1f}s")
    del v_full
    torch.cuda.empty_cache()

    # ---- 3. Decode tokens individually ----
    tokens: list[str] = []
    for i in range(n_test):
        tok = tokenizer.decode([int(turn_ids[0, i].item())], skip_special_tokens=False)
        tokens.append(tok)

    # ---- 4. Build rank maps ----
    ppl_rank = ranks_descending(perplexity)
    uniq_rank_by_k: dict[int, dict[int, int]] = {
        k: ranks_descending(v_uniq_by_k[k]) for k in NEIGHBORHOOD_KS
    }

    # ---- 5. Top-20 tables ----
    top_n = 20
    ppl_top = torch.topk(perplexity, min(top_n, n_test)).indices.tolist()
    uniq_top_primary = torch.topk(
        v_uniq_by_k[PRIMARY_K], min(top_n, n_test)
    ).indices.tolist()

    print()
    print(f"  [{label}] Top {min(top_n, n_test)} side-by-side")
    print(f"  {'Rank':>4} | {'By Perplexity':<48} | {'By V-Uniqueness (k=' + str(PRIMARY_K) + ')':<48}")
    print("  " + "-" * 110)
    for r in range(min(top_n, n_test)):
        p_pos = ppl_top[r]
        u_pos = uniq_top_primary[r]
        p_score = perplexity[p_pos].item()
        u_score = v_uniq_by_k[PRIMARY_K][u_pos].item()
        p_str = f"pos {p_pos:>4} {tokens[p_pos]!r:<22} (ppl={p_score:.3f})"
        u_str = f"pos {u_pos:>4} {tokens[u_pos]!r:<22} (uniq={u_score:.3f})"
        print(f"  {r + 1:>4} | {p_str:<48} | {u_str:<48}")

    top20_ppl_json = [
        {
            "pos": int(p),
            "token": tokens[p],
            "score": float(perplexity[p].item()),
        }
        for p in ppl_top
    ]
    top20_uniq_json = [
        {
            "pos": int(p),
            "token": tokens[p],
            "score": float(v_uniq_by_k[PRIMARY_K][p].item()),
        }
        for p in uniq_top_primary
    ]

    # ---- 6. Failure-token rank table ----
    failure_positions = find_token_positions(tokens, failure_targets)
    print()
    print(f"  [{label}] Failure-token ranks (N = {n_test} positions)")
    header = (
        f"  {'Token':<14} | {'Pos':>4} | "
        f"{'ppl rank':>9} | {'ppl score':>9} | "
        f"{'uniq k=' + str(PRIMARY_K) + ' rank':>13} | {'uniq score':>10} | "
        f"{'k=3 rank':>9} | {'k=10 rank':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    failure_token_json: dict[str, dict] = {}
    for target in failure_targets:
        positions = failure_positions[target]
        per_position_json: list[dict] = []
        if not positions:
            print(f"  {target!r:<14} | {'—':>4} | (no matches)")
        for p in positions:
            ppl_score = float(perplexity[p].item())
            ppl_r = ppl_rank[p]
            uniq_p_score = float(v_uniq_by_k[PRIMARY_K][p].item())
            uniq_p_rank = uniq_rank_by_k[PRIMARY_K][p]
            uniq_3_rank = uniq_rank_by_k[3][p]
            uniq_10_rank = uniq_rank_by_k[10][p]
            line = (
                f"  {target!r:<14} | {p:>4} | "
                f"{ppl_r:>4}/{n_test:<4} | {ppl_score:>9.3f} | "
                f"{uniq_p_rank:>4}/{n_test:<8} | {uniq_p_score:>10.3f} | "
                f"{uniq_3_rank:>4}/{n_test:<4} | {uniq_10_rank:>4}/{n_test:<5}"
            )
            print(line)
            per_position_json.append({
                "pos": int(p),
                "token": tokens[p],
                "ppl_rank": int(ppl_r),
                "ppl_score": ppl_score,
                f"v_uniq_rank_k{PRIMARY_K}": int(uniq_p_rank),
                f"v_uniq_score_k{PRIMARY_K}": uniq_p_score,
                "v_uniq_rank_k3": int(uniq_3_rank),
                "v_uniq_score_k3": float(v_uniq_by_k[3][p].item()),
                "v_uniq_rank_k10": int(uniq_10_rank),
                "v_uniq_score_k10": float(v_uniq_by_k[10][p].item()),
            })
        failure_token_json[target] = {
            "positions": [int(p) for p in positions],
            "per_position": per_position_json,
        }

    # ---- 7. Overlap analysis (top 10%) ----
    top_n_cut = max(1, n_test // 10)
    top_ppl_set = set(torch.topk(perplexity, top_n_cut).indices.tolist())
    top_uniq_set = set(
        torch.topk(v_uniq_by_k[PRIMARY_K], top_n_cut).indices.tolist()
    )
    both_set = top_ppl_set & top_uniq_set
    ppl_only_set = top_ppl_set - top_uniq_set
    uniq_only_set = top_uniq_set - top_ppl_set

    def positions_to_records(positions_iter, max_n: int = 10) -> list[dict]:
        # Sort by ppl + uniq combined to surface the "most interesting" first;
        # use uniq score as primary so v_uniq_only / both buckets sort sensibly.
        sorted_pos = sorted(
            positions_iter,
            key=lambda p: -float(v_uniq_by_k[PRIMARY_K][p].item()),
        )
        return [
            {
                "pos": int(p),
                "token": tokens[p],
                "ppl_score": float(perplexity[p].item()),
                "ppl_rank": int(ppl_rank[p]),
                f"v_uniq_score_k{PRIMARY_K}": float(v_uniq_by_k[PRIMARY_K][p].item()),
                f"v_uniq_rank_k{PRIMARY_K}": int(uniq_rank_by_k[PRIMARY_K][p]),
            }
            for p in sorted_pos[:max_n]
        ]

    overlap_json = {
        "top_n_threshold": int(top_n_cut),
        "n_in_both": len(both_set),
        "n_ppl_only": len(ppl_only_set),
        "n_v_uniq_only": len(uniq_only_set),
        "both_top10pct": positions_to_records(both_set),
        "ppl_only_top10pct": positions_to_records(ppl_only_set),
        "v_uniq_only_top10pct": positions_to_records(uniq_only_set),
    }

    print()
    print(f"  [{label}] Overlap analysis (top {top_n_cut} of {n_test})")
    print(f"    in both:                {len(both_set):>3}  "
          f"e.g. {[tokens[p] for p in list(both_set)[:6]]!r}")
    print(f"    perplexity-only:        {len(ppl_only_set):>3}  "
          f"e.g. {[tokens[p] for p in list(ppl_only_set)[:6]]!r}")
    print(f"    v-uniq-only (k={PRIMARY_K}):    {len(uniq_only_set):>3}  "
          f"e.g. {[tokens[p] for p in list(uniq_only_set)[:6]]!r}")

    # ---- 8. Build per-turn JSON dict ----
    return {
        "turn_index": turn_record["turn_index_1based"],
        "role": turn_record["role"],
        "token_count": n_test,
        "content_type": turn_record.get("content_type", ""),
        "tokens": tokens,
        "perplexity": [float(x.item()) for x in perplexity],
        f"v_uniqueness_k{PRIMARY_K}": [float(x.item()) for x in v_uniq_by_k[PRIMARY_K]],
        "v_uniqueness_k3": [float(x.item()) for x in v_uniq_by_k[3]],
        "v_uniqueness_k10": [float(x.item()) for x in v_uniq_by_k[10]],
        "failure_tokens": failure_token_json,
        "top20_ppl": top20_ppl_json,
        f"top20_v_uniq_k{PRIMARY_K}": top20_uniq_json,
        "overlap_analysis": overlap_json,
    }


# -------------------------------- main -------------------------------

def main():
    print("=" * 78)
    print("  Token importance diagnostic: perplexity vs V-uniqueness")
    print("=" * 78)

    # ---- Load model ----
    print(f"\nLoading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    device = model.device
    print(f"Model loaded. {vram_str()}")

    # ---- Load transcript ----
    raw_lines = TRANSCRIPT_PATH.read_text(encoding="utf-8").splitlines()
    turns = [json.loads(line) for line in raw_lines if line.strip()]
    print(f"Loaded {len(turns)} turns from {TRANSCRIPT_PATH.name}")

    # ---- Pre-tokenize selected turns ----
    selected: list[dict] = []
    for spec in TURNS_TO_ANALYZE:
        ti = spec["turn_index_1based"]
        turn = turns[ti - 1]
        role = turn.get("role", "?")
        content = _stringify_content(turn.get("content", ""))
        text = turn_marker(ti, role, content)
        ids_list = tokenizer(text, add_special_tokens=False).input_ids
        record = {
            "turn_index_1based": ti,
            "role": role,
            "n_tokens": len(ids_list),
            "ids": torch.tensor([ids_list], dtype=torch.long),
            "content_type": spec["content_type"],
        }
        selected.append({"spec": spec, "record": record})
        print(f"  T{ti:>2} ({role}): {len(ids_list)} tokens  "
              f"[failure_tokens={spec['failure_tokens']}]")

    # ---- Build prefix_kv ----
    prefix_turn = turns[PREFIX_TURN_INDEX]
    prefix_role = prefix_turn.get("role", "?")
    prefix_content = _stringify_content(prefix_turn.get("content", ""))
    prefix_text = turn_marker(PREFIX_TURN_INDEX + 1, prefix_role, prefix_content)
    prefix_ids = tokenizer(
        prefix_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    n_prefix = int(prefix_ids.shape[1])
    print(f"\nBuilding prefix KV (turn {PREFIX_TURN_INDEX + 1}, {n_prefix} tokens) ...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        prefix_out = model(input_ids=prefix_ids, use_cache=True)
    prefix_kv = detach_kv(to_legacy_kv(prefix_out.past_key_values))
    del prefix_out
    torch.cuda.empty_cache()
    prefix_len = kv_seq_len(prefix_kv)
    print(f"  prefix_kv kv_seq_len={prefix_len}. {vram_str()}")

    # ---- RoPE source check (warn only — irrelevant to this diagnostic) ----
    rope_verified, rope_finding = verify_rope_for_qwen()
    if rope_verified:
        print(f"[rope_check] CONFIRMED. line: {rope_finding}")
    else:
        print(f"[rope_check] WARNING: {rope_finding}")

    # ---- Analyze each turn ----
    turns_results: list[dict] = []
    for entry in selected:
        spec = entry["spec"]
        record = entry["record"]
        label = f"T{record['turn_index_1based']}"
        try:
            res = analyze_turn(
                model, tokenizer, prefix_kv, prefix_len,
                record, spec["failure_tokens"], label,
            )
            turns_results.append(res)
        except Exception as e:
            print(f"[ERROR] {label} analysis crashed: {e}")
            traceback.print_exc()
            turns_results.append({
                "turn_index": record["turn_index_1based"],
                "error": f"{type(e).__name__}: {e}",
            })
        torch.cuda.empty_cache()
        print(f"  [{label}] done. {vram_str()}")

    # ---- Save JSON ----
    results = {
        "config": {
            "model_id": MODEL_ID,
            "neighborhood_sizes": NEIGHBORHOOD_KS,
            "primary_k": PRIMARY_K,
            "turns_analyzed": [t["turn_index_1based"] for t in TURNS_TO_ANALYZE],
            "prefix_turn_index": PREFIX_TURN_INDEX + 1,
            "prefix_tokens": n_prefix,
            "rope_verified": rope_verified,
            "rope_finding": rope_finding,
        },
        "turns": turns_results,
    }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote {RESULTS_PATH}")

    # ---- Cross-turn overlap summary ----
    print()
    print("=" * 78)
    print("  CROSS-TURN OVERLAP SUMMARY (top 10%)")
    print("=" * 78)
    print(f"  {'Turn':>4} | {'Tokens':>6} | {'Top-N':>5} | "
          f"{'in both':>7} | {'ppl-only':>8} | {'uniq-only':>9}")
    print("  " + "-" * 60)
    for t in turns_results:
        if "error" in t:
            print(f"  {t['turn_index']:>4} | {'—':>6} | {'—':>5} | "
                  f"{'—':>7} | {'—':>8} | {'—':>9}  (ERROR)")
            continue
        oa = t.get("overlap_analysis", {})
        n = t.get("token_count", 0)
        print(f"  {t['turn_index']:>4} | {n:>6} | "
              f"{oa.get('top_n_threshold', '—'):>5} | "
              f"{oa.get('n_in_both', '—'):>7} | "
              f"{oa.get('n_ppl_only', '—'):>8} | "
              f"{oa.get('n_v_uniq_only', '—'):>9}")


if __name__ == "__main__":
    main()
