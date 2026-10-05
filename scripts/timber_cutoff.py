#!/usr/bin/env python
"""Does the cutoff in Timber's edit matter? eRank vs stable rank.

Timber (arXiv:2509.23595) edits an instruct model by keeping the top K singular
directions of dW = W_instruct - W_base and scaling the rest by lambda:

    W = W_base + U diag(s_1..s_K, lambda*s_K+1, ..., lambda*s_r) V^T

with K = ceil(eRank(dW)), which keeps ~85% of the directions. Our measurements
say eRank sits close to the noise value while stable rank tracks where the
energy actually is, and ceil(stable rank) keeps far fewer directions. So:
same edit, two cutoffs, scored by pass@k on GSM8K.

Pair is Qwen3-0.6B-Base -> Qwen3-0.6B (in Timber's table 1), non-thinking mode,
with Qwen's recommended sampling for that mode (T=0.7, top_p=0.8, top_k=20).

    uv run python scripts/timber_cutoff.py                      # full default grid
    uv run python scripts/timber_cutoff.py --variants instruct erank_l0.5
    uv run python scripts/timber_cutoff.py --limit 4 --n 2      # smoke test

Variant names: "instruct" (unedited), or "<cutoff>_l<lambda>" with cutoff in
{erank, stable}. Resumable: each variant appends one line per problem to
results/timber_cutoff/<variant>.jsonl and skips problems already there.
Summary goes to results/timber_cutoff.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    entropy_rank,
    is_analyzable_matrix,
    load_state_dict,
    stable_rank,
)

BASE = "Qwen/Qwen3-0.6B-Base"
INSTRUCT = "Qwen/Qwen3-0.6B"
DEFAULT_VARIANTS = ["instruct", "erank_l0.0", "erank_l0.5", "stable_l0.0", "stable_l0.5"]
PROMPT = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
OUT_DIR = Path("results/timber_cutoff")


def parse_variant(name: str) -> tuple[str | None, float]:
    if name == "instruct":
        return None, 1.0
    m = re.fullmatch(r"(erank|stable)_l([0-9.]+)", name)
    if not m:
        raise SystemExit(f"bad variant {name!r}, want instruct or erank_l0.5 / stable_l0.0 etc")
    return m.group(1), float(m.group(2))


@torch.no_grad()
def apply_edit(model, cutoff: str, lam: float, device: str) -> dict:
    """Timber's edit in place. Returns how much of dW the cutoff kept."""
    base = load_state_dict(
        Path(snapshot_download(BASE, allow_patterns=["*.safetensors", "*.json"]))
    )
    params = dict(model.named_parameters())
    kept_dirs, kept_energy = [], []
    for key, w_b in base.items():
        if key not in params or not is_analyzable_matrix(key):
            continue
        p = params[key]
        w_b = w_b.to(device, torch.float32)
        delta = p.data.to(torch.float32) - w_b
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        sv = s.cpu()
        size = entropy_rank(sv) if cutoff == "erank" else stable_rank(sv)
        k = min(math.ceil(size), len(s))
        scale = torch.full_like(s, lam)
        scale[:k] = 1.0
        p.data.copy_((w_b + (u * (s * scale)) @ vh).to(p.dtype))
        kept_dirs.append(k / len(s))
        kept_energy.append(float((sv[:k] ** 2).sum() / (sv**2).sum()))
    del base
    return {
        "matrices": len(kept_dirs),
        "kept_dirs_frac": sum(kept_dirs) / len(kept_dirs),
        "kept_energy_frac": sum(kept_energy) / len(kept_energy),
    }


def gold_answer(text: str) -> str:
    return text.split("####")[-1].strip().replace(",", "")


def extract(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = boxed[-1] if boxed else None
    if cand is None:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        cand = nums[-1] if nums else None
    if cand is None:
        return None
    cand = cand.replace(",", "").replace("$", "").strip().rstrip(".")
    m = re.search(r"-?\d+\.?\d*", cand)
    return m.group(0) if m else cand


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from Chen et al. 2021."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def run_variant(name: str, problems, args, device: str) -> None:
    out = OUT_DIR / f"{name}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["idx"] for line in out.read_text(encoding="utf-8").splitlines()}
    todo = [i for i in range(len(problems)) if i not in done]
    if not todo:
        print(f"{name}: all {len(problems)} problems done")
        return

    tok = AutoTokenizer.from_pretrained(INSTRUCT, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16).to(device)
    model.eval()
    cutoff, lam = parse_variant(name)
    if cutoff:
        stats = apply_edit(model, cutoff, lam, device)
        stats.update(variant=name, cutoff=cutoff, lam=lam)
        (OUT_DIR / f"{name}.edit.json").write_text(json.dumps(stats, indent=2))
        print(f"{name}: kept {stats['kept_dirs_frac']:.1%} of directions, "
              f"{stats['kept_energy_frac']:.1%} of dW energy")  # fmt: skip

    per_batch = max(1, args.batch_seqs // args.n)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), per_batch):
            idxs = todo[start : start + per_batch]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": PROMPT.format(q=problems[i]["question"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for i in idxs
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(device)
            # same seed per problem for every variant, so differences come from the weights
            torch.manual_seed(args.seed + idxs[0])
            gen = model.generate(
                **enc,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.n,
                pad_token_id=tok.pad_token_id,
            )
            outs = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            for j, i in enumerate(idxs):
                gold = gold_answer(problems[i]["answer"])
                samples = outs[j * args.n : (j + 1) * args.n]
                preds = [extract(s) for s in samples]
                rec = {
                    "idx": i,
                    "gold": gold,
                    "preds": preds,
                    "correct": [is_correct(p, gold) for p in preds],
                    # no boxed answer usually means it ran out of tokens
                    "boxed": ["\\boxed{" in s for s in samples],
                    "first_sample": samples[0],
                }
                f.write(json.dumps(rec) + "\n")
            f.flush()
            n_done = start + len(idxs)
            rate = (time.time() - t0) / n_done
            print(f"{name}: {n_done}/{len(todo)} problems, "
                  f"~{rate * (len(todo) - n_done) / 60:.0f} min left", flush=True)  # fmt: skip
    del model
    torch.cuda.empty_cache()


def summarize(n_problems: int) -> None:
    rows = []
    per_problem = {}  # variant -> {k: per-problem pass@k}, for the paired comparison
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        recs = sorted((r for r in recs if r["idx"] < n_problems), key=lambda r: r["idx"])
        if not recs:
            continue
        n = len(recs[0]["correct"])
        row = {"variant": path.stem, "problems": len(recs), "samples": n}
        per_problem[path.stem] = {}
        for k in [1, 2, 4, 8, 16]:
            if k <= n:
                vals = np.array([pass_at_k(n, sum(r["correct"]), k) for r in recs])
                per_problem[path.stem][k] = vals
                row[f"pass@{k}"] = vals.mean()
        row["boxed"] = sum(sum(r["boxed"]) for r in recs) / (n * len(recs))
        edit = path.with_suffix(".edit.json")
        if edit.exists():
            e = json.loads(edit.read_text())
            row.update(kept_dirs_frac=e["kept_dirs_frac"], kept_energy_frac=e["kept_energy_frac"])
        rows.append(row)

    # every variant saw the same problems with the same seeds, so compare per problem
    ref = per_problem.get("instruct")
    rng = np.random.default_rng(0)
    for row in rows:
        if ref is None or row["variant"] == "instruct" or row["problems"] != len(ref[1]):
            continue
        idx = rng.integers(0, row["problems"], size=(5000, row["problems"]))
        for k in (1, 8):
            if k in ref and k in per_problem[row["variant"]]:
                d = per_problem[row["variant"]][k] - ref[k]
                boot = d[idx].mean(axis=1)
                row[f"d_pass@{k}"] = d.mean()
                row[f"d_pass@{k}_lo"] = np.percentile(boot, 2.5)
                row[f"d_pass@{k}_hi"] = np.percentile(boot, 97.5)
    if rows:
        s = pd.DataFrame(rows)
        s.to_csv("results/timber_cutoff.csv", index=False)
        print()
        print(s.round(4).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--limit", type=int, default=500, help="first N GSM8K test problems")
    parser.add_argument("--n", type=int, default=8, help="samples per problem")
    parser.add_argument("--batch-seqs", type=int, default=32, help="sequences per generate call")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    for v in args.variants:
        parse_variant(v)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems = load_dataset("openai/gsm8k", "main", split="test").select(range(args.limit))
    for name in args.variants:
        run_variant(name, problems, args, device)
    summarize(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
