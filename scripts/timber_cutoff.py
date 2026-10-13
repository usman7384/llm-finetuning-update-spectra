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

--task gpqa runs GPQA-Diamond instead, the benchmark where Timber reports its
largest gain for Qwen3-0.6B (non-thinking mode), into
results/timber_cutoff_gpqa/ and results/timber_cutoff_gpqa.csv. The dataset is
gated on the Hub: accept its terms and log in first.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from align_geometry.analysis.svd_delta import (  # noqa: E402
    entropy_rank,
    is_analyzable_matrix,
    load_state_dict,
    stable_rank,
)
from align_geometry.eval.gsm8k import extract, gold_answer, is_correct, pass_at_k  # noqa: E402
from align_geometry.hub import weights_dir  # noqa: E402

BASE = "Qwen/Qwen3-0.6B-Base"
INSTRUCT = "Qwen/Qwen3-0.6B"
DEFAULT_VARIANTS = ["instruct", "erank_l0.0", "erank_l0.5", "stable_l0.0", "stable_l0.5"]
PROMPT = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
OUT_DIR = Path("results/timber_cutoff")
GPQA_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER "
    "is one of ABCD. Think step by step before answering.\n\n{q}\n\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def load_task(task: str, limit: int) -> list[dict]:
    """Problems as {"prompt", "gold"}, the same for every variant."""
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.select(range(min(limit, len(ds))))
        return [{"prompt": PROMPT.format(q=r["question"]), "gold": gold_answer(r["answer"])}
                for r in ds]  # fmt: skip
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    problems = []
    for i, r in enumerate(ds.select(range(min(limit, len(ds))))):
        options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"],
                   r["Incorrect Answer 3"]]  # fmt: skip
        order = list(range(4))
        random.Random(i).shuffle(order)  # fixed shuffle per question
        opts = {L: options[j].strip() for L, j in zip("ABCD", order, strict=True)}
        gold = "ABCD"[order.index(0)]
        problems.append({"prompt": GPQA_PROMPT.format(q=r["Question"].strip(), **opts),
                         "gold": gold})  # fmt: skip
    return problems


def extract_letter(text: str) -> str | None:
    """The chosen option: the last "Answer: X", else the last \\boxed{X}."""
    m = re.findall(r"(?i)answer\s*[:\uff1a]\s*\(?\**\s*([ABCD])\b", text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\\boxed\{\s*\(?([ABCD])\)?\s*\}", text)
    return m[-1] if m else None


def parse_variant(name: str) -> tuple[str | None, float, bool]:
    """-> (cutoff, lambda, uniform). "uniform_" variants scale all of dW by one
    factor per matrix, chosen so the edited update is as large as the matching
    tail edit: same amount removed, but spread evenly instead of from the tail."""
    if name == "instruct":
        return None, 1.0, False
    m = re.fullmatch(r"(uniform_)?(erank|stable)_l([0-9.]+)", name)
    if not m:
        raise SystemExit(
            f"bad variant {name!r}, want instruct, erank_l0.5, stable_l0.0, uniform_stable_l0.5 etc"
        )
    return m.group(2), float(m.group(3)), m.group(1) is not None


@torch.no_grad()
def apply_edit(model, cutoff: str, lam: float, device: str, uniform: bool = False) -> dict:
    """Timber's edit in place, or its norm-matched uniform version.

    Returns how much of dW the cutoff kept, and the mean uniform factor."""
    base = load_state_dict(weights_dir(BASE))
    params = dict(model.named_parameters())
    kept_dirs, kept_energy, alphas, survived = [], [], [], []
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
        if uniform:
            # one factor for the whole update, matching the tail edit's Frobenius norm
            alpha = float(((s * scale) ** 2).sum().sqrt() / (s**2).sum().sqrt())
            edited = w_b + alpha * delta
            alphas.append(alpha)
        else:
            edited = w_b + (u * (s * scale)) @ vh
        # the model runs in bf16, so check how much of the intended change survives rounding
        original = p.data.to(torch.float32)
        p.data.copy_(edited.to(p.dtype))
        intended = float(torch.linalg.norm(edited - original))
        if intended > 0:
            survived.append(float(torch.linalg.norm(p.data.to(torch.float32) - edited)) / intended)
        kept_dirs.append(k / len(s))
        kept_energy.append(float((sv[:k] ** 2).sum() / (sv**2).sum()))
    del base
    return {
        "matrices": len(kept_dirs),
        "kept_dirs_frac": sum(kept_dirs) / len(kept_dirs),
        "kept_energy_frac": sum(kept_energy) / len(kept_energy),
        "uniform_alpha_mean": sum(alphas) / len(alphas) if alphas else None,
        # rounding error / intended change, per matrix; near 0 is good, near 1
        # means bf16 storage erased the edit
        "rounding_error_vs_edit": sum(survived) / len(survived) if survived else None,
    }


def run_variant(name: str, problems, args, device: str) -> None:
    out = args.out_dir / f"{name}.jsonl"
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
    cutoff, lam, uniform = parse_variant(name)
    if cutoff:
        stats = apply_edit(model, cutoff, lam, device, uniform)
        stats.update(variant=name, cutoff=cutoff, lam=lam, uniform=uniform)
        (args.out_dir / f"{name}.edit.json").write_text(json.dumps(stats, indent=2))
        print(f"{name}: kept {stats['kept_dirs_frac']:.1%} of directions, "
              f"{stats['kept_energy_frac']:.1%} of dW energy")  # fmt: skip
        # the edit's SVDs leave memory reserved on the GPU; on a 6GB card under
        # Windows that pushes generation into slow shared memory, then OOM
        gc.collect()
        torch.cuda.empty_cache()

    per_batch = max(1, args.batch_seqs // args.n)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), per_batch):
            idxs = todo[start : start + per_batch]
            texts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": problems[i]["prompt"]}],
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
                gold = problems[i]["gold"]
                samples = outs[j * args.n : (j + 1) * args.n]
                if args.task == "gpqa":
                    preds = [extract_letter(s) for s in samples]
                    correct = [p == gold for p in preds]
                    answered = [p is not None for p in preds]
                else:
                    preds = [extract(s) for s in samples]
                    correct = [is_correct(p, gold) for p in preds]
                    answered = ["\\boxed{" in s for s in samples]
                rec = {
                    "idx": i,
                    "gold": gold,
                    "preds": preds,
                    "correct": correct,
                    # no final answer usually means it ran out of tokens
                    "boxed": answered,
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


def summarize(n_problems: int, out_dir: Path = OUT_DIR) -> None:
    rows = []
    per_problem = {}  # variant -> {k: per-problem pass@k}, for the paired comparison
    for path in sorted(out_dir.glob("*.jsonl")):
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
    for row in rows:
        if ref is None or row["variant"] == "instruct" or row["problems"] != len(ref[1]):
            continue
        # one generator per variant, so adding a variant leaves the others' intervals alone
        rng = np.random.default_rng(0)
        idx = rng.integers(0, row["problems"], size=(5000, row["problems"]))
        for k in (1, 4, 8):  # only the k both variants sampled enough for
            if k in ref and k in per_problem[row["variant"]]:
                d = per_problem[row["variant"]][k] - ref[k]
                boot = d[idx].mean(axis=1)
                row[f"d_pass@{k}"] = d.mean()
                row[f"d_pass@{k}_lo"] = np.percentile(boot, 2.5)
                row[f"d_pass@{k}_hi"] = np.percentile(boot, 97.5)
    if rows:
        s = pd.DataFrame(rows)
        s.to_csv(out_dir.with_suffix(".csv"), index=False)
        print()
        print(s.round(4).to_string(index=False))


def edit_stats_only(name: str) -> None:
    """Rewrite <name>.edit.json without generating, on the CPU (the GPU may be busy)."""
    cutoff, lam, uniform = parse_variant(name)
    if not cutoff:
        return
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT, dtype=torch.bfloat16)
    stats = apply_edit(model, cutoff, lam, "cpu", uniform)
    stats.update(variant=name, cutoff=cutoff, lam=lam, uniform=uniform)
    (OUT_DIR / f"{name}.edit.json").write_text(json.dumps(stats, indent=2))
    print(name, stats)


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
    parser.add_argument("--edit-only", action="store_true",
                        help="only recompute the .edit.json statistics, on the CPU")  # fmt: skip
    parser.add_argument("--task", choices=["gsm8k", "gpqa"], default="gsm8k")
    parser.add_argument("--tag", default="",
                        help="suffix for the output directory, e.g. 'long' for a rerun with a longer token limit")  # fmt: skip
    args = parser.parse_args()
    args.out_dir = OUT_DIR if args.task == "gsm8k" else Path(f"results/timber_cutoff_{args.task}")
    if args.tag:
        args.out_dir = Path(f"{args.out_dir}_{args.tag}")
    for v in args.variants:
        parse_variant(v)
    if args.edit_only:
        for name in args.variants:
            edit_stats_only(name)
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    problems = load_task(args.task, args.limit)
    for name in args.variants:
        run_variant(name, problems, args, device)
    summarize(len(problems), args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
