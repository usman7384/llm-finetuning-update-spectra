#!/usr/bin/env python
"""Same-size random updates, scored on a task instead of perplexity.

random_update_check.py showed that a random update the size of Qwen3-0.6B's
post-training update moves eRank(W) as little as the real one, while wikitext
perplexity rises. But the real update raises that perplexity even more, so
perplexity cannot tell damage from useful change. Here every model is scored
on GSM8K with the same prompt, so only the weights differ:

  base mode (8-shot plain-text prompt, greedy):
    base                 Qwen3-0.6B-Base
    base+real            base + its real post-training update, same matrices
    base+gaussian_s{0,1} base + iid Gaussian of the same norm, matrix by matrix
    base+spectrum_s0     base + the real singular values on random directions

  chat mode (Qwen3 chat template, thinking off, greedy):
    instruct             Qwen3-0.6B
    instruct+gaussian_s0 instruct + iid Gaussian of the size of the real update

Only the attention/MLP matrices are touched, in fp32, and the model then runs
in bf16. Resumable per problem.

    uv run python scripts/random_update_tasks.py
    uv run python scripts/random_update_tasks.py --variants base base+real --limit 100

Writes results/random_update_tasks/<variant>.jsonl and
results/v2/random_update_tasks.csv (accuracy, and the difference from the
unedited model of the same mode with a 95% paired bootstrap interval).
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from random_update_check import BASE, INSTRUCT, make_update  # noqa: E402

from align_geometry.analysis.svd_delta import LazyWeights, is_analyzable_matrix  # noqa: E402
from align_geometry.eval.gsm8k import extract, gold_answer, is_correct  # noqa: E402
from align_geometry.hub import weights_dir  # noqa: E402

OUT_DIR = Path("results/random_update_tasks")
VARIANTS = ["base", "base+real", "base+gaussian_s0", "base+gaussian_s1", "base+spectrum_s0",
            "instruct", "instruct+gaussian_s0"]  # fmt: skip
N_SHOT = 8


def parse(variant: str) -> tuple[str, str | None, int]:
    """'base+gaussian_s1' -> ('base', 'gaussian', 1)."""
    start, _, edit = variant.partition("+")
    if not edit:
        return start, None, 0
    kind, _, seed = edit.partition("_s")
    return start, kind, int(seed or 0)


def few_shot_prefix() -> str:
    train = load_dataset("openai/gsm8k", "main", split="train").select(range(N_SHOT))
    shots = []
    for ex in train:
        reasoning, answer = ex["answer"].split("####")
        # drop the <<a*b=c>> calculator annotations
        reasoning = "".join(part.split(">>")[-1] for part in reasoning.split("<<"))
        shots.append(f"Question: {ex['question']}\nAnswer: {reasoning.strip()} "
                     f"The answer is {answer.strip()}.")  # fmt: skip
    return "\n\n".join(shots) + "\n\n"


@torch.no_grad()
def apply_update(model, kind: str, seed: int, device: str) -> None:
    """Add a same-size update to every attention/MLP matrix of the loaded model."""
    base, inst = LazyWeights(weights_dir(BASE)), LazyWeights(weights_dir(INSTRUCT))
    params = dict(model.named_parameters())
    for key in base.keys():
        if key not in params or key not in inst or not is_analyzable_matrix(key):
            continue
        w_b = base[key].to(device, torch.float32)
        delta = inst[key].to(device, torch.float32) - w_b
        g = torch.Generator().manual_seed(zlib.crc32(key.encode()) + 1000 * seed)
        upd = make_update(kind, delta.cpu(), g).to(device)
        p = params[key]
        p.data.copy_((p.data.float() + upd).to(p.dtype))


def run_variant(variant: str, problems, args, device: str) -> None:
    out = OUT_DIR / f"{variant}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["idx"] for line in out.read_text(encoding="utf-8").splitlines()}
    todo = [i for i in range(len(problems)) if i not in done]
    if not todo:
        print(f"{variant}: all {len(problems)} problems done")
        return
    start, kind, seed = parse(variant)
    model_id = BASE if start == "base" else INSTRUCT
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    if kind:
        apply_update(model, kind, seed, device)
    prefix = few_shot_prefix() if start == "base" else ""

    def prompt(q: str) -> str:
        if start == "base":
            return f"{prefix}Question: {q}\nAnswer:"
        msg = [{"role": "user", "content": q + "\nPlease reason step by step, and put "
                "your final answer within \\boxed{}."}]  # fmt: skip
        return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)  # fmt: skip

    with out.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch):
            idxs = todo[b : b + args.batch]
            enc = tok([prompt(problems[i]["question"]) for i in idxs], return_tensors="pt",
                      padding=True).to(device)  # fmt: skip
            # a base model would go on to write the next few-shot question itself
            stop = {"stop_strings": ["\n\nQuestion:"], "tokenizer": tok} if start == "base" else {}
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id, **stop)  # fmt: skip
            for j, i in enumerate(idxs):
                text = tok.decode(gen[j, enc.input_ids.shape[1] :], skip_special_tokens=True)
                if start == "base":
                    text = text.split("\n\nQuestion:")[0]
                    m = text.rsplit("The answer is", 1)
                    pred = extract(m[1]) if len(m) == 2 else extract(text)
                else:
                    pred = extract(text)
                gold = gold_answer(problems[i]["answer"])
                rec = {"idx": i, "pred": pred, "gold": gold, "correct": is_correct(pred, gold),
                       "tokens": int((gen[j, enc.input_ids.shape[1]:] != tok.pad_token_id).sum())}  # fmt: skip
                f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{variant}: {min(b + args.batch, len(todo))}/{len(todo)}", flush=True)
    del model
    torch.cuda.empty_cache()


def summarize(limit: int) -> None:
    rng = np.random.default_rng(0)
    acc = {}
    for path in sorted(OUT_DIR.glob("*.jsonl")):
        recs = {r["idx"]: r["correct"] for r in map(json.loads, path.read_text().splitlines())}
        if all(i in recs for i in range(limit)):
            acc[path.stem] = np.array([recs[i] for i in range(limit)], dtype=float)
    rows = []
    for v, a in acc.items():
        ref = "base" if v.startswith("base") else "instruct"
        row = {"variant": v, "problems": limit, "accuracy": a.mean()}
        if v != ref and ref in acc:
            d = a - acc[ref]
            boot = d[rng.integers(0, limit, (5000, limit))].mean(1)
            row.update(diff=d.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/v2/random_update_tasks.csv", index=False)
    print(df.round(4).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--limit", type=int, default=1319)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    problems = load_dataset("openai/gsm8k", "main", split="test").select(range(args.limit))
    for v in args.variants:
        run_variant(v, problems, args, device)
    summarize(args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
