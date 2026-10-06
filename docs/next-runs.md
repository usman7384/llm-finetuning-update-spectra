# Follow-up runs for the paper

The runs added after the first write-up, to close the gaps a reviewer would
point at. All of them have been run and the results are in
[report.md](report.md); this page is kept as the exact commands used, with
the practical notes that came up along the way.

Before starting anything long, stop Windows from sleeping:

```
powercfg /change standby-timeout-ac 0
```

Every training command can be stopped and rerun; it resumes from the last
checkpoint. Run one GPU job at a time.

## 1. Released base/instruct pairs (no GPU, ~1 hour)

Measures public base -> instruct deltas with both the thresholded rank and
stable rank, next to a random-matrix baseline. This is the direct check
against Timber's setting.

```
uv run python scripts/released_pairs.py
```

Downloads about 17GB in total. The two biggest pairs (qwen2.5-1.5b and
smollm2-1.7b) are most of that, so if disk is tight start with:

```
uv run python scripts/released_pairs.py --pairs qwen2.5-0.5b smollm2-135m smollm2-360m
```

It's CPU and RAM heavy rather than GPU heavy (the 1.7B pair holds ~7GB of
weights in memory), so don't run the big pairs alongside a training job.

Output: `results/released_pairs.csv`, `results/released_pairs.png`, and a
per-matrix CSV per pair in `results/released_pairs/`.

## 2. Second model family: SmolLM2-360M (~3 to 5 hours of GPU)

Same recipe as the Qwen runs: SFT plus the two controls, then SVD. Order
matters. The controls read their token budget from the SFT run's manifest, so
SFT has to finish first.

```
uv run python scripts/save_reference_models.py --model HuggingFaceTB/SmolLM2-360M --prefix smollm2_ --random
uv run python scripts/train_sft.py --config configs/smollm2_sft.yaml
uv run python scripts/train_sft.py --config configs/smollm2_control_lm.yaml
uv run python scripts/train_sft.py --config configs/smollm2_control_code.yaml
```

Then the SVDs and the comparison (each takes seconds to a few minutes):

```
uv run python scripts/svd_deltas.py --before checkpoints/smollm2_base --after checkpoints/smollm2_sft --tag smollm2_sft --out results/svd_smollm2_sft
uv run python scripts/svd_deltas.py --before checkpoints/smollm2_base --after checkpoints/smollm2_control_lm --tag smollm2_control_lm --out results/svd_smollm2_control_lm
uv run python scripts/svd_deltas.py --before checkpoints/smollm2_base --after checkpoints/smollm2_control_code --tag smollm2_control_code --out results/svd_smollm2_control_code
uv run python scripts/svd_deltas.py --before checkpoints/smollm2_random_init_a --after checkpoints/smollm2_random_init_b --tag smollm2_random --out results/svd_smollm2_random
uv run python scripts/stable_rank.py --family smollm2
```

Output: `results/stable_rank_smollm2.csv` and `.png`. Kept separate from the
Qwen plot on purpose, since a different architecture has its own noise floor.

If SmolLM2's base tokenizer has no chat template, the SFT run borrows the one
from `SmolLM2-360M-Instruct` automatically (set in
`configs/model_smollm2_360m.yaml`).

## 3. Second seeds for the Qwen controls (~4 to 5 hours of GPU)

Same as the original controls with both seeds changed, so the controls get
error bars like SFT did.

```
uv run python scripts/train_sft.py --config configs/control_lm_seed43.yaml
uv run python scripts/train_sft.py --config configs/control_code_seed43.yaml
uv run python scripts/svd_deltas.py --before checkpoints/base --after checkpoints/control_lm_seed43 --tag control_lm_seed43 --out results/svd_control_lm_seed43
uv run python scripts/svd_deltas.py --before checkpoints/base --after checkpoints/control_code_seed43 --tag control_code_seed43 --out results/svd_control_code_seed43
uv run python scripts/stable_rank.py
```

The last command redoes `results/stable_rank.csv` and `.png` with the two new
runs included. The numbers for the existing runs won't change.

## Disk

You have roughly 54GB free. Everything above needs about 40GB: ~17GB of
downloads for task 1, ~12GB for task 3's two checkpoints, and ~10GB for task 2.
That's tight. The easy win is deleting the intermediate `checkpoint-*` folders
inside each finished run (only `manifest.json`, `model.safetensors` and the
tokenizer files are needed afterwards). The 24 LoRA runs in
`checkpoints/lora_sweep/` are also mostly reclaimable once you no longer need
them.

## What to look for

- Task 1: do the released deltas also read ~99% on thresholded rank but well
  below the random baseline on stable rank? If yes, the metric point holds
  beyond our own runs, and it bears directly on Timber's result.
- Task 2: are SmolLM2's SFT and control deltas all well below its own random
  baseline, and close to each other? If yes, "concentrated but not
  alignment-specific" isn't a Qwen quirk.
- Task 3: is the gap between the two seeds of each control small next to the
  gap between wikitext and code? If yes, the task differences have proper
  error bars.

## 4. Does Timber's cutoff matter? (~9 to 10 hours of GPU for the first pass)

Timber edits Qwen3-0.6B by keeping the top K singular directions of dW and
scaling the rest by lambda, with K = ceil(eRank(dW)). That cutoff keeps ~83% of
the directions and ~99% of dW's energy, so Timber-L removes only ~1% of the
update. ceil(stable rank) keeps ~12% of the directions and ~41% of the energy.
Same edit, both cutoffs, scored by pass@k on GSM8K (8 samples per problem,
non-thinking mode).

First pass, 300 problems, five variants (instruct, erank_l0.0, erank_l0.5,
stable_l0.0, stable_l0.5):

```
uv run python scripts/timber_cutoff.py --limit 300
```

About 90 seconds per 4 problems, so ~1.9 h per variant. Keep `--batch-seqs` at
32: 64 sequences already need ~5GB of KV cache and spill into shared memory,
which is far slower, not faster.

It's resumable, and the problems are always the first N of the test set, so a
variant worth a closer look can be extended to the full 1319 later without
redoing anything:

```
uv run python scripts/timber_cutoff.py --limit 1319 --variants instruct stable_l0.5
```

Worth knowing before reading the results: Timber's gains were 0.2-1.7 points.
With 300 problems the standard error on pass@1 is ~2.8 points, so a small
difference won't be visible. Every variant uses the same sampling seed per
problem, so comparisons between variants are paired and tighter than that,
but a Timber-sized effect still needs the full set.

## 5. The Qwen3-8B pair (no GPU needed for the download, ~6 hours total)

Same two scripts as the smaller pairs. The weights are read one matrix at a
time, so 16GB of RAM is enough; disk is the limit (~33GB of downloads), so
point the Hugging Face cache at a drive with room:

```
HF_HUB_CACHE=D:/hf-cache uv run python scripts/released_pairs.py --pairs qwen3-8b
HF_HUB_CACHE=D:/hf-cache uv run python scripts/timber_check.py --runs qwen3-8b
```

What it took here: 2-4 hours to download depending on the connection, about
45 minutes for `released_pairs.py` and 3.5 hours for `timber_check.py` (three
SVDs of every full 4096×12288 weight matrix). `timber_check.py` now prints
progress and saves finished matrices as it goes, so it can be stopped and
rerun.

The Hub's Xet downloader does not resume a half-finished file, so switching
networks mid-download costs whatever was in flight. Finished files are kept.
Delete the cache afterwards; nothing needs the weights once the CSVs exist.
