# Follow-up runs for the paper

Three additions that close the main gaps a reviewer would point at. All the
code and configs are in place; these are just the commands.

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
