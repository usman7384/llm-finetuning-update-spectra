# The plan

> **Note.** The pre-registration for the first round of runs. Predictions (b)
> and (d), the LoRA sweep and the evaluation suite were dropped from the paper
> after that round turned out to be invalid (bf16 weights, see
> `docs/report.md`); the paper's question became the measurement itself. Kept
> as a record of what was predicted before training.

This is what I wrote down before training anything: the question, what I
expected, and what would count as being wrong. I've cleaned up the wording
since, but the predictions and the falsification criteria are unchanged. What
actually happened is in [report.md](report.md). Several of these predictions
didn't hold, and two experiments got added along the way that aren't listed
here (see the note at the bottom).

## The question

LIMA's "superficial alignment hypothesis" says a model learns basically
everything during pretraining, and alignment (SFT, then DPO or RLHF) just
teaches it which style to answer in. The evidence for that has mostly been
behavioral: people compare outputs. I wanted to look at the weights instead.

If alignment really is mostly style, the change it makes to the weights should
be small and structured, sitting in a few directions rather than spread across
everything.

## Hypothesis

**H1.** Alignment training concentrates its weight change in a small subspace,
low-rank and/or in the later layers, rather than spreading it across the whole
network.

That gives four things to check:

- **(a)** SVD of ΔW_sft = W_sft − W_base, per weight matrix, decays fast (low
  effective rank), and more so in later layers.
- **(b)** LoRA at low rank nearly matches full fine-tuning on the style metrics
  but falls behind on the capability ones, and there's some rank r* where the
  capability gap closes.
- **(c)** ΔW_dpo = W_dpo − W_sft is lower-rank than ΔW_sft, on the idea that a
  preference signal is even more about style than supervised targets are. If
  that held it would also help separate "alignment is unusually low-rank" from
  "all fine-tuning is low-rank".
- **(d)** (bonus) The top singular directions of the full fine-tune's ΔW_sft
  line up with the subspace a LoRA run finds at the same rank.

**What would count as wrong.** The spectrum doesn't decay noticeably faster
than a random control. Effective rank doesn't differ between attention and MLP
or by depth. The LoRA sweep shows no crossover (either everything is already
recovered at rank 4, or nothing is below rank 64). Or DPO's delta isn't
lower-rank than SFT's. Any of those gets reported as-is, without reshaping the
hypothesis afterwards.

## The hardware constraint

One RTX 3060 Laptop GPU with 6GB of VRAM. That decides most things:

| Model | Full SFT/DPO here? | LoRA here? |
|---|---|---|
| Qwen2.5-0.5B base | yes, just. Needs bf16, 8-bit AdamW, gradient checkpointing and gradient accumulation | yes, easily |
| Qwen2.5-1.5B base | no, the AdamW states alone want 20GB+ | yes |

So the whole result is on 0.5B. Repeating it at 1.5B on a rented GPU is a
stretch goal, not something the write-up depends on.

## Data

Small enough to iterate on with 6GB.

| For | Dataset | Size | Why |
|---|---|---|---|
| SFT | `yahma/alpaca-cleaned` | 10k subsample | standard, clean instruction data |
| DPO | `HuggingFaceH4/ultrafeedback_binarized` | 10k pairs | standard preference set, works with TRL as-is |
| style | IFEval | all 541 | instruction-following that a script can check, no judge model |
| style | XSTest | all 450 (250 safe-sounding, 200 genuinely unsafe) | measures over- and under-refusal |
| capability | GSM8K test | all 1319 | reasoning a 0.5B base model is weak at |
| capability | ARC-Challenge | all | a second capability axis, so one benchmark's quirks don't decide it |

None of the eval sets overlap with the training data. XSTest is only used to
count refusals. Nothing harmful gets generated or graded.

## Steps

1. **Baseline and SFT.** Eval the untouched base model. Full-parameter SFT on
   Alpaca. Eval again. Save both as full state dicts, since the SVD needs both
   in full rather than as adapter diffs.
2. **SVD of the SFT delta.** For every linear layer, SVD of ΔW_sft, effective
   rank as the count of σᵢ ≥ 0.01·σ₁ (I thought this was the convention Timber uses; it isn't, they use an
   entropy-based eRank. See report.md section 6. Also check
   0.05 and 0.1, since the threshold is a choice, not a law). Plot decay
   curves and effective rank by depth. As a sanity check on the tooling, run
   the same thing on two random inits, so the pipeline isn't reporting "low
   rank" for noise.
3. **Style vs capability table.** SFT minus base on each metric. This is what
   the LoRA results get compared to.
4. **LoRA sweep.** Ranks 1 to 128, same data and token count as the SFT run,
   all from W_base. Eval each. Plot fraction of the full-FT gain recovered
   against rank and look for r*. Sweep the learning rate per rank (3 values)
   rather than reusing one, because Biderman et al. found these comparisons
   are LR-sensitive.
5. **DPO.** DPO on top of W_sft, eval it, SVD ΔW_dpo the same way, and overlay
   it on ΔW_sft. That overlay is the test for (c).
6. **Subspace overlap.** Principal angles between ΔW_sft's top-r singular
   vectors and the rank-r adapter's A/B matrices, against a random-subspace
   baseline of the same size. The baseline is required; the raw number means
   nothing without it.
7. **Write-up.** Report (a) to (d) separately. Each can hold or fail on its own,
   so they don't get averaged into one headline.

## Budget and bookkeeping

- Any comparison between ranks or methods has to be on the same number of
  training **tokens**, not steps, since batch size and sequence length differ.
- Every run writes a manifest: git commit, config hash, seed, the exact data
  rows used, package versions, wall-clock time.
- Seeds fixed at 42. Anything I'd call a finding should hold across at least
  two seeds, because small model plus small data is noisy.

## Not doing

- Building a good chatbot. This is about measuring, not quality.
- PPO. DPO only; PPO doubles the training setup for little extra relevance.
- Generating or grading harmful content. XSTest is a refusal counter here and
  nothing more.

## Added after the fact

Two things changed the conclusions and weren't in this plan:

- The effective-rank count in step 2 turned out to be the wrong tool. It reads
  ~99% for almost anything. Stable rank was used instead, from the same saved
  spectra.
- Step 5 was meant to separate "alignment is special" from "all fine-tuning
  does this". It can't, since SFT and DPO are both alignment. So two
  non-alignment controls were added (plain LM on wikitext and on Python code,
  matched on tokens and settings), plus a second SFT seed for error bars.

Both are written up in [report.md](report.md).
