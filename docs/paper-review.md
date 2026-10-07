# Internal review of the paper draft

A reviewer-style pass over `paper/` before submission: every number recomputed
from the raw results, every citation checked against its source, every figure
and table inspected. Each point says what was found and what was done.

## Summary of the paper

Rank measures that count singular directions (threshold count, 90%-energy
rank, entropy effective rank) give nearly the same values for fine-tuning
updates as for random matrices, so read alone they make updates look
unstructured. Stable rank against a noise baseline shows the updates are
several times more concentrated than noise. The concentration is not specific
to alignment, holds on released models up to 8B, and undercuts one recent
weight-level argument that post-training is superficial (Timber).

## Verdict

Sound and useful after the fixes below. The core empirical claim is solid and
well controlled. The main risks were overclaiming and one misattribution of
prior work, both fixed. As a methods/critique paper it fits TMLR, which judges
correctness over novelty. Main-conference reviewers would still push on scale
and on how much of the "noise baseline" lesson is new.

## Major issues

1. **The title and headline overclaimed.** "Cannot tell a fine-tuning update
   from noise" is false as stated: all four measures separate updates from
   noise with intervals that exclude zero. The counting measures compress the
   difference to 0.4-31% while stable rank shows 65-87%.
   *Fixed:* retitled to "Rank Measures of Fine-Tuning Updates Need a Noise
   Baseline"; abstract, introduction and results now state the measured gaps;
   new Table 2 gives every gap with a layer-bootstrap interval.

2. **Timber's 85% was misattributed.** Their eRank/r of about 0.85
   (interquartile range 0.75-0.95) is for the weight matrices W (their §2.2,
   Fig. 2), not for the update. The draft said they report eRank(ΔW) of 85% and
   that our 83% reproduced it.
   *Fixed:* we now reproduce the right quantity: eRank(W_base)/r is 0.84 for
   Qwen3-0.6B and 0.86 for Qwen3-8B, with interquartile ranges 0.76-0.91 and
   0.76-0.94. The same fix went into the README, report and related work.

3. **No reference besides Gaussian noise.** A reviewer will ask whether ΔW is
   unusually concentrated or just as concentrated as any trained matrix.
   *Fixed:* added the spectra of the pretrained weights
   (`scripts/weight_spectra.py`). Their stable rank is 0.178 (Qwen) and 0.140
   (SmolLM2); the updates are 1.2-1.6x and 1.6-3.1x more concentrated. This is
   stated in the paper, and added as a limitation ("which null").

4. **"Noise-like tail" was wrong.** The new spectrum figure shows the update's
   tail decays slowly, like the pretrained weights, not like a Gaussian's flat
   spectrum.
   *Fixed:* now described as "a long, slowly decaying tail that holds most of
   the energy", in the paper and all docs.

5. **Missing fair comparison with Timber's own ablation.** Their Table 3 already
   compares truncation at fixed rank and energy ratios (all keeping at least 95%
   of the energy), and reports that Timber keeps 98.82% of ΔW's energy on
   Qwen3-8B.
   *Fixed:* both are cited; their 98.82% independently confirms our 98.9% on
   Qwen3-0.6B, and we state that our cutoff (41% kept) is far outside what they
   tested.

6. **The story was hard to follow.** The abstract was about 260 words, the main
   text mixed the measurement result with LoRA, DPO and eval side results, and
   paragraphs were dense with numbers.
   *Fixed:* abstract cut to about 170 words with one storyline; the introduction
   is now question, puzzle, answer, findings; the behavioural results moved to
   an appendix; one point per subsection.

## Numerical errors found

| Where | Draft said | Correct | Status |
|---|---|---|---|
| Timber cutoff comparison | "about 40x more of the update" | tail energy 0.586 vs 0.011 = 53x | fixed (paper, report) |
| SFT GSM8K drop | "3.6 standard errors" | 4.7 / sqrt(1.31² + 1.26²) = 2.6 SE of the difference | fixed (paper, report) |
| eRank(W) real vs random on released models | "2-4x less" | 1.5-4x (Qwen2.5-1.5B: 1.5x) | fixed (paper, report) |
| Our update size relative to W | "0.08-1.4%" | 0.05-1.4% (SmolLM2 code run) | fixed |
| DPO vs SFT stable rank | "same, matrix type by matrix type" | same on average; differs by type (k_proj 0.220 vs 0.135) | fixed (paper, report) |
| Top-8 energy | written as "top 1% of directions" | top 8 directions: 12.0% vs 4.9% | fixed |
| LoRA overlap ratio, rank 1 | 11.5 (from rounded means) | 11.3 | fixed |

All other numbers in the main text and appendix were recomputed from
`results/` and match.

## Citations

Every entry was checked against arXiv or the publisher page.

| Key | Check | Result |
|---|---|---|
| ryu2023efficient | authors | was credited to Kalajdzievski in the docs; the paper is by Ryu, Seo and Yoo. Fixed. |
| codeparrot | dataset | pointed to `codeparrot/github-code`; we trained on `codeparrot/codeparrot-clean-valid`. Fixed. |
| thinkingmachines2025lora | author | byline is John Schulman (with Thinking Machines). Fixed. |
| roy2007effective | pages | venue confirmed (EUSIPCO 2007); page numbers could not be confirmed, so removed. |
| cui2024ultrafeedback | data source | we used the binarized split released with Zephyr; added tunstall2023zephyr. |
| wu2025timber, chaudhury2025alignment, ping2024deltacome, paul2026spectral, biderman2024lora, zhou2023lima, lin2024urial, rottger2024xstest, qwen2.5, yang2025qwen3, allal2025smollm2, zhao2024galore | title, authors | match |
| shuttleworth2025lora | venue | NeurIPS 2025, matches |
| all others | well-known; checked against memory of the published versions | no issues found |

Claims attributed to cited papers were also checked: URIAL's "differences are
mostly stylistic tokens", Biderman et al.'s 90%-variance rank and their note
that small Gaussian noise leaves W's spectrum similar, Shuttleworth et al.'s
intruder dimensions, Delta-CoMe's finding that low-rank truncation hurts, and
Timber's eRank definition (γ = 1), cutoff K = ⌈eRank(ΔW)⌉ and benchmarks
(GSM8K is not among them).

## Figures and tables

- **New figure:** `spectrum_head_tail.pdf`, cumulative energy and singular
  value decay for the updates, noise and the pretrained weights. It carries
  the head/tail argument that was previously made in prose only.
- **Figure 1** now computes the noise eRank from the random-init spectra, as
  the caption says (same values as before).
- **Released-pairs figure:** legend moved so it no longer covers the ratio
  labels. The left axis starts at 0.9; the caption says so.
- **New tables:** relative gaps with intervals (Table 2), pretrained-W rows in
  Table 1, threshold sensitivity (0.01/0.05/0.1), the Marchenko-Pastur check
  for the noise baseline, and all five cutoff variants on 300 problems.
- Every table and appendix section is now referenced from the main text.
- Wide tables use `\small`; check them first after compiling on Overleaf.

## Remaining weaknesses (need new experiments, not edits)

1. **Scale of our own training.** Fine-tunes are 0.36B and 0.5B with about two
   million tokens. Released pairs reach 8B, but only for measurement.
2. **Behavioural results are single-seed and Qwen-only**, and the LoRA
   learning-rate grid was too high. They are now in the appendix and framed as
   context.
3. **The cutoff test covers one model and one benchmark.** Timber's reported
   gains are on MATH, GPQA and others, which we did not run.
4. **No mechanism is tested.** The discussion offers low-rank gradients
   (GaLore) as a likely reason, without evidence from our runs.
5. **Novelty.** None of the measures is new, and Biderman et al. made a related
   noise observation for W. The contribution is the systematic per-matrix
   baseline, the controls, and the correction to a specific published
   argument. The paper says this plainly.

## Before submission

- Compile on Overleaf and check the wide tables and figure placement.
- Keep `\usepackage{tmlr}` (anonymous); no name or repository link in the
  submitted version.
- Check TMLR's current policy on disclosing LLM assistance and add a statement
  if required.
- Optional: email the Timber authors when the preprint goes up.
