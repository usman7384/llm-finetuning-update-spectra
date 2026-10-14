# Related work

> **Note.** Written during the first round. The paragraph "What I think is new
> here" and the claims that counting measures "can't tell a fine-tuning delta
> from noise" and that stable rank is the right measure were softened in the
> paper: every measure separates updates from noise, with intervals that
> exclude zero; what differs is the size of the gap, and the paper recommends
> no measure over the others. The LoRA material refers to first-round runs
> that the paper does not report. `paper/sections/related.tex` is current.

Notes on what was already out there when I started (September 2026), and where
this project fits. Most of it is about whether alignment changes knowledge or
just style, and whether fine-tuning updates are low-rank.

## Where the idea comes from

**LIMA** (Zhou et al. 2023, [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)).
The source of the "superficial alignment hypothesis", in their words: *"A model's
knowledge and capabilities are learnt almost entirely during pretraining, while
alignment teaches it which subdistribution of formats should be used when
interacting with users."* They fine-tuned 65B LLaMA on just 1,000 curated
examples and got outputs people often rated as good as GPT-4's. All the evidence
is about outputs, though, and it has been pushed back on: a
[LessWrong critique](https://www.lesswrong.com/posts/uyk5nn93HxJMsio98/metaai-less-is-less-for-alignment-1)
argues the result leans on that particular 65B model and hand-picked data and
doesn't carry over to smaller ones. Nobody looked at the weights.

## The output-level evidence

**URIAL** (Lin et al. 2023, [arXiv:2312.01552](https://arxiv.org/pdf/2312.01552),
[project page](https://allenai.github.io/re-align/urial.html)). The most direct
test of the idea before this, done on outputs. They compare a base model's
next-token distribution with its aligned version's at every position of some
held-out answers. The two agree on the top token about 92-95% of the time, and
where they disagree it's mostly discourse words and disclaimers ("However",
"cannot", "Here"). That held for three 7B base/chat pairs. They then show that
three in-context examples plus a system prompt gets a base model most of the
way to the aligned one, with no training at all.

Strong evidence, but it's entirely about the output distribution. It never
looks at ΔW, never asks if the shift is low-rank, and never splits capability
benchmarks from style ones. That's the gap this project goes after.

## Weight-space and layer evidence

This is the fast-moving part, and the most relevant to how the experiments
were set up.

**Aghajanyan et al. 2021**, *Intrinsic Dimensionality Explains the Effectiveness
of Language Model Fine-Tuning* ([ACL](https://aclanthology.org/2021.acl-long.568/)).
Pretraining makes fine-tuning cheap: about 200 randomly projected parameters got
RoBERTa to 90% of full fine-tuning on MRPC. It's the idea LoRA grew out of, and
it already says fine-tuning *in general* is low-dimensional, not just alignment.
That's the confound this project had to deal with (see the end of this page).

**Timber** (2025), *Training-free Instruct Model Refining with Base via Effective
Rank* ([arXiv:2509.23595](https://arxiv.org/html/2509.23595)). The closest
paper to the SVD part of this project. They take ΔW = W_instruct − W_base for
released Llama and Qwen3 pairs and measure Roy & Vetterli's entropy effective
rank (eRank, γ=1). Finding: eRank of each whole weight matrix is nearly the
same in base and instruct (about 85% of full rank), which they read as
post-training being superficial. They use eRank(ΔW) as a cutoff to trim
tail components of ΔW and improve pass@k without training. What they don't do:
they use whatever checkpoints labs released (unknown recipes, SFT and RLHF and
DPO mixed together), so they can't separate SFT from DPO, they don't connect
the spectrum to style vs capability evals or compare against a LoRA run, and
there's no noise or same-size-perturbation baseline. It did make me more
confident Qwen models would show something measurable.

(Hindsight correction: I originally wrote that the 0.01·σ₁ threshold count
came from Timber. It doesn't; I misread it. Report section 6 checks their
actual eRank and finds the same problem in a milder form: an unchanged
eRank(W) is what any small perturbation gives, and eRank(ΔW) sits within a few
percent of noise.)

**Chaudhury 2025**, *Alignment is Localized: A Causal Probe into Preference
Layers* ([arXiv:2510.16167](https://arxiv.org/abs/2510.16167)). Closest to the
DPO question. Uses activation patching between Llama-3.2-1B and its
preference-tuned version, layer by layer, to find which layers carry the
alignment effect. Answer: a narrow band in the middle of the stack, captured by
a few activation directions. Their summary is that preference alignment is "a
directional, low-rank process, rather than diffuse and parametric". This is
causal evidence about activations, a different thing from SVD of weights, and
the two should be read together. They don't SVD ΔW and don't compare a DPO delta
with an SFT delta on the same model.

**Zhao et al. 2026 (ACL)**, *A Layer-wise Analysis of Supervised Fine-Tuning*
([arXiv:2604.11838](https://arxiv.org/abs/2604.11838)). SFT only, 1B to 32B.
The first ~80% of layers stay structurally stable and the last few change
sharply. They turn that into a mid-block tuning method that beats LoRA on GSM8K
by up to 10 points for OLMo2-7B. Gave me a prior to check against: expect the
change to sit late. No DPO, no SVD of ΔW, no rank sweep against a style split.

**Massive SFT experiments** (2025, [arXiv:2506.14681](https://arxiv.org/pdf/2506.14681)).
Large correlational study of which layers change most vs which matter most.
Slightly surprising result: the layers with the biggest weight edits aren't the
ones driving quality; middle layers matter more even when they change less. A
useful warning against reading "biggest singular value" as "most important",
and the reason the eval split matters more than the SVD plots alone.

**Ryu, Seo & Yoo 2023**, *Efficient Storage of Fine-Tuned Models via Low-Rank
Approximation of Weight Residuals* ([arXiv:2305.18425](https://arxiv.org/pdf/2305.18425)).
Shows ordinary task fine-tuning deltas are low-rank enough to store as rank-r
approximations with little loss. Again evidence the low-rank thing is general,
not alignment-specific.

**Qi et al. 2024**, *Safety Alignment Should Be Made More Than Just a Few Tokens
Deep* ([arXiv:2406.05946](https://arxiv.org/pdf/2406.05946)). Safety alignment
often works by changing just the first few output tokens (learning to open with
"I cannot..."), which is why it breaks so easily under prefill or fine-tuning
attacks. A different kind of shallowness (token position rather than weight
rank), but plausibly related: a refusal implemented as "nudge the first-token
logits" is exactly the sort of narrow edit an SVD of ΔW might pick up.

## LoRA rank vs capability

**Biderman et al. 2024**, *LoRA Learns Less and Forgets Less*
([arXiv:2405.09673](https://arxiv.org/abs/2405.09673)). The main LoRA vs full
fine-tuning comparison, on code and math. LoRA learns the target domain less
well but forgets less of everything else. Their explanation: learning new
knowledge needs more capacity than turning a base model into an
instruction-follower. Close to prediction (b), but framed as domain transfer
rather than a style/capability split with a rank sweep. They also note the gap
partly closes with properly tuned learning rates, which is why the sweep here
tries three LRs per rank.

**Planning vs Reasoning: Ablations to Test Capabilities of LoRA Layers** (2024,
[arXiv:2412.00029](https://arxiv.org/html/2412.00029)) and **How Much is Too Much?
Exploring LoRA Rank Trade-offs** ([ACL Findings](https://aclanthology.org/2025.findings-ijcnlp.58.pdf)).
Some abilities come back at surprisingly low rank and others need a lot more, so
task-dependent crossover ranks do exist. Neither uses a style vs capability split
or ties it back to the SVD of a full fine-tune.

**Thinking Machines Lab**, *LoRA Without Regret* ([blog](https://thinkingmachines.ai/blog/lora/)).
Practitioner post arguing LoRA matches full fine-tuning for post-training style
workloads when rank and LR are set right. Supportive, not a controlled study.

## Measuring the shape of a weight update

Added after the main experiments, once it was clear the measurement itself was
the interesting part.

**The measures.** Stable rank, ‖A‖²_F / ‖A‖²₂, is from Rudelson & Vershynin
2007 (they call it numerical rank), introduced precisely because it ignores
tiny singular values. The entropy effective rank Timber uses is Roy & Vetterli
2007. Neither is new; the question here is only which one can tell a weight
update from noise.

**Biderman et al. 2024** also measure the spectrum of full fine-tuning updates
(Llama-2-7B, continued pretraining on code, 0.25B-20B tokens). Their measure is
the rank needed to explain 90% of the variance, and they find ΔW needs a rank
10-100x bigger than typical LoRA ranks, growing with more data. They also note
that the fine-tuned W has nearly the same spectrum as the base W, and that
adding small Gaussian noise to W gives a similar spectrum. That second point is
close to my same-size perturbation check on Timber (report section 6); they
made it about W, and I make it a per-matrix control and apply it to the
"superficial" argument. Their 90%-energy rank has the same blind spot as the
other count-style measures: on my runs it gives 0.66 of max rank for SFT and
0.72 for noise (Qwen), 0.50 vs 0.66 (SmolLM2), while stable rank gives 3x and 7x.
So "ΔW is high rank" and "ΔW is concentrated" are both true. The bulk of the
spectrum is a long, slowly decaying tail, and the top few directions hold
far more energy than noise would. Count-style measures see the first,
stable rank the second.

**Shuttleworth et al. 2025**, *LoRA vs Full Fine-tuning: An Illusion of
Equivalence* (NeurIPS 2025, [arXiv:2410.21228](https://arxiv.org/abs/2410.21228)).
Compare the singular vectors of fine-tuned W with those of the pretrained W.
LoRA introduces "intruder dimensions", new high-ranking singular vectors with
low cosine similarity to any pretrained one; full fine-tuning mostly doesn't.
They note effective-rank-style measures alone don't separate the two. Their
analysis is of W's singular vectors, not the spectrum of ΔW, and has no
non-alignment or noise control, but it's the strongest evidence that how
LoRA and full fine-tuning differ is about direction rather than rank. My
section 7 (LoRA's top directions overlap full SFT's, 11x above chance at rank
1) is compatible with it: overlap with ΔW's top directions is a different
question from whether LoRA adds intruders to W.

**Delta-CoMe** (Ping et al., NeurIPS 2024, [arXiv:2406.08903](https://arxiv.org/abs/2406.08903)).
Compresses fine-tuning deltas by spending more bits on the top singular
vectors, motivated by the long-tail distribution of singular values in ΔW.
Same observation as the "concentrated head, dense tail" picture here, used for
compression rather than as evidence about alignment, and it finds pure low-rank
truncation of ΔW hurts, which is a useful caution for any cutoff experiment.

**Paul 2026**, *Spectral Geometry of LoRA Adapters Encodes Training Objective
and Predicts Harmful Compliance* ([arXiv:2604.08844](https://arxiv.org/abs/2604.08844)).
Uses stable rank among other shape features of LoRA deltas (rank 8,
Llama-3.2-3B) to tell DPO objectives apart. Shape features separate
objectives where magnitude doesn't, which agrees with my finding that size and
concentration are separate things. LoRA only, one model, no noise baseline.

**What I think is new here,** after this pass: (1) a noise baseline for every
measure on every matrix, which shows three common measures (threshold count,
entropy eRank, 90%-energy rank) can't tell a fine-tuning delta from noise while
stable rank can; (2) applying that to the specific evidence for "superficial
post-training" in Timber and reproducing their number; (3) token-matched
non-alignment controls with seeds, showing the concentration isn't about
alignment; (4) the same across two model families and six released pairs.
None of the measures are new, and the "dense tail" observation isn't either.

## Tools that use SVD

PiSSA and LoftQ use SVD of the *base* weights to initialize LoRA; DoRA splits
weights into magnitude and direction; SVD-LLM and SVFit use SVD for compression
or parameter-efficient tuning. None of them measure the spectrum of a real
training run's ΔW, per layer and matrix type, as evidence about what alignment
does. Timber is the only one close to that.

## Where this project sits

I didn't find one paper that does all of:

1. full (not LoRA) SFT, then DPO on top, on the same small open model, with both
   checkpoints kept so ΔW can be attributed to one method;
2. SVD of ΔW per weight matrix, by depth and matrix type, for SFT and DPO
   separately;
3. a capability vs style eval split crossed with a LoRA rank sweep;
4. a check of whether LoRA's subspace matches the full fine-tune's top
   directions.

The closest work (Timber, Chaudhury, Zhao) is all from 2025-2026, so this is
still an open question rather than a settled one.

## The confound, and how it was actually handled

Aghajanyan and Ryu et al. both show low-rank ΔW is a property of fine-tuning
in general. So "the alignment delta is low-rank" proves nothing by itself; it
has to be compared against fine-tuning that isn't alignment.

The original plan tried to handle that by comparing the SFT delta with the DPO
delta on the same model. Looking back, that couldn't work: both are alignment,
so their agreeing says nothing about non-alignment training. What actually
settled it was two control runs on wikitext and Python code with the same token
budget and settings. Both came out as concentrated as SFT, so the concentration
isn't alignment-specific. Details in [report.md](report.md).

## References

- Zhou et al. 2023. LIMA: Less Is More for Alignment. <https://arxiv.org/abs/2305.11206>
- Lin et al. 2023. The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning (URIAL). <https://arxiv.org/pdf/2312.01552>
- Aghajanyan et al. 2021. Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. <https://aclanthology.org/2021.acl-long.568/>
- Timber 2025. Training-free Instruct Model Refining with Base via Effective Rank. <https://arxiv.org/html/2509.23595>
- Chaudhury 2025. Alignment is Localized: A Causal Probe into Preference Layers. <https://arxiv.org/abs/2510.16167>
- Zhao et al. 2026. A Layer-wise Analysis of Supervised Fine-Tuning. <https://arxiv.org/abs/2604.11838>
- 2025. Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality. <https://arxiv.org/pdf/2506.14681>
- Ryu, Seo & Yoo 2023. Efficient Storage of Fine-Tuned Models via Low-Rank Approximation of Weight Residuals. <https://arxiv.org/pdf/2305.18425>
- Qi et al. 2024. Safety Alignment Should Be Made More Than Just a Few Tokens Deep. <https://arxiv.org/pdf/2406.05946>
- Biderman et al. 2024. LoRA Learns Less and Forgets Less. <https://arxiv.org/abs/2405.09673>
- Shuttleworth, Andreas, Torralba, Sharma 2025. LoRA vs Full Fine-tuning: An Illusion of Equivalence. NeurIPS 2025. <https://arxiv.org/abs/2410.21228>
- Ping et al. 2024. Delta-CoMe: Training-Free Delta-Compression with Mixed-Precision for Large Language Models. NeurIPS 2024. <https://arxiv.org/abs/2406.08903>
- Paul 2026. Spectral Geometry of LoRA Adapters Encodes Training Objective and Predicts Harmful Compliance. <https://arxiv.org/abs/2604.08844>
- Rudelson & Vershynin 2007. Sampling from Large Matrices: An Approach through Geometric Functional Analysis. Journal of the ACM 54(4). <https://arxiv.org/abs/math/0503442>
- Roy & Vetterli 2007. The Effective Rank: A Measure of Effective Dimensionality. EUSIPCO 2007.
- 2024. Planning vs Reasoning: Ablations to Test Capabilities of LoRA Layers. <https://arxiv.org/html/2412.00029>
- 2025. How Much is Too Much? Exploring LoRA Rank Trade-offs. <https://aclanthology.org/2025.findings-ijcnlp.58.pdf>
- Thinking Machines Lab. LoRA Without Regret. <https://thinkingmachines.ai/blog/lora/>
- Qwen Team 2024. Qwen2.5 Technical Report. <https://arxiv.org/pdf/2412.15115>
- Zhou et al. 2023. Instruction-Following Evaluation for Large Language Models (IFEval). <https://arxiv.org/abs/2311.07911>
- Röttger et al. 2024. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours. <https://arxiv.org/abs/2308.01263>
- Cobbe et al. 2021. Training Verifiers to Solve Math Word Problems (GSM8K). <https://arxiv.org/abs/2110.14168>
- Clark et al. 2018. Think you have Solved Question Answering? Try ARC. <https://arxiv.org/abs/1803.05457>
- EleutherAI. LM Evaluation Harness. <https://github.com/EleutherAI/lm-evaluation-harness>
