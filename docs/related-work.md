# Related work

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
released Llama and Qwen pairs, SVD every linear layer and measure effective
rank. Finding: effective rank barely changes between base and instruct, so
post-training changes what's in the singular subspaces more than their size.
They use that to trim tail components of ΔW and improve pass@k without
training. What they don't do: they use whatever checkpoints labs released
(unknown recipes, SFT and RLHF and DPO mixed together), so they can't separate
SFT from DPO, and they don't connect the spectrum to style vs capability evals
or compare against a LoRA run. It did make me more confident Qwen models would
show something measurable. It's also where the 0.01·σ₁ effective-rank
convention came from, which turned out to be the weak link (see report.md).

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

**Kalajdzievski 2023**, *Efficient Storage of Fine-Tuned Models via Low-Rank
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

Aghajanyan and Kalajdzievski both show low-rank ΔW is a property of fine-tuning
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
- Kalajdzievski 2023. Efficient Storage of Fine-Tuned Models via Low-Rank Approximation of Weight Residuals. <https://arxiv.org/pdf/2305.18425>
- Qi et al. 2024. Safety Alignment Should Be Made More Than Just a Few Tokens Deep. <https://arxiv.org/pdf/2406.05946>
- Biderman et al. 2024. LoRA Learns Less and Forgets Less. <https://arxiv.org/abs/2405.09673>
- 2024. Planning vs Reasoning: Ablations to Test Capabilities of LoRA Layers. <https://arxiv.org/html/2412.00029>
- 2025. How Much is Too Much? Exploring LoRA Rank Trade-offs. <https://aclanthology.org/2025.findings-ijcnlp.58.pdf>
- Thinking Machines Lab. LoRA Without Regret. <https://thinkingmachines.ai/blog/lora/>
- Qwen Team 2024. Qwen2.5 Technical Report. <https://arxiv.org/pdf/2412.15115>
- Zhou et al. 2023. Instruction-Following Evaluation for Large Language Models (IFEval). <https://arxiv.org/abs/2311.07911>
- Röttger et al. 2024. XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours. <https://arxiv.org/abs/2308.01263>
- Cobbe et al. 2021. Training Verifiers to Solve Math Word Problems (GSM8K). <https://arxiv.org/abs/2110.14168>
- Clark et al. 2018. Think you have Solved Question Answering? Try ARC. <https://arxiv.org/abs/1803.05457>
- EleutherAI. LM Evaluation Harness. <https://github.com/EleutherAI/lm-evaluation-harness>
