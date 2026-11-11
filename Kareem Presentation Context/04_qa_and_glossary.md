# Likely questions, honest answers, and glossary

*Written for: Kareem, rehearsing the midway presentation. Extends sections 11 and 12 of `docs/PRESENTATION_GUIDE.md` with the 5% result (that guide is dated before the evaluation existed).*

## Questions about the result
**So which pipeline is better?**
Under our pre-registered rule, B passes narrowly on the truth that leans against it (GT#2: delta rho +0.44, CI [0.06, 0.70]). But B's own correlation with the truth is about zero (+0.07, p 0.75). The gap exists because A is negatively correlated with both truths. So: "no evidence yet that either pipeline tracks real polarization; B is not worse than A and passes our rule narrowly." Do not say "the network wins".

**B's correlation is zero. Why does the rule say B wins?**
The rule compares B to A and requires B > 0, which is a low bar. We wrote it before seeing the numbers and we did not tighten it afterwards (that would be post-hoc). We show the stronger conditions (B significantly above 0; the sample-matched truth) and they are not met.

**Why is A negatively correlated with the truth?**
Unexplained. Candidates: both truths fall from spring to summer with composition changes (composition explains about 79% of GT#1) while hostility scores rise later in the year; the 2016-06-27 level shift. Composition-only baselines and the pre/post split are in the spec but not run yet. This is a next step.

**Isn't 22% of the 5% tier too thin?**
Yes, and we say so. It is whole threads across all 53 weeks (no week is missing), 135,678 edges, about 2,500 per week, versus about 460 in the 1% run. rho_B against GT#2 stayed within 0.00 to +0.07 at every cut, so more data from this sample is unlikely to turn B into a strong tracker, but the A drift and the CI width could change. The full tier needs about 55 to 60 hours.

**The CI only just excludes 0. Did you cherry-pick the stopping point?**
No. The stop was driven by freeing the GPU, not by the result. All checkpoints (30k to 136k edges) are reported, including the ones where the CI includes 0 (77k and 102k). The lower bound is stable across bootstrap seeds (0.058 to 0.072). No Holm correction was applied across the two truths.

**Why not the sample-matched truth columns as primary?**
They were computed on all 5% threads, not on the processed prefix, so they do not match our subset (the same caveat as Tier 0). They are shown as secondary and give no winner.

**The LLM signs do not beat VADER in camp recovery. Doesn't that kill B?**
It shows the LLM signs add little at this sample size and model accuracy (about 72%). Camp recovery is weak for every source (0.531 to 0.540 vs a null of about 0.525). It is a limitation of sign quality and graph density, which the synthetic study predicted for sparse graphs.

**Why is P_B about zero?**
At about 1,450 user pairs and mean degree about 3, the synthetic study showed P_B is only 0.02 to 0.06 even at moderate polarization, and mild polarization is invisible. Mean P_B is -0.001: on average the weekly graphs are no more two-camp than their sign-shuffled copies.

**Did you tune anything to get this?**
No. The prompt and model were frozen on the human-labelled pilot before any ground-truth comparison (D-M7); the spec was written before the comparison (eval-spec-v1); the rule clarification was logged before the 5% numbers (D-E3). The two extra checks are labelled exploratory.

## Questions about process and engineering
**Why did the run stop at 22%?** We paused it to free the GPU for another project. It is fully resumable from the cache (`05_how_to_resume_and_reproduce.md`).

**What happened in the crash?** Ollama returned an HTTP 500 at about 98k pairs; the supervisor restarted the extraction in 6 minutes; nothing already stored was lost (the store held 129,533 pairs = 98,083 + 31,450 at the time of the check).

**Why didn't more GPU parallelism speed it up?** Ollama serves this model (qwen3.5, a hybrid architecture) on a single slot: 4, 8 and 16 slots all gave 1.30 edges/s and every request hit slot 0. Alternatives (the llama.cpp server, a different model) were not pursued because they would change the serving engine of the frozen primary model.

**Why is the second model not run?** gemma3:12b is pulled but not scored. The pilot comes first (primary: 0.72). It is a sensitivity analysis, never a replacement.

**Can you scale it?** The full 5% is about 55 to 60 h on this PC. 20% (2.7M edges) needs a faster serving setup or a bigger machine.

## Questions about the earlier parts (from the guide, still valid)
- *Why not tune the prompt until it correlates with the truth?* That would be leakage; the prompt was frozen on a pilot split made before any edits.
- *Isn't sharing one LLM call unfair or unrealistic?* It isolates structure from prompt design; the cost is that we are not exact replications of Chowdhury (A) or Guo (B) (D-S1).
- *Your synthetic checks failed.* Two of eight failed on pre-stated criteria; we report them and did not loosen them.
- *How do you know the frustration heuristic is right?* Optimal on 10/10 and 14/16 small exact-solver comparisons; not verifiable at 1,000+ nodes (D-B3).
- *How do you handle composition in GT#1?* Composition-only and volume-only baselines, partial correlations and the pre/post 2016-06-27 split are in the spec; the volume control is done, the rest is next.

## Glossary (additions to the guide's glossary)
- **Prefix sample:** the first k threads in a fixed random order per week; any prefix is a whole-thread subsample covering all weeks.
- **Checkpoint:** an evaluation of the pipelines on whatever pairs were in the store at that moment.
- **Delta rho:** rho_B minus rho_A; positive means B correlates better with the truth than A.
- **Block bootstrap (4-week):** resampling weeks in blocks of 4 to respect weekly autocorrelation when computing the CI of delta rho.
- **Circular-shift permutation p:** the chance of seeing a correlation this large when the series is shifted in time against the truth.
- **Camp recovery (balanced accuracy):** how well B's inferred two-camp split matches the real Dem/Rep labels, versus a label-shuffle null.
- **Single slot:** the server processes one request at a time for this model, so client parallelism does not help.
- **D-E3 / D-E4 / D-T2 / D-O1 / D-O2:** this session's decisions in `docs/llm_decisions.md` (rule clarification, final reporting, throughput, overnight run).
- Everything else: see section 12 of `docs/PRESENTATION_GUIDE.md`.
