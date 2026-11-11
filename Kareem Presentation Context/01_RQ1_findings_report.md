# RQ1 at the 5% sample (partial coverage): is the signed network worth building?

*Written for: Kareem and his co-author, to prepare the midway presentation. Final evaluation 2026-11-01. Spec: `docs/eval_spec.md` (tag eval-spec-v1) plus the rule clarification D-E3, logged before any 5% number existed. Nothing in the prompt, model, spec or thresholds was changed after seeing results.*

## 1. Bottom line
- **Literal result under the pre-stated rule:** on the primary truth that leans against B (full-week GT#2) the delta rho_B - rho_A is **+0.44, 95% CI [0.06, 0.70]** (4-week block bootstrap; lower bound stable at 0.06 to 0.07 across 5 seeds x 10,000 draws), and rho_B is positive (+0.07). The script's verdict is therefore "B wins on GT#2".
- **Why we do not read that as "B tracks polarization":** B itself is indistinguishable from zero (rho +0.07, permutation p 0.75 on GT#2; -0.05, p 0.81 on GT#1). The positive delta exists because **A is anti-correlated with both truths** (rho -0.38 on GT#2, -0.44 on GT#1, permutation p 0.11 and 0.08). The sample-matched secondary columns give **no winner** (delta CI includes 0). The B-minus-A gap crossed zero at the 77k and 102k checkpoints and only cleared it at 136k, and no Holm correction was applied across the two truths.
- **Defensible statement for the talk:** "On a 22% prefix of the 5% tier (135.7k edges, all 53 weeks), neither pipeline shows a positive, significant association with either ground truth. B passes our pre-registered decision rule narrowly on GT#2, but only relative to A, which runs in the wrong direction. We do not claim that the signed network tracks the ground truth. We claim that at this sample size it does not do worse than text scoring, and that both are weak."
- Stronger conditions we did NOT meet (reported, not hidden): rho_B significantly above 0; delta CI excluding 0 on the sample-matched truth; first-difference agreement (rho on first differences: B -0.25 on GT#2, A -0.15).

## 2. What was run
| Item | Value |
|---|---|
| Sample | 5% shared-venue tier, thread-prefix ordering (whole threads, all 53 weeks), **135,678 edges processed = 135.6k of 617,041 unique pairs (22%)**, 2,885 threads, median 2,512 edges/week (min 275 in the partial first week, max 5,788) |
| Model | `qwen3.5:9b-q4_K_M`, prompt v2, think off, logprobs, frozen under D-M7 (pilot: dev acc 0.733 / kappa 0.548; held-out 0.707 / 0.439; combined 0.72) |
| Extraction | 2026-10-31 23:22 to 2026-11-01 13:15 (about 14 h wall incl. one 6-minute crash/restart), 0 failed pairs, 4.4% truncated |
| Pipeline A | `runs/llm/20261101T131638_pipeA_5pct_final`: 53 weeks, P_A mean 3.08, sd 0.30, range 2.51 to 4.00; labels: 52% disagree, 30% agree, 17% neutral |
| Pipeline B | `runs/llm/20261101T131644_pipeB_5pct_final`: 53 of 53 weeks scored (33 of 53 at Tier 0 60/wk; 52 of 53 at 230/wk); giant component median about 1,000 nodes, about 1,450 user pairs per week; 61.6% of pair weights negative; sign source `continuous_llm` (headline) |
| Evaluation | `scripts/llm/rq1_compare.py` (spec-exact): Spearman rho on levels, circular-shift permutation p, first-difference rho, partial rho controlling for weekly volume, volume-only baseline, delta rho with 4-week block bootstrap (2,000 draws) |

**Coverage caveat:** this is a thread-complete prefix, not a random 22% of the tier, and it is thinner than a true 5% sample. The extraction was stopped by the user, not by completion (about 55 to 60 h would be needed for the whole tier at the observed 2.5 to 2.9 pairs/s).

## 3. Results (primary truth: full-week GT; n = 53 weeks)
| Series vs truth | Spearman rho | permutation p | rho on first differences | partial rho (volume) |
|---|---|---|---|---|
| P_A vs GT#1 | -0.44 | 0.075 | -0.09 | -0.46 |
| P_A vs GT#2 | -0.38 | 0.113 | -0.15 | -0.35 |
| P_B vs GT#1 | -0.05 | 0.811 | -0.03 | -0.05 |
| P_B vs GT#2 | +0.07 | 0.755 | -0.25 | +0.06 |
| edge volume vs GT#1 / GT#2 | +0.00 / -0.17 | 1.00 / 0.34 | | |

| Delta rho = rho_B - rho_A (4-week block bootstrap) | point | 95% CI | note |
|---|---|---|---|
| vs GT#1 (full-week) | +0.39 | [0.02, 0.65] | rule: this is the truth that leans toward B, so B beating A here is expected, not decisive |
| **vs GT#2 (full-week), the deciding comparison** | **+0.44** | **[0.06, 0.70]** | rule met literally |
| vs GT#1, 5%-matched (secondary) | +0.22 | [-0.11, 0.47] | includes 0 |
| vs GT#2, 5%-matched (secondary) | +0.29 | [-0.13, 0.52] | includes 0 |
| vs GT#2 alt AUC, 5%-matched (secondary) | +0.22 | [-0.19, 0.48] | includes 0 |

A vs B (weekly series): rho 0.27. The 5%-matched GT columns were computed on all 5% threads, not on the processed prefix, so they are a secondary check, not the primary comparison (D-E3).

### Trajectory (the result was not stable until the last cut)
| Edges processed | rho_A (GT#2) | rho_B (GT#2) | delta (GT#2) [95% CI] | usable B weeks |
|---|---|---|---|---|
| 30k | -0.19 | +0.07 | +0.31 [-0.17, 0.69] | 46 |
| 53k | -0.19 | 0.00 | +0.27 [-0.19, 0.59] | 51 |
| 77k | -0.34 | +0.04 | +0.41 [-0.01, 0.74] | 52 |
| 102k | -0.37 | +0.04 | +0.41 [-0.02, 0.70] | 53 |
| 136k (final) | -0.38 | +0.07 | +0.44 [0.06, 0.70] | 53 |

rho_B stays within +/-0.07 of zero at every cut. Only rho_A moves (more negative with more data), which is what widens the delta. Checkpoints were produced by the unattended supervisor and are the same code as the final.

### Camp recovery (does B's inferred partition match real Dem/Rep labels?)
| Sign source | mean balanced accuracy | mean shuffle null | weeks above null p95 (of 53) |
|---|---|---|---|
| LLM signs, continuous (headline) | 0.531 | 0.524 | 8 |
| LLM signs, binary | 0.540 | 0.527 | 9 |
| VADER on the reply only (no LLM) | 0.534 | 0.526 | 6 |
| Unsigned control | 0.500 | 0.500 | 0 |

About 450 labelled users per week. The LLM-signed camps are barely above the null, and **not better than free, context-free VADER sentiment**. At Tier 0 (1%) the continuous source gave 0.597 vs null 0.562; the two runs differ in graph size and labelled-user population (about 62 vs about 450 users per week), and we did not investigate the drop. P_B itself: mean -0.001, sd 0.11, positive in 26 of 53 weeks, i.e. on average **no more two-camp structure than a sign-shuffled copy of the same graph**.

### Exploratory checks (not in the spec, reported as such)
- Bootstrap seed stability: lower CI bound on GT#2 is 0.058 to 0.072 over 5 seeds (`results/llm/rq1_5pct/5pct_final/bootstrap_stability.txt`).
- Partial first/last weeks: dropping week 2015-12-28 (275 edges, partial) or both end weeks changes rho_A, rho_B and delta by 0.05 or less; the conclusion is not driven by them (`partial_week_sensitivity.txt`).

## 4. How to read it (honest reading)
1. **Neither pipeline tracks either truth.** All four correlations with the truths are weak; none beats a circular-shift null at 5% (A's are the closest: p 0.08 and 0.11).
2. **A runs the wrong way, and more strongly with more data.** This is unexplained. Candidates (not tested): both truths fall together from spring to summer with composition changes (Yasmina's `gt1_drivers`: composition explains about 79% of GT#1, and GT#1 and GT#2 correlate 0.80), while hostility scores drift upward from the summer; the 2016-06-27 level shift in GT#1 is also a candidate. The spec's composition baselines and pre/post-06-27 split were not run.
3. **B is at the floor.** At this density (about 1,450 user pairs, mean degree around 3) the synthetic study predicted P_B is only 0.02 to 0.06 even at moderate polarization (`phase_4_report.md`), so a weekly P_B near 0 is expected. B's problem at this sample size is the measure's sensitivity, not missing weeks.
4. **The deciding delta is positive but borderline and relative to a pipeline that is anti-correlated with truth.** Presenting it as "the network beats text" would overstate it.
5. **Sign quality is a confound we cannot resolve here:** model accuracy is about 72% on a 150-edge pilot and the LLM-signed camps do not beat VADER on camp recovery.

## 5. Not done / limitations (say these on stage)
- Partial coverage (22% of the 5% tier; prefix, not random); one prompt, one primary model, one sample.
- Not done from the spec: Holm correction, composition-only baselines, before/after 2016-06-27 split, split-half noise ceilings.
- A and B share one LLM call (D-S1), so they are not independent replications of Chowdhury or Guo. P_A uses our additive rule table (A-rules-v1, `TODO(verify)`).
- Frustration heuristic gap at 1,000+ nodes not verified against an exact solver (D-B3).
- Camp labels provisional (D-S6); 1% and 5% samples are independent draws (D-S5).
- Robustness model (`gemma3:12b`) pulled but **not scored or run**; the 5% extraction was stopped to free the GPU for another project.

## 6. What to do next
1. Resume the extraction (see `05_how_to_resume_and_reproduce.md`) toward a larger prefix; the result is stable in rho_B, so more data mainly tests whether rho_A keeps drifting.
2. Run the composition-only and volume baselines, the pre/post 2016-06-27 split, noise ceilings and Holm correction (all stated up front as not done).
3. Score `gemma3:12b` on the pilot; if not clearly worse than 0.72, run it for about 2 hours and report per-model delta rho and inter-model agreement of P_A and P_B (sensitivity analysis, never a replacement for the primary).
4. Investigate why A is negatively related to the truths (composition and 06-27 are the first suspects).
5. Sign-quality experiments (noise injection, binary vs continuous vs VADER signs) now that B has all 53 weeks.

Files: `results/llm/rq1_5pct/5pct_final/` (correlations, delta, summary, stability, sensitivity), checkpoint folders `5pct_ckpt1..4`, run folders above, supervisor log `logs_llm/overnight.log`, decisions D-E3, D-T2, D-O1, D-O2, D-E4 in `docs/llm_decisions.md`.
