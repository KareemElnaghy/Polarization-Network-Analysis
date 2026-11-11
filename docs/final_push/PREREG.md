# Pre-registration entries D-E5 and D-E6

Paste both entries into `docs/llm_decisions.md` **before computing any new correlation with ground truth**, fill in the SHA, commit, and tag `prereg-final-v1`. These entries were drafted on 2026-11-10. At that point the only GT comparisons that had been seen were the ones recorded in handbook §12 (Tier 0, the 5% prefix, the interim run, and the full 5%). Nothing below has been computed yet.

---

## D-E5: How the full-5% RQ1 result is reported (final)

**Date:** 2026-11-10. **Status:** final. **Applies to:** `results/llm/rq1_5pct/rq1_5pct_full_*` (A `20261107T121812_pipeA_5pct_full`, B `20261107T121822_pipeB_5pct_full`, git `022dba0`).

1. The full-5% result is **the** pre-registered RQ1 result under eval-spec-v1 and D-E3. It is reported as: *"The pre-registered decision rule is met on both the primary and the secondary truth (delta rho on GT#2 full = +0.65, 95% block-bootstrap CI [0.21, 0.94]). However, rho_B is not distinguishable from zero (+0.17, circular-shift p = 0.38), and the difference comes mainly from Pipeline A being negatively associated with both truths."*
2. It is never reported as "B tracks polarization" or "network structure is better" without the qualification in point 1.
3. The items eval-spec-v1 listed as not done (Holm, composition baselines, the 06-27 split, noise ceilings) are now computed (D-E6 §A) and reported beside it, whatever they show.
4. Seed stability (`rq1_bootstrap_stability.py`, 5 seeds × 10,000 draws) and the partial-week check (`rq1_partial_week_sensitivity.py`) are run on the full-5% runs and reported as robustness checks.

---

## D-E6: Final-milestone analyses (confirmatory unless marked otherwise)

**Date:** 2026-11-10. **Git SHA at freeze:** `<fill in>`. **Tag:** `prereg-final-v1`.

### General rules
- **Truths.** Primary: full-week `gt1_excess` and `gt2` from `results/ground_truth/ground_truth_weekly.csv`. Rate-matched truths are secondary where they exist.
- **Statistics.** These are unchanged from eval-spec-v1: Spearman rho on levels, circular-shift permutation p, 4-week moving-block bootstrap (2,000 draws, seed 0) for every delta and every CI.
- **Holm correction.** Applied within each family listed below, at α = 0.05. A "significant" claim requires the Holm-adjusted p.
- **"B tracks a truth"** means: rho_B > 0 and Holm-adjusted circular-shift p < 0.05 on that truth.
- **"Pipeline X beats Y"** keeps the D-E3 form: the delta CI excludes 0, and rho_X > 0, on the truth that leans against X. For B and C that truth is GT#2. For A it is GT#1.
- **Variants.** Each pipeline has one headline variant per density, listed below. Every other variant is a robustness check and never a headline.

### §A Completion of eval-spec-v1 on the full 5% (family A: 4 tests = {A, B} × {GT#1, GT#2})
- A1. Holm-adjusted p for rho_A and rho_B on the two primary truths.
- A2. Composition-only baselines: weekly r/politics share, rep/dem labelled-user ratio, new-user share (from `results/gt1_drivers/drivers_weekly.csv`, or recomputed per `SPEC_ANALYSES.md` A1 if that file lacks a column). Each baseline's rho with each truth is reported beside rho_A and rho_B. We also report partial rho_A and partial rho_B, controlling for each baseline.
- A3. Before/after 2016-06-27: rho_A and rho_B computed separately within each segment (descriptive, with bootstrap CIs; no win rule).

### §B Mechanism tests on the existing cache
- **H-NC (noise ceiling, descriptive).** Split-half reliability (Spearman-Brown) of P_A and P_B, computed as in SPEC_ANALYSES A2. Disattenuated rho = rho / sqrt(rel_pipeline × rel_GT), using rel_GT1 = 0.93 from the handbook, and rel_GT2 from its sample stability as a lower bound. These are reported with no decision rule.
- **H-AL (alignment, confirmatory).** On edges with both endpoints labelled (main labels, dem/rep), the continuous_llm sign source has a lower mean Guo weight on cross-camp than on same-camp edges: Δw = mean w(cross) − mean w(same) < 0, with a 95% CI from a thread-cluster bootstrap (1,000 draws, seed 0) that excludes 0. Secondary outcomes, reported with no rule: P(disagree), hostility and personal attack, and the same Δ for the binary_llm and VADER sources. Family H-AL contains this one test.
- **H-AN (why A runs negative, confirmatory).** (i) The weekly cross-camp share s_t (among labelled-both edges in the 5% extraction) is negatively associated with GT#1 (rho < 0). (ii) Controlling for s_t moves rho_A(GT#1) toward 0: |partial rho_A| < |rho_A|, with a bootstrap CI on the difference that excludes 0. Family H-AN: 2 tests.

### §C Pipeline C (SPEC_PIPELINE_C.md)
- **Variants (both confirmatory, fixed in SPEC_PIPELINE_C.md):** C1 is the intensity-weighted Guo sign, w_C1 = w_B · (1 + h + a + 0.5e). C2 is a frozen `bge-base-en-v1.5` pair embedding with a ridge head trained on w_C1, using out-of-fold predictions on the 5% tier. Both use the giant component, P_C = 1 − F/F_null, and R = 50. The headline C at each density is C1. C2 is reported beside it, and as a rung of the RQ3 ladder. C3 is exploratory.
- **Decision rules.** "C beats B" requires delta (rho_C − rho_B) CI > 0 **on GT#1**, and rho_C > 0. GT#1 is used here because GT#2 is a language-based truth, and text features could improve rho on GT#2 without any structural gain (circularity). "C beats A" uses the D-E3 form on GT#2. Family C: 2 truths × 2 comparisons = 4 tests.

### §D Density ladder (H1, the main new confirmatory test)
- **Sign source:** the encoder selected under SPEC_ENCODER (frozen at gate G-ENC; its checksum is recorded in an addendum D-E6a written *before* any encoder-based GT comparison).
- **Densities:** 1%, 5%, 20% thread samples (the existing independent draws) and 100% (all cleaned shared-venue edges). Mean degree per density is recorded from the giant components.
- **H1a (trend).** Across the four densities, rho_B(GT#2 full) increases with density: the Spearman correlation between density rank and rho_B is +1 (monotone), **or** rho_B(100%) − rho_B(5%) > 0 with a bootstrap CI excluding 0. The same test is run on GT#1. Family D-trend: 2 tests.
- **H1b (tracking at full density).** At 100%, B tracks GT#2 and/or GT#1 (definition above, Holm over the 2 truths).
- **H1c (head-to-head at full density).** The D-E3 rule (B vs A on GT#2), with A and B both computed from the encoder at 100%.
- **Fallback.** If full-density B cannot finish for all weeks, the weeks are the 27 odd-numbered ISO weeks of 2016 (fixed now), and every test is reported on those weeks at all densities.

### §E RQ3: sign quality (H3)
- Sign-quality ladder: unsigned, VADER, C2 (embedding), distilled encoder, binary_llm, continuous_llm. Each rung is placed on the x-axis by its *measured* pilot accuracy, not by an assumed order. Each rung's accuracy is measured on the 150-item human pilot (accuracy of agree vs disagree, excluding neutral). For VADER the threshold is fixed at compound > 0.05 = agree, < −0.05 = disagree.
- **H3 (confirmatory).** On continuous_llm weights at 5%, P_B's camp recovery and rho_B(GT#2) both decrease as the noise-injection rate p ∈ {0, 0.05, 0.1, 0.2, 0.4} increases (Spearman over p ≤ −0.9, 10 seeds per p, averaged). Family: 2 tests.

### §F RQ2: conditions (all descriptive; no win rules)
Rolling 2- and 4-week windows, calendar months, thinning to 50% and 25% of threads, high- vs low-GT quartile weeks, and the discordant-ally share (negative edges within a camp) against the per-week A−B divergence. Results are reported as tables and figures with bootstrap CIs.

### §G RQ4: forecasting
- **Target:** ΔGT_t = GT_t − GT_{t−1} (GT#1 and GT#2 separately).
- **Models:** persistence (ΔGT = 0); A (ΔGT_t ~ ΔP_A_{t−1}); B (ΔGT_t ~ ΔP_B_{t−1}); A+B. All are OLS refit at each origin, with rolling origins t = 24 … 53, one step ahead.
- **Metric:** MAE. Diebold–Mariano test against persistence (two-sided, HAC variance), reported with the caveat about the small sample.
- **Expectation stated in advance:** no model beats persistence (the first-difference rho is about 0 in §12.6). A null result is reported as a finding.

### §H Confirmatory replication on 2017 (only if Phase 3 runs)
Every rule, threshold, the encoder checkpoint, the label rule (the 58 supporter subreddits; any subreddit absent in 2017 is noted, not replaced) and the GT code are frozen as of the D-E6a addendum. The H1b, H1c, H-AL and C tests are repeated on 2017 at full density. A 2017 result counts as a replication only if it has the same sign and is significant under the same rule. Pooled 2016+2017 (106 weeks) is reported as secondary.

### What is exploratory (every output is labelled `exploratory`)
- C3 (edge-embedding coherence), the per-venue split, PolitiSky24, SRQ, `gemma3:12b`, and 2015.
- Any analysis not listed in this entry.
