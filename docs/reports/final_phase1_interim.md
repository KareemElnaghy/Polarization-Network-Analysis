# Phase 1 interim report (CP1)

Date: 2026-11-13. Branch `llm_implementation`, pre-registration tag `prereg-final-v1` (D-E5/D-E6). All numbers are on the full 5% tier (53 weeks of 2016, 617,568 extracted edges, qwen3.5:9b-q4_K_M, prompt v2) unless stated. Truths: GT#1 = structural (`gt1_excess`), GT#2 = linguistic (`gt2`). Statistics: Spearman rho, circular-shift permutation p, 4-week moving-block bootstrap (2,000 draws, seed 0), Holm within each pre-registered family. Confirmatory and exploratory results are labelled.

## Verdict: which explanation does the evidence favour?

| Explanation | What Phase 1 shows | Verdict |
|---|---|---|
| **Noise** (B is unreliable at mean degree about 3) | Split-half reliability of P_B is about zero (Spearman-Brown median -0.08, 5-95% [-0.46, 0.25]); the observed frustration sits at the sign-shuffle null (mean P_B +0.005); thinning does not reduce rho_B; injected sign noise lowers camp recovery and rho_B(GT#2) strictly but slightly (H3 passes). | **Favoured.** Consistent with everything in Phase 1. The decisive test is the Phase 2 density ladder. |
| **Signs** (LLM signs carry little partisan information) | Better signs do not help: pilot accuracy rises from 48% (VADER) to 73% (binary LLM), 75% (C2) and 79% (continuous LLM), C1 adds intensity, and rho_B stays null for every rung (no delta vs B, all Holm p = 1.0). Every sign source shows cross-camp concentration of negativity, strongest for the LLM. | **Disfavoured.** If signs were the bottleneck, better or richer signs would move rho_B. They do not. |
| **Mechanism** (disagreement is not aligned with the camps) | H-AL passes: cross-camp replies carry a lower Guo weight (delta-w -0.092, 95% CI [-0.106, -0.080], 139,015 labelled-both edges, 6,312 threads). The effect is small (about 0.16 SD; disagree share 50.3% cross vs 45.1% same; weekly alignment excess about 2.3 points above the label-permutation null). | **Rejected in its strong form.** Alignment exists, but it is weak. Weak alignment on a graph with mean degree 3.3 is exactly the setting where the noise explanation predicts no measurable balance. |

**Current claim rung (CLAIMS.md): 2.** "The signed network is never worse than direct scoring and passes our pre-registered comparison rule, but neither measure demonstrably tracks the ground truths at the densities we could afford." Phase 2 decides whether it rises to rung 3 or 4.

### A cross-cutting finding: venue composition

The share of shared-venue edges that come from r/politics (versus r/PoliticalDiscussion) correlates -0.72 with GT#1 and -0.76 with GT#2, which is stronger than any pipeline. The same trend drives three Phase 1 results:
- Pipeline A's negative association: partial rho_A given the r/politics share is -0.25 on GT#1 [-0.58, +0.03] and +0.07 on GT#2 [-0.26, +0.26].
- The H-AN cross-camp share: s_t correlates 0.85 with the r/politics share.
- The weekly alignment-GT correlation: -0.61 / -0.68 raw, -0.14 / -0.25 given the r/politics share, about 0 given the cross-camp share (diagnostic only).

Pipeline B is not driven by it (partial rho_B given the r/politics share: -0.16 on GT#1, +0.13 on GT#2, both CIs spanning 0).

## Judgment calls made at CP1 (they replace the STATUS.md handoffs)

1. **H-AN wording.** The family H-AN has two Holm-corrected tests. (i) s_t vs GT#1: rho -0.66, adjusted p 0.13, **not significant**. (ii) Controlling for s_t shrinks |rho_A| by +0.27 [+0.04, +0.43], adjusted p 0.03, **significant**. H-AN is therefore **partially supported**, and the CLAIMS.md pass sentence ("as the camps mixed ... cross-camp replies are more hostile") is **not** used. The shift-share decomposition attributes only 3% of P_A's weekly variance to the changing mix of edge types, and s_t is mostly a proxy for venue composition. The paper says: "Pipeline A's negative association with the ground truths is a venue-composition effect. Weeks with more r/politics traffic score as more hostile and as less polarized. It is not explained by hostility on cross-camp replies, since the edge-type mix accounts for 3% of A's weekly variance."
2. **D-E5 headline after the composition baselines.** The D-E5 rule result stands as pre-registered and is reported unchanged. It is always reported next to the composition baseline, with the sentence: "B's advantage over A comes from A's composition-driven negative association, not from B tracking the ground truths. Neither pipeline beats a one-variable composition baseline." The same caveat attaches to every "C beats A" verdict (C1 and C2 beat A with Holm p ≤ 0.003, but neither beats B).
3. **H-AL holds at the edge level, but weekly alignment correlates negatively with GT.** These are not in conflict. H-AL is a pooled edge-level property (confirmatory, passes). Its week-to-week variation is diagnostic, and it is largely venue composition (see above). Report H-AL as the CLAIMS.md pass sentence with the effect size, plus "the effect is small (about 0.16 SD)". Report the weekly series only as a diagnostic, with its composition partials.
4. **Noise curve next to near-zero reliability (H3 resolved with 10 seeds).** H3 **passes**: averaged over 10 seeds, both camp recovery (0.518, 0.516, 0.514, 0.512, 0.511) and rho_B(GT#2) (0.177, 0.160, 0.110, 0.087, 0.059) fall strictly as p rises from 0 to 0.4 (Spearman over p = −1.0 for both; the exact one-sided p for n = 5 is 1/120, Holm-adjusted 0.017). The CLAIMS.md pass sentence is used with an effect-size qualifier: "B degrades monotonically as sign noise is injected, but the effect is small (camp recovery −0.007; rho_B(GT#2) −0.12, inside one seed SD at each level)." This is consistent with the noise explanation: P_B carries a small sign-dependent signal that noise erodes, and it sits close to the floor at 5%. That explains why cleaner signs (the ladder) cannot lift it, while degraded signs still pull it down.

## One paragraph per task

**P1.1 Spec completion (A1).** Family A (B vs A, 4 tests): 0 of 4 significant after Holm (smallest adjusted p 0.226). The composition baselines outperform both pipelines (r/politics share: rho -0.72 GT#1, -0.76 GT#2). The 06-27 split: no CI excludes 0 in the first half. In the second half rho_B(GT#2) is +0.42 [+0.06, +0.70] and rho_A(GT#1) is -0.47 [-0.75, -0.09] (exploratory). The bootstrap seed is stable (lower bound of the GT#2 delta CI 0.20-0.22 across seeds). `results/llm/final/rq1_spec_completion/`

**P1.2 Noise ceilings (H-NC, descriptive).** 20 thread splits, R = 20 (10 on WIN, 10 on MAC). Spearman-Brown reliability: P_A 0.56 [0.44, 0.69]; P_B -0.08 [-0.46, 0.25]. rel_B < 0.3, so the observed rho_B is bounded by noise. Disattenuated rho is defined for A only: -0.85 (GT#1), -0.72 (GT#2). `results/llm/final/noise_ceiling/`

**P1.3 Alignment (H-AL, confirmatory).** Passes: delta-w (cross - same) = -0.092 [-0.106, -0.080]. Secondary: binary LLM -0.154 [-0.178, -0.130], VADER -0.057 [-0.065, -0.048]; P(disagree) +0.051, P(hostility > none) +0.070, P(attack) +0.029, all CIs excluding 0; odds ratio of disagree on cross 1.32. LLM signs carry more alignment than sentiment. `results/llm/final/alignment/`

**P1.4 Why A runs negative (H-AN, confirmatory).** Partially supported (judgment 1 above). (i) -0.66, Holm p 0.13; (ii) +0.27 [+0.04, +0.43], Holm p 0.03; edge-type mix 3% of variance. `results/llm/final/a_negative/`

**P1.5 Pipeline C1 and C2 (family C, confirmatory).** Neither beats B. C1 on GT#1: delta +0.10 [-0.08, +0.31], Holm p 0.55. C2 on GT#1: +0.20 [-0.23, +0.40], Holm p 0.79. Both beat A (Holm p ≤ 0.003), with the caveat in judgment 2. Camp recovery: C1 0.516, C2 0.520, B 0.518. The C2 head was trained on w_C1 with OOF R² 0.29 and pilot accuracy 74.6%. Its quality report was logged before any GT comparison. CLAIMS.md fail sentence: "Richer edge information does not improve on the sign alone." `results/llm/final/pipeline_c/`, `data/llm/c2/c2_quality.json`

**P1.6 RQ3 sign quality.** The ladder (exploratory) is flat: no rung differs from B (all Holm p = 1.0), even though pilot accuracy runs from 48% to 79%. The noise curve (H3, confirmatory, 10 seeds per p) **passes**: camp recovery 0.518 → 0.511 and rho_B(GT#2) 0.18 → 0.06 both fall strictly with p (Spearman −1.0 each), with small effects (judgment 4). `results/llm/final/rq3/`, `results/llm/final/rq3_rq2_jobs/`

**P1.7 RQ2 conditions (exploratory).** Rolling 4-week windows: rho_B +0.09 [-0.33, +0.50] on GT#1, +0.17 [-0.24, +0.50] on GT#2. Calendar months (13 points): -0.02 / +0.31. Thinning to 25% / 50% of threads: rho_B 0.21 / 0.09 (GT#1), 0.24 / 0.17 (GT#2); no density gain at these levels. Top vs bottom GT quartiles: no difference in camp recovery. The discordant-ally share correlates +0.66 / +0.67 with GT, but it is the mirror image of the composition-driven alignment series. Rolling 2-week windows (52 points): rho_B +0.24 [+0.00, +0.49] on GT#1 (perm p 0.29), +0.40 [+0.15, +0.62] on GT#2 (perm p 0.10). This is the strongest B association in Phase 1; it is exploratory and not significant under the circular-shift permutation. `results/llm/final/rq2/`, `results/llm/final/rq3_rq2_jobs/`

**P1.8 RQ4 forecasting.** No model beats persistence. MAE on GT#1: 0.0170 persistence, 0.0176 A, 0.0173 B. A is significantly worse than persistence on GT#2 (DM p 0.036). CLAIMS.md fail sentence applies. It is the expected result. `results/llm/final/rq4/`

## What this means for Phase 2

The noise explanation is the one left standing, and the encoder is what makes the density ladder affordable. The ladder (1%, 5%, 20%, full) is the decisive test of H1a/H1b/H1c. If rho_B does not rise with density, the paper sits at rung 2 with a clear mechanism story: weak alignment, a sparse graph, and a composition confound that explains the text-scoring result.
