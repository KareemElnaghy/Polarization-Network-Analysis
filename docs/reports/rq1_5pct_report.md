# RQ1 at the 5% sample, partial coverage (auto-drafted 2026-11-01 07:48; needs human reading)

Spec: `docs/eval_spec.md` (eval-spec-v1) plus clarification D-E3 (logged before any 5% number). Primary model `qwen3.5:9b-q4_K_M`, prompt v2, frozen (D-M7).
Extraction run `runs/llm\20261031T232220_tier5pct`: **82575 of 617041 unique pairs processed (13.4% of the 5% tier)**,
thread-prefix ordering (whole threads, all weeks), 0 failures, state `running`.
Pipeline A run `runs/llm\20261101T070735_pipeA_5pct_ckpt3`; Pipeline B run `runs/llm\20261101T070746_pipeB_5pct_ckpt3`. Coverage: {'weeks_A': 53, 'weeks_B': 52, 'edges_A': 76678, 'edges_B_input': 76698, 'min_coverage_A': 0.006746626686656672}.

**This is a partial 5% sample: thinner than a true 5% tier, so power is limited. PRIMARY truth = full-week GT; the 5%-matched columns are secondary (computed on all 5% threads, not on the processed prefix).**

## Correlations with ground truth
| truth_set | truth | series | n_weeks | rho | perm_p | rho_diff | partial_rho_vol |
|---|---|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | P_A | 53 | -0.369 | 0.038 | -0.222 | -0.37 |
| PRIMARY-full | gt1_excess | P_B | 52 | 0.047 | 0.774 | 0.136 | 0.046 |
| PRIMARY-full | gt1_excess | volume_A | 53 | -0.017 | 0.925 | 0.154 | nan |
| PRIMARY-full | gt1_excess | volume_B | 53 | -0.017 | 0.925 | 0.154 | nan |
| PRIMARY-full | gt2 | P_A | 53 | -0.345 | 0.019 | -0.113 | -0.346 |
| PRIMARY-full | gt2 | P_B | 52 | 0.042 | 0.868 | -0.018 | 0.041 |
| PRIMARY-full | gt2 | volume_A | 53 | -0.014 | 0.943 | -0.032 | nan |
| PRIMARY-full | gt2 | volume_B | 53 | -0.014 | 0.943 | -0.03 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_A | 53 | -0.105 | 0.509 | 0.151 | -0.102 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_B | 52 | 0.051 | 0.755 | -0.067 | 0.05 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_A | 53 | -0.026 | 0.887 | -0.149 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_B | 53 | -0.026 | 0.887 | -0.15 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | P_A | 52 | -0.196 | 0.321 | 0.139 | -0.199 |
| SECONDARY-5pct-matched | gt2_0.05 | P_B | 52 | -0.013 | 0.962 | -0.276 | -0.012 |
| SECONDARY-5pct-matched | gt2_0.05 | volume_A | 52 | 0.033 | 0.868 | 0.202 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | volume_B | 52 | 0.033 | 0.868 | 0.202 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_A | 52 | -0.21 | 0.226 | 0.141 | -0.21 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_B | 52 | -0.146 | 0.491 | -0.25 | -0.146 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_A | 52 | -0.009 | 1.0 | 0.216 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_B | 52 | -0.009 | 1.0 | 0.215 | nan |

## Delta rho (B minus A), 4-week block bootstrap
| truth_set | truth | delta_rho_B_minus_A | ci_lo | ci_hi | n_weeks_both |
|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | 0.48 | 0.015 | 0.83 | 52 |
| PRIMARY-full | gt2 | 0.415 | -0.011 | 0.739 | 52 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | 0.183 | -0.268 | 0.49 | 52 |
| SECONDARY-5pct-matched | gt2_0.05 | 0.183 | -0.245 | 0.459 | 52 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | 0.064 | -0.377 | 0.37 | 52 |

A vs B correlation: 0.109.

## Decision (rule: B must beat A on GT#2 with the delta CI excluding 0 AND rho_B > 0)
- PRIMARY (full-week GT#2): **NO WINNER (delta CI includes 0)**
- SECONDARY (5%-matched GT#2): NO WINNER (delta CI includes 0)

## Camp recovery
See `runs/llm/*_pipeB_5pct_ckpt3` (eval_camp_recovery output in `logs_llm/overnight_eval_5pct_ckpt3.log`).

## Robustness model
Not run yet (extraction still going; robustness model is scored after the primary run stops).

## Honest reading (to be completed by a human)
- Apply the rule above literally; a pipeline that is only less negative is not a winner.
- Coverage is partial; weeks with thin data are flagged low_n and Pipeline B weeks can be NaN. Report the number of usable B weeks every time.
- A positive result here would still be one sample, one model, and 1 prompt; the Tier 0 null (docs/reports/rq1_tier0_b230_report.md) stays the baseline.
