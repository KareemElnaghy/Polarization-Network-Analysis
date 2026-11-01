# RQ1 at the 5% sample, partial coverage (auto-drafted 2026-11-01 10:36; needs human reading)

Spec: `docs/eval_spec.md` (eval-spec-v1) plus clarification D-E3 (logged before any 5% number). Primary model `qwen3.5:9b-q4_K_M`, prompt v2, frozen (D-M7).
Extraction run `runs/llm\20261101T092612_tier5pct`: **11225 of 518971 unique pairs processed (2.2% of the 5% tier)**,
thread-prefix ordering (whole threads, all weeks), 0 failures, state `running`.
Pipeline A run `runs/llm\20261101T094910_pipeA_5pct_ckpt4`; Pipeline B run `runs/llm\20261101T094916_pipeB_5pct_ckpt4`. Coverage: {'weeks_A': 53, 'weeks_B': 53, 'edges_A': 101906, 'edges_B_input': 101919, 'min_coverage_A': 0.03074489304628929}.

**This is a partial 5% sample: thinner than a true 5% tier, so power is limited. PRIMARY truth = full-week GT; the 5%-matched columns are secondary (computed on all 5% threads, not on the processed prefix).**

## Correlations with ground truth
| truth_set | truth | series | n_weeks | rho | perm_p | rho_diff | partial_rho_vol |
|---|---|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | P_A | 53 | -0.416 | 0.057 | -0.123 | -0.416 |
| PRIMARY-full | gt1_excess | P_B | 53 | -0.018 | 0.962 | -0.165 | -0.023 |
| PRIMARY-full | gt1_excess | volume_A | 53 | -0.027 | 0.83 | 0.302 | nan |
| PRIMARY-full | gt1_excess | volume_B | 53 | -0.027 | 0.83 | 0.307 | nan |
| PRIMARY-full | gt2 | P_A | 53 | -0.37 | 0.038 | -0.177 | -0.372 |
| PRIMARY-full | gt2 | P_B | 53 | 0.04 | 0.83 | -0.297 | 0.014 |
| PRIMARY-full | gt2 | volume_A | 53 | -0.172 | 0.358 | -0.077 | nan |
| PRIMARY-full | gt2 | volume_B | 53 | -0.172 | 0.358 | -0.07 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_A | 53 | -0.115 | 0.491 | 0.275 | -0.113 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_B | 53 | 0.117 | 0.358 | 0.079 | 0.101 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_A | 53 | -0.114 | 0.434 | -0.156 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_B | 53 | -0.114 | 0.434 | -0.154 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | P_A | 52 | -0.245 | 0.17 | 0.048 | -0.235 |
| SECONDARY-5pct-matched | gt2_0.05 | P_B | 52 | 0.077 | 0.679 | -0.102 | 0.059 |
| SECONDARY-5pct-matched | gt2_0.05 | volume_A | 52 | -0.178 | 0.34 | -0.06 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | volume_B | 52 | -0.178 | 0.34 | -0.06 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_A | 52 | -0.257 | 0.151 | 0.073 | -0.248 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_B | 52 | -0.067 | 0.66 | -0.157 | -0.086 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_A | 52 | -0.166 | 0.453 | 0.017 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_B | 52 | -0.166 | 0.453 | 0.019 | nan |

## Delta rho (B minus A), 4-week block bootstrap
| truth_set | truth | delta_rho_B_minus_A | ci_lo | ci_hi | n_weeks_both |
|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | 0.398 | 0.026 | 0.698 | 53 |
| PRIMARY-full | gt2 | 0.41 | -0.024 | 0.696 | 53 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | 0.232 | -0.161 | 0.509 | 53 |
| SECONDARY-5pct-matched | gt2_0.05 | 0.322 | -0.122 | 0.611 | 52 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | 0.191 | -0.269 | 0.495 | 52 |

A vs B correlation: 0.265.

## Decision (rule: B must beat A on GT#2 with the delta CI excluding 0 AND rho_B > 0)
- PRIMARY (full-week GT#2): **NO WINNER (delta CI includes 0)**
- SECONDARY (5%-matched GT#2): NO WINNER (delta CI includes 0)

## Camp recovery
See `runs/llm/*_pipeB_5pct_ckpt4` (eval_camp_recovery output in `logs_llm/overnight_eval_5pct_ckpt4.log`).

## Robustness model
Not run yet (extraction still going; robustness model is scored after the primary run stops).

## Honest reading (to be completed by a human)
- Apply the rule above literally; a pipeline that is only less negative is not a winner.
- Coverage is partial; weeks with thin data are flagged low_n and Pipeline B weeks can be NaN. Report the number of usable B weeks every time.
- A positive result here would still be one sample, one model, and 1 prompt; the Tier 0 null (docs/reports/rq1_tier0_b230_report.md) stays the baseline.
