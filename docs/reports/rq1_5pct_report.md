# RQ1 at the 5% sample, partial coverage (auto-drafted 2026-11-01 02:39; needs human reading)

Spec: `docs/eval_spec.md` (eval-spec-v1) plus clarification D-E3 (logged before any 5% number). Primary model `qwen3.5:9b-q4_K_M`, prompt v2, frozen (D-M7).
Extraction run `runs/llm\20261031T232220_tier5pct`: **32525 of 617041 unique pairs processed (5.3% of the 5% tier)**,
thread-prefix ordering (whole threads, all weeks), 0 failures, state `running`.
Pipeline A run `runs/llm\20261101T022309_pipeA_5pct_ckpt1`; Pipeline B run `runs/llm\20261101T022314_pipeB_5pct_ckpt1`. Coverage: {'weeks_A': 53, 'weeks_B': 46, 'edges_A': 30033, 'edges_B_input': 30043, 'min_coverage_A': 0.0002443494196701283}.

**This is a partial 5% sample: thinner than a true 5% tier, so power is limited. PRIMARY truth = full-week GT; the 5%-matched columns are secondary (computed on all 5% threads, not on the processed prefix).**

## Correlations with ground truth
| truth_set | truth | series | n_weeks | rho | perm_p | rho_diff | partial_rho_vol |
|---|---|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | P_A | 53 | -0.227 | 0.17 | -0.042 | -0.229 |
| PRIMARY-full | gt1_excess | P_B | 46 | 0.17 | 0.132 | 0.347 | 0.202 |
| PRIMARY-full | gt1_excess | volume_A | 53 | 0.06 | 0.717 | -0.115 | nan |
| PRIMARY-full | gt1_excess | volume_B | 53 | 0.06 | 0.717 | -0.116 | nan |
| PRIMARY-full | gt2 | P_A | 53 | -0.189 | 0.208 | -0.087 | -0.198 |
| PRIMARY-full | gt2 | P_B | 46 | 0.067 | 0.585 | 0.24 | 0.172 |
| PRIMARY-full | gt2 | volume_A | 53 | 0.153 | 0.415 | -0.043 | nan |
| PRIMARY-full | gt2 | volume_B | 53 | 0.153 | 0.415 | -0.043 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_A | 53 | -0.118 | 0.396 | 0.029 | -0.122 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_B | 46 | 0.005 | 0.962 | -0.205 | 0.044 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_A | 53 | 0.101 | 0.528 | 0.146 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_B | 53 | 0.101 | 0.528 | 0.145 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | P_A | 52 | -0.096 | 0.604 | -0.035 | -0.095 |
| SECONDARY-5pct-matched | gt2_0.05 | P_B | 46 | 0.04 | 0.849 | -0.137 | 0.089 |
| SECONDARY-5pct-matched | gt2_0.05 | volume_A | 52 | 0.083 | 0.736 | 0.285 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | volume_B | 52 | 0.083 | 0.736 | 0.285 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_A | 52 | -0.078 | 0.736 | 0.114 | -0.078 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_B | 46 | 0.01 | 0.925 | -0.116 | 0.077 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_A | 52 | 0.089 | 0.642 | 0.236 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_B | 52 | 0.089 | 0.642 | 0.236 | nan |

## Delta rho (B minus A), 4-week block bootstrap
| truth_set | truth | delta_rho_B_minus_A | ci_lo | ci_hi | n_weeks_both |
|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | 0.482 | 0.079 | 0.802 | 46 |
| PRIMARY-full | gt2 | 0.315 | -0.169 | 0.686 | 46 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | 0.201 | -0.188 | 0.451 | 46 |
| SECONDARY-5pct-matched | gt2_0.05 | 0.185 | -0.192 | 0.485 | 46 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | 0.123 | -0.25 | 0.438 | 46 |

A vs B correlation: 0.231.

## Decision (rule: B must beat A on GT#2 with the delta CI excluding 0 AND rho_B > 0)
- PRIMARY (full-week GT#2): **NO WINNER (delta CI includes 0)**
- SECONDARY (5%-matched GT#2): NO WINNER (delta CI includes 0)

## Camp recovery
See `runs/llm/*_pipeB_5pct_ckpt1` (eval_camp_recovery output in `logs_llm/overnight_eval_5pct_ckpt1.log`).

## Robustness model
Not run yet (extraction still going; robustness model is scored after the primary run stops).

## Honest reading (to be completed by a human)
- Apply the rule above literally; a pipeline that is only less negative is not a winner.
- Coverage is partial; weeks with thin data are flagged low_n and Pipeline B weeks can be NaN. Report the number of usable B weeks every time.
- A positive result here would still be one sample, one model, and 1 prompt; the Tier 0 null (docs/reports/rq1_tier0_b230_report.md) stays the baseline.
