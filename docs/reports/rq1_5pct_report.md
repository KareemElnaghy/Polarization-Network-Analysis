# RQ1 at the 5% sample, partial coverage (auto-drafted 2026-11-01 05:07; needs human reading)

Spec: `docs/eval_spec.md` (eval-spec-v1) plus clarification D-E3 (logged before any 5% number). Primary model `qwen3.5:9b-q4_K_M`, prompt v2, frozen (D-M7).
Extraction run `runs/llm\20261031T232220_tier5pct`: **56775 of 617041 unique pairs processed (9.2% of the 5% tier)**,
thread-prefix ordering (whole threads, all weeks), 0 failures, state `running`.
Pipeline A run `runs/llm\20261101T043947_pipeA_5pct_ckpt2`; Pipeline B run `runs/llm\20261101T043954_pipeB_5pct_ckpt2`. Coverage: {'weeks_A': 53, 'weeks_B': 51, 'edges_A': 52665, 'edges_B_input': 52677, 'min_coverage_A': 0.006683719778859642}.

**This is a partial 5% sample: thinner than a true 5% tier, so power is limited. PRIMARY truth = full-week GT; the 5%-matched columns are secondary (computed on all 5% threads, not on the processed prefix).**

## Correlations with ground truth
| truth_set | truth | series | n_weeks | rho | perm_p | rho_diff | partial_rho_vol |
|---|---|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | P_A | 53 | -0.222 | 0.113 | -0.053 | -0.208 |
| PRIMARY-full | gt1_excess | P_B | 51 | -0.018 | 0.887 | 0.14 | -0.043 |
| PRIMARY-full | gt1_excess | volume_A | 53 | -0.149 | 0.491 | -0.124 | nan |
| PRIMARY-full | gt1_excess | volume_B | 53 | -0.149 | 0.491 | -0.124 | nan |
| PRIMARY-full | gt2 | P_A | 53 | -0.188 | 0.151 | -0.128 | -0.176 |
| PRIMARY-full | gt2 | P_B | 51 | 0.002 | 0.962 | 0.204 | -0.012 |
| PRIMARY-full | gt2 | volume_A | 53 | -0.117 | 0.585 | -0.108 | nan |
| PRIMARY-full | gt2 | volume_B | 53 | -0.117 | 0.585 | -0.108 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_A | 53 | -0.073 | 0.623 | 0.019 | -0.073 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | P_B | 51 | -0.121 | 0.396 | -0.226 | -0.122 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_A | 53 | -0.006 | 1.0 | 0.074 | nan |
| SECONDARY-5pct-matched | gt1_excess_0.05 | volume_B | 53 | -0.006 | 1.0 | 0.074 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | P_A | 52 | -0.067 | 0.66 | -0.016 | -0.058 |
| SECONDARY-5pct-matched | gt2_0.05 | P_B | 51 | -0.076 | 0.547 | -0.189 | -0.088 |
| SECONDARY-5pct-matched | gt2_0.05 | volume_A | 52 | -0.119 | 0.604 | 0.191 | nan |
| SECONDARY-5pct-matched | gt2_0.05 | volume_B | 52 | -0.119 | 0.604 | 0.191 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_A | 52 | -0.082 | 0.566 | 0.09 | -0.072 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | P_B | 51 | -0.105 | 0.377 | -0.129 | -0.121 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_A | 52 | -0.15 | 0.396 | 0.124 | nan |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | volume_B | 52 | -0.15 | 0.396 | 0.124 | nan |

## Delta rho (B minus A), 4-week block bootstrap
| truth_set | truth | delta_rho_B_minus_A | ci_lo | ci_hi | n_weeks_both |
|---|---|---|---|---|---|
| PRIMARY-full | gt1_excess | 0.308 | -0.137 | 0.654 | 51 |
| PRIMARY-full | gt2 | 0.269 | -0.187 | 0.589 | 51 |
| SECONDARY-5pct-matched | gt1_excess_0.05 | 0.011 | -0.376 | 0.322 | 51 |
| SECONDARY-5pct-matched | gt2_0.05 | 0.044 | -0.313 | 0.322 | 51 |
| SECONDARY-5pct-matched | gt2_alt_auc_0.05 | 0.028 | -0.363 | 0.277 | 51 |

A vs B correlation: 0.028.

## Decision (rule: B must beat A on GT#2 with the delta CI excluding 0 AND rho_B > 0)
- PRIMARY (full-week GT#2): **NO WINNER (delta CI includes 0)**
- SECONDARY (5%-matched GT#2): NO WINNER (delta CI includes 0)

## Camp recovery
See `runs/llm/*_pipeB_5pct_ckpt2` (eval_camp_recovery output in `logs_llm/overnight_eval_5pct_ckpt2.log`).

## Robustness model
Not run yet (extraction still going; robustness model is scored after the primary run stops).

## Honest reading (to be completed by a human)
- Apply the rule above literally; a pipeline that is only less negative is not a winner.
- Coverage is partial; weeks with thin data are flagged low_n and Pipeline B weeks can be NaN. Report the number of usable B weeks every time.
- A positive result here would still be one sample, one model, and 1 prompt; the Tier 0 null (docs/reports/rq1_tier0_b230_report.md) stays the baseline.
