# Numbers registry: every number to use on slides, with its source

*Written for: Kareem and the Claude Code session building the deck. Paths are relative to the repo root; copies of the cited results are inside this folder under `results/`, `reports/`, `pipeline_outputs/`. "Final" = the 135,678-edge evaluation of 2026-11-01. Fills the empty `docs/midway_numbers.md`.*

## A. RQ1 final result (5% prefix)
| Number | Value | Source |
|---|---|---|
| Edges processed (A and B input) | 135,678 (unique pairs in store about 135.6k of 617,041 = 22%) | `results/llm/rq1_5pct/5pct_final/rq1_5pct_final_summary.json` (coverage) |
| Threads / weeks | 2,885 threads / 53 weeks (A and B both 53) | `runs/llm/20261101T131638_pipeA_5pct_final` edge_scores; summary.json |
| Median / min / max edges per week | 2,512 / 275 (partial first week) / 5,788 | `pipeline_outputs/pipelineA_final/weekly_A.csv` (column n) |
| P_A mean, sd, range | 3.08, 0.30, 2.51 to 4.00 | same |
| Agreement labels | 52% disagree, 30% agree, 17% neutral | edge_scores.parquet (run A final) |
| Pipeline B graph size | giant component median about 1,000 nodes, about 1,450 pairs; 61.6% of pair weights negative | `pipeline_outputs/pipelineB_final/weekly_B.csv` |
| P_B (continuous) mean, sd, positive weeks | -0.001, 0.11, 26 of 53 | same |
| rho P_A vs GT#1 / GT#2 | -0.44 (p 0.075) / -0.38 (p 0.113) | `rq1_5pct_final_correlations.csv` |
| rho P_B vs GT#1 / GT#2 | -0.05 (p 0.811) / +0.07 (p 0.755) | same |
| rho on first differences A / B (GT#2) | -0.15 / -0.25 | same (rho_diff) |
| Partial rho controlling for volume | A -0.46 / -0.35; B -0.05 / +0.06 | same (partial_rho_vol) |
| Volume-only rho vs GT#1 / GT#2 | +0.00 / -0.17 (n.s.) | same |
| delta rho (B-A), GT#2 full-week (deciding) | +0.44, CI [0.06, 0.70] | `rq1_5pct_final_delta.csv` |
| delta rho (B-A), GT#1 full-week | +0.39, CI [0.02, 0.65] | same |
| delta rho 5%-matched (secondary) | GT#1 +0.22 [-0.11, 0.47]; GT#2 +0.29 [-0.13, 0.52]; GT#2 alt AUC +0.22 [-0.19, 0.48] | same |
| A vs B weekly rho | 0.27 | summary.json |
| Rule verdict | primary: B wins on GT#2 literally; secondary: no winner | summary.json |
| Bootstrap stability (GT#2 lower CI bound, 5 seeds x 10k) | 0.058 to 0.072 | `5pct_final/bootstrap_stability.txt` |
| Partial-week sensitivity | dropping first/last week moves rho/delta by <= 0.05 | `5pct_final/partial_week_sensitivity.txt` |

## B. Checkpoint trajectory (GT#2, full-week)
| Edges | rho_A | rho_B | delta [CI] | B weeks | Source |
|---|---|---|---|---|---|
| 30,033 | -0.19 | +0.07 | +0.31 [-0.17, 0.69] | 46 | `results/llm/rq1_5pct/5pct_ckpt1` |
| 52,665 | -0.19 | 0.00 | +0.27 [-0.19, 0.59] | 51 | ckpt2 |
| 76,678 | -0.34 | +0.04 | +0.41 [-0.01, 0.74] | 52 | ckpt3 |
| 101,906 | -0.37 | +0.04 | +0.41 [-0.02, 0.70] | 53 | ckpt4 |
| 135,678 | -0.38 | +0.07 | +0.44 [0.06, 0.70] | 53 | 5pct_final |

## C. Camp recovery (final)
| Sign source | balanced accuracy | null | weeks above p95 | Source |
|---|---|---|---|---|
| LLM continuous | 0.531 | 0.524 | 8/53 | `pipeline_outputs/pipelineB_final/camp_recovery.csv`, `logs/final_evaluation.log` |
| LLM binary | 0.540 | 0.527 | 9/53 | same |
| VADER | 0.534 | 0.526 | 6/53 | same |
| Unsigned | 0.500 | 0.500 | 0/53 | same |
| Labelled users per week | about 450 (continuous) | same |

## D. Tier 0 (1% sample, 230 edges/week)
24,037 edges, 0 failed; B in 52 of 53 weeks (33 at 60/wk); rho A -0.43 / -0.37, B +0.11 / +0.12; delta GT#1 +0.54 [0.20, 0.85], GT#2 +0.49 [0.10, 0.83]; A vs B 0.11; camp recovery 0.597 vs null 0.562 (8 of 43 weeks above p95). Source: `reports/rq1_tier0_b230_report.md`. The 60/wk run: 12,733 edges, 4.75 h, `reports/tier0_real_data_report.md`. (The Tier 0 run folders live on the Mac, not here.)

## E. Model and pilot
Primary `qwen3.5:9b-q4_K_M`, prompt v2, think off, logprobs. Pilot 150 edges (83 disagree, 55 agree, 12 neutral); llama3.1:8b v1 0.587 / kappa 0.205; v2 dev 0.627 / 0.307; qwen dev 0.733 / 0.548 / AUC 0.87; **held-out 0.707 / kappa 0.439**; combined 108/150 = 0.72. Source: `docs/llm_decisions.md` (D-P2, D-M6, D-M7), `reports/phase_1_report.md`. gemma3:12b pulled (8.2 GB), not scored.

## F. Throughput
| Number | Value | Source |
|---|---|---|
| Mac | 0.64 to 0.81 edges/s; 5% = about 245 h | `reports/phase_1_report.md` |
| RTX 3090 benchmark, 4 / 8 / 16 server slots | 1.30 / 1.31 / 1.30 edges/s (200 edges); VRAM about 10 GB | `results/llm/bench/qwen3.5_9b-q4_K_M_sweep{4,8,16}.csv` |
| Real 5% run speed | 2.5 to 2.9 pairs/s | `logs/overnight_supervisor.log` |
| Whole 5% tier | 617,041 unique pairs (617,568 edges): about 55 to 60 h at that speed | `status.json`, log |
| 1% / 5% / 20% shared-venue edges | 152,496 / 617,633 / 2,736,934 | `reports/phase_1_report.md` |
| Crash | 1 HTTP-500 at about 98,050 pairs (09:20), restart 09:26, no loss | `logs/overnight_supervisor.log`, `docs/LOG.md` |

## G. Data and ground truth (Yasmina)
42.4M comments / 855k users / 28.1M reply edges; 53 weeks; labelled users 100,108 (35,244 Dem, 64,864 Rep); about 80% label vs self-declared vote agreement; GT#1 mean 0.174 (0.109 to 0.246), split-half 0.86, composition explains about 79%, drop about 0.21 -> 0.15 near 2016-06-27; GT#2 mean 0.514 vs null 0.500, AUC about 0.70; GT#1 vs GT#2 rho 0.80; 1% sample fidelity 0.38, 5% 0.72. Sources: `results/ground_truth/summary.md`, `docs/PRESENTATION_GUIDE.md` section 4, `docs/handoff_issues.md`.

## H. Synthetic validation
6 of 8 checks pass; monotone rho 0.964; planted-camp balanced accuracy 0.51 / 0.81 / 0.97 at lam .3 / .6 / .9; heuristic optimal 10/10; failing: size flatness CV 0.46 (bar 0.15), separation P_B 0.172 (bar 0.2; 26.8 null-sd below shuffle null). P_B at real density 0.02 to 0.06. Source: `results/llm/synthetic/checks.md`, `reports/phase_4_report.md`.

## I. Engineering facts
Tests: 21 of 22 pass on Windows (1 Unix-only `termios` test). Decisions logged: D-E3 (rule clarification), D-E4 (final reporting), D-T2 (throughput), D-O1/D-O2 (overnight run). Branch `llm_implementation`, last commit of results: `022dba0`.
