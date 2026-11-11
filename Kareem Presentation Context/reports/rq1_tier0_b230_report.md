# RQ1 first look: Tier 0 at 230 edges/week (weak-power, not confirmatory)

Spec: `docs/eval_spec.md` (written before the comparison). Extraction `runs/llm/20261031T182808_tier0_b230` (24,037 edges, 0 failed); A `20261031T193814_pipeA_b230`; B `20261031T193815_pipeB_b230`. GT = full-week `ground_truth_weekly.csv`. B usable in 52 of 53 weeks (33 at 60 edges/week).

| Series vs truth | Spearman rho (levels) | circular-shift p |
|---|---|---|
| P_A vs GT#1 | -0.43 | 0.02 |
| P_A vs GT#2 | -0.37 | 0.06 |
| P_B vs GT#1 | +0.11 | 0.14 |
| P_B vs GT#2 | +0.12 | 0.18 |
| edge volume vs GT#1 / GT#2 | +0.04 / -0.01 | n.s. |

First differences: A -0.11 / -0.05; B +0.10 / +0.19 (n = 51). Partial rho controlling for volume: unchanged. A vs B: 0.11.
Delta = rho_B - rho_A (4-week block bootstrap): GT#1 +0.54 [0.20, 0.85]; GT#2 +0.49 [0.10, 0.83].
Camp recovery (continuous): balanced accuracy 0.597 vs null 0.562, above p95 in 8 of 43 weeks; VADER 0.561 (no signal); unsigned 0.500.

## Reading
- Neither pipeline tracks either truth: B is positive but indistinguishable from zero; A is **negatively** related to both. Delta > 0 is driven by A going the wrong way, not by B succeeding. By the pre-stated rule, no winner.
- A and B are nearly uncorrelated (0.11): they measure different things at this sample size.
- The negative A correlation is unexplained; candidates are the shared composition trend in the GTs (spring-to-summer fall) and sampling noise. Not investigated.
- Not done (spec): Holm, composition baselines, pre/post 06-27 split, noise ceilings, 5% tier. 1% samples are weak against both truths.
