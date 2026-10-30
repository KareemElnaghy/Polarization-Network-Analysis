# Synthetic validation checks (criteria: docs/synthetic_spec.md)

| check | observed | pass |
|---|---|---|
| 1 monotone in lam | Spearman 0.909; max adjacent drop 0.006 | FAIL |
| 2 size flatness (lam .5) | P_B by n {500: 0.036, 2000: 0.055, 4000: 0.037, 10000: 0.007, 40000: 0.001}; CV 0.838; raw F ratio max/min 98.9 | FAIL |
| 2 density (reported only) | P_B by mean degree {2.5: 0.019, 3.2: 0.037, 5.0: 0.123, 7.0: 0.188} | PASS |
| 3 separation from nulls | lam0 P_B 0.004; lam.6 P_B 0.190, z vs shuffle 25.5, P_B_rewire 0.195 | FAIL |
| 4 heuristic vs exact (calibrated) | proved 10/10; mean gap 0.0000; max 0.0000 | PASS |
| 5 discordant allies | P_B by share {0.0: 0.548, 0.1: 0.363, 0.2: 0.215, 0.3: 0.044, 0.5: -0.004}; Spearman -1.000 | PASS |
| 6 imbalance | {0.05: {'P_B': 0.588, 'BA': 0.952}, 0.14: {'P_B': 0.568, 'BA': 0.937}, 0.3: {'P_B': 0.559, 'BA': 0.926}, 0.5: {'P_B': 0.548, 'BA': 0.928}} | PASS |
| 7 planted-camp recovery | BA by lam {0.3: 0.517, 0.6: 0.799, 0.9: 0.967} | PASS |

replicates per cell: 5; runtime 790 s
