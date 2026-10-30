# Synthetic validation checks (criteria: docs/synthetic_spec.md)

| check | observed | pass |
|---|---|---|
| 1 monotone in lam | Spearman 0.964; max adjacent drop 0.005 | PASS |
| 2 size flatness (lam .5) | P_B by n {500: 0.036, 2000: 0.051, 4000: 0.06, 10000: 0.056, 40000: 0.011}; CV 0.461; raw F ratio max/min 89.5 | FAIL |
| 2 density (reported only) | P_B by mean degree {2.5: 0.02, 3.2: 0.06, 5.0: 0.12, 7.0: 0.172} | PASS |
| 3 separation from nulls | lam0 P_B 0.002; lam.6 P_B 0.172, z vs shuffle 26.8, P_B_rewire 0.178 | FAIL |
| 4 heuristic vs exact (calibrated) | proved 10/10; mean gap 0.0000; max 0.0000 | PASS |
| 5 discordant allies | P_B by share {0.0: 0.531, 0.1: 0.339, 0.2: 0.193, 0.3: 0.079, 0.5: -0.002}; Spearman -1.000 | PASS |
| 6 imbalance | {0.05: {'P_B': 0.578, 'BA': 0.954}, 0.14: {'P_B': 0.553, 'BA': 0.939}, 0.3: {'P_B': 0.543, 'BA': 0.926}, 0.5: {'P_B': 0.531, 'BA': 0.928}} | PASS |
| 7 planted-camp recovery | BA by lam {0.3: 0.514, 0.6: 0.809, 0.9: 0.969} | PASS |

replicates per cell: 5; runtime 2488 s
