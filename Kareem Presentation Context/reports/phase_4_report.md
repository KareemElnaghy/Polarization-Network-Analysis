# Phase 4 report: calibrated synthetic validation

Spec and pass criteria were fixed before running: `docs/synthetic_spec.md`. Calibration from the sampled shared-venue user-pair graphs (`results/llm/synthetic/calibration_weekly.csv`). Plots with their CSVs: `results/llm/synthetic/run2/`. Both runs kept (`run1_before_budget_fix/`, `run2/`). This is tool validation and a behaviour study; it supports no claim about Reddit.

## Outcome (same criteria, two runs)
| Check | Run 1 | Run 2 (final tool) |
|---|---|---|
| 1 monotone in polarization lam | FAIL (Spearman 0.909) | **PASS** (0.964; max drop 0.005) |
| 2 flat across graph size at lam .5 | FAIL (CV 0.84) | **FAIL** (CV 0.46) |
| 2 density (reported only) | P_B 0.019 / 0.037 / 0.12 / 0.19 at mean degree 2.5 / 3.2 / 5 / 7 | 0.020 / 0.060 / 0.12 / 0.17 |
| 3 separation from nulls | FAIL (P_B .19 at lam .6) | **FAIL** (P_B 0.172 < 0.2; 26.8 null sd below shuffle null; rewiring null agrees, P_B_rewire 0.178; lam 0 gives 0.002) |
| 4 heuristic vs exact, calibrated | PASS | **PASS** (optimal 10/10) |
| 5 discordant allies | PASS | **PASS** (Spearman -1.0; P_B 0.53 to 0 as share goes 0 to .5) |
| 6 imbalance | PASS | **PASS** (6:1 gives 0.553 vs 0.531 balanced; camp BA 0.94) |
| 7 planted-camp recovery | PASS | **PASS** (BA 0.51 / 0.81 / 0.97 at lam .3 / .6 / .9) |

**Final: 6 of 8 pass (counting the report-only density line as a pass); 2 fail on the pre-stated criteria.** Criteria were not changed after seeing results.

## What the failures mean
- **Run 1 failures were mostly the tool.** `scripts/llm/synth_diag_scale.py` showed the annealing was under-converged on weakly polarized graphs above about 4k nodes (P_B 0.019 vs 0.080 at 10k nodes, lam .5, with 5000 sweeps). Strongly polarized graphs (lam .8) were already converged. Fix: 5000 sweeps up to 15k nodes (D-B3 amended); null draws parallelised. Check 1 then passes.
- **Check 2 (size) still fails, narrowly informative.** At lam .5, P_B by nodes is 0.036 / 0.051 / 0.060 / 0.056 / 0.011 (500 / 2k / 4k / 10k / 40k). The 40k value is likely still under-converged (sweeps drop to 1875 there). For 500-10k nodes (the real range: median 4.2k, max 9.4k at 5%) the spread is 0.036-0.060, CV about 0.19, still above the 0.15 bar, and the quantity is small, so relative spread is large. Raw frustration varies 90x over the same range, so the normalisation does remove the size effect to first order.
- **Check 3 fails by an absolute threshold (0.2) that I set arbitrarily.** The measure separates polarized from random graphs strongly (26.8 sd), but the absolute level of P_B is low in sparse graphs. If kept, a dated spec v2 should phrase this criterion by the z-score against the null, not by the level; I have not applied that (it would be a post-hoc change).

## Findings that matter for Reddit
1. **P_B depends strongly on mean degree.** Real shared-venue graphs have mean degree 2.6 (1% sample) to 3.2 (5%); at that density P_B is 0.02-0.06 even for lam .5. Compare weeks by rank, not by level, and expect noise.
2. **Mild polarization is invisible.** P_B is about 0 for lam <= 0.4 and rises after that; planted-camp recovery is chance at lam .3 and 0.81 at .6.
3. **Imbalance is not a problem** for the measure at 6:1.
4. **Discordant allies lower P_B as intended**, which is the behaviour the proposal was worried about.
5. **Heuristic gap at scale is unverified**: exact checks cover only 36-100 nodes; convergence at 10k+ nodes is checked only against a longer anneal.

## Limitations
Generator is a stylised block model (two camps, random-or-aligned signs, no thread structure, no temporal dynamics, no text); mean degree of the real graphs is lower than the 4.7-7.4 in the brief (probably another graph definition; to confirm with Yasmina); only 5 replicates per cell.

**Open for you:** accept the two failures as documented, or approve a dated `synthetic_spec` v2 with a z-score based separation criterion and a size-flatness bar defined over 500-10k nodes only.
