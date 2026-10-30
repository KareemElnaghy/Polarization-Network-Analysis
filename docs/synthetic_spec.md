# Phase 4 spec: calibrated synthetic validation (written before running the checks, 2026-10-30)

**Purpose.** (a) Tool validation: does normalised frustration P_B behave sensibly on graphs with known structure? (b) Behaviour study: supporting evidence only. Nothing here is a result about Reddit; all claims come from real data.

**Calibration** (`results/llm/synthetic/calibration_weekly.csv`, from the sampled shared-venue undirected user-pair graphs): 5% sample: 480-9,400 nodes (median 4.2k), mean degree 2.7-4.1 (median 3.2), median degree 2, p99 19, giant share 0.89-0.97. 1% sample: 260-4,300 nodes, mean degree 2.0-3.4, giant 0.54-0.97. These are lower than the 4.7-7.4 mean degree in the brief (probably a different graph definition: directed or multi-edge); flagged in `docs/handoff_issues.md`. Degrees are resampled from a pooled empirical sequence (`degree_pool.npy`).

**Generator** (degree-corrected signed block model). Nodes get a degree propensity from the empirical degree pool and a camp (two camps, share `rep_share` in camp 1; `intermediate_share` of nodes have no camp). Endpoints are drawn proportional to propensity; an edge is within-camp with probability `homophily`. Edge sign: with probability `lam` (polarization) the sign is camp-aligned (+ within, - across), otherwise random (negative with probability 0.5). `discordant_share`: share of within-camp edges forced negative (allies fighting). `flip`: share of all edges whose sign is flipped independently. Edges touching a no-camp node get random signs. Weights are sign times Uniform(0.3, 1). Pairs are unique, no self-loops.

**Checks and pre-stated pass criteria** (graphs of about 4,000 nodes, mean degree 3.2, rep_share 0.5, homophily 0.5, unless varied; 5 replicates; headline P_B on the giant component; R_null = 20, restarts 10):
1. **Monotone in lam.** lam in {0, .1, ..., 1}: Spearman of mean P_B with lam >= 0.95 and no decrease larger than 0.03 between adjacent lam.
2. **Size/density flatness.** At lam = 0.5, n in {500, 2k, 10k, 40k} and mean degree in {2.5, 3.2, 5, 7}: raw F_obs varies by more than 10x across n; P_B coefficient of variation across n (at degree 3.2) < 0.15, and across degree reported (P_B is expected to depend on density, and this is shown, not hidden; no pass criterion).
3. **Separation from nulls.** At lam = 0: |mean P_B| < 0.05; at lam = 0.6: P_B > 0.2 and F_obs more than 5 null sd below the null mean, for both nulls (sign shuffle, rewiring).
4. **Heuristic vs exact on calibrated small graphs** (n 60-100, degrees from the pool, ILP with proved optimality): mean relative gap < 1%, max < 5%.
5. **Discordant allies.** discordant_share in {0, .1, .2, .3, .5} at lam = 0.8: Spearman of P_B with discordant_share <= -0.9.
6. **Imbalance.** rep_share in {0.5, 0.3, 0.14 (about 6:1), 0.05} at lam = 0.8: report P_B and planted-camp recovery (balanced accuracy); pass if P_B stays above 0.5 of its balanced-camps value at 6:1.
7. **Planted-camp recovery** at lam in {0.3, 0.6, 0.9}: balanced accuracy of the partition against the planted camps (synthetic labels, not Reddit labels); reported, with pass if > 0.8 at lam = 0.9.

Outputs: `results/llm/synthetic/` (CSV of every replicate, plots with their CSVs, `checks.md` pass/fail table).
