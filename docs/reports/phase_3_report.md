# Phase 3 report: Pipeline B and structural measures

**Built:** sign sources (continuous, binary, VADER sentiment, unsigned, noise wrapper); weekly undirected signed graph with giant component; frustration heuristic (spectral + annealing + greedy, numba) with an exact ILP for validation; P_B = 1 - F_obs/F_null with sign-shuffle null and a signed-degree-preserving rewiring null; camp assignments; separate camp-recovery module (the only reader of labels); runner scripts `run_pipeline_b.py`, `eval_camp_recovery.py`, `bench_frustration.py`.
**Tests (9 in `tests/test_pipeline_b.py`, all pass; full suite also run):** balanced two-camp graph has F=0 and P_B=1 and its camps are recovered; all-positive graph has P_B=0; random signs give P_B near 0; heuristic never below the exact optimum and optimal on most tiny graphs; sign shuffle keeps topology and weight multiset; rewiring keeps every node's positive and negative degree, no self-loops or duplicates; sign-source formulas; camp recovery on planted and random partitions; no pipeline module touches labels or the eval module.

## Optimality-gap table (`results/llm/bench/frustration_gap.csv`)
16 random-sign graphs, 36-77 nodes (hardest case, equal to the null). The ILP proved optimality in 16/16. Heuristic equals the optimum in 14/16 (88%); mean relative gap 0.08%; worst 1.2%. Not verifiable at 40k+ nodes.

## Cost (`results/llm/bench/frustration_cost.csv`, synthetic graphs with mean degree 5-6)
| nodes | seconds per frustration run (10 restarts) | seconds per rewire |
|---|---|---|
| 2,000 | 0.34 | 0.24 |
| 10,000 | 0.29 | 1.3 |
| 40,000 | 0.63 | 6.9 |
| 150,000 | 3.0 | 28 |

Projection for all weekly graphs (continuous source: 1 observed + 50 nulls + 25 full-graph runs + 10 rewires; other sources: 20 nulls, no rewires): 5% tier (about 40k nodes/week) about 1.5 h for the continuous source and about 0.5 h each for the others (about 3 h total); 20% tier (about 150k nodes) about 7 h for the continuous source, dominated by the pure-Python rewiring (28 s per rewire), which should be moved to numba before the 20% tier. Tier 0 graphs are tiny, so cost there is seconds.

## Plumbing check on real partial data
Run `runs/llm/20261030T235157_pipeB_plumbing` (854 extracted edges, thresholds lowered to 8 pairs / 6 nodes, only to exercise the code): all four sources ran in 17 s. P_B values there are meaningless (graphs of about 8 nodes). Camp recovery returned no weeks (needs >= 5 labelled users per side), so **there is no first camp-recovery number yet**.

## Concerns for Tier 0
Tier 0 has a median of about 104 edges per week, so weekly graphs are small and tree-like. Frustration only exists on cycles, so P_B will be noisy and many weeks may be NaN or near 0 (threshold: 40 user pairs). This is a plumbing result; a credible P_B needs the 5% tier or more threads per week (the Tier 0 thread order extends in place).

**Decisions:** D-B1 to D-B5. **Open:** exact Guo normalisation and pair aggregation (TODO(verify)); heuristic gap at scale; rewiring speed.
