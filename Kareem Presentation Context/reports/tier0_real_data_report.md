# Tier 0 real-data run (plumbing result; no ground-truth comparison made)

Extraction run `runs/llm/20261030T233105_tier0_b60`: qwen3.5:9b-q4_K_M, prompt v2, seed 0, 1% shared sample, whole threads, 60 edges/week budget. 12,733 edges, 12,720 unique pairs, 220 threads, all 53 weeks. **0 failures**, 4.75 h, 4% of texts truncated. Pipeline A run `20261031T103856_pipeA_tier0`; Pipeline B run `20261031T103858_pipeB_tier0` (continuous, binary, VADER, unsigned). Not compared with GT#1 or GT#2: the evaluation spec (`docs/eval_spec.md`, tag `eval-spec-v1`) must exist first.

## Pipeline A
All 53 weeks scored. Weekly edges: median 104, min 60, max 1,334. P_A (mean edge score 0-10): mean 2.96, range 1.34-4.29. 53% of edges are labelled disagree, 21% score >= 7, median edge score 1. The series is jumpy week to week (sd about 0.7), which fits the small weekly n (standard error about 0.3-0.4 for typical weeks). No `low_n` weeks (all n >= 60).

## Pipeline B
Only 33 of 53 weeks pass the thresholds for each sign source (18 weeks have fewer than 40 user pairs, 2 have a giant component under 20 nodes). Median graph: 60 pairs, 90 nodes in the giant component (max 591). P_B is extremely noisy: sd 0.68 for continuous, values from -3.2 to +1.0, mean 0.05; 21 of 33 weeks above 0. The unsigned control is 0 by construction. Rewiring-null P_B (mean 0.295) differs from the shuffle-null P_B (0.049) on these tiny graphs, another sign that the null is unstable at this size. Weekly P_A vs continuous P_B: Spearman 0.12 (n = 33).
**Camp recovery** (balanced accuracy of the partition against the user labels, about 62 labelled users per week): continuous 0.598 vs null 0.565, above the null 95th percentile in 4 of 22 weeks; binary 0.602 (5 of 20); sentiment 0.581 (2 of 21); unsigned 0.500. That is a weak signal, only slightly above chance.

## Reading
Tier 0 works as a plumbing test and A is usable, but B cannot be evaluated at this size: weekly graphs are tiny and tree-like, so frustration is mostly noise (the synthetic validation predicted this). Next step for B is more edges per week (extend the same thread order to a larger budget, e.g. 230 edges/week = 24k edges, about 4.5 h more) and ultimately the 5% tier on larger hardware.
