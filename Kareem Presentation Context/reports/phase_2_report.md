# Phase 2 report: Pipeline A

**Built:** additive heuristic edge score 0-10 (`configs/llm/pipeline_a_rules.yaml`, A-rules-v1, total over 54 attribute combinations), weekly aggregation (mean = P_A, median, share >= 7, n, variants A_disagree and A_hostility, `low_n` flag, per-week `coverage` for partial runs), runner `scripts/llm/run_pipeline_a.py` (stored outputs only, run folder + registry + plot with CSV).
**Tests:** rule totality, hand-built aggregation example (13 engine tests pass).
**Data:** Tier 0 extraction (qwen3.5:9b, prompt v2, 1% shared sample, whole threads, 60 edges/week budget) is still running; at the time of writing only a few hundred edges exist, so the first real weekly P_A plot is not yet meaningful. Re-run `run_pipeline_a.py` on the Tier 0 run folder when it has finished (or at about 50% coverage).
**Decisions:** D-A1, D-A2. **Limitations:** the rules are an additive reading of Chowdhury et al. (TODO(verify)); P_A inherits the model's moderate pilot accuracy (about 0.72).
**Open:** real weekly P_A plot pending the extraction.
