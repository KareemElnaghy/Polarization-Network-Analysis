# Evaluation spec v1 (written 2026-10-31, before any comparison with GT)

Scope: first RQ1 look at the Tier 0 sample (1% shared sample, whole threads, 230 edges/week, run 20261031T182808_tier0_b230). A plumbing-level, weak-power analysis, not the confirmatory one (that needs 5%).
Not yet git-tagged (`eval-spec-v1` to be added at the next commit).

- Series: P_A (mean edge score), P_B continuous_llm (headline sign source; weeks with NaN dropped). Truths: GT#1 `gt1_excess`, GT#2 `gt2` (full-week, shared venue), from `results/ground_truth/ground_truth_weekly.csv`.
- Primary: Spearman rho on levels, per (pipeline x truth). Secondary: Spearman on first differences.
- Rank level, not size: only Spearman (scales differ).
- Inference: circular-shift permutation p (10,000 shifts) per rho; paired bootstrap over 4-week blocks (2,000 draws) for delta = rho_B - rho_A on the weeks where both exist.
- Confound check: partial Spearman controlling for weekly edge volume (`n`) of the pipeline input. Volume-only baseline rho reported.
- Decision rule: a pipeline wins convincingly only if it beats the other on the truth that leans against it (B on GT#2, A on GT#1) with the delta CI excluding 0. Otherwise: no winner.
- Not done in v1 (stated up front): Holm correction, composition-only baselines, before/after 2016-06-27 split, split-half noise ceilings, sample-matched GT. These need the 5% tier.
