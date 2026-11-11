# Analysis specifications

Put new code in `scripts/final/` (entry points) and `src/algorithms/analysis/` (reusable functions; the only new package allowed to read labels). Every script:
- takes the run folders as arguments,
- writes a run folder through the provenance helper,
- writes its CSVs and a `summary.json` with verdict strings to `results/llm/final/<topic>/`,
- labels exploratory outputs in both the file name and the JSON.

Seeds are fixed (0 unless stated). Reuse the existing helpers in `rq1_compare.py` (Spearman, circular-shift p, block bootstrap) rather than reimplementing them: import them, or move them into `src/algorithms/eval/metrics.py` and keep `rq1_compare.py` working.

Abbreviations: **LB** = edges with both endpoints carrying a main label (dem/rep). **Cache** = `data/llm/extractions.sqlite`, qwen3.5:9b-q4_K_M / prompt v2. **Full-5% runs** = `runs/llm/20261107T121812_pipeA_5pct_full` and `runs/llm/20261107T121822_pipeB_5pct_full`.

---

## A1. Completing the eval spec (D-E5, D-E6 §A)

1. **Holm correction.** For family A (rho_A and rho_B × GT#1 and GT#2, primary truths), take the circular-shift p values from `rq1_5pct_full_correlations.csv`, apply Holm, and write `holm.csv`.
2. **Composition baselines.** These are weekly covariates:
   - `share_politics`: share of the week's shared-venue comments that are in r/politics,
   - `rep_dem_ratio`: labelled rep users ÷ labelled dem users active in the shared venue,
   - `new_user_share`: share of active users whose first 2016 comment falls in this week.

   Use `results/gt1_drivers/drivers_weekly.csv` if it has these columns. Otherwise compute them from `data/processed/thread_user/` and the labels (`weekly_composition.csv` does not exist; handbook §17). For each covariate, report its rho with each truth, and partial rho_A and partial rho_B controlling for it (Spearman partial via rank residuals).
3. **The 06-27 split.** Weeks starting before 2016-06-27 make segment 1; the rest make segment 2. Report rho_A and rho_B per segment with block-bootstrap CIs, and the per-segment means of P_A and P_B.
4. **Robustness.** Run `scripts/llm/rq1_bootstrap_stability.py` and `rq1_partial_week_sensitivity.py` on the full-5% runs.

## A2. Split-half noise ceilings (H-NC)

- For each week, randomly assign each **thread** (`link_id`) in the 5% extraction to half H1 or H2 (seed s = 0…19).
- For each split, compute P_A (weekly mean edge score) and P_B (continuous_llm, giant component, R = 20 nulls; the same graph rules as `structural.py`) on each half separately.
- The reliability for split s is the Spearman rho across weeks between the H1 and H2 series. The Spearman-Brown correction is 2r/(1+r). Report the median and the 5–95% range over the 20 splits.
- Disattenuated rho = rho_full / sqrt(rel_pipeline × rel_GT), with rel_GT1 = 0.93 (handbook §6.1) and rel_GT2 = 0.79 (the 5% sample stability, used as a stand-in; label it as such).
- Interpretation, written in the summary: if rel_B < 0.3, the observed rho_B is bounded by noise and the density ladder is the decisive test.
- Cost: B on 2 × 20 half-graphs per week. Use `--sources continuous_llm` only and a 6-process pool. Fallback: 10 splits with R = 10.

## A3. Edge-level alignment (H-AL), the most important diagnostic

Inputs: the full-5% edges and their cache outputs, plus main labels. This analysis reads labels, so it lives in `src/algorithms/analysis/alignment.py`.

1. Keep the LB edges. For each edge record: week, thread, cross = (label_replier ≠ label_parent), and from the cache: Guo weight w, argmax agreement, hostility level (0 / 0.5 / 1), personal_attack, emotional_charge. Also record the VADER compound of the reply, and the encoder outputs once they exist.
2. **Edge-level effect.** Δw = mean w(cross) − mean w(same), with a 95% CI from a **thread-cluster bootstrap** (resample threads, 1,000 draws). Compute the same for P(disagree), P(hostility > none), P(attack), and VADER compound. Also fit a logistic regression of disagree on cross + week fixed effects with thread-clustered SEs (statsmodels), and report the odds ratio.
3. **Fraxanet decomposition per week.**
   - Antagonism_t is the share of LB edges with w < 0.
   - Alignment_t is the share of negative LB edges that are cross-camp, minus the same share under 200 label permutations within the week (an excess over the null).
   - Report both series. Correlate each with GT#1 and GT#2 (diagnostic only; not a pipeline).
4. **By sign source.** Repeat steps 2–3 for binary_llm, VADER and (later) the encoder. This directly tests whether the LLM signs carry more partisan alignment than VADER, which is the camp-recovery puzzle in handbook §13 point 12.
5. **By density** (in Phase 2): repeat at 1%, 5%, 20% and full with encoder signs.
6. Output `alignment_edges_summary.csv`, `alignment_weekly.csv`, and a figure: Δw with its CI by sign source.

## A4. Why Pipeline A runs negative (H-AN)

1. s_t = the share of LB edges that are cross-camp in week t (5% extraction).
2. Test (i): rho(s_t, GT#1) and rho(s_t, GT#2), with circular-shift p.
3. Test (ii): partial rho_A(GT#1 | s_t). Then bootstrap the difference |rho_A| − |partial rho_A| (block bootstrap over weeks). Do the same for GT#2.
4. **Decomposition (descriptive).** Compute P_A separately on same-camp LB edges, cross-camp LB edges, and edges with one or both ends unlabelled. Report rho with each truth for each subset, the weekly mean edge score by edge type, and how much of P_A's variation over time comes from the shift in mix (a shift-share decomposition: Σ_k Δshare_k · score_k versus Σ_k share_k · Δscore_k).
5. Also correlate the P_A components (`A_disagree`, `A_hostility`, `share_high`) with GT and with s_t.
6. The summary states in one sentence whether "the camps mix more, so they argue more, so P_A rises while segregation falls" is supported.

## A5. RQ3: the sign-quality ladder and the noise curve (H3)

1. **Noise curve.** For p ∈ {0, 0.05, 0.1, 0.2, 0.4} and seeds 0–9, run `noise_injection(w, p)` on the continuous_llm weights. Run Pipeline B (R = 20) on each, then camp recovery and the full RQ1 statistics. Plot mean P_B, rho_B(GT#1), rho_B(GT#2) and camp-recovery BA against p, with bands across seeds.
2. **All sign sources through the full RQ1 statistics.** Run `rq1_compare.py` with P_B from binary_llm, sentiment (VADER) and unsigned (the last one is a sanity check: it should be NaN or 0), plus C1, and C2 once G1.7 has finished. Until now only camp recovery compared these.
3. **Pilot accuracy per rung.** On the 150 pilot pairs (consensus labels; neutral excluded for binary accuracy), report the agree/disagree accuracy of VADER (fixed thresholds ±0.05), binary_llm, and the encoder once it exists. continuous_llm is scored by the sign of w.
4. **Ladder figure.** Pilot accuracy (x) against rho_B on each truth (y) and against camp recovery, one point per rung.

## A6. RQ2: conditions (descriptive)

1. **Windows.** Re-aggregate the existing cached edges into rolling 2-week and 4-week windows (step 1 week) and calendar months. Recompute P_A, P_B (R = 20) and the matching GT. GT#1 and GT#2 are weekly; for a window, use the mean of the weekly GT values (state this as an approximation). If the data-side code can compute GT on a multi-week window cheaply, use that instead and say so. Report rho per window size, with CIs; note that rolling windows overlap and handle this with block lengths ≥ the window length.
2. **Thinning.** Keep 50% and 25% of each week's threads (seeds 0–4), recompute P_B and rho_B, and record the giant-component mean degree. Together with the 1% Tier-0 points, this gives a density curve with no new extraction (Phase 2 extends it upward).
3. **Polarization level.** Split weeks into the top and bottom GT quartiles (per truth). Report A–B agreement, P_B level and camp recovery in each.
4. **Discordant allies.** Per week, the share of negative LB edges that are within-camp, correlated with |rank(P_A) − rank(P_B)| (the per-week A/B divergence).

## A7. RQ4: forecasting

As specified in D-E6 §G. Implement it in `scripts/final/rq4_forecast.py`. Output the MAE per model per truth, DM statistics, and a figure of predicted against actual ΔGT for the best model. If Phase 3 adds 2017, rerun on the pooled 106 weeks with a year break (no Δ across years).

## A8. The density ladder (Phase 2, H1)

1. **Inputs.** Encoder outputs for the 1%, 5% and 20% sampled tiers (`data/processed/edges_sampled/rate=*`) and the full cleaned shared venue (`data/processed/edges_clean/`). Apply the same filters as `data.py` (drop removed/empty, 1200-character truncation).
2. **Per density.** Run A (encoder edge score, same rule table), B (continuous, from the encoder's agreement probabilities; binary as robustness), C1 (from the encoder's attributes) and C2 (bge embedding with ridge head; SPEC_PIPELINE_C §C2). Use R = 50 at full density for continuous B, C1 and C2, and R = 20 otherwise.
3. **Record** per week and density: giant-component nodes and edges, mean degree, P_B, P_B_rewire (full density: R = 5 only), and camp recovery.
4. **Statistics.** rho per pipeline × truth × density with CIs, then the H1a, H1b and H1c tests (D-E6 §D).
5. **Fallback weeks.** The 27 odd-numbered ISO weeks (fixed in D-E6). Never choose weeks by result.
6. **Cost control.** Shard B by week (`--weeks` argument; add it if missing). Scale the annealing budget with the existing rule (7.5e7 node-visit cap). Note in the limitations that heuristic optimality is unverified at this size, and report the best-of-10 spread as a stability check.
7. **Figure** (headline of the paper): x = mean degree (log), y = rho with 95% CI. Lines for A, B and C. Two panels (GT#1, GT#2). Include the 5% LLM-based points as hollow markers (the anchor).
