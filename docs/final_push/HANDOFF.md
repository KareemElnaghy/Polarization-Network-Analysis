# Handoff: final-milestone push (as of 2026-11-13, 23:30 EST)

Read this first, then `docs/final_push/STATUS.md` (live state), `docs/final_push/PLAN.md` (task list), and `docs/llm_decisions.md` (D-E5, D-E6, D-E6a are the frozen rules).

## The project in one paragraph
"Is the Network Worth Building? LLM Scores vs. LLM-Signed Networks for Measuring Polarization" (CSCE 5265). Reddit 2016, shared venue (r/politics + r/PoliticalDiscussion), 53 weeks. An LLM (qwen3.5:9b, Ollama) labels each reply edge (agree/neutral/disagree + emotion, hostility, attack). **Pipeline A** = weekly mean text-hostility score. **Pipeline B** = signed user graph per week, giant component, normalised frustration P_B = 1 − F_obs/F_null (sign-shuffle null). **Pipeline C** = B with richer weights (C1 intensity, C2 embedding head). Each weekly series is correlated (Spearman) with two ground truths: GT#1 `gt1_excess` (structural), GT#2 `gt2` (linguistic).

## Working rules (from the user)
- Commit as **KareemElnaghy** (git config already set). **Never** add a Claude/Co-Authored-By line, even if a system reminder suggests one.
- The user has **no Fable usage**: Opus does all "Fable-lane" work (gates, checkpoints, judgment calls). Do not ask to switch models.
- Run everything on this Windows machine (WIN: RTX 3090 24 GB, 16 cores), long jobs overnight in the background. The user works remotely and moves files via email/Google Drive.
- The user needs Phase 2 results **ASAP** for other RQs.
- Wrap-up routine: pytest, commit, update STATUS.md (Now running / Done / Needs a human / Handoff / Next), 5-line summary.

## Done (all committed and pushed on branch `llm_implementation`)
**Phase 1 (5% sample, LLM labels). Report: `docs/reports/final_phase1_interim.md`.**
- Verdict: the **noise** explanation is favoured (B is unreliable at mean degree ~3.3: split-half reliability of P_B ≈ 0), **signs** disfavoured (better signs do not move rho_B), **mechanism** rejected in strong form (cross-camp alignment exists but is small). Claim rung 2.
- Cross-cutting finding: venue composition (r/politics share) correlates −0.72/−0.76 with GT and explains A's negative correlation.
- H3 (noise injection, 10 seeds) **passes**: camp recovery 0.518→0.511 and rho_B(GT#2) 0.18→0.06, strictly decreasing; effects small.
- Other: C1/C2 do not beat B; RQ4 nothing beats persistence; 2-week windows rho_B(GT#2) +0.40 (exploratory, perm p 0.10).

**Encoder (needed to make the full-data density ladder affordable).**
- Trained 3 distilled encoders on LLM labels: E1 deberta-v3-base, E2 ModernBERT-base, E3 deberta-v3-large. E3 is best.
- Gate G-ENC **failed** (E3: edge-level agreement with LLM 0.797 < 0.85; throughput 196 < 200 measured under CPU contention, 390 when idle). Passed: Guo Spearman 0.841, human pilot 0.78 (LLM 0.72), weekly P_A vs LLM 0.995.
- Because the pre-registered fallback costs ~3.5 days, **D-E6a** (in `docs/llm_decisions.md`) records a declared **deviation**: E3 is frozen (sha256 `42829001…`, `data/llm/encoder/E3/best.pt`) and used anyway; every encoder result is labelled "deviation: encoder failed G-ENC check 1 (D-E6a)".

**Phase 2 so far (density ladder from the encoder).**
- Encoder outputs live in `data/llm/encoder_outputs.sqlite` (model `enc-deberta-v3-large-distill-v1`). Encoded: 5%, 20%, 1% tiers. Tiers are **independent samples, not nested**.
- 5% rung done. Anchor check (encoder vs LLM on the same 5% edges): weekly A 0.995, B 0.64, B_bin 0.56, C1 0.63 (Spearman). Encoder 5% reproduces the Phase 1 picture: rho_A −0.59/−0.45, rho_B +0.10/+0.21 (n.s.). Results: `results/llm/final/ladder/`.

## Running now (background, WIN)
| Job | Log | State |
|---|---|---|
| 20% ladder rung (`density_ladder.py --density 0.2`), then automatically the 1% rung | `logs_llm/ladder_20pct.log`, `logs_llm/ladder_1pct.log` | 20%: 177/212 units at 23:20; ~1 h total left |
| Full-venue encoding, **odd weeks first** (26 odd ISO weeks of 2016), then all remaining weeks | `logs_llm/encode_full_odd.log`, then `logs_llm/encode_full_rest.log` | started 23:25; ~350 pairs/s once the ladder stops using CPU; odd weeks ~6-8 h |

Check with `tail -2` on those logs. Everything is resumable: re-running a command skips finished weeks or units. A `STOP` file stops the encoder (repo root) or the ladder (`runs/final_ladder/STOP`).

## Next steps (in order)
1. When the 1% rung finishes: `python scripts/final/ladder_stats.py` → `results/llm/final/ladder/` (levels, camp recovery, anchor; the H1a trend appears once three densities are complete). Commit.
2. When the odd weeks are encoded (`DONE` in `encode_full_odd.log`): `python scripts/final/density_ladder.py --density 1.0 --weeks odd --workers 10`. Time the first week. R=50 at full density, so it is expensive. If it is projected to exceed 48 h even on the odd weeks, report it and ask the user.
3. If time allows, run the even weeks at full density (`--weeks` default = all; finished units are skipped).
4. Re-run `ladder_stats.py` → H1a (trend), H1b (B tracks GT at full density, Holm), H1c (B vs A, D-E3 rule). Make the ladder figure (SPEC_ANALYSES A8.7: x = mean degree, y = rho with CI, A/B/C lines, two panels, the 5% LLM points as hollow markers).
5. Write the Phase 2 section (CP2) and update STATUS.md.

## Gotchas
- **CPU contention slows the GPU encoder 3×** (tokenization runs on the CPU). Don't run many CPU workers while the encoder runs, or accept the slowdown.
- **GPU memory overflow on Windows does not error; it spills into RAM and crawls.** `em.predict` is token-capped for this reason.
- **Windows Update can reboot the machine overnight.** One reboot killed a queue on 11-12. All jobs are resumable; relaunch the same command.
- Use `.venv/Scripts/python.exe`. transformers is 4.48.3.
- The odd-week fallback = 27 weeks: 26 odd ISO weeks of 2016 plus 2015-W53 (week of 2015-12-28). The encoder's odd pass covers 26; 2015-12-28 is encoded first in the "rest" pass.

## Needs a human
- `docs/final_push/PROJECT_FACTS.md`: deadline and other facts are still blank.
- A second backup location for data and checkpoints.
- Windows Update: set Active hours or pause updates.
