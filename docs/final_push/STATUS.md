# Status, final push (updated 2026-11-11, after P1.4)

Agent model this session: Sonnet 5.5 (cheap lane). Folder lives at `docs/final_push/` (spec text that says `final_push/` is the same folder).

## Now running
| Task | Machine | Started | Log | ETA |
|---|---|---|---|---|
| P2.2 encoder inference (E3, frozen D-E6a): 20% tier, then 1% tier, then the rest of the full shared venue (13.5M pairs). Resumable; chain waits on the 20% PID | WIN GPU | 2026-11-13 ~14:00 | `logs_llm/encode_20pct.log`, `encode_1pct.log`, `encode_full.log` | 20% ~2 h; full ~12 h total |
| P2.3 ladder at 5% (A, B, B_bin, C1; R=20; 8 workers) | WIN CPU | 2026-11-13 | `logs_llm/ladder_5pct.log` | ~1 h |

## Done
| Task | Headline result | File |
|---|---|---|
| P0.1 backup | `extractions.sqlite` sha256 `79f327ab17f0f274fd3cab658e14f3f7d6d225fb17c1d36d965660d0474e7ab8`, 617,041 rows (read-only open; source and copy hashes match). Pilot folder copied, per-file sha256 identical (excluding `._*` macOS junk). | `C:\Users\admin\SNA_backups\20261111\` (same disk as the source, **not** off-machine) |
| P0.2 commit | Full-5% results, interim results, `watch_5pct.py`, INDEX rows, handbook, presentation context committed. `final_push.zip` deliberately not committed (duplicate of the folder). | commit `f56e579` |
| P0.3 pre-register | D-E5 and D-E6 appended to `docs/llm_decisions.md`; proposal copied to `docs/proposal.pdf`. "Git SHA at freeze" recorded as `f56e579` (a commit cannot contain its own hash). | commit `533eb2d`, tag `prereg-final-v1` |
| P0.4 ops scripts | `ops/UBU/setup.sh`, `ops/MAC/setup.sh` (syntax-checked only, not run) | `ops/` |
| P0.5 | this file | |
| P1.8 RQ4 forecasting | No model beats persistence (30 one-step forecasts, t=24..53). MAE GT#1: persistence 0.0170, A 0.0176, B 0.0173; GT#2: 0.0017 / 0.0018 / 0.0017. A is significantly WORSE than persistence on GT#2 (DM p 0.036). Matches the pre-registered expectation. | `results/llm/final/rq4/` |
| P1.5 code | Pipeline C1 weights + resumable runner; parity check: runner fed B's own weight reproduces B's F_obs and P_B exactly on 2 weeks. C1 stats script `scripts/final/rq1_pipeline_c.py` written, to run when the run finishes. | `src/algorithms/pipeline_c/`, `scripts/llm/run_pipeline_c.py` |
| G1.1 distill export | 616,891 rows (150 pilot pairs removed; train 567,585 / val 26,416 / test 22,890; 11,160 threads; 53 weeks); sha256 `d0acab85…98666` | `data/llm/distill/distill_v1.parquet`, `manifest.json` (git-ignored; script `scripts/final/export_distill.py`) |
| P1.1 spec completion | Holm: 0 of 4 family-A tests significant (min adj p 0.226). Composition-only baseline `share_politics` has rho -0.72 (GT#1) / -0.76 (GT#2), larger in magnitude than rho_A or rho_B; partial rho_A(GT#2 given share_politics) = +0.07. 06-27 split: before, all CIs include 0; after, rho_B(GT#2) = +0.42 [0.06, 0.70], rho_A(GT#1) = -0.47 [-0.75, -0.09]. Seed stability: delta GT#2 CI lower bound 0.20-0.22 across 5 seeds. | `results/llm/final/rq1_spec_completion/` |
| P1.3 alignment (H-AL) | Delta-w (cross - same) = -0.092, 95% CI [-0.106, -0.080] over 139,015 labelled-both edges, 6,312 threads: rule met. Same sign for binary_llm (-0.154) and VADER (-0.057). Weekly alignment excess correlates -0.61 (GT#1) / -0.68 (GT#2) with truth (diagnostic only). | `results/llm/final/alignment/` |
| P1.4 why A is negative (H-AN) | (i) rho(s_t, GT#1) = -0.66, Holm p = 0.13; (ii) \|rho_A\| - \|partial\| = +0.27 CI [+0.04, +0.43]. Both conditions met as written. Shift-share: only 3% of P_A variance from edge-type mix. | `results/llm/final/a_negative/` (two run folders exist: first run and rerun after a wording fix; the second is canonical) |

Tests: `.venv/Scripts/python.exe -m pytest -q` gives 106 passed, 1 failed (known Windows `termios` failure in `test_label_tool_resume_back_skip`).
| P1.5 | C1 at 5%: does not beat B (GT#1 delta +0.10, CI [-0.08,+0.31], Holm p 0.55); beats A (A is negative, H-AN) | `results/llm/final/pipeline_c/C1/` |
| G1.7 | C2 head: OOF R2 0.29, rho 0.53, pilot acc 74.6% vs LLM 73.2%. C2 at 5%: does not beat B (GT#2 delta +0.20, CI [-0.21,+0.47]); beats A | `data/llm/c2/c2_quality.json`, `results/llm/final/pipeline_c/C2/` |
| P1.2 | Split-half noise ceilings (20 splits, R=20; 10 on WIN, 10 on MAC). Spearman-Brown reliability P_A 0.56 [0.44, 0.69]; P_B -0.08 [-0.46, 0.25], so rel_B < 0.3 (pre-registered rule: observed rho_B is bounded by noise; density ladder decisive). Disattenuation undefined for B. | `results/llm/final/noise_ceiling/` |


## Needs a human
- [ ] Fill in `docs/final_push/PROJECT_FACTS.md`: deadline, presentation date, hard stop; UBU CPU/RAM/GPU; MAC CPU; LAN/SSH/shared drive/internet; who runs what. (Detected on WIN: RTX 3090, 24 GB.)
- [ ] Copy the backup `C:\Users\admin\SNA_backups\20261111\` to a second physical location (external SSD or UBU). It currently sits on the same disk as the original.
- [ ] Say whether I may `git push` (local branch is 2 commits ahead of origin, plus the tag). UBU/MAC setup scripts clone from origin, so they need the push first.
- [ ] (UBU) after push: `bash ops/UBU/setup.sh`
- [ ] (MAC) after push: `bash ops/MAC/setup.sh`

## Blocked
- Nothing blocking in-lane work.

## Gates
- prereg-final-v1 tag: **done** (2026-11-11, commit `533eb2d`)
- G-ENC: **FAILED** after the retry (E3: check 1 test agreement 0.797 < 0.85; check 5 196 pairs/s under CPU contention, 390 idle). Phase 2 proceeds with E3 as a declared deviation (D-E6a, 2026-11-13), because the timeline cannot absorb the 3.5-day LLM fallback.
- Phase 3 gate: pending

## Handoff
- **For CP1 (Fable): H-AN reporting.** `results/llm/final/a_negative/summary.json`. Pre-registered conditions (i) rho(s_t,GT#1)<0 and (ii) |partial|<|rho_A| are met as written, but (i) is not significant (Holm p 0.13) and the descriptive shift-share gives mix only 3% of P_A variance. Decision needed: which CLAIMS.md wording (pass or fail) applies. I did not choose.
- **For CP1 (Fable): composition confound.** `share_politics` alone out-correlates A and B with both truths (-0.72 / -0.76, `composition_baselines.csv`). Decision needed: how this reframes the RQ1 headline in D-E5 point 3.
- **For CP1: alignment vs tracking.** H-AL holds at the edge level (delta-w -0.092) yet weekly alignment excess is negatively correlated with GT. Needs interpretation against the three-explanation table in PLAN.md.

## Next
1. When the 20% and 1% tiers are encoded: `scripts/final/density_ladder.py --density 0.2` and `--density 0.01` (CPU, 8-10 workers while the GPU encodes).
2. When the full venue is encoded: `--density 1.0` (R=50; measure the per-week time first; if > 48 h projected, use `--weeks odd`).
3. P2.4 statistics per density (rho with CIs, H1a/H1b/H1c, camp recovery, alignment) and P2.5 anchor (encoder vs LLM A/B/C at 5%, mandatory under D-E6a).
4. Ladder figure (A8.7).

> **SHA note (2026-11-11):** before the first push, local commits were rewritten (author set to KareemElnaghy, co-author trailers removed). Content is unchanged. Old `f56e579` is now `d16b880`; old `533eb2d` (tag `prereg-final-v1`) is now `ec83d8c`. `docs/llm_decisions.md` D-E5/D-E6 still print the old SHA `f56e579`; it is left as frozen text and this note is the mapping.

### Handoff (P1.2 -> CP1, Fable)
P_B split-half reliability is about zero at 5% (halves have half the edges, Spearman-Brown on a negative r is not meaningful). The pre-registered rule fires. Open for CP1: how to word it (noise ceiling vs. no signal) and whether it reframes the headline. Not interpreted further here.

### Partial P1.6 / P1.7 (cheap parts done)
- P1.6 steps 2-4 (`results/llm/final/rq3/`): no rung beats B (all Holm p = 1.0). Pilot accuracy rises from VADER 48% to binary 73% to continuous 79% (C2 75%) while rho_B stays null on both truths, so the ladder does not predict GT correlation. Secondary/exploratory.
- P1.7 parts 3-4 (`results/llm/final/rq2/`): within-camp share of negative edges correlates +0.66 (GT#1) / +0.67 (GT#2) with GT, mirroring the negative alignment-excess correlation; A/B rank divergence unrelated to it (rho -0.17, p 0.13). Exploratory.

### P1.6 noise curve and P1.7 thinning/windows (queue finished 2026-11-13; `results/llm/final/rq3_rq2_jobs/`)
Run with R=20 (baseline p=0 reproduces the R=50 run: rho -0.024/+0.177 vs -0.030/+0.173). A forced Windows Update restart on 11-12 03:33 cost ~9 h; relaunched, ran 5.95 h of budget (23 jobs). Seeds finished: noise p=0.05/0.2: 3, p=0.1/0.4: 4 (p=0 is deterministic); thinning: 3 seeds each. Fallback taken: not the specified 10 seeds. 2-week windows not run.
- Noise curve is flat: mean P_B moves -0.008 to -0.028 and rho_B shows no monotone decline with p (GT#1: 0.16, 0.04, 0.04, 0.08; GT#2: 0.28, 0.14, 0.10, 0.13; seed sd 0.03-0.23). Camp-recovery BA falls only 0.518 -> 0.511.
- Thinning: 25% of threads gives higher rho (GT#1 0.21, GT#2 0.24) than 50% (0.09/0.17) and the full 5% (-0.03/+0.17); no density gain visible, giant-component mean degree 2.76 / 2.96 / 3.29.
- Rolling 4-week windows (50): rho GT#1 +0.09 [-0.33, 0.50], GT#2 +0.17 [-0.24, 0.50]. Months (13): -0.02 / +0.31 (descriptive).
- Handoff to CP1 (Fable): how to word a flat noise curve alongside rel_B about 0 (P_B insensitive to sign quality, not just noisy?). Not interpreted here.

### CP1 done (2026-11-13, Opus standing in for Fable)
`docs/reports/final_phase1_interim.md`. Verdict: noise explanation favoured, signs disfavoured, mechanism rejected in strong form (weak alignment). Rung 2. The four CP1 handoffs above are resolved there (H-AN partially supported, composition caveat on every "beats A", weekly alignment is composition, flat noise curve). Pending: H3 final verdict after 10 seeds (queue relaunched, `logs_llm/b_jobs.log`).

### G1.3-G1.5 (Opus standing in for Fable)
- E1 deberta-v3-base: val agree acc 0.763, Guo Spearman 0.796 (64 min). E2 ModernBERT-base: 0.767 / 0.803 (45 min). transformers upgraded 4.46.3 -> 4.48.3 for ModernBERT (E1 reload verified).
- Selection (val only): within 0.5 pp -> faster: E2 (1,099 vs 1,015 pairs/s).
- G-ENC on E2 (`results/llm/final/encoder/gate.json`): (1) test acc 0.771 FAIL (>= 0.85); (2) Guo Spearman 0.798 FAIL (>= 0.80); (3) pilot 3-class 0.747 pass (LLM 0.72 on the same items); (4) weekly P_A Spearman 0.994 pass; (5) 1,119 pairs/s pass. Per SPEC_ENCODER section 7: one retry (E3, deberta-v3-large) running; then re-select among E1/E2/E3 and re-run the gate. If it fails again: G-ENC fails and the Phase 2 fallback (LLM on 20% tier, 12 fixed weeks) applies.
- 5%-tier encoder outputs for E2 are in `data/llm/encoder_outputs.sqlite` (model `enc-ModernBERT-base-distill-v1`).
- **G1.5 retry (2026-11-13):** E3 deberta-v3-large val 0.793 / Guo 0.844 (4.9 h). Re-selection picked E3 (higher val accuracy). Gate on E3: (1) 0.797 FAIL, (2) 0.841 pass, (3) 0.780 pass (LLM 0.72), (4) 0.995 pass, (5) 196.5 FAIL as recorded (idle re-measure 376-390 pairs/s). G-ENC fails. User needs Phase 2 results immediately for the other RQs, so D-E6a records a deviation: E3 frozen (sha256 42829001...), every encoder result labelled. 5%-tier E3 rows are in `data/llm/encoder_outputs.sqlite`.
- **Tiers are independent thread samples, not nested** (5% threads in 20%: 23%; 1% in 5%: 5.5%). Each tier is encoded separately; the full venue covers all.
### H3 final (10 seeds, 2026-11-13)
H3 **passes**: camp recovery 0.518 -> 0.511 and rho_B(GT#2) 0.177 -> 0.059 strictly decreasing over p (Spearman -1.0 each), small effects. 2-week windows (exploratory): rho_B GT#2 +0.40 [+0.15, +0.62], perm p 0.10. CP1 report updated.
