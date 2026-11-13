# Status, final push (updated 2026-11-11, after P1.4)

Agent model this session: Sonnet 5.5 (cheap lane). Folder lives at `docs/final_push/` (spec text that says `final_push/` is the same folder).

## Now running
| Task | Machine | Started | Log | ETA |
|---|---|---|---|---|
| (none) | | | | |

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
- G-ENC: pending
- Phase 3 gate: pending

## Handoff
- **For CP1 (Fable): H-AN reporting.** `results/llm/final/a_negative/summary.json`. Pre-registered conditions (i) rho(s_t,GT#1)<0 and (ii) |partial|<|rho_A| are met as written, but (i) is not significant (Holm p 0.13) and the descriptive shift-share gives mix only 3% of P_A variance. Decision needed: which CLAIMS.md wording (pass or fail) applies. I did not choose.
- **For CP1 (Fable): composition confound.** `share_politics` alone out-correlates A and B with both truths (-0.72 / -0.76, `composition_baselines.csv`). Decision needed: how this reframes the RQ1 headline in D-E5 point 3.
- **For CP1: alignment vs tracking.** H-AL holds at the edge level (delta-w -0.092) yet weekly alignment excess is negatively correlated with GT. Needs interpretation against the three-explanation table in PLAN.md.

## Next
1. Cheap lane, stay on this model: P1.8 (RQ4), P1.5 (Pipeline C1), P1.6, P1.7, P1.2 (long compute), G1.7, G1.6.
2. **Then needs Fable:** G1.3-G1.5 (encoder training, selection, G-ENC gate).


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
