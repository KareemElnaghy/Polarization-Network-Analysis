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
