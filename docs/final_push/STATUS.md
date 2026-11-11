# Status, final push (updated 2026-11-11)

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

Tests: `.venv/Scripts/python.exe -m pytest -q` gives 106 passed, 1 failed (known Windows `termios` failure in `test_label_tool_resume_back_skip`).

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
- (none)

## Next
1. G1.1 distillation export and G1.6 llama-server benchmark (cheap lane, WIN GPU) and Phase 1-CPU P1.1, P1.3, P1.4 (cheap lane), all stay on this model.
2. **Then needs Fable:** G1.3-G1.5 (encoder training, selection, G-ENC gate).
