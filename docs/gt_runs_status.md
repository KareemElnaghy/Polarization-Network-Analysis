# Ground-truth runs: status report (paused, Oct 28 about 19:10)

Spec: `docs/gt_runs_spec.md`. **Paused on request because the PC is saturated by Vivado.** Nothing of this spec is running and nothing is half-written.

## Why it stopped
A Vivado batch job eats the CPU: `vivado.exe` pid 141412 (20 threads, about 1200% CPU, 2.7 GB, over 26,000 CPU-seconds since 18:31), started by `cmd.exe /c vivado.bat -mode batch -source ../../scripts/gl_write.tcl`, working folder `C:\work\pipe\verif\netlist\work`. Its parent chain is bash.exe, then **claude.exe pid 138712** (VS Code extension), then Code.exe. My own session is a different claude.exe (pid 139324), so **this Vivado run was started by another Claude Code session**, not this one; the same session also started xelab and gcc for a simulation (`tb_hal_axi`, `legacy_hal`) at 19:02. Two older `vivado.exe` (pids 112536 and 118936, started yesterday 21:51 from the Vivado GUI launcher) are also still open.
Effect on this work: even `import` took 20 s, and shell and PowerShell commands timed out, so I could not check the first launch of the new `labels` job reliably (it showed no `status.json` for minutes). I did not find a code bug; starvation is the likely cause, but that is not proven. I also made it worse by deleting its lock file while an old runner was still alive, which left two runners on one folder. I killed both and deleted `runs/labels/`.

## Done
- Preconditions (spec section 0): comment_flags, phrase_counts, thread_user, clean_edges_and_sampling, label_versions, within_camp_setup and label_coverage are all `done`. Inputs inspected: `label_versions/user_counts.parquet` (author, subreddit, n_all, n_ge1, n_ge2, n_above_median, w_log), `thread_user` weekly parts, `edges_sampled/sampled_threads.parquet` (week_start as a string, subreddit, link_id, rate), `phrase_counts/month=*/counts.parquet` (dictionary-encoded subreddit and author, 37 row groups per month), `vocab.parquet` and `vocab_hashes.parquet`, `within_camp/users.parquet`, the declarations table.
- Written, **uncommitted** (`git status`):
  - `config/labels.yaml` (exactly the template) and `config/gt_runs.yaml` (venues, `label_sets_extra: []`, n_perm 200 for GT1, 100 and 100 for GT2, `min_phrases` 20, the `exclude_tokens` contraction list, events, memory rules, worker split).
  - `src/gt_runs/` (`config.py`, `labels.py` with leave-the-venue-out and gold, `cells.py` with the per-week readers and the GT1 and GT2 cell functions).
  - `src/prep_jobs/jobs/labels.py` (Job L; builds main and gold per venue, `heavy_posters.parquet`, `camp_recovery/labels.parquet`, copies the rule to `results/ground_truth/labels_used.yaml`).
  - Job stubs for T1 to T5 and S added to `config/prep_jobs.yaml` (`depends_on` as in the spec); `scripts/launch_job.py` now snapshots the whole `config/` folder; alias `leave_out_polarization = leave_out_estimator` in `src/ground_truth/leave_out_estimator.py`.
- Not yet run or tested: all of the above. Only the import of `labels.py` was checked (it loads; its `phases()` returns one unit).

## Left to do
1. Run Job L (light) and check its printed users per side. Use `python scripts/launch_job.py labels --fresh --workers 1`.
2. Write jobs: T1 `gt1_dissimilarity` (unit = week; GT1 cells for shared and the five special venues, main on full plus the three sample versions, gold on full), T2 `gt2_leave_out_full` (unit = week, contraction filter on, bootstrap only for main/shared), T3 `gt2_leave_out_samples` (unit = week; re-count phrases for sampled threads from the raw dumps with Job A's reader and counting functions, extended with a time window and `link_id`; rates 0.01, 0.05, 0.20; no counts saved), T4 `gt_within_camp`, T5 `gt_variants`, S `gt_summary` (plots, null checks, `ground_truth_weekly.csv`).
3. Measure one real week for T2 and T3 (peak memory per worker), set workers so workers x peak stays below 60% of free RAM, and log it in `runs/<job>/NOTES.md`. Add the "halve workers after two memory kills" rule to the runner.
4. Foreground: `scripts/camp_recovery_eval.py` with tests (perfect, random, swapped), tests for L, T1, T2 (leave-venue-out subtraction, low-N gives NaN with `low_n`), the "Ground truth" section of `docs/decisions.md`, then commit and push to PR #1 (no data, `runs/` or results).
5. Final report in the format of spec section 5.

## Decisions already taken (from your instruction and my defaults)
- Only the `main` label set runs, plus `gold` in T1 and T2 on the full week; `broad` and `strict` are not built. No deadline; ETAs are still printed. Leave-the-venue-out: the five special subs are not in Job G's counts, so their labels are already outside the venue; for extra venues the venue's counts are subtracted. Gold excludes anyone with a Johnson/Stein vote or more than one distinct vote label.
- Heavy-poster list was not saved by Job C, so Job L recomputes it from `author_week_replies` (Job C already verified that column equals a recount, 0 differences in 28.1M edges).
- Job names: the spec says `leave_out_polarization`; the real function is `leave_out_estimator` (alias added).

## To resume
When Vivado is finished or throttled (or you tell me to proceed anyway), say "continue the ground-truth runs". I will start at step 1 above. To see who is using the CPU: `Get-Process vivado | Select Id,CPU,WS` and the parent chain as above; the offending session is claude.exe pid 138712.
