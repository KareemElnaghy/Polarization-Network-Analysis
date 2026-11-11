# Prep jobs spec (Oct 28)

Read this whole file before doing anything. Then build the job framework, build and launch jobs A, B, C, E, G and H as **detached background jobs**, do jobs D and F in the foreground while they run, and report back. Do not wait for the background jobs to finish.

## 0. Goal and rules

Context: Politosphere 2016 (42.4M comments, 855k users) is parsed, reply edges are built (28.1M), per-user tables exist. We are preparing everything for two ground-truth measures and for the edge slice that Kareem's LLM pipelines will read. The labelling rule (who counts as dem or rep) is **not decided yet**, so nothing here may depend on labels.

Rules:
- **Never ask me anything.** If something is ambiguous, pick the most conservative default, write the choice and your reasoning to `runs/<job>/NOTES.md`, and continue.
- **Do not trust the names in this file.** Column names and paths below come from memory of earlier reports. Inspect the real schemas (`parse_comments.py`, `build_edges.py`, `windowing.py`, `user_profiles.py`, the Parquet files) and adapt.
- **Read-only on existing outputs** (`data/interim/comments/`, `data/interim/edges/`, `data/processed/user_profiles/`). Write new outputs only to new folders. Another Claude Code session may be running a score check at the same time.
- Every subreddit set, threshold, rate, seed and path lives in `config/prep_jobs.yaml`. No magic numbers in code.
- Use the existing week definition (Monday 00:00 UTC) by **importing the repo's windowing function**, not reimplementing it.
- Add new dependencies (`psutil`, anything else) to `pyproject.toml`. No internet downloads at run time (use sklearn's built-in English stopword list).
- Platform: Windows, VS Code, PowerShell. Use `pathlib`, `multiprocessing` with spawn, `if __name__ == "__main__":` guards.

## 1. Job framework (build this first, test it, then use it for all jobs)

### 1.1 Detached
- `scripts/launch_job.py <job> [--workers N]` starts the job as a **fully detached process** and returns immediately, printing the PID and the status command.
  - Windows: `subprocess.Popen` with `creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`, `close_fds=True`, stdin = DEVNULL, stdout and stderr redirected to `runs/<job>/stdout.log` and `stderr.log`. POSIX: `start_new_session=True`.
  - The job must **not** be a child of the Claude Code shell, the VS Code terminal or the chat. Closing VS Code, ending the chat or killing the parent shell must not stop it.
  - On launch, **copy `src/` and the config to `runs/<job>/snapshot/`** and run from the snapshot, so later edits and commits cannot change a running job.
  - Best-effort sleep prevention inside the job process (Windows: `ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`). Run at below-normal priority so the machine stays usable.
- **Prove it works** with a dummy 90-second job: launch it, kill the launching shell, confirm `status.json` keeps updating and the job finishes. Report the result.

### 1.2 Work units, checkpoints, resume
- Each job is a list of **units** (a month, a week, a month-pass). A pool of worker processes runs them.
- Each unit writes to a temporary name, then **atomically renames**. On success it writes `runs/<job>/checkpoints/<unit>.done` (rows, bytes, seconds, peak memory).
- `scripts/launch_job.py <job> --resume` skips units with a checkpoint. Re-running is always safe.
- A failed unit is retried up to 3 times with backoff, then marked failed; the job continues with the others. A job ends as `done` or `done_with_failures` (listing the failed units).
- `runs/<job>/STOP` (an empty file) makes the job stop cleanly after the current units.
- PID in `runs/<job>/pid`. No two copies of one job may run: use a lock file and exit with a clear message if it exists and the PID is alive.
- A job can declare `depends_on: [jobs]`. It then starts in state `queued` (the status table shows "waiting for X") and starts itself when every dependency is `done`.

### 1.3 Heartbeat and ETA
- The main job process writes `runs/<job>/status.json` **every 15 seconds** (atomic write) with: `state` (queued, running, paused, done, done_with_failures, failed, crashed), `phase`, `started_at`, `updated_at`, `units_done`, `units_total`, `work_done`, `work_total` (**measured in input bytes or rows, not unit count**, so big months weigh more), `percent`, `workers`, `current_units`, `rows_per_second` (rolling 5-minute), `output_bytes_so_far`, `peak_memory_gb`, `eta_seconds`, `eta_clock_time`, `deadline_at`, `at_risk` (true if the ETA is past the deadline), `last_error`, `failed_units`, `pid`.
- `scripts/jobs_status.py [--watch SECONDS] [--json]` reads every `runs/*/status.json` and prints one table row per job: name, state, percent, units, workers, elapsed, ETA (duration and clock time), heartbeat age, output size, flags. Flags: `STALE` if the heartbeat is older than 90 s while `running`; `CRASHED` if the PID is not alive while `running` (also rewrite the state to `crashed`); `AT RISK` if the ETA passes the deadline. Under the table, print the overall ETA (the latest of all jobs) and free disk space.
- `--deadline-hours` (default 9) on launch sets `deadline_at`. Compute the ETA after the first units finish, and if it is at risk, say which setting to cut (see each job).

### 1.4 CPU and memory allocation
- Detect `cpu_count`, total and free RAM. **Total workers across all jobs = min(cpu_count − 2, floor(0.8 × free_RAM_GB / peak_GB_per_worker))**. Peak memory per worker comes from a quick measurement on one unit, falling back to 4 GB if unmeasured.
- Print an **allocation plan table** before launching. Starting split of the available workers: A 45%, C 25%, B 10%, G 15%, E 5% (minimum 1 each); H waits for B and C, then takes 1–2 workers. Static for the life of each job; the split can be changed with `--workers` on a resume.

### 1.5 Load guard (protects the PC)
The machine is shared, so the jobs must never take it down.
- Every job process runs a governor thread that samples **system-wide** CPU, RAM (`psutil.virtual_memory()`) and free disk on the output drive every 10 seconds. Thresholds live in `config/prep_jobs.yaml` under `load_guard`.
- **RAM above 85%:** stop dispatching new units, and halve the worker count as running units finish (state `throttled`, with the reason). **RAM above 92% for 30 seconds:** pause dispatch completely until RAM falls below 75%.
- **Free disk below 25 GB or below 10%:** pause with the reason written to the status.
- **CPU:** do not throttle on CPU alone, since jobs run at below-normal priority, but always keep `reserved_cores` (default 2) free.
- **Memory-safe by construction:** never load a whole month into memory. Read Parquet in batches (`iter_batches`, default 250k rows) and aggregate incrementally. Each worker checks its own memory use; above `worker_rss_limit_gb` it flushes and halves its batch size. A worker killed for memory has its unit retried with a smaller batch (counts as an attempt).
- Several jobs run at once and each sees the same system-wide numbers, so they back off together. Add 0–15 seconds of random jitter before restarting dispatch so they don't all resume at the same instant.
- `runs/PAUSE_ALL` (an empty file) pauses dispatch for every job; delete it to resume. `runs/<job>/STOP` still stops one job cleanly.
- `scripts/jobs_status.py` also prints a system line (CPU %, RAM used %, free disk) and shows `throttled` with the reason for any job that is.
- At launch, print a notice that the PC is running heavy jobs at low priority, so other users know.

## 2. Background jobs

All four write logs to `runs/<job>/job.log` and notes to `runs/<job>/NOTES.md`.

### Job A: `phrase_counts` (heaviest; start first)
Purpose: the two-word-phrase counts that ground truth #2 needs. Labels are applied later.

- **Input:** `data/interim/comments/comments_2016-MM.parquet` (12 files); `user_summary.parquet` (read-only, for each author's comment count).
- **Comments to count:** all in-scope subreddits (config option `subreddits`, default all; if the pilot projection shows the output will not fit, the fallback list is r/politics, r/PoliticalDiscussion and all party-tagged subs). Exclude `[removed]`/`[deleted]` bodies, privacy-script overwritten text (reuse the detector from `build_edges`; move it to a shared function), and comments with `distinguished` set.
- **Text:** use `body_cleaned` (already tokenised and lower-cased; inspect its exact format). Keep alphabetic tokens (apostrophes allowed) of length ≥ 2. Drop URL fragments and pure numbers.
- **Phrases:** adjacent token pairs within one comment. Drop a pair if **both** tokens are stopwords (sklearn `ENGLISH_STOP_WORDS`).
- **Pass 1 (per month, parallel):** per phrase, the number of distinct authors and total count. Merge across months, keep phrases with ≥ 20 distinct authors, then the top 100,000 by distinct authors. Save `data/processed/phrase_counts/vocab.parquet` (phrase_id, phrase, n_users, total_count).
- **Pass 2 (per month, parallel):** for authors with ≥ 5 in-scope comments in 2016, sparse counts per (week_start, subreddit, author, phrase_id), and a totals table per (week_start, subreddit, author) with `m_vocab` (sum of in-vocabulary phrase counts) and `n_comments`. Write `data/processed/phrase_counts/month=2016-MM/{counts,totals}.parquet`.
- **Weeks span months.** After all months finish, a `finalise` step sums rows with the same key that appear in two adjacent monthly files (boundary weeks only) and rewrites those weeks once. Unit-test this with a synthetic case.
- **Pilot and disk check:** after the first pass-2 month, project the total output size. If it exceeds 70% of free disk, set `paused`, write the reason, and stop; do not fill the disk.
- **If at risk of the deadline:** cut in this order: vocabulary to 50,000, minimum comments per author to 10, restrict subreddits to the fallback list.
- **Validation (write to `runs/phrase_counts/VALIDATION.md`):** recount phrases independently for 1,000 random comments and compare; print the 30 most frequent vocabulary phrases; confirm totals equal the sum of counts.

### Job B: `thread_user`
Purpose: the table that ground truth #1 (thread dissimilarity index) is computed from.

- **Input:** the monthly comment Parquet files.
- Exclude removed/deleted/privacy-script comments and `distinguished` comments.
- Per (week_start, subreddit, link_id, author): `n_comments`, `n_replies`, `n_accepted` (score ≥ 1), `score_sum`.
- Monthly units, then the same boundary-week `finalise` step as Job A, output partitioned by week to `data/processed/thread_user/week=YYYY-MM-DD/part.parquet`.
- **Validation:** the sum of `n_comments` equals the number of in-scope comments after exclusions (print both).

### Job C: `clean_edges_and_sampling`
Purpose: the cleaned edge set and the thread samples Kareem will run on.

- **Step 1, per month (row-level cleaning).** Read the existing edge files. Drop edges whose reply or parent text is `[removed]`/`[deleted]` or privacy-script overwritten. Add `body_clean` and `parent_body_clean` with `html.unescape` applied (keep the originals). Keep short replies. Join `distinguished` from the comments table by `comment_id` and keep it as a flag column (do not drop the edge).
- **Step 2, per week (53 units, parallel).** Using all edges of that week:
  1. **Heavy posters:** an author who sent more than 500 replies that week (across all subreddits, measured on the edge set **before** step-1 cleaning; use `author_week_replies` only if it matches this definition, otherwise recompute) is excluded: drop edges they **sent** and edges **sent to them** (replies to a bot or troll are mostly people pointing that out). Authors who merely receive many replies are kept.
  2. **Minimum activity k = 3**, computed **once** (not iterated) on the cleaned week: a user is kept if sent + received ≥ 3; an edge is kept if both endpoints are kept. Match the logic already in `weekly_threshold_sweep` exactly.
  3. Write the **full cleaned week** to `data/processed/edges_clean/week=YYYY-MM-DD/part.parquet`.
  4. **Thread sampling** at each rate in `config.sampling.rates` (default 0.01, 0.05 and 0.20: tiny, small and large, because we do not yet know how many edges Kareem's pipelines can process; 20% is roughly 3.8M edges over the year and may be too much for the LLM). The thread unit is (week, subreddit, link_id), sized by its edge count in that week. **Strata** = subreddit group (the 25 largest subreddits individually plus "other") × size bucket (1, 2–3, 4–7, 8–15, 16–31, 32–99, 100–499, 500+). In each stratum take round(rate × threads) threads with **random rounding** (so the total is unbiased), seeded by a hash of (rate, week_start). Keep every edge of a sampled thread. **Do not re-apply k = 3 after sampling.** Write `data/processed/edges_sampled/rate=0.05/week=.../part.parquet` and `sampled_threads.parquet` (week_start, subreddit, link_id, rate) so ground truth can later be computed on exactly the same sample.
- **Reports:** `results/edge_filters/filter_report.csv` (edges dropped per rule per week, with each rule's effect shown on its own and in sequence); `results/edge_filters/sampling_report.csv` (per week and rate: threads sampled of total, realised share of edges vs the rate, nodes, edges, mean degree and largest-component share against the full cleaned week).
- **Validation:** with the new filters off and only k = 3 on, reproduce the existing threshold-sweep numbers (users and edges per week at k = 3) within 0.5%; print any week that differs. Check that the realised edge share is within 3 percentage points of the rate in typical weeks.
- **If at risk of the deadline:** cut the 0.20 sampling rate first.

### Job E: `within_camp_setup` (light)
Purpose: set up the Sanders-vs-Clinton within-camp test.

- Scan the comments in the Sanders and Clinton subs and count each user's comments under every version in config `count_versions` (default: `all`, `score_ge_1` = not net-downvoted, `score_ge_2` = net-upvoted by someone else, `above_sub_median` = above that subreddit's median score). Label users under **each** version, so we compare posting alone with posting plus score; `score_ge_1` is the primary version for the summary tables. Use the config lists `sanders_subs` (default `SandersForPresident`) and `clinton_subs` (default `hillaryclinton`). Leave the special subreddits (`enoughsandersspam`, `RepublicansForSanders`, `AskTrumpSupporters`, `AskDemocrats`, `AskThe_Donald`) out of labelling.
- Leaning = share of comments in Sanders subs among comments in Sanders + Clinton subs. Classified at ≥ 0.75 (Sanders), ≤ 0.25 (Clinton), with a minimum of `min_comments` (default 5). Both thresholds are config values and are **not final**.
- Output `data/processed/within_camp/users.parquet` (author, then n_sanders, n_clinton, share and label for each count version).
- A light pass over the comments for these users only: per week, how many labelled users of each side were active in r/politics and in r/PoliticalDiscussion, and in total. Save `results/within_camp/weekly_counts.csv` and a plot against the primary calendar (Iowa Feb 1, New Hampshire Feb 9, California Jun 7, Sanders endorses Clinton Jul 12).
- Cross-tabulate with the existing posting-based dem/rep leaning (how many Sanders and Clinton users are also classed dem). Report it. Also report, for each pair of count versions, how many users are labelled in one but not the other and how often labels flip.

### Job G: `label_versions` (medium)
Purpose: settle the labelling rule with numbers. Compare score cutoffs and thresholds, test whether the score rule removes the *wrong* users, and measure coverage among people who post in the supporter subs.

- **Input:** the monthly comment Parquet files, supporter-community subs only (config `supporter_subs`: the party-tagged subs minus `enoughsandersspam`, `RepublicansForSanders`, `AskTrumpSupporters`, `AskDemocrats`, `AskThe_Donald`; 58 subs earlier), and the existing self-declaration table (read-only). Exclude `distinguished`, removed and deleted comments.
- **Counts:** per user per sub under `all`, `score_ge_1` (not net-downvoted), `score_ge_2` (net-upvoted by someone else), `above_sub_median` (above the 2016 median score of that sub) and `weighted` (existing definition). Save `data/processed/label_versions/user_counts.parquet`.
- **Grid:** for each version × `min_comments` in {5, 10, 20} × `share` in {0.70, 0.75, 0.80}: users labelled left, right and mixed, as a share of (a) users who posted in the supporter subs and (b) all users, plus the agreement with self-declarations (and the number of declarers) for each cell, so we can see the coverage-versus-accuracy trade-off and pick the threshold at the knee of the curve. Save as CSV.
- **Agreement and flips** between versions.
- **Direct test of the score rule:** take users labelled by `all` but dropped or changed by each score version (the removed set). Compare their agreement with self-declarations against the agreement of users the version keeps. Report the number of declarers in each group, because they will be small. If the removed set agrees less, the score rule is removing mislabelled users.
- **Coverage:** labelled share among users who posted in supporter subs; comment-count distribution of labelled vs unlabelled users.
- Output `results/label_versions/summary.md` and the tables.

### Job H: `label_coverage` (`depends_on: [thread_user, clean_edges_and_sampling]`)
Purpose: find out whether the labelled pool is big enough **where we measure**, not just overall.

- For `score_ge_1` and `score_ge_2` at 10 comments and 75% (config list), per week: labelled left and right users active in r/politics, r/PoliticalDiscussion and the two together, and in each supporter sub.
- The share of cleaned edges in r/politics + r/PoliticalDiscussion whose author or target is labelled (one side or both), and the share where both ends are labelled.
- Labelled users per side per week inside the 1%, 5% and 20% samples.
- Flag weeks where either side has fewer than `min_users` (default 50) labelled users active in the venue.
- Output `results/label_coverage/weekly.csv`, `summary.md` and a plot over time.

## 3. Foreground jobs (do these while A, B, C, E and G run)

### Job D: ground-truth code and tests
Create `src/ground_truth/` with two modules and tests in `tests/ground_truth/`. Use **only synthetic data**; never real labels.

1. **`dissimilarity.py`:** `dissimilarity_index(rows, n_perm=200, seed=0, min_users=50)` where rows have (thread, user, label). Count **distinct labelled users** per thread. With d_t and r_t the dem and rep users in thread t and D, R their totals over all threads, the index is `0.5 * sum_t |d_t/D − r_t/R|`. Use sparse matrices (a user × thread incidence matrix). Permutation null: shuffle labels across users (same group sizes), recompute. Return `observed`, `null_mean`, `null_sd`, `excess = observed − null_mean`, group sizes and thread count. Return NaN with a reason if either side has fewer than `min_users`.
2. **`leave_out_estimator.py`:** the leave-out estimator of Gentzkow, Shapiro and Taddy (2019). Input: a sparse users × phrases count matrix and a label vector (1 = rep, 0 = dem). For each side P, phrase frequencies `rho_P_j = sum_{i in P} c_ij / sum_{i in P} m_i`. For user i, recompute both with user i removed, `q_{-i,j} = rho_R_{-i,j} / (rho_R_{-i,j} + rho_D_{-i,j})`, and `qhat_i = sum_j (c_ij / m_i) q_{-i,j}`. Then `pi = 0.5 * mean_{i in R} qhat_i + 0.5 * mean_{i in D} (1 − qhat_i)`. Implement over the non-zero entries only (cost proportional to nnz), not with dense loops. Add (a) a **shuffled-label null** (permute labels, keep group sizes) returning mean and SD, and (b) **bootstrap confidence bands** (resample users). Return NaN with a reason below `min_users` per side.
3. **Tests:** (a) index near 1 for camps in fully separate threads, near the null mean for fully mixed ones, and `excess` near 0 for mixed; (b) estimator on synthetic users where both sides draw phrases from the same distribution gives pi ≈ 0.5, and with a mixing parameter from 0 (identical) to 1 (disjoint vocabularies) pi rises monotonically towards 1; (c) the shuffled-label null is ≈ 0.5 even for strongly separated data; (d) a speed check on 50,000 users × 100,000 phrases (about 5M non-zeros) finishing in minutes.

### Job F: housekeeping (do last, after D)
- Commit the `user_profiles` code and tests (including the score-aware labels, `score_lag_scan.py`, `score_aware_labels.py` and `user_signal_coverage.py`), the framework, jobs A, B, C and E code, `src/ground_truth/` and tests, and `config/prep_jobs.yaml` to the existing branch. **Never commit data, `runs/` or results.** Verify `.gitignore` covers `runs/`, `data/processed/` and `results/`.
- Append a "Decisions, Oct 28" section to `docs/decisions.md` (do not rewrite what is there):
  - weekly windows, Monday start, all of 2016; monthly only as a check
  - minimum activity k = 3, computed once on the cleaned full week, before sampling
  - authors sending more than 500 replies a week are excluded together with replies sent to them (replies to a bot or troll are mostly people pointing that out); authors who only receive many replies are kept
  - `[removed]`/`[deleted]` and privacy-script edges are skipped; HTML symbols are unescaped; short replies are kept
  - sampling: whole threads stratified by size, a fixed **rate** per week (not a fixed budget), so activity trends survive
  - special subreddits stay in the data and measurement but are not used for labelling
  - ground truth #1 = thread dissimilarity index with a shuffled-label null; ground truth #2 = leave-out estimator with a shuffled-label null (about 0.5) and bootstrap bands; run on all weeks and flag weeks with few labelled users
  - the data files live on a shared PC that Kareem has access to, so no transfer is needed; **open:** who runs integration (expected to be Kareem, with Yasmina helping online if she can). Kareem makes no decisions; Yasmina has full authority and sends them to him
  - **the labelling rule is still open** (candidate: posting share in the 58 supporter-community subs, counted only on comments with score ≥ 1, at least 10 comments and 75% on one side: 107,730 users labelled, about 80% agreement with self-declarations; scores are a single reading about four weeks after posting, and the choice between the all / accepted / weighted versions barely changes labels); thresholds are not locked, and the score cutoff (≥ 1 vs ≥ 2 vs above the subreddit median) is also open; Job G compares them
- Push and open a PR into main. Give me the link.

## 4. Order of work
1. Read this file, inspect the repo, write `config/prep_jobs.yaml`.
2. Build the framework (section 1) and prove detachment with the dummy job.
3. Print the allocation plan.
4. Build and launch **A first**, then **C**, **B**, **G** and **E**, then **H** (it waits for B and C). Confirm each reports a healthy first heartbeat (state `running`, no error) before moving on; wait up to 3 minutes per job.
5. While they run, do **D**, then **F**.
6. Do **not** wait for A, B, C, E, G or H to finish.

## 5. Final report (print this when you stop)
- A table: job, PID, workers, state, ETA (duration and clock time), deadline status.
- The exact PowerShell commands, in a code block: check progress, watch progress, stop a job, resume after a reboot.
- Anything you assumed, and anything that failed or looks at risk.
- The PR link.
