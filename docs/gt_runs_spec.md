# Ground-truth runs spec (Oct 28), second spec

Read this whole file before doing anything. It computes the weekly ground truths. It is the follow-up to `docs/prep_jobs_spec.md`, whose outputs it consumes. Build and launch everything here as **detached background jobs** with the existing job framework, do the foreground work while they run, and report back. Do not wait for the jobs to finish.

## 0. Preconditions (check first, stop with a clear message if any fails)

1. The job framework from the first spec exists and works (`scripts/launch_job.py`, `scripts/jobs_status.py`, heartbeat, ETA, resume, load guard, detached launch, `depends_on`). **Reuse it. Do not rebuild it.**
2. These outputs exist and their jobs are `done` (check `runs/<job>/status.json`): `phrase_counts` (Job A, including `vocab.parquet`), `thread_user` (Job B), `clean_edges_and_sampling` (Job C, including `sampled_threads.parquet` for each rate), `label_versions` (Job G), `within_camp_setup` (Job E). If one is not done, say which and stop. Do not start work on partial inputs.
3. **`config/labels.yaml` holds the labelling rule.** There is no lock step and no approval gate. If the file is missing, create it from the template below with exactly these values and carry on. If I give different values at launch, use mine. Template:

```yaml
# count_version names must match the ones Job G actually wrote
main:   {count_version: score_ge_1, min_comments: 10, share: 0.75}
broad:  {count_version: score_ge_1, min_comments: 5,  share: 0.75}
strict: {count_version: score_ge_2, min_comments: 10, share: 0.80}
gold:   {source: vote_declarations, exclude: [johnson, stein, conflicting]}
exclude_heavy_posters: true     # authors who sent > 500 replies in a week
use_additions: false            # Conservative, HillaryForPrison, EnoughTrumpSpam; only if Job G counted them
min_users_per_side: 50
```

Rules for this spec are the same as the first: never ask me anything, pick the conservative default and log it in `runs/<job>/NOTES.md`, inspect real schemas instead of trusting the names written here, read-only on existing outputs, every setting in `config/gt_runs.yaml` (no magic numbers), import the repo's week function, Windows and PowerShell, add dependencies to `pyproject.toml`, no internet downloads.

## 1. Definitions

- **Sides:** `rep` and `dem` (the left/right wording in earlier outputs maps to dem/rep). Users who are mixed or unlabelled are excluded from ground truth.
- **Which label sets run: only `main`**, plus `gold` in T1 and T2 on the full week only (it is small and cheap). `broad` and `strict` stay in `config/labels.yaml` as noted options and are **not run** unless `label_sets_extra` in `config/gt_runs.yaml` lists them (default: empty list). Everywhere below that says "label set" means the sets that actually run.
- **Label sets defined:** `main`, `broad`, `strict` (posting-based, from the supporter-community subs, using Job G's per-user counts and the parameters in `config/labels.yaml`) and `gold` (self-declared votes: Trump means rep; Clinton or Sanders means dem; Johnson, Stein and conflicting declarations excluded).
- **Venues** (where polarization is measured), in `config/gt_runs.yaml`:
  - `shared`: r/politics + r/PoliticalDiscussion.
  - one venue per special subreddit: `AskTrumpSupporters`, `enoughsandersspam`, `RepublicansForSanders`, `AskDemocrats`, `AskThe_Donald`.
  - an optional `extra_venues` list (default empty), for example The_Donald, SandersForPresident, hillaryclinton.
- **Leave-the-venue-out labelling.** For a venue made of supporter or special subs, each user's side is computed from their comments **outside that venue**, by subtracting the venue's counts before applying the rule. For `shared` nothing needs subtracting. This is what lets us measure inside tagged subreddits without building the answer into the labels. Save each venue's labels separately.
- **Heavy posters:** if `exclude_heavy_posters`, drop users who sent more than 500 replies in that week (use the list Job C produced; if it was not saved, recompute it from `author_week_replies` in the raw edge table).
- **Sample versions:** `full` (the whole cleaned week), and one per sampling rate (`0.01`, `0.05`, `0.20`) restricted to the same sampled threads the pipelines will read (`sampled_threads.parquet`).
- A (week, venue, label set) cell with fewer than `min_users_per_side` labelled users on either side gets `NaN` and a `low_n` flag. **Never drop the row.**

## 2. Jobs

All jobs use the framework: detached, resumable, per-unit checkpoints, heartbeat and ETA, load guard, `depends_on`, own `runs/<job>/` folder, snapshot of the code at launch.

### Job L: `labels` (light; start first)
- Apply `config/labels.yaml` to Job G's per-user counts and write `data/processed/labels/venue=<venue>/set=<main|broad|strict|gold>.parquet` (author, side, share, n_comments) and `data/processed/labels/summary.csv` (users per side per venue per set).
- Write `data/processed/camp_recovery/labels.parquet` (author, side_main, side_broad, side_strict, side_gold) for the **shared** venue. This folder is for evaluating Pipeline B only; **nothing the pipelines read may import or depend on it.**
- Copy `config/labels.yaml` into `results/ground_truth/labels_used.yaml` so the exact rule is stored with the results.
- Validation: print users per side per set; assert that no user has two sides within one (venue, set).

### Job T1: `gt1_dissimilarity` (`depends_on: [labels]`) — priority 1
- Unit = one week. For each (week, venue, label set, sample version): from `thread_user`, restricted to the venue's subs, the labelled users and (for sample versions) the sampled threads, compute `dissimilarity_index` from `src/ground_truth/dissimilarity.py` with `n_perm` (default 200) and a fixed seed.
- Output `results/ground_truth/gt1_weekly.parquet` with columns: week_start, venue, label_set, sample, n_dem, n_rep, n_threads, n_threads_both, observed, null_mean, null_sd, excess, low_n.

### Job T2: `gt2_leave_out_full` (`depends_on: [labels]`) — priority 2
- Unit = one week. For each (week, venue, label set) on the **full** week: build the users × phrases sparse matrix from `phrase_counts` (sum over the venue's subs; only labelled users with at least `min_phrases` in-vocabulary phrases that week, default 20), then run `leave_out_polarization` from `src/ground_truth/leave_out_estimator.py` with `n_perm` (default 100) shuffled-label runs and, for `main` on `shared` only, `n_boot` (default 100) bootstrap resamples.
- **Contraction filter (no rerun of Job A):** `body_cleaned` splits contractions into fragments, so the vocabulary is full of phrases like "do n't". Before building the matrix, drop every vocabulary phrase where either word is a contraction fragment (`n't`, `'s`, `'re`, `'ve`, `'ll`, `'m`, `'d`). Keep the list in `config/gt_runs.yaml` as `exclude_tokens`, apply the identical filter in T3, and log how many phrases were dropped. Run `main` / `shared` / `full` both with and without the filter in T5 and report how far `pi` moves.
- Output `results/ground_truth/gt2_weekly_full.parquet`: week_start, venue, label_set, n_dem, n_rep, pi, null_mean, null_sd, boot_lo, boot_hi, low_n.

### Job T3: `gt2_leave_out_samples` (`depends_on: [labels, gt2_leave_out_full]`) — priority 3
- `phrase_counts` is aggregated per (week, subreddit, author), so it **cannot** be restricted to sampled threads. For each sampling rate and week, re-count phrases for the **sampled threads only**: read the comments belonging to the sampled threads (join on week, subreddit, link_id), restrict to labelled users, and count phrases with Job A's counting function and **the same `vocab.parquet`**. The interim Parquet has no `body_cleaned`, so read the comment text the same way Job A does (from the raw dumps, joined to the sampled threads), and reuse Job A's reader and counting function rather than writing a new one. Do not save the raw counts; compute the estimator immediately and write only the result. Same exclusions as Job A.
- Output `results/ground_truth/gt2_weekly_samples.parquet` (same columns as T2, plus `sample`).
- Run `main` on `shared` first, then the other sets and venues.

### Job T4: `gt_within_camp` (`depends_on: [labels, gt2_leave_out_full]`) — priority 3
- Using Job E's Sanders-vs-Clinton labels (main count version, `score_ge_1`), compute both ground truths per week on the **shared** venue, treating Sanders-leaning as one group and Clinton-leaning as the other. Output `results/ground_truth/within_camp_weekly.parquet` and a plot with the primary calendar marked (Iowa Feb 1, New Hampshire Feb 9, California Jun 7, Sanders endorses Clinton Jul 12). Expectation to check, not assume: it rises through the primaries and eases after the endorsement.

### Job T5: `gt_variants` (`depends_on: [gt1_dissimilarity, gt2_leave_out_full]`) — priority 4, drop first if short of time
- Rerun the `main` / `shared` / `full` cell for `gt1` and `gt2` with: `exclude_heavy_posters: false`; and weeks restricted to those with `n` above the median. Output `results/ground_truth/variants_weekly.parquet`.
- Report how far each ground-truth series moves compared with the main run (Spearman between series, mean absolute difference).

### Job S: `gt_summary` (`depends_on: [gt1_dissimilarity, gt2_leave_out_full]`, and the others if done)
Write `results/ground_truth/summary.md` and plots, including:
- GT#1 `excess` and GT#2 `pi` over the 53 weeks for `main` / `shared` / `full`, with the null band, the bootstrap band for GT#2, `low_n` weeks marked, and these events annotated: Iowa (Feb 1), California (Jun 7), conventions (Jul 18–28), first debate (Sep 26), Access Hollywood tape (Oct 7), Comey letter (Oct 28), Election Day (Nov 8).
- **Null checks:** the shuffled-label GT#2 mean should be about 0.5 and GT#1 `excess` about 0 under shuffling; flag it loudly if not.
- Labelled users per side per week.
- Agreement between GT#1 and GT#2 (Spearman across weeks) and stability across label sets (`main` vs `broad` vs `strict` vs `gold`) and across sample versions (`full` vs each rate).
- The **Kareem-ready file** `results/ground_truth/ground_truth_weekly.csv`: week_start, gt1_excess, gt2_pi, n_dem, n_rep, low_n for `main` / `shared`, with extra columns for each sample rate. Keep a column guide at the top of `summary.md`.

## 3. Foreground work (while the jobs run)

- **Camp-recovery evaluation script** `scripts/camp_recovery_eval.py <pipeline_b_user_sides.csv>`: reads a file of (author, side) from Pipeline B, joins to `data/processed/camp_recovery/labels.parquet`, and reports accuracy, adjusted Rand index and confusion counts, for each label set. Side labels in Pipeline B's output may be swapped (a graph split has no inherent direction), so compute accuracy both ways and report the better one. Tests with synthetic data (perfect, random and swapped).
- Tests for Jobs L, T1 and T2 on small synthetic inputs, including that leave-the-venue-out labelling subtracts the venue's counts and that a low-N cell returns NaN with `low_n` set.
- Append to `docs/decisions.md` a section "Ground truth (<date>)" with the labelling rule used (copy of `config/labels.yaml`), the venue definition and leave-the-venue-out labelling, the sample versions, and the null checks. Commit code and tests (never data, `runs/`, or `results/`), push and open a PR.

## 3b. Memory and disk rules (Job A was memory-killed several times; these jobs must not repeat that)

- **One week per unit, never more than one week in memory per worker.** Release every large object (matrices, dataframes) at the end of the unit and run `gc.collect()`.
- **Measure before scaling.** Run one real week for T2 and one for T3, record peak memory per worker (`psutil`), and set the worker count so that workers × measured peak stays below 60% of free RAM. Log the numbers in `runs/<job>/NOTES.md`. Never use the 2 GB default without measuring.
- **T3 order:** rate 0.01 first, then 0.05, then 0.20 last. At most 2 workers for T3 until a 0.05 week has been measured. If a 0.20 week's peak exceeds the budget, read its threads in batches and sum the counts, or skip 0.20 and say so.
- **Never write phrase counts for the samples.** T3 computes the estimator immediately and writes only result rows (one row per week, venue, label set and rate). Expected disk use for all ground-truth outputs: under 1 GB. Temporary files go in one folder and are deleted at the end of each unit.
- **Crash safety:** a unit writes to a temporary file and renames it only when complete, so a kill leaves no half-written output; a killed unit is simply redone on `--resume`. If a job is memory-killed twice, halve its workers automatically and note it.
- Keep the framework's load guard on (RAM 85% pause, 92% stop dispatch) and keep priority below normal.

## 4. Order, deadline and priorities

1. Check preconditions (section 0). Print an allocation plan using the framework's rules.
2. Launch **L**, then **T1** and **T2** together, then **T3** and **T4**, then **T5**, then **S**. They wait on `depends_on` by themselves.
3. Suggested worker split among T1, T2, T3, T4: 25%, 40%, 25%, 10%.
4. Take `--deadline-hours` from my instruction at launch; if I give none, run with no deadline (still print ETAs, and make no cut suggestions). If any job's ETA is past the deadline, say which setting to cut, in this order: T5 first; then the 0.20 sample in T3; then permutations (`n_perm` 100 to 50); never T1, T2 for `main` / `shared` / `full`.
5. Do not wait for the jobs to finish.

## 5. Final report (print when you stop)
- A table: job, PID, workers, state, ETA (duration and clock time), deadline status.
- The PowerShell commands, in a code block: check progress, watch, stop, resume.
- Anything you assumed, anything that failed or looks at risk, and the PR link.
