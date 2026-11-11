# SNA Handbook: Polarization-Network-Analysis

*Handover document. Snapshot of branch `llm_implementation` as of 2026-11-08 (last commit `022dba0`, plus the uncommitted full-5% results listed in section 15). Every number below was read from a file in this repo, and its source is named next to it. Where a number appears here for the first time (the full 5% evaluation of 2026-11-07, section 12.6), this handbook says so.*

---

## Contents

0. [Status in one page](#0-status-in-one-page)
1. [Research question and study design](#1-research-question-and-study-design)
2. [People, machines and environment](#2-people-machines-and-environment)
3. [Repository map](#3-repository-map)
4. [Data pipeline: raw dumps to cleaned weekly reply edges](#4-data-pipeline-raw-dumps-to-cleaned-weekly-reply-edges)
5. [Camp labels: who is a Democrat or a Republican](#5-camp-labels-who-is-a-democrat-or-a-republican)
6. [Ground truths GT#1 and GT#2](#6-ground-truths-gt1-and-gt2)
7. [The LLM engine (shared extraction)](#7-the-llm-engine-shared-extraction)
8. [Pipeline A: text scoring](#8-pipeline-a-text-scoring)
9. [Pipeline B: signed network and frustration](#9-pipeline-b-signed-network-and-frustration)
10. [Synthetic validation of Pipeline B](#10-synthetic-validation-of-pipeline-b)
11. [Evaluation spec and the RQ1 comparison script](#11-evaluation-spec-and-the-rq1-comparison-script)
12. [Every real-data run, in order, with results](#12-every-real-data-run-in-order-with-results)
13. [Findings: what we can and cannot claim](#13-findings-what-we-can-and-cannot-claim)
14. [Tests](#14-tests)
15. [Run book: how to reproduce and continue](#15-run-book-how-to-reproduce-and-continue)
16. [Decision log summary](#16-decision-log-summary)
17. [Limitations and known issues](#17-limitations-and-known-issues)
18. [Comparison with the proposal](#18-comparison-with-the-proposal)
19. [Where to go next](#19-where-to-go-next)
20. [Glossary](#20-glossary)

---

## 0. Status in one page

**Question (RQ1).** Does a structure-aware pipeline (LLM agree/disagree signs → weekly signed user graph → normalised frustration, "Pipeline B") track a ground-truth polarization proxy better than direct LLM text scoring averaged per week ("Pipeline A")?

**Data.** Reddit Politosphere, calendar year 2016, 53 weekly windows (Monday 00:00 UTC; first and last weeks partial). Measurement venue = r/politics + r/PoliticalDiscussion ("shared venue").

**What exists and works**

| Component | State | Where |
|---|---|---|
| Download, parse, reply edges for 2016 (28.1M edges, 42.4M comments, 855k users) | done | `src/dataset/` |
| Detached, resumable background-job framework with load guard | done | `src/prep_jobs/`, `scripts/launch_job.py` |
| Cleaning (text rules, heavy posters, k = 3) and stratified thread sampling at 1% / 5% / 20% | done | job `clean_edges_and_sampling` |
| Camp labels (posting share in 58 supporter subs, score ≥ 1, ≥ 10 comments, 75%) | done, rule fixed in `config/labels.yaml` | job `labels` |
| GT#1 thread dissimilarity (excess over shuffled null) + robustness, drivers, fixed-panel checks | done | `src/ground_truth/dissimilarity.py`, jobs `gt1_*` |
| GT#2 partisan language (transfer design, phrase estimator, cap 100) | done | job `gt2_transfer` |
| LLM engine (Ollama, constrained JSON, logprobs, SQLite cache, provenance) | done | `src/algorithms/llm/` |
| Primary model chosen on a human-labelled pilot without GT: `qwen3.5:9b-q4_K_M`, prompt v2 | frozen | `docs/llm_decisions.md` D-M7 |
| Pipeline A (additive 0–10 edge score, weekly mean) | done | `src/algorithms/pipeline_a/` |
| Pipeline B (4 sign sources, signed graph, frustration heuristic + ILP check, 2 nulls, camp recovery) | done | `src/algorithms/pipeline_b/`, `src/algorithms/eval/` |
| Synthetic validation of P_B (pre-stated criteria) | done, 6 of 8 checks pass | `docs/reports/phase_4_report.md` |
| Evaluation spec, written before any GT comparison | frozen (eval-spec-v1 + D-E3) | `docs/eval_spec.md`, `scripts/llm/rq1_compare.py` |
| **Full 5% shared-venue tier extracted and evaluated** (617,568 edges, all 53 weeks, 0 failures) | **done 2026-11-07, not yet written up or committed** | `results/llm/rq1_5pct/rq1_5pct_full_*` |

**Headline result (full 5% tier, n = 53 weeks; section 12.6).**

| | rho vs GT#1 | rho vs GT#2 |
|---|---|---|
| P_A (text) | −0.61 (p 0.06) | −0.48 (p 0.21) |
| P_B (network) | −0.03 (p 0.87) | +0.17 (p 0.38) |
| delta rho_B − rho_A (95% block-bootstrap CI) | +0.58 [0.27, 0.92] | **+0.65 [0.21, 0.94]** |

The pre-registered rule ("B wins if the delta CI on GT#2 excludes 0 and rho_B > 0") is met on both the primary (full-week GT) and the secondary (5%-sample-matched GT) truth. **But** rho_B is not distinguishable from zero, the gap comes mostly from A being *negatively* related to both truths, no Holm correction was applied, and first-difference correlations are about zero for both pipelines. The honest reading is: *neither pipeline demonstrably tracks either ground truth; B is not worse than A, and A runs the wrong way for reasons not yet investigated.*

**Not done** (sections 18–19): robustness model, composition baselines, pre/post 2016-06-27 split, Holm correction, noise ceilings, sign-quality experiments (RQ3), window/density analysis (RQ2), forecasting (RQ4), anything beyond 2016 or beyond the shared venue.

---

## 1. Research question and study design

### 1.1 The question
Two families of methods measure online polarization:

1. **Text scoring** (Pipeline A, after Chowdhury et al. 2026): an LLM rates each interaction (stance, affect, hostility); the weekly polarization score is an aggregate of those ratings.
2. **Signed networks** (Pipeline B, after Guo et al. 2026): an LLM labels each reply as agree / disagree; replies become positive / negative edges between users; polarization is how cleanly the signed graph splits into two opposing camps (low *frustration*).

B costs much more to build. RQ1 asks whether the structure buys anything. Working title in the docs: *"Is the Network Worth Building? LLM Scores vs. LLM-Signed Networks for Measuring Polarization."*

Secondary questions from the README: RQ2 (when does structure help: window size, density, polarization level), RQ3 (how does sign-inference quality affect results), RQ4 (does either pipeline improve forecasting), RQ5 (drivers of polarization shifts; cut). See section 18.

### 1.2 The controlled-comparison design (decision D-S1)
**One LLM call per reply edge** returns all four attributes (agreement, emotional_charge, hostility, personal_attack) plus agreement probabilities. Pipeline A and Pipeline B both read this same cached output and differ **only in aggregation** (mean of an edge score vs. signed graph + frustration). Any A/B difference is therefore about structure, not prompts or models. Cost of this choice: A and B are not independent replications of the two cited papers.

### 1.3 The referee: LLM-free ground truths
Two weekly series computed without any LLM, from user camp labels derived from subreddit membership:
- **GT#1** (who talks to whom): do Democrats and Republicans comment in different threads? Thread dissimilarity index minus a shuffled-label null.
- **GT#2** (how they talk): can you tell the camps apart by their two-word phrases in the neutral venue? Phrase-leaning estimator trained in partisan subs, applied in the shared venue.

GT#1 is structural (it "leans towards" B), GT#2 is linguistic (it "leans towards" A). The decision rule therefore requires each pipeline to win on the truth that leans *against* it (section 11).

### 1.4 Firewall between labels and pipelines (D-B5)
Camp labels enter only the ground truths and the camp-recovery evaluation (`src/algorithms/eval/camp_recovery.py`). A test fails if any module in `pipeline_a`, `pipeline_b` or `llm` references labels or the eval module. The LLM never sees usernames, subreddits, labels or GT fields (`FORBIDDEN_FIELDS` in `src/algorithms/llm/schema.py`, also tested).

### 1.5 Pre-registration discipline
Model, prompt, Pipeline A rule table, Pipeline B thresholds and the evaluation spec were all fixed **before** any comparison with GT was computed. The model was chosen on human pilot labels, never on GT (D-M1). The synthetic-validation criteria were written before the checks were run, and were not changed after failures. Every decision is in `docs/llm_decisions.md` (LLM side) or `docs/decisions.md` (data side).

---

## 2. People, machines and environment

### 2.1 Ownership
| Person | Owns | Code |
|---|---|---|
| Yasmina | Download, parsing, reply graph, user profiles, labels, both ground truths, background-job framework | `src/dataset/`, `src/prep_jobs/`, `src/gt_runs/`, `src/ground_truth/`, `config/*.yaml` |
| Kareem | LLM engine, Pipelines A and B, synthetic validation, evaluation, integration runs | `src/algorithms/`, `configs/llm/`, `scripts/llm/` |

Two config folders exist and are easy to confuse: `config/` (data/GT side) and `configs/llm/` (LLM side). A standing rule from the LLM-side brief: LLM code never edits the data-side packages; problems found in data outputs go to `docs/handoff_issues.md`.

### 2.2 Machines used
| Machine | Used for | Notes |
|---|---|---|
| MacBook, 24 GB unified memory | Phase 1–4 development, pilot, Tier 0 extractions | llama3.1:8b baseline 0.77 edges/s; qwen3.5:9b 0.64–0.81 edges/s; 5% tier infeasible there (~220 h) |
| Windows 11 PC, RTX 3090 24 GB, 64 GB RAM, i9-11900F | data jobs, ground truths, the 5% extraction | qwen3.5:9b ~1.9–2.9 pairs/s; GPU is shared with other projects (Open-Jev, Vivado jobs from another session have competed for it) |

### 2.3 Environment
- Python 3.12 venv at `.venv` on Windows (`pip install -e .`, dependencies in `pyproject.toml`; the Mac used Python 3.14). Key packages: pandas, pyarrow, numpy, scipy (HiGHS MILP), numba, networkx, scikit-learn, psutil, requests, tqdm, vaderSentiment, matplotlib, PyYAML.
- Ollama 0.40.11 (Mac) / 0.40.13 (Windows). Models pulled on Windows: `qwen3.5:9b-q4_K_M` (6.6 GB, primary) and `gemma3:12b` (8.2 GB, robustness, not yet scored). `qwen3.5:27b-q4_K_M` exists on a portable SSD only (D-M8).
- Windows server launch: `scripts/llm/start_ollama.ps1` sets `OLLAMA_NUM_PARALLEL=4`, `MAX_LOADED_MODELS=1`, `FLASH_ATTENTION=1`, `CONTEXT_LENGTH=4096`, `KEEP_ALIVE=30m`.
- `config/config.yaml` (window 30 days, `llama3.2`, seed 42) is the original scaffold and is **not used** by the current pipelines.

---

## 3. Repository map

```
config/               data & GT settings: prep_jobs.yaml, gt_runs.yaml, labels.yaml (Yasmina)
configs/llm/          llm.yaml (engine), pipeline_a_rules.yaml, pipeline_b.yaml (Kareem)
src/dataset/          download_politosphere, parse_comments, build_edges, windowing, graph_stats, user_profiles, ground_truth (stub)
src/common/           text_filters (removed/deleted/privacy-script detector), io, types
src/prep_jobs/        job framework (runner, governor, allocation, util, data) + jobs/ (one module per job)
src/gt_runs/          labels (leave-the-venue-out), cells (GT1/GT2 per week), transfer (GT#2 transfer), gt1_robust, gt1_panel, fix_options
src/ground_truth/     dissimilarity.py (GT#1 index), leave_out_estimator.py (Gentzkow-Shapiro-Taddy)
src/algorithms/llm/   schema (prompts, JSON schema), backends (Ollama, Mock), extract, store (SQLite), data (input loader), provenance, label_tool
src/algorithms/pipeline_a/  aggregate.py (edge score + weekly aggregation); stance_scoring.py is an unused stub
src/algorithms/pipeline_b/  sign_inference.py, structural.py (graph, nulls, P_B), frustration.py (heuristic + ILP)
src/algorithms/eval/  camp_recovery.py (only label reader), metrics.py; null_model.py is an unused stub
src/algorithms/synthetic/   generator.py (signed block model), calibration.py
scripts/              launch_job, jobs_status, plan_jobs, plan_gt_jobs, camp_recovery_eval, gt2_fix_*, measure_unit
scripts/llm/          run_extraction, run_pipeline_a, run_pipeline_b, eval_camp_recovery, rq1_compare, run_synthetic,
                      pilot_*, bench*, validate_probs, sweep_parallel, overnight, watch_5pct, make_slide_figures,
                      rq1_bootstrap_stability, rq1_partial_week_sensitivity, synth_*, start_ollama.ps1
experiments/phase2_data_pipeline_check/  score-lag scan, score-aware labels, user-signal coverage, sampling checks
tests/                pytest suite (section 14)
docs/                 decisions, specs, reports, logs (this file)
results/              small result files (only results/llm/ is committed; the rest is git-ignored)
runs/                 job and run folders (git-ignored)
data/                 raw, interim, processed, llm (git-ignored)
logs_llm/             extraction, supervisor and evaluation logs (git-ignored)
Kareem Presentation Context/  midway-presentation bundle (snapshot 2026-11-01, uncommitted)
```

Stubs that raise `NotImplementedError` and are **not** part of the working system: `src/algorithms/pipeline_a/stance_scoring.py`, `src/algorithms/eval/null_model.py`, the first function in `src/algorithms/synthetic/generator.py` (`generate_signed_graph`; the working one is `generate`), and `docs/decisions.md` sections "Ground-truth mapping rule", "Null-model definition", "Open questions" (placeholders superseded by later sections).

---

## 4. Data pipeline: raw dumps to cleaned weekly reply edges

### 4.1 Source and download
- Reddit Politosphere (Zenodo record 5851729): monthly bz2 JSON-lines comment dumps 2008-01 to 2019-12 plus `subreddits_metadata.json` and `users_metadata.json`.
- `src/dataset/download_politosphere.py`: resumable downloader; 158/158 files complete with checksums (`results/dataset_size/summary.md`).
- Full dataset: 288.8M raw comments, 605 subreddits, 3.66M distinct authors, 35.6 GiB compressed. 2016 alone: 51.1M raw comments (2015 → 2016 jumps 4.3×). Proposal volume figures (12.0M in 2015, 71.0M in 2019) match the **raw** counts.
- No author flair exists in the raw files (`results/flair_check/summary.md`), so flair cannot be a label source.

### 4.2 Scope (decided 2026-10-21, `docs/decisions.md`)
- **Time:** all of 2016 (midway scope was June–Dec; expanded because one month parses in ~1.5 min).
- **Subreddits:** all in `subreddits_metadata.json` except non-US regions (`ca`, `eu`, `me`, `uk`, `world`) → 522 US political subreddits.
- **Comment filters** while streaming (`src/dataset/parse_comments.py`): `created_utc` in range; US subreddit; author not `[deleted]`, not flagged bot/automoderator in `users_metadata.json`; language `en` or `un` (undetermined, ~14%, kept).
- Result: 42.4M cleaned comments, 855,399 users (~17% of 2016 raw removed). Output: `data/interim/comments/comments_2016-MM.parquet`.

### 4.3 Reply edges (`src/dataset/build_edges.py`, `windowing.py`, `graph_stats.py`)
- **Edge** = one comment replying to another comment (`parent_id` starts `t1_`), directed replier → parent author. Replies to posts (`t3_`) are never edges. Parents are looked up across all parsed months.
- **Dropped at build time** (counts in `data/interim/edges/stats/edges_summary.json`): of 30,427,105 replies, parent author deleted 2,124,628; parent bot 65,572; parent non-English 68,538; parent not in data (pre-2016) 6,554; self-replies 62,792 → **28,099,021 edges**, 12 monthly files, 6.4 GB.
- **Flag columns kept:** `reply_removed`, `parent_removed`, `author_week_replies`, `body_repeat_count`; plus `week_start` (Monday 00:00 UTC) and `month_start`. Column guide: `docs/edge_files.md`.
- **Full weekly reply graphs** (`results/graph_stats/weekly_stats.csv`): mean 79k nodes, 417k directed user pairs, 530k replies per week; reciprocity 0.41; largest component 98%; r/politics ≈ 47% of replies, r/PoliticalDiscussion ≈ 5.5%.

### 4.4 Per-user signals (`src/dataset/user_profiles.py`)
Per-user and per-user-per-subreddit tables in `data/processed/user_profiles/`: comment counts under several score rules, self-declarations (vote and identity, regex with negation guard), linked domains. Findings (`results/user_signals/summary.md`, `results/score_aware/summary.md`):
- 30,272 users (3.5%) self-declare; 26,506 with one clear label. 29% post a link.
- Scores are a single reading ~4 weeks after posting (retrieval lag median 27.8 days), not early snapshots; ~30% of comments sit at score 1.
- Posting-based lean agrees with self-declarations 77.8% (all party subs) to 80.4–80.6% (58 supporter subs).

### 4.5 Background-job framework (`src/prep_jobs/`, spec `docs/prep_jobs_spec.md`)
Every heavy data job runs through it.
- `scripts/launch_job.py <job> [--workers N] [--resume] [--fresh]` starts a fully **detached** process (survives closing VS Code / the shell), copies `src/` and `config/` into `runs/<job>/snapshot/` and runs from the snapshot.
- Units (month/week) with atomic temp-file-then-rename writes; per-unit checkpoints in `runs/<job>/checkpoints/`; retry 3× with backoff; `done` / `done_with_failures`.
- Heartbeat `runs/<job>/status.json` every 15 s with ETA measured in input bytes; `scripts/jobs_status.py [--watch S]` shows a table with STALE / CRASHED / AT RISK flags.
- `depends_on` chains; lock file prevents duplicate runs; `runs/<job>/STOP` stops one job; `runs/PAUSE_ALL` pauses all.
- **Load guard** (`governor.py`, thresholds in `config/prep_jobs.yaml`): RAM > 85% throttles, > 92% for 30 s pauses until < 75%; free disk < 25 GB or < 10% pauses; 2 cores reserved; per-worker RSS limit 5 GB (halve batch), kill above 1.6× and retry smaller; halve workers after two memory kills; below-normal priority; sleep prevention.

### 4.6 Prep jobs (Oct 28) and what each produced

| Job | Purpose | Main output |
|---|---|---|
| `comment_flags` | `distinguished` (moderator) flags missing from the parsed Parquet | `data/interim/comment_flags/` |
| `phrase_counts` (A) | bigram counts for GT#2. Tokens from `body_cleaned`, alphabetic ≥ 2 chars, URL fragments dropped, pairs dropped if both are stopwords. Vocabulary = top 100,000 phrases with ≥ 20 distinct authors; counts per (week, subreddit, author) for authors with ≥ 5 in-scope comments; boundary weeks merged in a finalise step | `data/processed/phrase_counts/` (vocab + month=*/counts, totals) |
| `thread_user` (B) | per (week, subreddit, thread, author): n_comments, n_replies, n_accepted, score_sum, for GT#1 | `data/processed/thread_user/week=*/` |
| `clean_edges_and_sampling` (C) | cleaning + thread sampling (4.7, 4.8) | `data/processed/edges_clean/`, `data/processed/edges_sampled/rate=*/`, `results/edge_filters/` |
| `within_camp_setup` (E) | Sanders vs Clinton labels | `data/processed/within_camp/users.parquet`, `results/within_camp/` |
| `label_versions` (G) | compare label rules (section 5) | `data/processed/label_versions/user_counts.parquet`, `results/label_versions/` |
| `label_coverage` (H) | labelled users per side per week where we measure | `results/label_coverage/` |

### 4.7 Cleaning (Job C step 1–2; decided Oct 28)
Applied per week on the full edge set, in this order (`results/edge_filters/filter_report.csv`, totals over 53 weeks):

| Step | Rule | Edges removed |
|---|---|---|
| start | 28,099,021 | |
| text rules | reply or parent text `[removed]`/`[deleted]` or overwritten by a privacy script (regex in `config/prep_jobs.yaml`; detector `src/common/text_filters.py`); HTML entities unescaped into `body_clean` / `parent_body_clean` | 19,139 |
| heavy posters | authors sending > 500 replies in the week (395 author-weeks): drop edges they sent **and** received | 477,268 |
| minimum activity k = 3 | user kept if sent + received ≥ 3 in the week, computed once; edge kept if both ends kept | 2,258,228 |
| **final** | | **25,344,390** |

Short replies are kept; `distinguished` is a flag column (17,933 edges), not a filter.

### 4.8 Thread sampling (Job C step 3–4)
- Unit = thread (week, subreddit, link_id); every edge of a sampled thread is kept; k = 3 is **not** re-applied after sampling.
- Strata = subreddit group (25 largest individually + "other") × thread-size bucket (1, 2–3, 4–7, 8–15, 16–31, 32–99, 100–499, 500+). Take round(rate × threads) per stratum with random rounding, seed = hash(rate, week).
- Rates 1%, 5%, 20%. Realised edge shares 1.07% / 4.77% / 20.05%. Mean per week (all venues): 5.1k / 22.8k / 96.3k edges; mean degree 3.0 / 3.8 / 6.2; largest component 69% / 90% / 97% (`results/edge_filters/sampling_report.csv`).
- **The three samples are independent draws, not nested** (570 of 10,356 1% threads are in the 5% sample; D-S5).
- Validation (`results/sampling_check_production/summary.md`): the thread sampler preserves the full-week degree distribution better than uniform edge sampling of the same size (degree KS vs the cleaned week, shared venue: 0.35 vs 0.61 at 5%), in 9/9 test weeks.
- Shared-venue edges per rate (input to the LLM): **1% 152,496; 5% 617,633; 20% 2,736,934** (before dedup; Phase 1 report).

---

## 5. Camp labels: who is a Democrat or a Republican

### 5.1 Rule used (`config/labels.yaml`, copied to `results/ground_truth/labels_used.yaml`)
- **Supporter subs:** the 63 party-tagged subreddits (`docs/party_subreddits.md`: 43 dem, 20 rep) minus 5 special ones (`AskTrumpSupporters`, `enoughsandersspam`, `RepublicansForSanders`, `AskDemocrats`, `AskThe_Donald`) = **58 subs**.
- **main:** count each user's comments in dem vs rep supporter subs, counting only comments with score ≥ 1; need ≥ 10 such comments; rep if rep share ≥ 0.75, dem if ≤ 0.25, else mixed (excluded).
- **gold:** self-declared votes (Trump → rep; Clinton or Sanders → dem; Johnson, Stein and conflicting declarations excluded). Used as a check only.
- `broad` and `strict` are defined but never run (`label_sets_extra: []`).
- Heavy posters (> 500 replies in a week) excluded from every GT cell; a cell with < 50 labelled users on a side is NaN with `low_n` (never dropped).
- **Leave-the-venue-out:** a user's side for a venue is computed from comments outside that venue. For the shared venue nothing is subtracted (r/politics and r/PoliticalDiscussion are not supporter subs).

### 5.2 Numbers
- main labels 107,705 users at the default cell, of which **100,108 dem/rep** (35,244 dem, 64,864 rep; the other 7,597 are "mixed"). 107,705 vs 100,108 was a source of confusion (`docs/handoff_issues.md` #3).
- Agreement with self-declarations ≈ 80.6% (11,919 declarers); vote declarations 85.7%; identity declarations 74.4%.
- Score versions (all / ≥ 1 / ≥ 2 / above sub median / weighted) never flip a user left ↔ right; they only move users between a pole and mixed/unlabelled (`results/label_versions/summary.md`).
- Coverage where we measure (`results/label_coverage/summary.md`): median per week in the shared venue 7,782 dem and 5,623 rep active users; 74% of shared-venue edges touch a labelled user, 25% have both ends labelled. In the 5% sample: median 1,219 dem, 593 rep.

### 5.3 Within-camp (Sanders vs Clinton) setup
`results/within_camp/summary.md`: SandersForPresident vs hillaryclinton, threshold 0.75/0.25, ≥ 5 comments, score ≥ 1 → 47,873 Sanders-side and 7,354 Clinton-side users. Used for a within-camp GT (T4) that still uses the old volume-weighted GT#2 (section 6.3).

---

## 6. Ground truths GT#1 and GT#2

Specs: `docs/gt_runs_spec.md`; settings `config/gt_runs.yaml`; results `results/ground_truth/` (the file Pipeline evaluation reads is **`results/ground_truth/ground_truth_weekly.csv`**, column guide in `results/ground_truth/summary.md`).

### 6.1 GT#1: thread dissimilarity (`src/ground_truth/dissimilarity.py`, job `gt1_dissimilarity`)
- Per week, from `thread_user` restricted to the venue and labelled users: d_t, r_t = distinct dem/rep users in thread t; D, R totals. **Index = 0.5 · Σ_t |d_t/D − r_t/R|** (0 = identical spread over threads, 1 = never share a thread).
- Null: shuffle labels across users (same group sizes), 200 permutations. **`gt1_excess` = observed − null mean.**
- Versions: full week and restricted to the 1/5/20% sampled threads (`gt1_excess_0.01` etc.).
- Series (main/shared/full): mean 0.174, range 0.109–0.246; 0 of 53 weeks low_n; null check OK (excess under one extra shuffle = 0.0006).

**GT#1 robustness (`results/gt1_robustness/summary.md`):** split-half reliability (20 thread halves) Pearson 0.86, Spearman-Brown 0.93; stratified null (users shuffled only within subreddit × day × thread-size cells) keeps 86% of the excess, Spearman 0.98 with main; r/politics-only 0.96; removing top-1% users changes nothing (0.98); gold labels give a lower but parallel series (Spearman 0.85; 20% label flips cut excess to 30%). **Event weeks (Iowa … Election Day) show no effect** (diff −0.007, one-sided p 0.69).

**GT#1 drivers (`results/gt1_drivers/summary.md`):** 11 composition variables explain R² 0.79 (leave-one-out 0.62) of weekly variation; strongest single relations: share of comments in r/politics (rho −0.76), share of new users (+0.68), number of dem users (+0.64), rep/dem ratio (−0.62). Month effects alone R² 0.80; best single mean shift at **2016-06-27** (0.207 before → 0.146 after; R² 0.71, p < 0.001).

**GT#1 fixed panel (`results/gt1_panel/summary.md`):** the 06-27 drop **survives** every control: fixed panels of the same users (106–110% of the main drop), balanced sides (79–109%), reweighting the after-weeks to the before-weeks composition (141%). So the fall is behaviour of the same users, not a change in who participates. No variant shows an event effect.

### 6.2 GT#2: partisan language, the path to the adopted design
1. **Original (T2/T3): Gentzkow–Shapiro–Taddy leave-out estimator** (`src/ground_truth/leave_out_estimator.py`): phrase frequencies per side, leave user i out, q_{−i,j} = ρ_R/(ρ_R + ρ_D), q̂_i = Σ_j (c_ij/m_i) q_{−i,j}, π = 0.5·mean_R q̂ + 0.5·mean_D (1 − q̂). Computed on the shared venue. **It failed:** π = 0.428, *below* its shuffled null (≈ 0.50) in 51 of 53 weeks, and agreed with its own samples at only 0.00 / 0.02 / 0.28 (1/5/20%). Cause: a few very active users dominate pooled frequencies.
2. **Fix exploration** (`results/gt2_fix_options/summary.md`): equal-weight users (A) lifts every week above the null; phrase threshold (B), caps (C300/C1000) and 4-week pooling (D) do not fix it. A held-out-user logistic classifier (E) separates camps at AUC 0.71–0.76, so the signal is real.
3. **Adopted: transfer design (2026-10-29)** (`src/gt_runs/transfer.py`, job `gt2_transfer`, `results/gt2_transfer/summary.md`):
   - Learn how each camp talks in the **supporter subs** (main-labelled users, heavy posters excluded, same vocabulary and contraction filter), apply the frozen model to users' comments in the **neutral shared venue**.
   - Users split into 5 piles; a user is scored only by the model trained on the other four.
   - **`gt2` (chosen) = phrase estimator:** phrase leaning = share of the phrase's use by Republican supporters (training weighting: per-user-week cap of 100 phrases); a user's weekly estimate = mean leaning of the phrases they used that week (≥ 20 phrases); **gt2 = 0.5 · mean over Republicans + 0.5 · mean over Democrats of (1 − estimate)**. 0.5 = no information.
   - **`gt2_alt_auc` (secondary):** L2 logistic regression (C chosen by CV = 0.3; grid widened to 0.003–3 to confirm an interior optimum), equal-weight users; weekly AUC against main labels.
   - Contraction filter: phrases containing `n't`, `'s`, `'re`, `'ve`, `'ll`, `'m`, `'d` are dropped (937 of 100,000), because `body_cleaned` splits contractions.
   - Nulls: 100 label shuffles; bands: 100 user bootstraps.

**GT#2 numbers:**
- Held-out balanced accuracy in supporter subs: 0.93 (main labels), 0.80 (self-declared voters).
- `gt2` mean 0.5139 (null 0.5000; range 0.5086–0.5206); above null + 2 sd in **53 of 53 weeks**. AUC companion mean 0.697 (0.646–0.749).
- Sample stability (full vs sample Spearman): 0.68 / 0.79 / 0.91 for 1% / 5% / 20% (phrase, cap 100); 11 of 53 weeks NaN at 1%.
- GT#1 vs GT#2: rho 0.80. gt2 vs gold-label series: 0.95. gt2 vs old volume-weighted: 0.18.

**GT#2 caveats** (from `docs/decisions.md`): (1) transfer loss, gold voters AUC 0.86 in supporter subs but 0.66 in neutral subs; (2) the level is tiny (0.514) so use it as a ranking; (3) GT#2 shares GT#1's composition trend (rho with r/politics share −0.74), so the 0.80 agreement is partly a common trend; (4) the three GT#2 candidates do not corroborate each other; (5) 1% sample is weak; (6) the transfer job reads a cache under `results/gt2_fix_options/cache`, do not delete it; (7) T4 within-camp and T5 variants still use the old volume-weighted estimator.

### 6.3 Other GT outputs
- `results/ground_truth/within_camp_weekly.parquet` (T4, Sanders vs Clinton, GT#1 excess mean 0.143).
- `results/ground_truth/variants_weekly.parquet` (T5: keeping heavy posters changes GT#1 by nothing, rho 0.999; contraction filter matters for the old GT#2).
- Venue table: special venues are mostly too thin (AskTrumpSupporters 45 weeks usable; others ≤ 22).

---

## 7. The LLM engine (shared extraction)

Code `src/algorithms/llm/`; config `configs/llm/llm.yaml`; Phase 1 report `docs/reports/phase_1_report.md`.

### 7.1 Input (`data.py`, D-S2, D-S3, D-S4)
- Sampled edges at a given rate, restricted to the **shared venue** (`politics`, `PoliticalDiscussion`; 51–57% of sampled edges).
- Edges with `reply_removed`/`parent_removed` or empty cleaned text are dropped.
- Text = `parent_body_clean` and `body_clean`, each truncated to **1200 characters** (truncation rate 3.5–4.4% per run). No usernames, subreddits, labels or GT fields.

### 7.2 Prompt and schema (`schema.py`)
- System prompt first (static, so Ollama can reuse the KV prefix), user message `PARENT:\n…\n\nREPLY:\n…`.
- JSON schema enforced by Ollama constrained decoding, fields in this order: `agreement` ∈ {agree, neutral, disagree} **first** (so its token probabilities are interpretable), `emotional_charge` ∈ {low, medium, high}, `hostility` ∈ {none, mild, strong}, `personal_attack` (bool).
- **Prompt v2** (frozen) vs v1: defines agreement as siding with the *parent's point*; states that hostility towards third parties (candidates, parties) does not imply disagreement; sarcasm = intended meaning; 3 invented examples; asks for one-line compact JSON (cut output tokens 36 → 29.5 per edge). Hostility/personal attack are about the parent's author or position only.
- Request options: `temperature 0`, `seed 0`, `num_predict 60`, `think: false` (D-M2), `logprobs: true`, `top_logprobs: 10`.

### 7.3 Agreement probabilities (D-P1, `backends.py`)
Take the token right after `"agreement":"`, map it and its top-10 alternatives to labels by prefix ("ag" → agree, "dis" → disagree, "neu" → neutral), exponentiate, normalise over the three labels. Validated on llama3.1:8b against 10 samples at T = 0.7 on 100 pilot edges: argmax = greedy 100%, MAE 0.059, Spearman 0.54–0.73 (`results/llm/bench/probs_llama3.1_8b.json`). qwen3.5:9b returns logprobs for 100% of edges (mean top p 0.73) but was not separately validated against sampling.

### 7.4 Extraction loop (`extract.py`) and cache (`store.py`, D-E1)
- Hash the visible (truncated) text pair; **dedup identical pairs**; skip pairs already in the store; call the backend in a thread pool (`parallel: 4`); retry invalid JSON up to 3 times; write each result immediately.
- Store = append-only **SQLite** `data/llm/extractions.sqlite`, primary key (pair_hash, model, quant, prompt_version, temperature, seed); columns fields JSON, p_agree/p_neutral/p_disagree, latency, retries, n_tokens. Any change of model/prompt/temperature/seed is a new cache key. Parquet export with sha256 manifest available.
- Ordering: random within week + round-robin across weeks by default; for long runs, **thread-prefix ordering** (`frac` column in `run_extraction.py`): within each week threads are put in a fixed random order and processed in ascending cumulative position, so *any* stopping point is a set of whole threads at about the same fraction in every week.
- Safe stop: Ctrl-C (finishes in-flight requests) or a `STOP` file in the run folder; rerunning the same command resumes from the cache.

### 7.5 Provenance (`provenance.py`)
Every run gets an immutable folder `runs/llm/<timestamp>_<name>/` with `manifest.json` (config, git SHA and dirty flag, input file sha256s, hardware, model, prompt version, headline numbers, output hashes) and a row in `results/llm/INDEX.md` / `index.csv`.

### 7.6 Model selection (D-M1, D-M6, D-M7)
Rule fixed in advance: best agreement with human consensus labels on a 150-edge pilot, subject to ≥ 99% valid JSON and a throughput budget; never chosen on GT.
- **Pilot sheet:** 150 shared-venue edges, 50 per lexical-cue stratum (agree-ish / disagree-ish / other), all 53 weeks (`scripts/llm/make_pilot_sheet.py`). Labelled by Kareem and Yasmina as an **agreed consensus set** (no independent inter-rater kappa exists, D-P3). Class mix 83 disagree / 55 agree / 12 neutral. Rubric `docs/pilot_rubric.md`. Labelling tools: `label_tool.py` (terminal), Word/Markdown sheets (`pilot_doc.py`).
- **Split fixed before any prompt edit:** 75 dev / 75 held-out (`data/llm/pilot/pilot_split.csv`). Prompts edited on dev only; held-out scored once.

| Model | Prompt | Split | Accuracy | kappa | agree-vs-disagree AUC |
|---|---|---|---|---|---|
| llama3.1:8b | v1 | all 150 | 0.587 | 0.205 | 0.76 |
| llama3.1:8b | v2 | dev | 0.627 | 0.307 | 0.75 |
| qwen3.5:9b-q4_K_M | v2 | dev | 0.733 | 0.548 | 0.87 |
| **qwen3.5:9b-q4_K_M** | **v2** | **held-out (once)** | **0.707** | **0.439** | n/a |

Combined 108/150 = 0.72. v1 failure mode: third-party negativity read as disagreement (35 of 55 human-"agree" predicted "disagree"). Held-out confusion (rows human): agree 16/10/2, disagree 7/35/1, neutral 0/2/2 (agree/disagree/neutral). 100% valid JSON first try. `gemma3:12b` and `qwen3:8b` were never scored.

### 7.7 Throughput (D-T1, D-T2)
- Mac: llama3.1:8b 0.77 edges/s regardless of client threads (server serial); qwen3.5:9b 0.64–0.81 edges/s → 5% tier ≈ 210–270 h, infeasible.
- Windows 3090: `OLLAMA_NUM_PARALLEL` 4 / 8 / 16 all gave 1.30 edges/s on a 200-edge bench (every request on slot 0; qwen3.5's hybrid architecture appears not to batch in Ollama 0.40.13). Real runs: ~2.9 pairs/s early, **1.94 pairs/s average over the final 61.5 h run**. Not pursued: llama.cpp `llama-server`, qwen3.5:27b.

---

## 8. Pipeline A: text scoring

Code `src/algorithms/pipeline_a/aggregate.py`; rules `configs/llm/pipeline_a_rules.yaml` (A-rules-v1, D-A1); runner `scripts/llm/run_pipeline_a.py`.

- **Edge score (0–10)** = agreement (disagree +1) + emotional_charge (low 0, medium 1, high 2) + hostility (none 0, mild 3, strong 6) + personal_attack (+1), clipped to [0, 10]. Max reachable = 10. Total over all 54 attribute combinations (tested). This is an additive reading of Chowdhury et al., `TODO(verify)` against their table.
- **Weekly P_A** = mean edge score over the week's extracted edges (counted per edge, duplicates included). Also reported: median, `share_high` (score ≥ 7), n, `A_disagree` (share of disagree labels), `A_hostility` (hostility only, 0/5/10), `low_n` (n < 30, kept), `coverage` (for partial runs).
- Reads only the store; no LLM calls. Outputs `weekly_A.csv/parquet`, `edge_scores.parquet`, plot.

---

## 9. Pipeline B: signed network and frustration

Code `src/algorithms/pipeline_b/`; config `configs/llm/pipeline_b.yaml`; runner `scripts/llm/run_pipeline_b.py`; Phase 3 report `docs/reports/phase_3_report.md`.

### 9.1 Sign sources (D-B1, `sign_inference.py`)
| Source | Edge weight | Role |
|---|---|---|
| `continuous_llm` (**headline**) | w = (p_agree − p_disagree) · (1 − p_neutral) (Guo weight) | primary |
| `binary_llm` | argmax → +1 / −1, neutral dropped | robustness |
| `sentiment` | VADER compound of the reply only (no LLM, no parent context) | free, context-free baseline |
| `unsigned` | all +1 | degenerate control (P_B = 0 by construction) |
| `noise_injection(w, p)` | flips a random share p of signs | for RQ3, **implemented, never run** |

### 9.2 Weekly graph (D-B2, `structural.py`)
Per week: undirected; a user pair's weight = **mean** of its interaction weights (`TODO(verify)` vs Guo); self-loops and zero-weight pairs dropped. Headline on the **giant component**; full-graph P_B also stored. A week is NaN if < 40 user pairs or the giant component < 20 nodes (reason recorded).

### 9.3 Frustration (D-B3, D-B3a, `frustration.py`)
- For a partition s ∈ {−1, +1}^n: **F(s) = Σ |w_e| over violated edges** (positive edge across camps, negative edge inside a camp) = ½ (A − Σ w_ij s_i s_j), A = Σ|w|. Minimising F is NP-hard.
- **Heuristic:** 1 spectral start (top eigenvector of W) polished by greedy single-node flips, plus 9 random starts each run through **simulated annealing** (geometric cooling 1.5 → 0.05 × mean|w|; 5,000 sweeps up to 15k nodes, then a 7.5e7 node-visit budget, floor 1,000) and greedy polish; best of 10. numba-compiled.
- **Exact check:** ILP via scipy HiGHS (`exact_frustration`). On 16 random-sign graphs of 36–77 nodes (hardest case): heuristic optimal in 14/16, mean gap 0.08%, max 1.2% (`results/llm/bench/frustration_gap.csv`); calibrated 60–100-node graphs: optimal 10/10. **Not verifiable at 1,000+ nodes.** The first budget (100 sweeps) was only 38% optimal; it was raised before use.

### 9.4 Normalisation and nulls (D-B4)
- **P_B = 1 − F_obs / mean(F_null)**; F_null over R sign-shuffled copies (same topology, weights permuted among edges). R = 50 for continuous, 20 otherwise. P_B = 0 if F_null = 0. `TODO(verify)` exact normalisation in Guo.
- Secondary null: **signed-degree-preserving rewiring** (double edge swaps inside the positive and negative edge sets separately, 10 swaps per edge; no self-loops or duplicates) → `P_B_rewire`, R = 10 for continuous only.
- Null draws run in a 6-process pool (identical results, ~2.4× faster). Cost for all four sources: ~1 h (3,551 s) at 135k edges, **3.6 h (12,933 s) for the full 617k edges**.

### 9.5 Camp recovery (D-B5, `eval/camp_recovery.py`, `scripts/llm/eval_camp_recovery.py`)
The frustration partition of each week's giant component is saved (`camp_assignments.parquet`). Evaluation: among users with main labels, **balanced accuracy** of the partition vs dem/rep, taking the better of the two orientations, against 200 label shuffles; weeks with < 5 labelled per side skipped. This is the only code that reads labels.

---

## 10. Synthetic validation of Pipeline B

Spec (written before running) `docs/synthetic_spec.md`; report `docs/reports/phase_4_report.md`; results `results/llm/synthetic/run2/`; runner `scripts/llm/run_synthetic.py`.

- **Calibration** from the real sampled shared-venue user-pair graphs (`calibration_weekly.csv`): 5% sample 480–9,400 nodes (median 4.2k), mean degree 2.7–4.1 (median 3.2), giant share 0.89–0.97. Degrees resampled from the pooled empirical sequence (`degree_pool.npy`).
- **Generator** (`generator.generate`): degree-corrected signed block model. Two camps (`rep_share`), optional no-camp nodes, `homophily` (within-camp edge probability), `lam` = polarization (probability a sign is camp-aligned: + within, − across, else random), `discordant_share` (within-camp edges forced negative), `flip` (global sign noise); weights = sign × U(0.3, 1).
- Default cell: ~4,000 nodes, mean degree 3.2, rep_share 0.5, homophily 0.5, 5 replicates.

| Check (pre-stated) | Run 1 | Run 2 (final tool) |
|---|---|---|
| 1 monotone in lam (Spearman ≥ 0.95) | FAIL 0.909 | **PASS 0.964** |
| 2 flat across size at lam 0.5 (CV < 0.15) | FAIL 0.84 | **FAIL 0.46** (500–10k nodes CV ≈ 0.19; 40k likely under-converged) |
| 2 density (report only) | | P_B 0.020 / 0.060 / 0.12 / 0.17 at mean degree 2.5 / 3.2 / 5 / 7 |
| 3 separation from nulls (P_B > 0.2 at lam 0.6) | FAIL | **FAIL** P_B 0.172 (but 26.8 null sd below the shuffle null; rewire null agrees 0.178; lam 0 gives 0.002) |
| 4 heuristic vs exact, calibrated | PASS | **PASS** 10/10 optimal |
| 5 discordant allies (Spearman ≤ −0.9) | PASS | **PASS** −1.0 (P_B 0.53 → 0) |
| 6 imbalance (6:1 keeps ≥ 0.5 of balanced) | PASS | **PASS** 0.553 vs 0.531 |
| 7 planted-camp recovery (BA > 0.8 at lam 0.9) | PASS | **PASS** 0.51 / 0.81 / 0.97 at lam 0.3 / 0.6 / 0.9 |

Run 1 failures were the tool (under-converged annealing on weak-signal graphs above ~4k nodes, found by `scripts/llm/synth_diag_scale.py`); fixed by D-B3a. Criteria were not changed after results. **Lessons for real data:** P_B is small (0.02–0.06) at real densities even at moderate polarization, mild polarization (lam ≤ 0.4) is invisible, compare weeks by rank not level, imbalance is not a problem, discordant allies lower P_B as intended. Open: accept the two failures, or a dated spec v2 (z-score separation, size bar over 500–10k only).

---

## 11. Evaluation spec and the RQ1 comparison script

Spec `docs/eval_spec.md` (v1, written 2026-10-31 before any GT comparison) + clarification **D-E3** (logged before any 5% number). Implementation `scripts/llm/rq1_compare.py`.

- **Series:** P_A (weekly mean edge score) and P_B (`continuous_llm`, giant component). **Truths:** primary = full-week `gt1_excess` and `gt2` from `ground_truth_weekly.csv`; secondary = 5%-sample-matched `gt1_excess_0.05`, `gt2_0.05`, `gt2_alt_auc_0.05`.
- **Statistics:** Spearman rho on levels per (pipeline × truth); **circular-shift permutation p** (all T−1 shifts; preserves autocorrelation); Spearman on first differences; partial Spearman controlling for weekly edge volume; volume-only baseline; **delta = rho_B − rho_A** with a **4-week moving-block bootstrap** (2,000 draws, seed 0) over weeks where both exist.
- **Decision rule:** a pipeline wins only if it beats the other on the truth that leans against it (B on GT#2, A on GT#1) with the delta CI excluding 0. D-E3 adds: B must also have rho_B > 0 ("less negative than A" is never a win); for partial runs the full-week truth is primary.
- Outputs per evaluation: `rq1_<label>_correlations.csv`, `rq1_<label>_delta.csv`, `rq1_<label>_summary.json` (verdict strings).
- **Stated up front as not done in v1:** Holm correction, composition-only baselines, before/after 2016-06-27 split, split-half noise ceilings.
- Exploratory add-ons (labelled as such): `scripts/llm/rq1_bootstrap_stability.py` (5 seeds × 10,000 draws), `scripts/llm/rq1_partial_week_sensitivity.py` (drop partial first/last weeks).

---

## 12. Every real-data run, in order, with results

All run folders are under `runs/llm/` (git-ignored); the registry `results/llm/INDEX.md` lists every one with its headline.

### 12.1 Plumbing runs (2026-10-30)
`20261030T233322_pipeA_partial` (88 edges) and `20261030T235157_pipeB_plumbing` (854 edges, thresholds lowered): code paths exercised, numbers meaningless.

### 12.2 Tier 0, 60 edges/week (Mac, 2026-10-30/31)
`20261030T233105_tier0_b60`: 1% sample, whole threads, budget 60 edges/week → 12,733 edges, 12,720 unique pairs, 220 threads, 53 weeks, 0 failures, 4.75 h, 3.5% truncated. Report `docs/reports/tier0_real_data_report.md`.
- A: all 53 weeks; P_A mean 2.96 (1.34–4.29); 53% disagree; jumpy (sd ~0.7).
- B: only 33/53 weeks usable; median giant 90 nodes; P_B sd 0.68 (range −3.2 to +1.0) → noise.
- Camp recovery: continuous 0.598 vs null 0.565 (4/22 weeks above p95).

### 12.3 Tier 0, 230 edges/week (Mac, 2026-10-31), first RQ1 look
`20261031T182808_tier0_b230` (24,037 edges, 0 failed; extended in place from the same thread order); A `..._pipeA_b230`, B `..._pipeB_b230` (52/53 weeks). Report `docs/reports/rq1_tier0_b230_report.md`.

| | GT#1 | GT#2 |
|---|---|---|
| rho_A (p) | −0.43 (0.02) | −0.37 (0.06) |
| rho_B (p) | +0.11 (0.14) | +0.12 (0.18) |
| delta [CI] | +0.54 [0.20, 0.85] | +0.49 [0.10, 0.83] |

Verdict then: **no winner** (B indistinguishable from zero; delta driven by A being negative). Camp recovery 0.597 vs 0.562; VADER 0.561; A vs B rho 0.11.

### 12.4 5% tier extraction (Windows 3090, 2026-10-31 → 2026-11-05)
Selection: the whole 5% shared-venue tier, thread-prefix ordering (`--budget 1000000000`) → **617,568 edges, 617,041 unique pairs, 53 weeks, 4.4% truncated.** One cache, five run folders (each restart reselects the same edges and skips cached pairs):

| Run folder | Window | Pairs added | How it ended |
|---|---|---|---|
| `20261031T232220_tier5pct` | Oct 31 23:22 → Nov 1 ~09:20 | 98,050 | Ollama HTTP 500, process crashed |
| `20261101T092612_tier5pct` | Nov 1 09:26 → 13:15 | 37,508 | STOP file (GPU needed elsewhere) → the "5% prefix" evaluation |
| `20261101T182821_tier5pct` | Nov 1 18:28 → ~22:15 | 37,900 | HTTP 500 crash |
| `20261102T134006_tier5pct` | Nov 2 13:40 → 15:01 | 13,175 | HTTP 500 crash; `scripts/llm/watch_5pct.py` restarted it |
| `20261102T150142_tier5pct` | Nov 2 15:01 → **Nov 5 04:29** | 430,360 | **done**, 61.5 h, 1.94 pairs/s, 0 failed |

The SQLite store now holds exactly **617,041 rows for qwen3.5:9b-q4_K_M / prompt v2** (checked 2026-11-08). Total failures: 0. Supervision: `scripts/llm/overnight.py` (D-O1/D-O2: checkpoint evaluations at 30k pairs then every 2 h, auto-restart, draft report refresh) for Nov 1; `scripts/llm/watch_5pct.py` (relaunch-on-death watcher, log `logs_llm/watch_5pct.log`) from Nov 2.

### 12.5 5% prefix result, 135,678 edges (2026-11-01, the midway result, D-E4)
A `20261101T131638_pipeA_5pct_final`, B `20261101T131644_pipeB_5pct_final`; results `results/llm/rq1_5pct/5pct_final/`; report `docs/reports/rq1_5pct_report.md`. 22% of the tier, 2,885 threads, median 2,512 edges/week, B scored in 53/53 weeks, giant component median ~1,000 nodes.

| | GT#1 | GT#2 |
|---|---|---|
| rho_A (p) | −0.44 (0.08) | −0.38 (0.11) |
| rho_B (p) | −0.05 (0.81) | +0.07 (0.75) |
| delta [CI] | +0.39 [0.02, 0.65] | +0.44 [0.06, 0.70] |
| delta, 5%-matched | +0.22 [−0.11, 0.47] | +0.29 [−0.13, 0.52] |

Primary verdict "B wins on GT#2" literally; secondary "no winner". Seed stability: GT#2 lower bound 0.058–0.072 over 5 seeds. Dropping partial weeks moves rho/delta by ≤ 0.05. Camp recovery: continuous 0.531 vs null 0.524 (8/53 weeks above p95), binary 0.540, VADER 0.534, unsigned 0.500. P_B mean −0.001 (sd 0.11).

### 12.6 Full 5% tier, 617,568 edges (evaluated 2026-11-07) — NEW, not in any other doc
Interim evaluation on 2026-11-03 at 370,008 edges (`20261103T231046_pipeA_5pct_interim_nov3`, `..._pipeB_5pct_interim_nov3`; results `results/llm/rq1_5pct/5pct_interim_nov3/`), then the full tier: A `runs/llm/20261107T121812_pipeA_5pct_full`, B `runs/llm/20261107T121822_pipeB_5pct_full` (git SHA 022dba0, dirty), results **`results/llm/rq1_5pct/rq1_5pct_full_{correlations,delta}.csv`, `rq1_5pct_full_summary.json`** (uncommitted). Coverage 1.0 in every week.

**Pipeline A (full):** P_A mean 2.99, sd 0.22, range 2.56–3.55; median edge score 1–2; share ≥ 7: 0.20; disagree share 0.51; median 11,130 edges/week (min 1,334 in the partial first week, max 25,345).

**Pipeline B (full), per week medians/means over 53 weeks:**

| Sign source | P_B mean (sd) | weeks P_B > 0 | P_B_rewire | giant nodes / pairs (median) | share negative |
|---|---|---|---|---|---|
| continuous_llm | 0.0047 (0.033) | 24 | 0.040 | 4,056 / 6,876 | 0.61 |
| binary_llm | 0.0235 (0.018) | 49 | | 3,509 / 5,148 | 0.59 |
| sentiment (VADER) | 0.0005 (0.022) | 23 | | 3,634 / 5,457 | 0.45 |
| unsigned | 0 | 0 | | 4,056 / 6,876 | 0 |

**RQ1 comparison (full tier):**

| Truth | rho_A | perm p | rho_B | perm p | delta rho_B − rho_A [95% CI] |
|---|---|---|---|---|---|
| GT#1 full (primary) | −0.61 | 0.057 | −0.03 | 0.87 | +0.58 [0.27, 0.92] |
| **GT#2 full (primary, deciding)** | **−0.48** | 0.21 | **+0.17** | 0.38 | **+0.65 [0.21, 0.94]** |
| GT#1 5%-matched | −0.39 | 0.17 | +0.05 | 0.72 | +0.44 [0.06, 0.84] |
| GT#2 5%-matched | −0.38 | 0.26 | +0.16 | 0.40 | +0.54 [0.11, 0.83] |
| GT#2 AUC 5%-matched | −0.39 | 0.21 | +0.14 | 0.47 | +0.52 [0.14, 0.83] |

- First differences (GT#2 full): A +0.01, B −0.02. Partial rho controlling for volume: A −0.47, B +0.18 (volume explains nothing). Volume alone vs GT#2: −0.11.
- A vs B across weeks: rho 0.10 (they measure different things).
- Verdicts in the summary JSON: **PRIMARY "B wins on GT#2"; SECONDARY "B wins on GT#2"**.
- **Camp recovery (full):** continuous 0.518 vs null 0.511 (14/53 weeks above p95; ~1,733 labelled users/week), binary 0.519 vs 0.512 (13), VADER 0.515 vs 0.512 (7), unsigned 0.500.
- Not yet run for the full tier: bootstrap seed stability, partial-week sensitivity, a written report, a D-E decision entry, slide figures.

**Trajectory of the deciding comparison (GT#2 full):**

| Edges | rho_A | rho_B | delta [95% CI] |
|---|---|---|---|
| 30k (ckpt1) | −0.19 | +0.07 | +0.31 [−0.17, 0.69] |
| 53k (ckpt2) | −0.19 | 0.00 | +0.27 [−0.19, 0.59] |
| 77k (ckpt3) | −0.34 | +0.04 | +0.41 [−0.01, 0.74] |
| 102k (ckpt4) | −0.37 | +0.04 | +0.41 [−0.02, 0.70] |
| 136k (prefix final) | −0.38 | +0.07 | +0.44 [0.06, 0.70] |
| 370k (interim Nov 3) | −0.30 | +0.21 | +0.51 [0.09, 0.78] |
| **617k (full tier)** | **−0.48** | **+0.17** | **+0.65 [0.21, 0.94]** |

---

## 13. Findings: what we can and cannot claim

**Data and ground truth**
1. Two LLM-free ground truths exist for all 53 weeks with zero low-n weeks in the shared venue, both above their nulls every week.
2. GT#1 is reliable (split-half 0.93) and robust to topic, label, panel and composition controls; it shows a **behavioural fall after 2016-06-27** (0.204 → 0.146) and **no event-week effects**.
3. GT#2 via the volume-weighted leave-out estimator was broken by heavy users; the transfer design works but has a tiny level and lossy transfer (AUC 0.66 on neutral-sub gold voters).
4. GT#1 and GT#2 agree (0.80), partly through a shared composition trend.

**LLM and pipelines**
5. An open 9B model at 4-bit reaches 0.72 accuracy (kappa 0.44 held-out) on Reddit stance; thinking off; 100% valid JSON; 0 failures in 617k pairs.
6. Throughput, not model quality, was the binding constraint (5% tier = 61.5 h on a 3090 after crashes).
7. Synthetic: P_B rises monotonically with planted polarization, separates strongly from nulls, handles imbalance and discordant allies, but is **small and density-dependent** at real sparsity.

**RQ1**
8. **A is negatively related to both ground truths, and more so with more data** (GT#1 −0.44 → −0.61; GT#2 −0.38 → −0.48 from 136k to 617k edges). Unexplained.
9. **B is positive on GT#2 but not significant** (+0.17, p 0.38) and ~0 on GT#1.
10. The pre-registered rule is now met on both primary and secondary truths (delta GT#2 +0.65 [0.21, 0.94]). What that supports: *B does not do worse than A, and A goes the wrong way.* What it does not support: *B tracks polarization.*
11. Neither pipeline tracks week-to-week **changes** (first-difference rho ≈ 0).
12. **Camp recovery is barely above chance and LLM signs are no better than context-free VADER** (0.518 vs 0.515), which questions whether the LLM signs carry partisan structure at all at this density.
13. P_B is ~0 on average (no more two-camp structure than a sign-shuffled copy), consistent with the synthetic floor.

Suggested statement: *"On the full 5% sample (617k replies, 53 weeks), neither pipeline shows a significant positive association with either ground truth. The signed-network pipeline passes our pre-registered decision rule against text scoring, but mainly because text scoring is anti-correlated with both truths; the network's own correlation is weakly positive and not significant."*

---

## 14. Tests

Run: `.venv\Scripts\python.exe -m pytest -q`. Current state on Windows (2026-11-08): **107 tests, 106 pass, 1 fails** (`tests/test_llm_engine.py::test_label_tool_resume_back_skip`, because `termios` is Unix-only; irrelevant to runs).

| File | What it checks |
|---|---|
| `tests/dataset/test_parse_comments.py`, `test_build_edges.py`, `test_windowing.py`, `test_graph_stats.py`, `test_user_profiles.py` | parsing filters, edge construction and drop reasons, Monday-week windows, graph stats, declaration regex and domains |
| `tests/prep_jobs/test_framework.py` | detached runner, checkpoints, resume, lock, STOP |
| `tests/prep_jobs/test_boundary_and_sampling.py` | boundary-week merge, stratified thread sampling with random rounding |
| `tests/prep_jobs/test_phrase_counts.py` | phrase counting, finalise step |
| `tests/ground_truth/test_dissimilarity.py` | index ≈ 1 for separate camps, ≈ null for mixed, excess ≈ 0 when mixed |
| `tests/ground_truth/test_leave_out_estimator.py` | π ≈ 0.5 with identical vocab, monotone rise with separation, shuffled null ≈ 0.5, speed on 50k × 100k |
| `tests/gt_runs/test_labels.py`, `test_cells.py`, `test_transfer.py`, `test_gt1_robust.py`, `test_camp_recovery_eval.py` | leave-the-venue-out subtraction, low-n → NaN, transfer folds, robustness pieces, camp recovery (perfect / random / swapped) |
| `tests/test_llm_engine.py` | caching, retry, resume after kill, dedup, cache-key sensitivity (model, prompt, temperature, seed), probability normalisation, logprob parsing, schema validation, no forbidden fields in the prompt, Parquet export, run folder/registry, Pipeline A rule totality and aggregation |
| `tests/test_pipeline_b.py` | balanced two-camp graph F = 0 / P_B = 1 and camps recovered; all-positive P_B = 0; random signs P_B ≈ 0; heuristic never below the ILP optimum; sign shuffle keeps topology; rewiring keeps signed degrees, no self-loops/duplicates; sign-source formulas; camp recovery; **no pipeline module touches labels** |
| `tests/test_real_modules.py` | evaluation metrics (Spearman, Pearson, MAE, direction of change) and the shared `SignedEdge` / `Window` types |

Non-unit validations that count as tests: the pilot (7.6), logprob validation (7.3), heuristic-vs-ILP benchmarks (9.3), synthetic checks (10), GT null checks (6), sampling validation (4.8), phrase-count recount (`runs/phrase_counts/VALIDATION.md`).

---

## 15. Run book: how to reproduce and continue

All commands from the repo root on the Windows PC, PowerShell, venv `.venv`.

### 15.1 Data and ground truth (already done; only if rebuilding)
```powershell
python scripts/launch_job.py <job> --workers N      # comment_flags, phrase_counts, thread_user, clean_edges_and_sampling,
                                                    # label_versions, within_camp_setup, label_coverage, labels, gt1_dissimilarity,
                                                    # gt2_leave_out_full, gt2_leave_out_samples, gt_within_camp, gt_variants,
                                                    # gt2_transfer, gt1_robustness, gt1_drivers, gt1_panel, gt_summary
python scripts/launch_job.py <job> --resume         # after a reboot / kill
python scripts/jobs_status.py --watch 30            # progress table
New-Item runs\<job>\STOP                            # stop one job cleanly
```
Order and dependencies are in `config/prep_jobs.yaml` (`depends_on`). The GT file the evaluation needs is `results/ground_truth/ground_truth_weekly.csv`.

### 15.2 LLM server
```powershell
Start-Process powershell -ArgumentList "-NoProfile","-File","scripts/llm/start_ollama.ps1","-Parallel","4" -WindowStyle Hidden
```
Do not use `Start-Process -Wait` on it (hangs). Check another project is not holding the GPU first.

### 15.3 Extraction (any rate; resumable)
```powershell
.\.venv\Scripts\python.exe scripts/llm/run_extraction.py --rate 0.05 --budget 1000000000 --name tier5pct --plan-only   # counts and ETA only
Start-Process .\.venv\Scripts\python.exe -ArgumentList "scripts/llm/run_extraction.py","--rate","0.05","--budget","1000000000","--name","tier5pct" -RedirectStandardOutput logs_llm/tier5pct.log -RedirectStandardError logs_llm/tier5pct.err -WindowStyle Hidden
.\.venv\Scripts\python.exe scripts/llm/watch_5pct.py   # optional: relaunch on crash (up to 30 attempts)
```
`run_extraction.py` refuses to start while a prep job is running. Stop with `New-Item runs\llm\<run>\STOP`. The 5% tier is complete in the store, so rerunning it costs nothing. The 20% tier (2.74M shared-venue edges) would take roughly 2.7M / 2 pairs/s ≈ 16 days on this setup.

### 15.4 Evaluation on whatever is in the store
```bash
RUN=$(ls -d runs/llm/*_tier5pct | tail -1)
python scripts/llm/run_pipeline_a.py $RUN --name pipeA_5pct_full
python scripts/llm/run_pipeline_b.py $RUN --name pipeB_5pct_full                 # ~3.6 h at 617k edges
python scripts/llm/eval_camp_recovery.py $(ls -d runs/llm/*_pipeB_5pct_full | tail -1)
python scripts/llm/rq1_compare.py $(ls -d runs/llm/*_pipeA_5pct_full | tail -1) $(ls -d runs/llm/*_pipeB_5pct_full | tail -1) --out results/llm/rq1_5pct --label 5pct_full
python scripts/llm/rq1_bootstrap_stability.py ...    # exploratory, see script docstring
python scripts/llm/rq1_partial_week_sensitivity.py ...
python scripts/llm/make_slide_figures.py             # slide PNGs (A orange, B blue, GT neutral)
```
Use `--sources continuous_llm,binary_llm,...` on `run_pipeline_b.py` to restrict sign sources.

### 15.5 Pilot, benchmarks and synthetic
```bash
python scripts/llm/pilot_run.py gemma3:12b --prompt v2 --split dev      # then --split heldout --final (scored once)
python scripts/llm/pilot_eval.py                                        # accuracy, kappa vs consensus labels
python scripts/llm/validate_probs.py <model>                            # logprobs vs k samples
python scripts/llm/bench.py <model>  /  scripts/llm/sweep_parallel.py   # throughput
python scripts/llm/bench_frustration.py                                 # heuristic gap + cost
python scripts/llm/run_synthetic.py                                     # Phase 4 checks (~40 min)
```

### 15.6 What is NOT in git (back it up)
`data/` (including `data/llm/extractions.sqlite`, the whole 617k-pair cache, and `data/llm/pilot/` with the human labels and split), `runs/`, `logs_llm/`, and everything in `results/` except `results/llm/`.

**Uncommitted right now:** `results/llm/rq1_5pct/rq1_5pct_full_*`, `results/llm/rq1_5pct/5pct_interim_nov3/`, `scripts/llm/watch_5pct.py`, the `results/llm/INDEX.md`/`index.csv` rows for the interim and full runs, the `Kareem Presentation Context/` folder, and this handbook.

### 15.7 Windows gotchas
`termios` test failure is expected; `pkill -f` style kills can kill their own shell (Mac); keep the PC awake (`powercfg /change standby-timeout-ac 0`, undo with 30); `ollama pull` resumes partial downloads; Ollama occasionally returns HTTP 500 under long runs (3 crashes in 5 days); use the watcher.

---

## 16. Decision log summary

Full text: `docs/llm_decisions.md` (D-*) and `docs/decisions.md`.

| ID | Decision | Status |
|---|---|---|
| D-S1 | One shared LLM call per edge for A and B | frozen |
| D-S2 | Shared venue only (r/politics + r/PoliticalDiscussion) | frozen |
| D-S3 | Cleaned text, 1200 chars each, no metadata | open |
| D-S4 | Drop removed/empty edges before the LLM | frozen |
| D-S5 | Samples are independent, not nested | open (ask data side) |
| D-S6 | Labels only for camp recovery; 100,108 dem+rep users | open |
| D-M1 | Model chosen by pilot agreement + validity + throughput, never GT | frozen |
| D-M2 | Thinking off | frozen |
| D-M3/M5 | Candidate tags and download queue | open |
| D-M4 | Fallback to llama3.1:8b if < 2 models by 2026-11-02 (git tag `fallback-rule-v1`) | moot |
| D-M6/M7 | qwen3.5:9b-q4_K_M + prompt v2 frozen (dev 0.733, held-out 0.707) | frozen |
| D-M8 | qwen3.5:27b on SSD only, untested | open |
| D-P1 | Probabilities from first-token logprobs | open (validated on llama only) |
| D-P2/P4 | Prompt v1 fails; v2 tuned on dev split only | done |
| D-P3 | Pilot labels are a consensus set, no kappa | final |
| D-E1 | SQLite store keyed by text-pair hash | frozen |
| D-E2 | Attribute set (agreement, emotion, hostility, attack), TODO(verify) vs Chowdhury | open |
| D-A1/A2 | Additive 0–10 rule table; weekly mean | open / final |
| D-B1–B5 | Sign sources; mean pair aggregation; SA heuristic; P_B = 1 − F/F_null; label firewall | mostly open (TODO(verify) vs Guo) |
| D-B3a | Annealing budget raised after synthetic run 1 | open |
| D-T1/T2 | Throughput is binding; Ollama serves qwen3.5 single-slot | open |
| D-E3 | B needs delta CI > 0 **and** rho_B > 0; full-week GT primary | frozen |
| D-O1/O2 | Overnight supervisor; cutoff removed | done |
| D-E4 | How the 135k-edge result is reported (rule met literally, caveats) | final |
| *(missing)* | **How the full-5% result is reported** | **to log** |

Data-side decisions (`docs/decisions.md`): 2016 scope and filters (Oct 21); weekly windows, k = 3, heavy posters > 500, thread sampling by rate (Oct 28); labelling rule (Oct 28); GT#2 transfer design (Oct 29).

---

## 17. Limitations and known issues

**Design**
- A and B share one extraction, so they are not independent replications of Chowdhury or Guo; the A rule table, pair aggregation and P_B normalisation are our readings (`TODO(verify)` ×4).
- Single primary model at 4-bit on consumer hardware; robustness model never run.
- Only the shared venue, only 2016, only weekly windows, only the 5% sample.

**Sign quality**
- Pilot accuracy 0.72 (kappa 0.44 held-out, n = 75); only 12 neutral items; pilot labels are a consensus without inter-rater kappa; logprob probabilities validated only on llama3.1:8b.
- LLM-signed camps do no better than VADER on camp recovery.

**Pipeline B**
- P_B is low and density-dependent at mean degree ~3; synthetic size-flatness and absolute-separation checks fail; heuristic optimality unverified above ~100 nodes.

**Ground truths**
- Labels are a proxy (~20% disagreement with self-declarations).
- GT#1 is dominated by a slow trend and a 06-27 shift; event weeks show nothing; composition explains ~62% out of sample.
- GT#2 level is tiny, transfer is lossy, and it shares the composition trend; GT#2 candidates disagree with each other.
- 1% samples are weak; samples are independent draws; the 5%-matched GT shares sampling noise with the pipelines.

**Statistics**
- 53 weekly points with strong autocorrelation; no Holm correction; no noise ceilings; first-difference results null.

**Known issues / housekeeping** (`docs/handoff_issues.md`)
- `docs/edge_files.md` omits `body_clean`, `parent_body_clean`, `distinguished`, `week` columns of the sampled files.
- `weekly_composition.csv` does not exist; composition covariates must be computed from edges/thread_user.
- README "Current Status" still says 2026-10-19 midway week 1; `docs/PRESENTATION_GUIDE.md` predates the evaluation; `docs/midway_numbers.md` is empty.
- Unused stubs listed in section 3; `config/config.yaml` unused.
- The calibration mean degree (2.7–4.1) differs from the 4.7–7.4 quoted in an earlier brief (different graph definition), unresolved with the data side.

---

## 18. Comparison with the proposal

**Caveat:** the SNA proposal document is not in this repo (`README.md` says "see `docs/`", but it is not there), and `C:\Users\admin\Downloads\Project Proposal.pdf` is an unrelated document. The proposal items below are reconstructed from `README.md` (RQ1–RQ5, scope note), `docs/PRESENTATION_GUIDE.md`, `docs/llm_decisions.md` and `results/dataset_size/summary.md` ("Proposal check"). Check this table against the actual proposal and add anything missing.

| Proposal item | Status | Evidence / gap |
|---|---|---|
| **RQ1**: does structure-aware B track GT better than text-scoring A? | **Answered at 5%**: rule met, but B not significant, A anti-correlated | sections 12.6, 13 |
| Controlled comparison (same LLM output for both) | done (D-S1) | the proposal's "controlled-comparison gap" is closed |
| LLM sign inference (agree / neutral / disagree) | done, 0.72 pilot accuracy | section 7 |
| Signed graph + frustration / balance measure | done (heuristic + ILP check, 2 nulls) | section 9 |
| Parameterised synthetic graph generator / synthetic validation | done (calibrated, 7 pre-stated checks) | section 10 |
| Null-model control (degree-preserving sign/edge randomisation) | done (sign shuffle + signed-degree-preserving rewiring); `docs/decisions.md` placeholder never filled | sections 9.4, 17 |
| Ground truth from subreddit party tags | done, two independent truths + robustness | sections 5, 6 |
| Agreement metrics (Spearman, Pearson, MAE, direction of change) | Spearman + permutation + block bootstrap used; `metrics.py` has Pearson/MAE/direction but they are not reported | section 11 |
| **RQ2**: when does structure help (window size, density, polarization level) | **partial**: density/size/polarization studied only synthetically; real data only weekly windows at 1% (Tier 0) and 5% | monthly windows, 20% tier, and a high- vs low-polarization week split not run |
| **RQ3**: effect of sign-inference quality | **partial**: four sign sources run (continuous, binary, VADER, unsigned) with camp recovery; `noise_injection` built but never run; no second LLM | the noise-injection curve and the robustness model are the missing pieces |
| **RQ4**: forecasting | **not started** | no forecasting code |
| **RQ5**: drivers of polarization shifts | **cut** (README scope note); the GT#1 drivers/panel analyses partly cover the GT side | |
| Second dataset (PolitiSky24) | **cut** | |
| Multi-camp detection, GNNs, alternative LLMs | **cut** (alternative LLMs survive only as an unscored robustness model) | |
| Full Politosphere 2008–2019 | **narrowed** to 2016 (raw volume measured for all years) | section 4.1 |
| Monthly windows as primary | **changed**: weekly is primary (Oct 28); monthly only built as graphs/stats, never through the pipelines | |
| Chowdhury et al. replication (A) | approximated: additive rule table, 8–9B model vs their Llama-3.1-70B | D-A1, D-E2 |
| Guo et al. replication (B) | approximated: weight formula, mean pair aggregation, normalisation `TODO(verify)` | D-B2, D-B4 |
| Local LLM via Ollama | done | |

---

## 19. Where to go next

Ordered by value for the final paper; items 1–3 are cheap and close the RQ1 story.

1. **Write up and commit the full-5% result.** Log a D-E5 entry (how it is reported, same rules as D-E4), run `rq1_bootstrap_stability.py` and `rq1_partial_week_sensitivity.py` on the full runs, regenerate slide figures, commit the uncommitted files (section 15.6), update `results/llm/INDEX.md` and the README status.
2. **Run the spec's missing analyses** (all CPU-only, no new LLM calls): Holm correction across the two truths; composition-only baselines (r/politics share, rep/dem ratio, new-user share from `results/gt1_drivers/drivers_weekly.csv`) as competitors to A and B; before/after 2016-06-27 split; split-half noise ceilings for P_A and P_B (split threads within week).
3. **Explain why A is negative.** Correlate P_A components (`A_disagree`, `A_hostility`, `share_high`) with the composition variables; check whether hostility rises from summer as GT falls; check per-week P_A level shift at 06-27.
4. **RQ3 sign quality, cheaply:** run `noise_injection` at p = 0.05–0.4 on the continuous weights and plot P_B, rho_B and camp recovery against p; compare binary vs continuous vs VADER on the RQ1 statistics (currently only camp recovery compares them).
5. **Robustness model:** score `gemma3:12b` on the pilot (dev, then held-out once); if not clearly worse than 0.72, run a few hours on the same thread prefix and report per-model delta and inter-model agreement (D-M1, D-M7).
6. **RQ2 on real data:** monthly windows from the same 5% extraction (no new LLM calls: re-aggregate by month); density effect via the giant-component size per week; contrast high- vs low-GT weeks.
7. **Camp-recovery puzzle:** LLM signs ≈ VADER ≈ chance. Check the share of edges with both endpoints labelled, the label balance in giant components, and recovery restricted to high-degree users; this is the most direct test of whether the signed graph contains partisan structure at all.
8. **Optional, expensive:** 20% tier (~16 days at current speed; mean degree of the all-venue sample graphs rises from 3.8 at 5% to 6.2 at 20%, and synthetic P_B at mean degree 5–7 is 2–3× its value at 3.2), or a faster serving engine (requires re-scoring the pilot, D-T2).
9. **RQ4 forecasting** only if time remains: lagged regressions of GT on P_A / P_B vs an autoregressive baseline.

---

## 20. Glossary

| Term | Meaning |
|---|---|
| Shared venue | r/politics + r/PoliticalDiscussion, where both GT and pipelines are measured |
| Supporter subs | 58 party-tagged subreddits used to label users |
| main / gold labels | posting-based camp labels / self-declared votes |
| GT#1, `gt1_excess` | thread dissimilarity index minus shuffled-label null |
| GT#2, `gt2` | transfer phrase estimator (0.5 = no partisan language difference) |
| `gt2_alt_auc` | logistic-regression AUC companion of GT#2 |
| P_A | weekly mean LLM edge score (0–10) |
| P_B | 1 − F_obs / F_null on the week's signed giant component |
| Frustration F | total weight of edges violating the best two-camp split |
| Sign shuffle / rewiring null | permute weights on fixed topology / swap edges keeping each node's positive and negative degree |
| Camp recovery | balanced accuracy of B's partition vs main labels |
| Tier 0 | 1% sample, whole threads, 60 or 230 edges/week |
| 5% prefix / full 5% | first 135,678 edges of the thread-ordered 5% tier / all 617,568 |
| Thread-prefix ordering | processing threads in a fixed random order per week so any stop is a thread-complete subsample |
| Circular-shift p | permutation p from rotating one series against the other (keeps autocorrelation) |
| Block bootstrap | resampling 4-week blocks of weeks to get a CI for delta rho |
| Heavy poster | author sending > 500 replies in a week |
| k = 3 | minimum sent + received replies per user per week |
| D-xx | decision IDs in `docs/llm_decisions.md` |
