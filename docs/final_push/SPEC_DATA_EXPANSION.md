# Using more of the data

There are three ways to add evidence. In order of value per hour, they are: (1) denser 2016, (2) more years, (3) other datasets. RQ1's main weaknesses are the sparse graphs (mean degree ≈ 3) and only 53 autocorrelated weekly points. Denser graphs address the first weakness, and more years address the second. Other datasets mostly address generalisation, which matters less for this paper's claims than getting RQ1 right.

---

## 1. Denser 2016: all cleaned shared-venue edges (Phase 2, confirmatory: D-E6 §D)

- Source: `data/processed/edges_clean/` (25.3M edges, all venues). Filter to `politics` and `PoliticalDiscussion`, then apply the same removal and empty-text filters and the 1200-character truncation as `src/algorithms/llm/data.py`. Reuse its functions; don't rewrite the rules.
- Expected size: about 13M edges, about 250k per week. P2.1 records the real count per week in `results/llm/final/density/full_counts.csv`.
- Scoring: the encoder only (SPEC_ENCODER). The LLM is never run at this density.
- GT: the full-week truths are the *native* match at 100% density, so there is no sample mismatch any more. This is the one setting where the primary and the matched truth coincide.
- Cost to watch: Pipeline B on giant components of tens of thousands of nodes. Shard by week (MACHINES.md). Use R = 50 for the headline continuous B, C1 and C2, R = 20 for binary, and R = 5 for the rewiring null. Record the wall time per week, so that the projection in PLAN's fallback rule is based on measurements.

## 2. Other years (Phase 3, gated)

### Why it is worth it
- **Statistical power.** Pooling 2016 and 2017 gives 106 weeks. Every CI narrows by about √2.
- **A confirmatory replication.** Everything is frozen in D-E6/D-E6a before 2017 is touched. If the 2016 findings replicate on 2017 under identical rules, that is far stronger than any 2016-only result, and it heads off the objection that the analysis was tuned to 2016.
- **RQ4** gets a longer series.

### Which years
- **2017 first:** post-election, Trump's first year, high volume, and the same main communities exist. The 58 supporter subreddits: check how many have at least 1,000 comments in 2017 (some 2016-campaign subs, such as `hillaryclinton` and `SandersForPresident`, shrink after the election). Report this; do not replace any subreddit (D-E6 §H).
- **2015 second, only if more than 4 days remain after 2017:** primary season, cheaper (12.0M raw comments), but the supporter subs are thinner, so the labels may be sparser. Check the label coverage (labelled users per side per week in the shared venue) against the 2016 median of 7,782 dem and 5,623 rep before running the GT.
- Not 2012 or earlier (volume and label coverage are too thin), and not 2018–2019 (time).

### Feasibility check (P3.0, 2 hours, before committing to it)
1. The raw dumps for all years are already downloaded (158/158 files, handbook §4.1). Confirm that the 2017 files are present on WIN.
2. Read `config/prep_jobs.yaml`, `config/gt_runs.yaml` and the job modules. Is the year a config value, or hard-coded? List every place 2016 is assumed: the date range in `parse_comments`, the week calendar in `windowing`, the label and GT configs, and the output paths.
3. If a year parameter can be added backward-compatibly (outputs under `.../year=2017/`, with 2016 paths unchanged), do it, log it in `docs/handoff_issues.md`, and tell Yasmina in `STATUS.md`. If it needs a deeper refactor, **skip years**, record why, and spend the time on §3 instead.
4. Dry-run `parse_comments` on 2017-01 only, and time it.

### Steps for a year Y (P3.1)
1. Data jobs for Y in this order: parse, edges, comment_flags, thread_user, phrase_counts, `clean_edges_and_sampling` (only the full clean set is needed; sampling tiers are optional), labels, GT#1 (`gt1_dissimilarity`), GT#2 (`gt2_transfer`), `gt_summary`. Use the existing detached job framework and its load guard. Expect 1–2 days of machine time; run it on WIN while Phase 2's Pipeline B shards run on UBU and MAC.
2. **Encoder transfer check:** sample 2,000 shared-venue pairs from Y (stratified by week), score them with the LLM (the cache handles dedup; about 20 min on Ollama, or faster on G1.6's server), and compute the same checks 1, 2 and 4 as SPEC_ENCODER §6 on this sample. If check 1 drops below 0.80, report Y as *exploratory*, not a replication.
3. Encoder inference on the full shared venue for Y. Run A, B, C1 and C2 with the frozen settings.
4. Run D-E6 §H's tests on Y alone, then pooled with 2016 (week indices restarting per year; no first difference across the year boundary; block bootstrap within year).

## 3. Per-venue split of 2016 (P3.4, exploratory)

- Treat r/politics and r/PoliticalDiscussion as separate measurement venues. This gives 106 venue-weeks, and the two venues differ a lot in size and tone, which is useful for RQ2 ("conditions").
- Prerequisite: GT#1 and GT#2 cells per venue. Check whether `src/gt_runs/cells.py` already computes per-venue cells (the venue table in handbook §6.3 suggests it does for the special venues). If computing them needs a new GT job, time-box it to 3 h.
- r/PoliticalDiscussion is about 10% of the shared venue, so its weekly graphs are sparse even at full density. That is a natural low-density condition, which is useful for RQ2.

## 4. External datasets from the proposal (each behind a check of 1 hour or less)

| Dataset | Role in the proposal | What we do | Default |
|---|---|---|---|
| **SRQ** (Zenodo 3609277) | RQ3: sign-classifier error against human stance labels | **Check:** does the release include the tweet *text*? Twitter datasets are often released as IDs only, and hydration is not possible now. If text is present, score the reply/target pairs with the LLM (via the cache; about 5,200 pairs, under 1 h on a faster server), the encoder, VADER and C2. Map support → agree and denial → disagree, and drop comment and query. Report the accuracy of each, as an out-of-domain complement to the pilot. If there is no text, record that and use our in-domain 150-item pilot as the RQ3 error measurement (we already have it). | check, then likely skip |
| **Measured error → noise curve** | the proposal's RQ3 design | Mark each sign source's measured pilot error rate on the noise-injection curve (A5): "the LLM's measured error corresponds to p ≈ x on the curve". This is how the proposal intended SRQ to be used, and it works with the pilot alone. | **do it** (cheap) |
| **PolitiSky24** (Zenodo 15616911, 2.1 GB) | RQ1–RQ2 on a second platform | Stretch goal: only if Phase 3 §2 is done or skipped and at least 4 days remain. **Check:** download, inspect the schema, and count reply edges between labelled users per week. Plan if feasible: weekly reply graphs, Sep–Nov 2024; encoder signs after a 1,000-pair LLM transfer check (domain shift from Reddit to Bluesky); GT#1-style thread dissimilarity from the 8,022 user labels; the 445 human-labelled users used as a check on those labels. Every result is exploratory (the labels are 95% LLM-made, which the proposal already notes). | stretch |
| **Conflict-Bias-Eval** | RQ1 direct-measurement baseline (text only) | No reply links, so B and C cannot run. The only possible use is checking whether the LLM's hostility and emotion attributes differ by human-labelled stance, which says little about RQ1. | **cut**; say so in the limitations |
| **DEBAGREEMENT** | the ideal RQ3 resource | The download was unavailable at proposal time. Spend at most 10 minutes checking for a public mirror (for example on Hugging Face). If one exists with Reddit agree/disagree labels, it becomes the strongest RQ3 sign-accuracy benchmark (in-domain, human, with edge-level labels). Treat it like SRQ. | 10-minute check |

Record every check's outcome, including "not available", in `STATUS.md` and in the final report's data section, so the paper can explain what happened to each dataset in the proposal.
