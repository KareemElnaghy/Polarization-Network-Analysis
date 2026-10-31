# Project walkthrough: everything we've done, in plain language

*Written for: Kareem, to prepare for the midway presentation. Snapshot as of 2026-10-31. Every number below comes from a file in the repo; the source is named next to it so you can check it.*

---

## 0. How to use this document

- **Short on time?** Read section 1 (the story in one page), section 9 (numbers cheat sheet) and section 11 (hard questions).
- **Want to really understand it?** Read in order. Each section ends with a "say it like this" line you can reuse on a slide.
- **Terms you don't know** are in the glossary (section 12).

---

## 1. The whole story on one page

**The question.** Researchers measure political polarization online in two ways:

1. **Read the text** and ask an LLM how hostile or polarized each interaction is, then average (this is **Pipeline A**, after Chowdhury et al. 2026).
2. **Build a network** where people who agree are linked by "friend" edges and people who disagree by "foe" edges, then measure how well the network splits into two opposing camps (this is **Pipeline B**, after Guo et al. 2026).

Pipeline B is much more work. **Is the extra work worth it?** That is our paper: *"Is the Network Worth Building? LLM Scores vs. LLM-Signed Networks for Measuring Polarization."* (RQ1 in the README.)

**The data.** Reddit "Politosphere", calendar year 2016, cut into 53 weekly windows. An "interaction" is one comment replying to another comment.

**The fair-fight design.** Both pipelines read the *same* LLM output for each reply (decision D-S1). They differ only in how they combine it: A takes an average, B builds a signed graph and computes "frustration". So any difference between them is about *structure*, not about prompts.

**The referee.** We need an independent yardstick for "how polarized was this week really?" that uses no LLM. Yasmina built two (**GT#1** and **GT#2**) from the data itself, using subreddit membership to guess each user's camp (Democrat or Republican).

**Where we are.**
- Data and both ground truths: **done** (Yasmina).
- LLM engine, model choice, Pipeline A, Pipeline B, synthetic validation: **built and tested**.
- First real-data run ("Tier 0", 1% sample): **finished**. A works. B is too thin to judge, because the weekly graphs are tiny.
- The actual head-to-head comparison against the ground truths (RQ1): **not done yet.** The evaluation spec has not been written. Be honest about this on stage: we have a working pipeline and a plumbing result, not an answer.

> **Say it like this:** "We built both pipelines on a shared LLM extraction so the only difference is structure. The machinery works and has been validated on synthetic data. On the first small real slice, A is usable and B is data-starved. The head-to-head result needs more data, and we know exactly how much."

---

## 2. Who does what

| Person | Owns | Where |
|---|---|---|
| **Yasmina** | Data download and parsing, reply graph, labelling users into camps, both ground truths, background-job framework | `src/dataset/`, `src/prep_jobs/`, `src/gt_runs/`, `src/ground_truth/`, `config/*.yaml` |
| **Kareem (you)** | LLM engine, Pipeline A, Pipeline B, synthetic validation, evaluation, integration run | `src/algorithms/`, `configs/llm/`, `scripts/llm/` |

Rule from your brief: your code never edits Yasmina's packages. Problems found in her outputs go in `docs/handoff_issues.md`. Branch: `llm_implementation`.

Note the two config folders: `config/` (Yasmina's) and `configs/llm/` (yours). Easy to confuse.

---

## 3. The data (Yasmina's part)

### 3.1 Source
Reddit Politosphere (Zenodo 5851729). Scope decisions (`docs/decisions.md`, 2026-10-21):
- **Time:** all of 2016.
- **Subreddits:** all US political subreddits (522 of them; non-US regions removed).
- **Comment filters:** drop deleted authors, bots and AutoModerator, non-English comments. Keep "undetermined language" (about 14%, mostly short).

Scale: **42.4M comments, 855k users, 28.1M reply edges** (`docs/prep_jobs_spec.md`, `docs/edge_files.md`).

### 3.2 What an "edge" is
One comment replying to another comment, pointing from the replier to the author of the parent comment. Self-replies and replies whose parent was lost are dropped. Each edge carries both texts (`body`, `parent_body`), a week stamp (Monday 00:00 UTC) and flags like `reply_removed`. Edges are *unsigned*: nobody has said yet whether the reply is friendly or hostile. That is the LLM's job.

### 3.3 Cleaning rules decided on Oct 28
- **Weeks, not months,** are the main window (53 weeks; first and last are partial and flagged).
- **Minimum activity k = 3:** a user needs at least 3 replies sent plus received that week.
- **Heavy posters** (more than 500 replies sent in a week) are removed with the replies sent to them. These are mostly bots or trolls.
- Removed/deleted text and privacy-script spam are skipped.
- **Sampling:** because the full data is huge, we sample **whole threads** (not individual comments, so conversation structure survives) at **1%, 5% and 20%** per week.
  - Handoff finding: these three samples are **independent draws, not nested** (only 570 of 10,356 1%-threads also appear in the 5% sample). So extraction work cannot be shared across rates (D-S5).

### 3.4 Special subreddits
`AskTrumpSupporters`, `enoughsandersspam`, `RepublicansForSanders`, `AskDemocrats`, `AskThe_Donald` stay in the data but are not used to *label* users (they are not clean party signals).

> **Say it like this:** "28 million reply edges from 2016. We analyze whole-thread samples at 1, 5 and 20 percent so that conversation structure is preserved."

---

## 4. Ground truth (Yasmina's part, but you must be able to explain it)

Without a yardstick we cannot say which pipeline is "better". Both ground truths use **no LLM and no sentiment**, so neither pipeline gets an unfair edge.

### 4.1 Who is a Democrat or Republican? (the labels)
63 of 605 subreddits carry a party tag (`docs/party_subreddits.md`). The **labelling rule** (`config/labels.yaml`, "main"):
- Count a user's comments in the 58 supporter subreddits, only comments with score ≥ 1.
- Need at least **10** such comments, and **75%** on one side.
- Result: **100,108** labelled users (35,244 Dem, 64,864 Rep) in the main set.
- Sanity check: agrees with self-declared votes (e.g. "I voted Trump") about **80%** of the time.
- A "gold" label set uses only self-declared votes (Trump = Rep; Clinton/Sanders = Dem; Johnson/Stein/conflicting excluded).

⚠ The rule is **provisional** (thresholds not locked). The Republican side is about 94% r/The_Donald, so "Republican" here largely means Donald-supporter Redditor.

⚠ Don't mix up the numbers: the brief says 107,705 labelled users, but that figure includes 7,597 "neither side" users. The correct Dem+Rep count is 100,108 (`docs/handoff_issues.md` #3).

**Hard rule:** camp labels never enter either pipeline. They are used *only* to grade the pipelines afterwards (the camp-recovery evaluation, section 7.5).

### 4.2 GT#1: "Do the camps hang out in different threads?"
Analogy: residential segregation. If Democrats and Republicans comment in completely different threads, polarization is high. If they are mixed everywhere, it is low. This uses the **dissimilarity index** across threads, and subtracts what you'd get by randomly shuffling labels (the "null"). The leftover is `gt1_excess`.

- Mean **0.174**, range 0.109 to 0.246 over 53 weeks (`results/ground_truth/summary.md`).
- Leans toward **Pipeline B** (it is a structure/co-location measure).
- Properties found by Yasmina's robustness work:
  - **Reproducible:** split-half correlation 0.86 (0.93 corrected) (`results/gt1_robustness`).
  - **Does not track campaign events:** event weeks (Iowa, conventions, debates, Comey letter, Election Day) are not higher than other weeks (diff -0.007, p = 0.69).
  - **A level drop around 2016-06-27** (about 0.21 → 0.15) that survives fixed-panel controls (`results/gt1_panel`). It is not just people coming and going.
  - **Composition explains about 79% of the weekly variation** (`results/gt1_drivers`): share of r/politics comments (rho -0.76), share of new users (+0.68), ratio of Rep to Dem users (-0.62). So any pipeline that just tracks "who showed up" will correlate with GT#1.

### 4.3 GT#2: "Can you tell the camps apart by how they talk?"
Learn which two-word phrases Republicans vs. Democrats use in their own supporter subreddits, freeze that model, then apply it to users' comments in the neutral subs (r/politics + r/PoliticalDiscussion). If you can guess someone's camp from their language, the camps talk differently, so polarization is high.

- Leans toward **Pipeline A** (it is a text/language measure).
- It went through a real struggle. The first version (volume-weighted leave-out estimator) sat *below* its own null in 51 of 53 weeks, because a few very active users dominated. Several fixes were explored (`results/gt2_fix_options`); the adopted one is the **transfer design** with a **phrase estimator, capped at 100 phrases per user-week** (commit `0a25199`).
- Result: mean **0.514** versus a null of **0.500**. The effect is small. Use ranks and the null, not the size. The secondary **AUC** version (logistic regression) is easier to read: about **0.70** (range 0.65 to 0.75), i.e. a random Republican outranks a random Democrat 70% of the time.
- Held-out accuracy in supporter subs: 0.93; self-declared voters in neutral subs: AUC only 0.66. Signal is **lost in transfer** (honest caveat).

### 4.4 How the two truths relate
- GT#1 vs GT#2 Spearman **0.80**. But both fall from spring to summer with the same composition changes, so this agreement is partly a shared trend, not independent confirmation.
- GT#2 (chosen) vs the old volume-weighted version: only 0.18. The candidates do not corroborate each other, which is an open question Yasmina flagged.
- 1% samples are weak (GT#1 full-vs-1% Spearman 0.38; 5%: 0.72). **Treat 1% as plumbing, 5% as the first credible rate, 20% as best.**

### 4.5 The decision rule
A pipeline only wins convincingly if it beats the other on the ground truth that **leans against it**. So: can B beat A on GT#2? Can A beat B on GT#1? Anything else is expected.

> **Say it like this:** "We have two independent, LLM-free yardsticks: one measures how separated the camps are across threads (favors the network), the other how well language predicts camp (favors text scoring). B must beat A on the language yardstick, or A must beat B on the structure yardstick, for a convincing win."

---

## 5. The LLM engine (your part, Phase 1)

Code: `src/algorithms/llm/` (`schema.py`, `backends.py`, `store.py`, `extract.py`, `data.py`, `provenance.py`). Config: `configs/llm/llm.yaml`. Full decision table: `docs/llm_decisions.md`.

### 5.1 What the LLM is asked
For each (parent comment, reply) pair, one structured JSON answer with four fields (agreement first, on purpose, so its probabilities are readable):

| Field | Values | Used by |
|---|---|---|
| `agreement` | agree / neutral / disagree (reply's stance toward the parent's position) | A and B |
| `emotional_charge` | low / medium / high | A |
| `hostility` | none / mild / strong | A |
| `personal_attack` | false / true | A |

Prompt contains only the two texts (cleaned, truncated to 1200 characters; 4% get truncated). **No usernames, subreddit names, labels or GT fields** (a test enforces this). Edges with removed text are dropped before the LLM (D-S4).

### 5.2 Scope: shared venue only (D-S2)
Only edges from r/politics + r/PoliticalDiscussion, because that is where the ground truth is measured. Halves the cost. Edges at each rate (before dedup): 1% = 152,496; 5% = 617,633; 20% = 2,736,934.

### 5.3 Probabilities for Pipeline B (D-P1)
B needs *p(agree), p(neutral), p(disagree)*, not just a label. We read the **token log-probabilities** of the first word of the `agreement` answer and normalise over the three options. Validated on llama3.1:8b against 10 repeated samples on 100 edges: always gives the same top label as greedy decoding, Spearman 0.54 to 0.73 with the sample frequencies. Caveat: models are overconfident (mean top probability 0.90 for llama, 0.73 for qwen), so probabilities are close to hard labels. Not separately validated for qwen.

### 5.4 Reproducibility machinery
- **SQLite store** keyed by (text-pair hash, model, quantization, prompt version, temperature, seed); append-only; identical text pairs share one call; exports to Parquet with SHA-256 manifest. This is "the crown jewel": every number can be regenerated without re-running the LLM. **It lives only on this Mac and is git-ignored, so back it up** (`data/llm/extractions.sqlite`).
- **Immutable run folders** `runs/llm/<timestamp>_<name>/` with a manifest (git SHA, config, seed, model, hashes), and a registry in `results/llm/INDEX.md` / `index.csv`.
- 11 engine tests with a mock backend: caching, retry on bad JSON, resume after kill, dedup, cache-key sensitivity, no forbidden fields in prompt, etc.

### 5.5 The pilot and model choice (the part reviewers will probe)
**Pilot set:** 150 Reddit edges drawn from the shared venue, stratified by lexical cues, spanning all weeks; labelled by you and Yasmina (agreed consensus, so **no inter-rater kappa exists**; class mix 83 disagree, 55 agree, 12 neutral). Rubric: `docs/pilot_rubric.md`. Split once into **dev (75)** and **held-out (75)** *before* any prompt edit, so there is an honest accuracy number.

**Pre-registered rule (D-M1):** choose the model by pilot agreement, JSON validity ≥ 99% and throughput. **Never by agreement with ground truth** (that would be leakage / forking paths). A fallback date and rule (D-M4, git tag `fallback-rule-v1`) was also fixed in advance.

What happened:

| Step | Model | Prompt | Accuracy | Kappa |
|---|---|---|---|---|
| Baseline | llama3.1:8b | v1 | 0.587 (all 150) | 0.205 |
| Fixed wording | llama3.1:8b | v2 | 0.627 (dev) | 0.307 |
| Different family | qwen3.5:9b (4-bit) | v2 | 0.733 (dev) | 0.548 |
| **Final test, scored once** | **qwen3.5:9b** | **v2** | **0.707 (held-out)** | **0.439** |

- Why v1 failed: llama read negativity toward a *third party* (e.g. Trump) as disagreement with the parent. v2 explains that agreement is about the parent's position, not sentiment toward topics.
- **Primary model frozen (D-M7): `qwen3.5:9b-q4_K_M`, prompt v2.** Combined accuracy 108/150 = 0.72. 100% valid JSON, 4.8 GB loaded, fits the 24 GB Mac.
- Thinking mode is off (`think:false`) for speed and clean logprobs (D-M2).
- gemma3:12b and qwen3:8b were not tested; they are earmarked as robustness models.

⚠ **Moderate accuracy means noisy signs downstream.** About 28% of edges are mislabelled; "neutral" is rare in the pilot (12 of 150), so its accuracy is poorly measured.

### 5.6 The throughput wall
The server handles one request at a time, so client threads don't help. Measured: **0.64 to 0.81 edges/s**.

| Tier | Unique pairs | Hours at ~0.7/s |
|---|---|---|
| 1% | ~152k | 52 to 66 |
| 5% | ~617k | 210 to 270 (about 245) |
| 20% | ~2.7M | 940 to 1200 |

**Conclusion:** 5% and 20% are out of reach on this Mac. They belong on a larger machine. Hence the plan to pull `qwen3.5:27b` onto the portable SSD (D-M8). It is not run here; it must pass the same pilot test before it can replace the 9B model.

> **Say it like this:** "We froze the model on a pre-registered rule using a human-labelled pilot, never the ground truth. Accuracy is about 72%, so sign noise is a real limitation. Speed, not quality, is the binding constraint: the 5% tier would take about ten days on this laptop."

---

## 6. The pipelines

### 6.1 Pipeline A (text scoring) — Phase 2
Each edge gets an integer score 0 to 10 from an **additive rule table** (`configs/llm/pipeline_a_rules.yaml`, "A-rules-v1"):
- disagree: +1
- emotional charge: low 0, medium 1, high 2
- hostility: none 0, mild 3, strong 6
- personal attack: +1
- clipped to 0 to 10.

**P_A (week) = mean edge score.** Also stored: median, share of edges ≥ 7, n, and two simple variants (A_disagree: fraction labelled disagree; A_hostility: hostility only). Weeks with n < 30 are flagged. Rules are tested to be total over all 54 attribute combinations.

⚠ `TODO(verify)`: this is *our additive reading* of Chowdhury et al., not their exact table.

### 6.2 Pipeline B (signed network) — Phase 3
Steps, all from the cached extraction (no extra LLM calls):

1. **Sign each edge** (`pipeline_b/sign_inference.py`). The main one, `continuous_llm`: `w = (p_agree − p_disagree) × (1 − p_neutral)`. Positive means friend-like, negative means foe-like, near zero means unsure. Alternatives for ablations:
   - `binary_llm` (argmax to ±1, neutral dropped)
   - `sentiment` (VADER on the reply only, no LLM, no context; a weak baseline)
   - `unsigned` (all +1; a control where P_B must be 0)
   - `noise_injection` wrapper (flip signs, for later experiments)
2. **Build a weekly graph:** users are nodes, undirected, one link per user pair with weight = mean of their interactions. Use the **giant component** for the headline. Weeks with fewer than 40 user pairs (or giant component under 20 nodes) are NaN with a reason.
3. **Frustration F:** for the best split of users into two camps, F = total weight of edges that "break the rules" (a foe edge inside a camp, or a friend edge across camps). A perfectly polarized graph (friends inside, foes across) has F = 0. A graph where signs are random has high F. Finding the exact best split is intractable on big graphs, so we use a **heuristic**: spectral start, then simulated annealing from 10 starts, then greedy polishing.
4. **Normalise:** `P_B = 1 − F_obs / F_null`, where F_null is the frustration of the *same graph with signs shuffled* (average over 50 shuffles). So P_B ≈ 0 means "no more camp-structured than random", and P_B → 1 means "strongly two-camp". This removes the size/density effect. A second null (signed-degree-preserving rewiring) is a secondary check.
5. **Camps:** each week's partition is saved (`camp_assignments.parquet`).

**Camp-recovery evaluation** (`src/algorithms/eval/camp_recovery.py`) is the *only* module allowed to read user labels. It asks: does B's partition line up with the real Dem/Rep labels (balanced accuracy, better of the two orientations, vs a label-shuffle null)? A test fails if any pipeline module touches labels.

**Heuristic quality:** on 16 random-sign graphs (36 to 77 nodes), the heuristic matched the exact ILP optimum in 14/16 (88%), mean gap 0.08%, max 1.2%. Never beat a proved optimum (that would be a bug). ⚠ Gap at the real sizes (thousands of nodes) is **not verified** because the exact solver cannot run there.

**Tests:** 9 in `tests/test_pipeline_b.py`: balanced two-camp graph gives F = 0 and P_B = 1; all-positive gives 0; random signs give about 0; shuffle preserves topology; rewiring preserves degrees; no pipeline imports labels.

> **Say it like this:** "A averages a hostility score. B asks a structural question: can this week's users be split into two camps so that friends are inside and foes across? Frustration measures how badly the best split fails, and we normalise it against shuffled signs."

---

## 7. Synthetic validation (Phase 4)

**Why:** before trusting P_B on Reddit, check it behaves on graphs where we *know* the answer. This is tool validation and a behaviour study, **not** a result about Reddit.

**Generator** (`src/algorithms/synthetic/`): a degree-corrected signed block model calibrated to the real sampled graphs (degree sequences from real weekly graphs). Knobs: polarization level `lam` (0 = random signs, 1 = perfectly camp-aligned), camp balance, `discordant_share` (allies who still fight), noise.

**Pass criteria were written down before running** (`docs/synthetic_spec.md`) and not changed afterwards. Two runs:

| # | Check | Run 1 | Run 2 (final) |
|---|---|---|---|
| 1 | P_B rises monotonically with polarization | FAIL | **PASS** (Spearman 0.964) |
| 2 | P_B flat across graph size | FAIL (CV 0.84) | **FAIL** (CV 0.46; bar was 0.15) |
| 3 | Clear separation from nulls (P_B > 0.2 at lam .6) | FAIL | **FAIL** (P_B 0.172, though 26.8 null-sd below shuffle null) |
| 4 | Heuristic close to exact optimum | PASS | **PASS** (10/10 optimal) |
| 5 | Allies who fight lower P_B | PASS | **PASS** |
| 6 | Unequal camps (6:1) still fine | PASS | **PASS** |
| 7 | Planted camps recovered | PASS | **PASS** (balanced accuracy 0.51 / 0.81 / 0.97 at lam .3 / .6 / .9) |

**6 of 8 pass.** The story of run 1 → run 2 is a nice one for the talk: a diagnostic showed the annealing was **under-converged on large weakly-polarized graphs**; the budget was raised (D-B3a), null draws were parallelised, and check 1 went from FAIL to PASS. The criteria were *not* loosened to make things pass.

What the failures mean:
- Check 3 failed on an absolute level threshold (0.2) that you set somewhat arbitrarily. The measure *does* separate polarized from random graphs strongly (26.8 sd). A z-score criterion would pass, but changing it now would be a post-hoc edit. It's offered to you as a decision (a dated spec v2).
- Check 2: for realistic sizes (500 to 10k nodes) the spread is 0.036 to 0.060 (CV about 0.19), still above the bar. Raw frustration varies 90x over the same range, so normalisation does most of the work.

**Findings that matter for the Reddit run:**
1. **P_B depends strongly on mean degree.** Real shared-venue graphs have mean degree about 2.6 (1%) to 3.2 (5%). At that density P_B is only 0.02 to 0.06 even at moderate polarization. **Compare weeks by rank, not level.**
2. **Mild polarization is invisible** (lam ≤ 0.4 gives P_B ≈ 0).
3. Camp imbalance (6:1) is not a problem.
4. Discordant allies lower P_B as intended.
5. Real graphs have lower mean degree (2.6 to 3.2) than the 4.7 to 7.4 in the original brief. Probably a different graph definition; flagged for Yasmina.

> **Say it like this:** "We tested the measure on graphs with known answers. It goes up with polarization, recovers planted camps and handles unequal camps, but it is weak on sparse graphs, which is exactly what our Reddit samples are. Two of eight pre-stated checks fail and we report them as failures."

---

## 8. First real-data run: Tier 0 (Phase 6 start)

### 8.1 What it is
The cheapest meaningful run: **1% shared-venue sample, whole threads, about 60 edges per week budget.**
- Run: `runs/llm/20261030T233105_tier0_b60`
- **12,733 edges, 12,720 unique pairs, 220 threads, all 53 weeks, 0 failed, 4.75 hours**, 3.5% truncated.
- Threads were taken in a fixed random order so the set can be **extended in place** later without redoing work.

### 8.2 Pipeline A result
All 53 weeks scored. Median 104 edges/week (min 60, max 1,334). **P_A mean 2.96, range 1.34 to 4.29.** 53% of edges labelled disagree, 21% score ≥ 7. Week-to-week jumpiness (sd about 0.7) is consistent with small weekly n (standard error about 0.3 to 0.4). **A is usable.**

### 8.3 Pipeline B result
Only **33 of 53 weeks** pass the minimum-size thresholds (18 weeks have fewer than 40 user pairs). Median graph: 60 pairs, 90 nodes. P_B for the continuous source: sd 0.68, range -3.2 to +1.0, mean 0.05, positive in 21 of 33 weeks. The two nulls disagree (shuffle 0.049 vs rewiring 0.295): a sign of instability on tiny graphs. The unsigned control is exactly 0 as it should be.

Weekly P_A vs P_B Spearman: 0.12 (n = 33), basically no relationship at this size.

**Camp recovery** (about 62 labelled users per week): continuous 0.598 balanced accuracy vs a null of 0.565, above the null's 95th percentile in only 4 of 22 weeks. Binary 0.602, VADER 0.581, unsigned 0.500 (chance). **Weak but slightly above chance.**

### 8.4 How to read this
The synthetic study predicted it: weekly graphs this small are tree-like, and frustration only exists on cycles. **B cannot be judged at this size.** This is a plumbing success, not a scientific result. And it was **not compared against GT#1 or GT#2**, because the evaluation spec must exist first.

### 8.5 What's running now / next
- **Tier 0 extension** (`20261031T140346_tier0_b230`): same thread order, raised to about 230 edges/week (about 24k edges, +11.3k new pairs, about 4.5 h). The last status I saw was 2,175 pairs done with about 2.5 h to go. Check `results/llm/INDEX.md` and the run folder for the current state before the talk.
- **5% run** (`20261031T111015_tier5pct`): was started, with thread-prefix ordering so partial results are whole-thread subsamples, a progress bar with ETA and safe Ctrl-C. About 617k pairs and about 245 h, so you decided to pause it and extend Tier 0 instead.
- **qwen3.5:27b** being pulled onto the SSD "Backup Plus" for the larger machine (D-M8).

> **Say it like this:** "The first real run finished cleanly: 12.7k edges, zero failures, 4.75 hours. A is stable. B is starved because 1% weekly graphs have only about 60 user pairs. We are extending the sample and moving the 5% tier to bigger hardware."

---

## 9. Numbers cheat sheet

| What | Value | Source |
|---|---|---|
| Comments / users / reply edges | 42.4M / 855k / 28.1M | `docs/prep_jobs_spec.md` |
| Weeks | 53 (first and last partial) | `docs/decisions.md` |
| Labelled users (main) | 100,108 (35,244 Dem, 64,864 Rep) | `docs/handoff_issues.md` |
| Label vs self-declared vote agreement | about 80% | `docs/decisions.md` |
| GT#1 excess | mean 0.174 (0.109 to 0.246) | `results/ground_truth/summary.md` |
| GT#1 split-half reliability | 0.86 (0.93 corrected) | `results/gt1_robustness` |
| GT#1 explained by composition | about 79% | `results/gt1_drivers` |
| GT#1 level drop near 2016-06-27 | about 0.21 → 0.15 | `results/gt1_panel` |
| GT#2 (phrase) | mean 0.514 vs null 0.500 | `results/ground_truth/summary.md` |
| GT#2 AUC (secondary) | mean 0.697 | same |
| GT#1 vs GT#2 Spearman | 0.80 | same |
| 1% sample fidelity (GT#1 full vs 1%) | Spearman 0.38 (5%: 0.72) | same |
| Shared-venue edges | 1%: 152k; 5%: 618k; 20%: 2.74M | `docs/reports/phase_1_report.md` |
| Primary model | qwen3.5:9b-q4_K_M, prompt v2 | D-M7 |
| Pilot (held-out, n=75) | accuracy 0.707, kappa 0.439 | Phase 1 report |
| Pilot (all 150) | 108/150 = 0.72 | same |
| Throughput | 0.64 to 0.81 edges/s | same |
| Projected hours: 1% / 5% / 20% | about 55 / 245 / 1000 | same |
| Heuristic vs exact optimum | 14/16 optimal, mean gap 0.08% | Phase 3 report |
| Synthetic checks passed | 6 of 8 | Phase 4 report |
| P_B at real density (deg 2.6 to 3.2) | 0.02 to 0.06 at moderate polarization | Phase 4 report |
| Tier 0 | 12,733 edges, 0 failed, 4.75 h, 53 weeks | Tier 0 report |
| P_A range (Tier 0) | 1.34 to 4.29, mean 2.96 | same |
| P_B usable weeks (Tier 0) | 33 of 53 | same |
| Camp recovery (Tier 0) | 0.598 vs null 0.565; 4 of 22 weeks above p95 | same |
| Tests | 85 pre-existing + 11 engine + 9 Pipeline B + A tests | reports |

`docs/midway_numbers.md` is supposed to list every number used on slides with run id and file, but it is **empty**. Fill it in as you build the slides.

---

## 10. Limitations (say these before someone else does)

1. **A and B share one LLM call** (D-S1), so they are not independent replications of Chowdhury or Guo. Differences reflect aggregation only. A separate-prompt robustness check is planned.
2. **One primary model** at 4-bit, with about 72% accuracy on a small pilot (150 edges, no inter-rater kappa, only 12 neutral items).
3. **Ground truths are provisional:** labels have about 20% disagreement with self-declared votes and the rule is not locked; GT#2 effect is small and loses signal in transfer; GT#1 doesn't track events and is mostly composition.
4. **1% samples are weak** against both truths. 5% is the first credible rate.
5. **Sample-matched GT** shares sampling noise with the pipelines; the full-week GT should be shown too.
6. **P_B is low and density-dependent in sparse graphs;** mild polarization is undetectable; the heuristic is only verified against the exact optimum up to about 100 nodes.
7. **Pipeline A rule table** and **Pipeline B normalisation / pair aggregation** are our readings of the papers (`TODO(verify)`).
8. **Truncation** of long comments at 1200 characters (about 4%).
9. **Reddit-specific:** the Republican side is about 94% r/The_Donald; Reddit demographics.
10. Samples are independent draws, not nested.

Running list: `docs/limitations.md`.

---

## 11. Questions you should expect, and honest answers

**"So which pipeline is better?"**
We can't say yet. The comparison needs the evaluation spec, the 5% tier, and confound controls. If the two pipelines end up closer than the noise ceilings, the right conclusion is "no winner at this stage".

**"Why not tune the prompt until it correlates with ground truth?"**
That would be leakage. We froze the prompt and model on a human-labelled pilot split made before any edits, and scored the held-out half once.

**"Isn't it unfair that both use the same LLM call?"**
It's the point: it isolates structure from prompt design. The cost is that we aren't replicating either paper exactly, which we state as a limitation.

**"Why is the network measure so noisy?"**
Frustration only exists on cycles; a 1% weekly graph has about 60 user pairs and is nearly a tree. Synthetic validation predicted this.

**"72% accuracy is low. Doesn't that wreck B?"**
It adds noise to signs. The synthetic noise experiments and the sign-source ablation (continuous, binary, VADER, unsigned) are how we plan to quantify sensitivity. Not run on real data yet.

**"How do you know your frustration heuristic is right?"**
Exact solver agreement on small graphs (88% optimal, 0.08% mean gap) plus a planted-camp recovery test. At real scale it's unverified; we compared only against a longer anneal.

**"Your synthetic checks failed."**
Two of eight did, on criteria fixed in advance. One is about absolute level in sparse graphs (itself an important finding), the other is size flatness. We kept the criteria unchanged and report the failures.

**"How do you handle the fact that GT#1 is mostly composition?"**
Mandatory confound controls in the planned evaluation: composition-only and volume-only baselines that must be beaten, partial correlations, ΔR², and separate analysis before/after 2016-06-27.

**"Can you scale it?"**
5% needs about 245 h on this Mac. A larger machine (or a rented GPU with vLLM) is the plan, using the 27B model if it passes the same pilot.

---

## 12. Glossary

- **Politosphere:** the Reddit political-subreddit dataset (2008 to 2019) we use for 2016.
- **Edge / interaction:** one comment replying to another.
- **Signed graph:** a network where links are + (friend/agree) or − (foe/disagree).
- **Frustration (F):** weight of "rule-breaking" edges under the best two-camp split. Zero for a perfectly balanced network.
- **P_B:** 1 − F_obs / F_null; our normalised polarization from the network.
- **P_A:** mean 0 to 10 hostility score; our polarization from text.
- **Null model:** the same thing recomputed on randomised data, to know what "nothing going on" looks like.
- **GT#1 / GT#2:** the two ground-truth polarization series (section 4).
- **Dissimilarity index:** how unevenly two groups are spread across units (here: threads).
- **Leave-out estimator:** measure of how well language predicts group, computed without letting a person's own words predict their own group.
- **Shared venue:** r/politics + r/PoliticalDiscussion, where polarization is measured.
- **Camp recovery:** does B's inferred partition match the real Dem/Rep labels?
- **Spearman ρ:** rank correlation. Insensitive to scale, good when levels aren't comparable.
- **Noise ceiling:** the maximum correlation you could expect given each series' own reliability.
- **Quantization (q4_K_M):** compressing model weights to about 4 bits so they fit in memory.
- **Logprobs:** the model's internal probability for each possible next token; our source of p(agree) etc.
- **Dev / held-out:** the two halves of the pilot; one for tuning, one scored once.
- **Tier 0:** the first small real run (1% sample, small per-week budget).
- **D-xx:** a numbered entry in `docs/llm_decisions.md`.
- **`TODO(verify)`:** a spot where we interpreted a cited paper and still need to check it.

---

## 13. Timeline

| Date | Milestone |
|---|---|
| Oct 19 | Midway sprint begins (README status; now out of date) |
| Oct 21 | Scope fixed: all 2016, all US political subs, edges built |
| Oct 28 | Cleaning/sampling/labelling decisions; prep-job framework; ground-truth runs |
| Oct 29 | GT#2 transfer design adopted (fix for the below-null problem) |
| Oct 30 | Phase R recon; Phase 1 engine built; pilot labelled; prompt v2; model comparison; **primary model frozen**; Tier 0 launched; Phases 2 and 3 built |
| Oct 31 | Synthetic validation runs 1 and 2; Tier 0 finished; Pipelines A and B on real data; 5% run paused; Tier 0 extension to 230 edges/week started |
| Nov 2 | Date of the pre-registered fallback-model rule (now moot, primary chosen on merit) |

---

## 14. What is NOT done yet (be upfront)

1. **`docs/eval_spec.md` / tag `eval-spec-v1`** does not exist. It must be written and tagged *before* any real comparison with GT.
2. **Phase 5 evaluation harness** (Spearman on levels and first differences, permutation tests, block bootstrap for Δρ, confound controls, noise ceilings, 2×2 decision table) is only partly scaffolded (`src/algorithms/eval/` has `metrics.py`, `null_model.py`, `camp_recovery.py`).
3. **No RQ1 result:** neither pipeline has been compared with GT#1 or GT#2.
4. **Robustness models** (a second and third family) not run.
5. **5% tier and 20% tier** not run (hardware).
6. **Back up** `data/llm/extractions.sqlite` off this Mac.
7. **Housekeeping:** README "Current Status" is from Oct 19; `docs/midway_numbers.md` is empty; `kareem_claude_code_prompt_v2 (1).md` and `results/llm/INDEX.md`/`index.csv` are uncommitted; `config/prep_jobs.yaml` has local edits (disk/RSS limits).
8. **Decision for you:** accept the two failed synthetic checks as documented, or approve a dated `synthetic_spec` v2 with a z-score separation criterion.

---

## 15. Suggested talk outline (about 10 minutes)

1. **The question** (1 min): Is the network worth building?
2. **Two pipelines, one extraction** (2 min): A averages, B builds a signed graph; shared LLM call.
3. **Data and the referee** (2 min): 2016 Reddit, weekly windows, GT#1 and GT#2 and the decision rule.
4. **Choosing the LLM without cheating** (1.5 min): pilot, held-out, freeze, pre-registration.
5. **Does the measure work?** (1.5 min): synthetic validation, 6/8, the sparse-graph finding.
6. **First real run** (1.5 min): A fine, B starved; why.
7. **What's next and honest limits** (0.5 min): evaluation spec, 5% tier on bigger hardware, robustness models.

Good figures to show (all in the repo with their CSVs):
- `results/ground_truth/gt_main_shared_full.png` (the two truths over 2016, with events)
- `results/gt1_panel/series_by_variant.png` (the June drop)
- `results/llm/synthetic/run2/` (P_B vs polarization, vs size, vs density)
- the Pipeline A run folder's weekly plot (`runs/llm/20261031T103856_pipeA_tier0`)

---

## 16. Where everything lives

| Looking for | Path |
|---|---|
| Every methodological decision | `docs/llm_decisions.md` |
| Day-by-day lab notebook | `docs/LOG.md` |
| Phase reports | `docs/reports/phase_{1,2,3,4}_report.md`, `tier0_real_data_report.md` |
| Running limitations list | `docs/limitations.md` |
| Issues in Yasmina's outputs | `docs/handoff_issues.md` |
| Pilot rubric | `docs/pilot_rubric.md` |
| Synthetic spec (pre-stated criteria) | `docs/synthetic_spec.md` |
| Ground-truth design and caveats | `docs/decisions.md` (section "Ground truth"), `results/ground_truth/summary.md` |
| Edge file columns | `docs/edge_files.md` |
| Run registry | `results/llm/INDEX.md`, `index.csv` |
| LLM engine / pipelines / eval / synthetic | `src/algorithms/{llm,pipeline_a,pipeline_b,eval,synthetic}/` |
| Run scripts | `scripts/llm/` |
| Settings | `configs/llm/{llm,pipeline_a_rules,pipeline_b}.yaml` |
| Reproduction commands | `README.md` ("LLM pipelines (Kareem) — reproduction") |
