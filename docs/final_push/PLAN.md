# Plan

Days are counted from the agent's start (D1 = first day). Time boxes are hands-on agent time. Background compute time is listed separately as "wall". Machine names are from `PROJECT_FACTS.md`. Every task ID (P0.1 and so on) is referenced in `STATUS.md`.

## Why this order

The full-5% result already meets the pre-registered rule, but it is weak: rho_B = +0.17 (p 0.38), A runs negative, and LLM signs do no better than VADER at recovering camps. Three explanations are possible, and each one predicts something different:

| Explanation | Predicts | Test | Phase |
|---|---|---|---|
| **Noise**: B is unreliable at mean degree ≈ 3 | low split-half reliability of P_B; rho_B rises with density | noise ceilings (1.2), density ladder (2.x) | 1, 2 |
| **Signs**: the LLM signs carry little partisan information | B improves steeply as sign quality improves; alignment is weak for every sign source | noise curve (1.6), alignment by sign source (1.3), Pipeline C (1.5, 2.x) | 1, 2 |
| **Mechanism**: disagreement in the shared venue is not aligned with the camps | P(disagree) is about equal on cross-camp and same-camp edges at every density | edge-level alignment (1.3) | 1 |

Phase 1 tells us which explanation holds, using data we already have. Phase 2 is the decisive test of the noise explanation, and it is only feasible because of the encoder. Phase 3 adds statistical power and a confirmatory replication. Whichever outcome we get, `CLAIMS.md` gives a defensible headline.

---

## Phase 0: safety and freeze (D1, first 2 hours, blocks everything else)

| ID | Task | Machine | Time box | Done when |
|---|---|---|---|---|
| P0.1 | Copy `data/llm/extractions.sqlite` to a backup location (external drive or `UBU`), record its sha256 and row count (617,041 for qwen3.5 / prompt v2) in `STATUS.md`. Also back up `data/llm/pilot/`. | WIN | 20 min | both checksums match |
| P0.2 | Commit the uncommitted files listed in handbook §15.6 (full-5% results, interim results, `watch_5pct.py`, INDEX rows, handbook, presentation context). | WIN | 15 min | `git status` is clean apart from ignored files |
| P0.3 | Paste D-E5 and D-E6 from `PREREG.md` into `docs/llm_decisions.md`, filling in the git SHA. Commit, then `git tag prereg-final-v1`. | WIN | 20 min | tag exists |
| P0.4 | Write `ops/UBU/setup.sh` (clone, venv with Python 3.12, `pip install -e .` plus torch/CUDA, transformers, datasets, accelerate, sentence-transformers) and `ops/MAC/setup.sh`. Put both commands under "Needs a human". | WIN | 30 min | scripts committed |
| P0.5 | Create `STATUS.md` from the template at the end of this file. | WIN | 5 min | |

---

## Phase 1-CPU: close RQ1 with the existing cache (D1–D3)

All tasks read `data/llm/extractions.sqlite` plus the existing run folders `runs/llm/20261107T121812_pipeA_5pct_full` and `runs/llm/20261107T121822_pipeB_5pct_full`. No LLM calls. Definitions are in `SPEC_ANALYSES.md`. Run them on MAC or WIN CPUs, whichever is free. Recommended order:

| ID | Task (spec section) | Time box | Wall | Output folder |
|---|---|---|---|---|
| P1.1 | Spec completion: Holm, composition baselines, 06-27 split, bootstrap stability and partial weeks on the full run (A1) | 3 h | <1 h | `results/llm/final/rq1_spec_completion/` |
| P1.2 | Split-half noise ceilings for P_A and P_B, plus disattenuated rho (A2) | 3 h | ~8 h (B runs on halves; R=20 nulls) | `results/llm/final/noise_ceiling/` |
| P1.3 | Edge-level alignment test and weekly antagonism/alignment series, per sign source (A3) | 4 h | <1 h | `results/llm/final/alignment/` |
| P1.4 | Why A runs negative: cross-camp share mediation and A on edge types (A4) | 3 h | <1 h | `results/llm/final/a_negative/` |
| P1.5 | Pipeline C1, the intensity-weighted sign (SPEC_PIPELINE_C §C1), through the full RQ1 statistics | 3 h | ~3 h | `runs/llm/*_pipeC1_5pct_full`, `results/llm/final/pipeline_c/` |
| P1.6 | RQ3: noise-injection curve and all sign sources through the RQ1 statistics; VADER on the pilot (A5) | 3 h | ~6 h | `results/llm/final/rq3/` |
| P1.7 | RQ2: rolling 2- and 4-week windows, calendar months, thinning to 50% and 25% of threads, high- vs low-GT weeks, discordant-ally share (A6) | 4 h | ~6 h | `results/llm/final/rq2/` |
| P1.8 | RQ4: rolling-origin forecasting of ΔGT (A7) | 3 h | <1 h | `results/llm/final/rq4/` |

**Fallbacks.** If P1.2's B-on-halves runs exceed 10 h of wall time, drop to R=10 nulls and 10 splits, and say so. If P1.7 runs short of time, keep only the thinning and rolling 4-week parts.

**Checkpoint CP1** (end of D2 or D3): write `docs/reports/final_phase1_interim.md` with one paragraph per task (headline number and file path), and state which of the three explanations in the table above the evidence favours.

---

## Phase 1-GPU: encoder (speed), Pipeline C2, and a faster LLM server (D1–D3, in parallel with Phase 1-CPU)

Definitions are in `SPEC_ENCODER.md`.

| ID | Task | Machine | Time box | Wall |
|---|---|---|---|---|
| G1.1 | Export the distillation dataset (617k pairs: truncated text, LLM targets, week, thread id, split column; pilot pairs excluded) to Parquet, with a sha256 manifest | WIN | 1.5 h | 20 min |
| G1.2 | Copy the export to UBU (shared folder, scp, or SSD) | human / WIN | 15 min | depends on link |
| G1.3 | Train candidate E1 (`microsoft/deberta-v3-base`) | WIN 3090 | 1.5 h setup | 3–5 h |
| G1.4 | Train candidate E2 (`answerdotai/ModernBERT-base`) | UBU | 1 h setup | 3–6 h |
| G1.5 | Select the encoder on the validation split (rule in SPEC_ENCODER §5), then run the acceptance checks (pilot, held-out, pipeline fidelity on the 5% tier) | WIN | 2 h | 1 h |
| G1.7 | C2: embed the unique texts of the 617k pairs with `bge-base-en-v1.5`, fit the ridge head with 5-fold out-of-fold predictions, run C2 at 5% through B's machinery (SPEC_PIPELINE_C §C2) | whichever GPU is free | 2 h | 1–2 h |
| G1.6 | `llama-server` benchmark: same q4_K_M GGUF, `-np 8/16`, continuous batching; a 1,000-pair parity check against the cache; throughput (SPEC_ENCODER §8) | WIN 3090, **before** G1.3 or while E2 trains | **2 h hard** | |

**Gate G-ENC.** The encoder passes every acceptance check in SPEC_ENCODER §6, and this is recorded in `STATUS.md` and in a D-E6 addendum. If it fails, see SPEC_ENCODER §7: Phase 2 then runs with the 20% stratified LLM extraction (12 weeks) instead of the full-density encoder ladder. C2 does not depend on the encoder, so it is still reported at 5% and on those 12 weeks.

**G1.6 outcome.** If throughput is ≥ 3× Ollama's 1.94 pairs/s with ≥ 99% label parity, it becomes the LLM server for everything left that needs the LLM: encoder validation samples, the 2017 check sample, and the optional `gemma3:12b` robustness check. Otherwise stay on Ollama and do not spend more time on it.

---

## Phase 2: the density ladder (D3–D6, gated on G-ENC)

Definitions are in SPEC_ANALYSES §A8 and SPEC_DATA_EXPANSION §1.

| ID | Task | Machine | Time box | Wall |
|---|---|---|---|---|
| P2.1 | Build the input loader for the **full** cleaned shared venue (`data/processed/edges_clean/`, venue filter, the same text rules and 1200-character truncation as `data.py`). Count edges per week. | WIN | 2 h | 30 min |
| P2.2 | Encoder inference on the 1%, 5% and 20% tiers and the full venue, sharded by week across WIN and UBU, then merged (MACHINES.md) | WIN + UBU GPUs | 1.5 h | about 13M pairs ÷ the measured rate (plan: 4–12 h) |
| P2.3 | Pipeline A, B (continuous, binary), C1 and C2 at all four densities from the encoder store. B is sharded by week across WIN/UBU/MAC CPUs. R=50 nulls at full density for the headline, R=20 elsewhere. | all CPUs | 3 h | 1 day or more at full density, so start as soon as P2.2 finishes for any week |
| P2.4 | RQ1 statistics at each density, the H1 test (density trend), camp recovery and the alignment test at each density | WIN/MAC | 3 h | 2 h |
| P2.5 | Anchor check: encoder-based vs LLM-based A, B and C at 5% (same edges). Report the weekly correlations. | MAC | 1 h | |

**Fallback, if full-density Pipeline B is too slow** (projected > 48 h of wall time after sharding): run full density on alternate weeks (27 of 53, chosen by the rule in SPEC_ANALYSES §A8, not by results) and say so.

**Checkpoint CP2:** `docs/reports/final_density_ladder.md` with the ladder figure (rho_A, rho_B, rho_C against mean degree, with CIs, one panel per GT) and the H1 verdict.

---

## Phase 3: more data (gated on Phase 2 done and at least 6 days left before the hard stop)

Definitions are in SPEC_DATA_EXPANSION §2–4. Run these in this order, each behind its own feasibility check.

| ID | Task | Time box | Wall |
|---|---|---|---|
| P3.0 | Feasibility check: can the data-side jobs run for another year by changing config only? Read `config/prep_jobs.yaml`, `config/gt_runs.yaml` and the job modules, then dry-run `parse_comments` on 2017-01. | 2 h | |
| P3.1 | 2017 confirmatory replication: data jobs, labels, GT#1 and GT#2 for 2017, the encoder (frozen) on the full shared venue, and A/B/C under frozen D-E6 rules. Before trusting the encoder on 2017, check it against 2,000 LLM-scored 2017 pairs. | 6 h | 2–3 days |
| P3.2 | Pooled 2016+2017 RQ1 (106 weeks) and RQ4 forecasting with the longer series | 2 h | |
| P3.3 | 2015 (only if more than 4 days are left after P3.1) | 4 h | 1–2 days |
| P3.4 | Per-venue split of 2016 (r/politics and r/PoliticalDiscussion as separate units), if per-venue GT cells exist | 3 h | 1 day |
| P3.5 | External datasets (SRQ first, then PolitiSky24), each behind a feasibility check of 1 hour or less | ≤ 1 h check each | |

---

## Phase 4: write-up (starts at the hard stop, or earlier if everything is done)

| ID | Task | Time box |
|---|---|---|
| P4.1 | `docs/reports/final_results.md`: every confirmatory test with its verdict; exploratory work separate | 4 h |
| P4.2 | Figures (A orange, B blue, C green, GT neutral, matching the existing slides): the density ladder, alignment by sign source, the noise curve, noise ceilings, the forecasting error table, the A-negative mediation | 3 h |
| P4.3 | Handbook: add sections 21+ (new runs, decisions, results), update §0, §13, §16, §18 and §19 in place, keeping the old text | 2 h |
| P4.4 | `docs/final_push/paper_sections.md`: Methods additions (Pipeline C, the encoder, the density ladder, new pre-registration), Results RQ1–RQ4, and Limitations, using `CLAIMS.md` wording | 4 h |
| P4.5 | `docs/final_push/slide_numbers.md`: every number a slide will show, with its source file | 1 h |

---

## STATUS.md template

```markdown
# Status, final push (updated YYYY-MM-DD HH:MM)

## Now running
| Task | Machine | Started | Log | ETA |

## Done
| Task | Headline result | File |

## Needs a human
- [ ] (machine) exact command, and why

## Blocked
-

## Gates
- G-ENC: pending / passed (date, numbers) / failed (fallback taken)
- Phase 3 gate: pending / open / skipped (reason)

## Next
1.
```
