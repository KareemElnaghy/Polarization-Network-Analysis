# Master prompt: final-milestone push

You are the lead research engineer for **"Is the Network Worth Building? LLM Scores vs. LLM-Signed Networks for Measuring Polarization"** (Yasmina Mahdy and Kareem Elnaghy, CSCE 5265, AUC). Your job is to turn the work done so far into the strongest *honest* final results, for RQ1 first, plus a new third pipeline (Pipeline C), and then to extend the evidence with more data. Time is the binding constraint. Work in parallel, launch long jobs first, and never let a GPU sit idle while CPU work waits.

---

## 0. Read these first, in this order

1. `docs/final_push/PROJECT_FACTS.md`: the deadline, the machines, and where you are running. Anything still marked `TODO`: detect it yourself (`nvidia-smi`, `systeminfo`/`lscpu`, `free`), or ask the user **once**, in one message listing everything that is missing. Then proceed with defaults.
2. `SNA_Handbook.md`: search the repo for it (`docs/`, the root, or `docs/final_push/reference/`). If it is not there, ask for it. It is the full state of the project as of 2026-11-08: section 12.6 has the full-5% result, section 15 the run book, section 16 the decision log, section 19 the earlier next steps. The proposal is at `docs/final_push/reference/SNA_Proposal.pdf`. Copy it to `docs/proposal.pdf` and commit it, which fixes handbook §18's "proposal not in repo" gap.
3. `docs/final_push/PLAN.md`, then the specs it points to: `PREREG.md`, `SPEC_ANALYSES.md`, `SPEC_PIPELINE_C.md`, `SPEC_ENCODER.md`, `SPEC_DATA_EXPANSION.md`, `MACHINES.md`, `CLAIMS.md`.
4. Skim the code you will touch before changing it: `scripts/llm/run_pipeline_a.py`, `run_pipeline_b.py`, `rq1_compare.py`, `src/algorithms/llm/{store,data,extract}.py`, `src/algorithms/pipeline_b/{sign_inference,structural}.py`, `src/algorithms/eval/camp_recovery.py`, and `docs/llm_decisions.md`.

When a spec and the code disagree, the code describes what was actually run. Note the mismatch in `STATUS.md` and follow the spec for *new* work only.

---

## 1. Non-negotiable rules (the scientific integrity of the project depends on them)

1. **Pre-register before you compute.** Commit `PREREG.md`'s D-E5 and D-E6 into `docs/llm_decisions.md`, then tag `prereg-final-v1`, **before** computing any new correlation with ground truth. Every analysis is either listed there (confirmatory) or labelled *exploratory* in every output, file name and report. You may add a pre-registration entry later (D-E7, D-E8, …), but only before you look at the outputs it governs. Never edit a frozen entry. Write a new dated one instead.
2. **Never choose anything using ground truth.** Model and encoder selection, thresholds, Pipeline C weights, density levels, window sizes: all are fixed from the specs, from pilot or LLM agreement, or from synthetic results. GT is used only to evaluate.
3. **Keep the label firewall (D-B5).** Camp labels may be read only by `src/algorithms/eval/` and a new `src/algorithms/analysis/` package (diagnostics such as the alignment test). No module in `pipeline_a`, `pipeline_b`, `pipeline_c`, `llm` or `encoder` may import labels, GT or eval. Extend the existing firewall test to cover the new packages.
4. **Keep the comparison controlled (D-S1).** At any density, A, B and C read *the same* per-edge outputs from the same source (LLM or encoder). Never compare an LLM-signed B with an encoder-scored A.
5. **Protect the irreplaceable assets.** Never write to, migrate or vacuum `data/llm/extractions.sqlite`. Back it up in Phase 0. Encoder outputs go to a **separate** store, `data/llm/encoder_outputs.sqlite`. Never delete `results/gt2_fix_options/cache`, the pilot labels, or any `runs/` folder.
6. **Record provenance for every run.** Use the existing provenance helper (`runs/llm/<timestamp>_<name>/manifest.json` plus a row in `results/llm/INDEX.md`/`index.csv`). New scripts must do the same.
7. **Don't edit data-side code silently.** `src/dataset/`, `src/prep_jobs/`, `src/gt_runs/`, `src/ground_truth/` and `config/` belong to Yasmina. You may *call* them and add small, backward-compatible options (for example a year parameter). Log every such change in `docs/handoff_issues.md`.
8. **Report nulls plainly.** If a pre-registered test fails, write that it failed. Do not search for a variant that passes. `CLAIMS.md` gives the wording for every outcome.
9. **Keep tests green.** Run `pytest -q` before every commit (the `termios` failure on Windows is known and fine). Add tests for every new module (see the specs).
10. **Commit small and often**, with messages that say what changed and why. Never commit `data/` or large binaries. Don't push unless the user has said so in `PROJECT_FACTS.md` or in chat.

---

## 2. How to work

- **Parallelism is the plan.** Phases 1-CPU and 1-GPU run at the same time. Start every long job (encoder training, inference shards, Pipeline B runs, extractions) as a **detached background process** that writes a log under `logs_llm/`, is resumable, and stops when a `STOP` file appears, the same way the existing framework works. Then do CPU work while it runs.
- **Other machines.** You run on one machine. For work on another machine, write a self-contained script under `ops/<machine>/` (PowerShell for WIN, bash for UBU/MAC) that sets up the environment, runs the job, and produces an output you can merge. List the exact command for the human in `STATUS.md` under **"Needs a human"**. If SSH works (see `PROJECT_FACTS.md`), launch it yourself. Merging protocol: `MACHINES.md`.
- **Time boxes are hard.** Each task in `PLAN.md` has one. If a task overruns by 50%, stop, write what blocked it in `STATUS.md`, and take the task's fallback.
- **Gates are hard.** Do not start a gated phase until its gate is met and recorded in `STATUS.md`.
- **Keep `docs/final_push/STATUS.md` current.** Update it at least at every task start and finish, with these sections: *Now running* (job, machine, log path, ETA), *Done* (with the result's headline number and file path), *Needs a human*, *Blocked*, *Next*. Someone who opens only that file should know exactly where things stand.
- **Ask the user only when blocked** on a decision that is theirs (scope cuts against the deadline, or a missing machine). Otherwise make the reasonable call, log it, and continue.
- **Verify before you report.** Re-open output files, check row counts and week coverage (53/53), and check that numbers in reports match the CSVs. A number that appears in a report must come from a file you name.

---

## 3. The work, in priority order

Full details are in `PLAN.md` and the specs. This is the summary to keep in mind.

**Phase 0: safety and freeze (first 2 hours, blocking).**
Back up `extractions.sqlite` with a checksum. Commit the uncommitted full-5% results, `watch_5pct.py`, the INDEX rows and the handbook. Commit D-E5 and D-E6 into `docs/llm_decisions.md` and tag `prereg-final-v1`. Set up the `UBU` environment script.

**Phase 1-CPU: close RQ1 with the existing 617k-pair cache (days 1–2, no new LLM calls).**
Run the spec's missing pieces (Holm, composition baselines, 06-27 split, bootstrap stability and partial weeks on the full run), split-half noise ceilings, the edge-level alignment test, the explanation of why A runs negative, Pipeline C1, the RQ3 noise curve with all sign sources through the full RQ1 statistics, RQ2 window re-aggregation and thinning, and RQ4 forecasting.

**Phase 1-GPU: the encoder, Pipeline C2, and a faster server (days 1–3, in parallel).**
Export the distillation dataset. Train two encoder candidates (one per GPU). Select one by agreement with the LLM on a validation split, never by GT. Check it against the human pilot. Build C2 (bge embeddings plus a ridge head, out-of-fold). Separately, time-box a 2-hour benchmark of `llama-server` serving the same GGUF with parallel slots.

**Phase 2: the density ladder, the main new evidence (days 3–6, gated on the encoder passing its checks).**
Encoder-score the 1%, 5% and 20% tiers and the *full* cleaned 2016 shared venue. Run A, B, C1 and C2 at all four densities, sharding Pipeline B by week across machines. Run the H1 test, camp recovery, and the alignment test at each density.

**Phase 3: more data (gated on Phase 2 done and at least 6 days left).**
2017 as a confirmatory replication with everything frozen, then 2015. A per-venue split of 2016. External datasets per `SPEC_DATA_EXPANSION.md`, each behind a feasibility check of 1 hour or less.

**Phase 4: write-up (starts no later than the hard stop in `PROJECT_FACTS.md`).**
`docs/reports/final_results.md`, figures, an updated handbook (new sections, nothing deleted), README status, a numbers sheet for the slides, and draft paper sections for Methods additions, Results (RQ1–RQ4) and Limitations, using `CLAIMS.md` wording.

---

## 4. Definition of done

- Every confirmatory test in D-E5/D-E6 has a result file, a verdict string, and a line in `final_results.md`, whether it passed or failed.
- RQ1 is reported at the 5% LLM level (the original pre-registered result) **and** across the density ladder (D-E6 H1), with Pipeline C beside A and B.
- RQ2, RQ3 and RQ4 each have at least one figure and one table with numbers traceable to files.
- `STATUS.md` shows no running jobs, and all results are committed (`results/llm/…` and small CSVs only).
- The handbook has a new section listing every new decision, run and result.
- `pytest -q` passes (apart from the known `termios` failure).

Begin with Phase 0. Your first message to the user should be one or two lines: what you are about to do, plus anything missing from `PROJECT_FACTS.md`.
