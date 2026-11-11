# Final-milestone runbook

This folder holds the plan, specs and pre-registration for the final milestone. It is written for an AI coding agent (Claude Fable in Claude Code) working inside this repo, and it is also readable by the two of us.

## How to use it

1. Unzip into the repo at `docs/final_push/` (so this file is `docs/final_push/README.md`).
2. If `SNA_Handbook.md` is not already in the repo, copy it into `docs/final_push/reference/`. The agent needs it, and it is not in this zip because it lives in the claude.ai project. The proposal PDF is already included in `reference/`.
3. Fill in `PROJECT_FACTS.md`: the deadline, the machines, and which machine the agent runs on. This takes five minutes, and the plan's time boxes depend on it.
4. Commit the folder before the agent starts:
   `git add docs/final_push && git commit -m "Add final-milestone runbook"`
5. Open Claude Code at the repo root on the data-host machine (the Windows PC, unless `PROJECT_FACTS.md` says otherwise) and paste:

   > Read `docs/final_push/PROMPT.md` and follow it exactly. Start with Phase 0.

6. The agent keeps `docs/final_push/STATUS.md` current. Check it when you come back. It names any command that you need to run on another machine.

## Files

| File | What it is | Who reads it |
|---|---|---|
| `PROMPT.md` | The master instructions for the agent | agent |
| `PROJECT_FACTS.md` | Deadline, machines, network, who runs where. **You fill this in.** | both |
| `PLAN.md` | Phased plan with time boxes, gates and machine assignment | both |
| `PREREG.md` | Decision entries D-E5 and D-E6, committed before any new result is computed | both |
| `SPEC_ANALYSES.md` | Exact definitions of every new RQ1–RQ4 analysis | agent |
| `SPEC_PIPELINE_C.md` | The third pipeline: edge features and text embeddings | both |
| `SPEC_ENCODER.md` | Distilling the LLM into a fast encoder: the speed lever that makes full density possible | agent |
| `SPEC_DATA_EXPANSION.md` | Denser 2016 data, other years, other datasets, with feasibility gates | both |
| `MACHINES.md` | How work is split across the three machines, and how shards are merged | both |
| `CLAIMS.md` | What we may say in the paper for each possible outcome | both |
| `STATUS.md` | Live progress log, kept current by the agent | both |
| `reference/` | The proposal PDF (and the handbook, if you add it) | agent |

## The plan in five lines

1. **Hour 0:** back up the 617k-pair cache, commit the uncommitted results, and freeze the new pre-registration (D-E5, D-E6).
2. **Days 1–2, on the CPU, using the cache we already have:** close RQ1 (noise ceilings, edge-level alignment test, why A runs negative, Holm, composition baselines), build Pipeline C1, and run RQ3's noise curve and RQ4.
3. **Days 1–3, on the GPUs, in parallel:** distil qwen3.5 into a small encoder that scores hundreds of pairs per second, and build Pipeline C2 (a text-embedding edge weight).
4. **Days 3–6:** score the 20% tier and the *full* 2016 shared venue with the encoder, then run A, B and C at four densities. This is the real test of "structure helps once the graph is dense enough".
5. **Remaining days:** replicate on 2017 as a confirmatory year (and 2015 if time allows), run cheap external checks, then write up. Each step is gated on the one before it.
