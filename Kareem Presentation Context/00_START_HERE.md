# Kareem Presentation Context: start here

*Written for: Claude Code (or any Claude session) that Kareem gives this folder to, to (1) explain where the RQ1 midway milestone stands and (2) help build the slide deck. Snapshot 2026-11-01. Branch `llm_implementation` of Polarization-Network-Analysis.*

## The situation in five lines
- Project: does a signed-network Pipeline B (LLM signs -> weekly signed graph -> normalised frustration P_B) track ground truth better than a text-scoring Pipeline A (mean LLM hostility score P_A)? Reddit Politosphere 2016, 53 weeks. Yasmina built the data and two LLM-free ground truths (GT#1, GT#2); Kareem built the LLM engine, both pipelines, validation and evaluation.
- Midway milestone: delivered a working pipeline on real data, a pre-registered evaluation spec, a model frozen without using the ground truth, synthetic validation (6/8 checks), and a preliminary RQ1 answer of "no clear winner" (1% Tier 0), then a 5% run.
- Final 5% result (22% prefix, 135,678 edges, all 53 weeks): rho A -0.44 / -0.38, rho B -0.05 / +0.07 (GT#1 / GT#2); delta rho_B - rho_A on GT#2 +0.44, CI [0.06, 0.70]. The pre-stated rule is met literally on GT#2, but B is itself about zero, the gap comes from A running the wrong way, the sample-matched check shows no winner, and the CI only cleared 0 at the last cut. Reported honestly as: no evidence yet that either pipeline tracks real polarization.
- The 5% extraction was stopped on purpose (GPU needed elsewhere) and is resumable.
- Still to do: more data, composition/volume baselines, pre/post 2016-06-27 split, noise ceilings, Holm correction, robustness model gemma3:12b, explain why A is negative.

## Read in this order
1. `01_RQ1_findings_report.md`: the final result, honest reading, limits, next steps (same as `reports/rq1_5pct_report.md`).
2. `02_storyline_and_slide_outline.md`: the five-act story (Yasmina's data -> Kareem's algorithms -> experiments -> next) and an 18-slide outline with talking points, a figure per slide and handoff lines.
3. `03_numbers_registry.md`: every number with its source file (use it for all slide numbers).
4. `04_qa_and_glossary.md`: likely questions and honest answers.
5. `05_how_to_resume_and_reproduce.md`: resume the extraction, re-run the evaluation, what is not in git.
6. Background: `docs/PRESENTATION_GUIDE.md` (a long explainer of the whole project; **dated before the evaluation existed**, so its "not done yet" items about the eval spec and RQ1 are out of date), `docs/llm_decisions.md` (decision log D-*), `docs/LOG.md` (lab notebook), `docs/eval_spec.md`.

## Folder map
| Path | Contents |
|---|---|
| `docs/` | every project markdown file (decisions, specs, log, handoff issues, limitations, pilot rubric) |
| `reports/` | phase 1 to 4 reports, Tier 0 reports, the final 5% report |
| `results/llm/rq1_5pct/` | correlations, delta, summary per checkpoint (`5pct_ckpt1..4`) and final (`5pct_final`, incl. stability and sensitivity text) |
| `results/llm/synthetic/`, `results/llm/bench/` | synthetic validation (checks.md, plots, CSV) and throughput benchmarks |
| `results/ground_truth/` | GT summary, weekly GT CSV, GT plots |
| `pipeline_outputs/` | final weekly A and B series, camp recovery CSV, native plots, and `figures/` (4 slide-ready PNGs) |
| `configs/` | LLM config, Pipeline A rule table, Pipeline B config |
| `scripts/` | the evaluation, figure, overnight and run scripts used |
| `logs/` | overnight supervisor log, final evaluation log, parallel sweep |

## Rules to respect when helping
- Do not change the prompt, the primary model (qwen3.5:9b-q4_K_M, prompt v2), the eval spec or thresholds after seeing results; log any new decision in `docs/llm_decisions.md`.
- Do not present "B wins". Present the rule outcome together with its caveats (see `01_RQ1_findings_report.md` section 1).
- State coverage ("22% of the 5% tier, 135.7k edges, 53 weeks") with every 5% number.
- Pipeline A = orange, Pipeline B = blue, ground truths = neutral ink, in every chart.
- Do not edit Yasmina's packages (`src/dataset`, `src/prep_jobs`, `src/gt_runs`, `src/ground_truth`, `config/*.yaml`).

## Prompt to give Claude Code
> Read `00_START_HERE.md`, then `01_RQ1_findings_report.md`, `02_storyline_and_slide_outline.md`, `03_numbers_registry.md` and `04_qa_and_glossary.md` in this folder. I am Kareem; I present the midway milestone on RQ1 (Pipeline A vs B). First, explain to me in plain language where we stand and what the honest answer to RQ1 is. Then help me build the slide deck following the five-act outline (Yasmina's data and ground truths, then my algorithms, then the experiments, then next steps), using the figures in `pipeline_outputs/figures/` and only numbers from `03_numbers_registry.md`. Keep the story smooth with a handoff line between acts, show coverage on every 5% slide, and do not claim that B wins.
