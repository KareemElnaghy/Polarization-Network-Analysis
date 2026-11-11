# How to resume the 5% run and reproduce every number

*Written for: Kareem (and any Claude Code session working for him) on the Windows PC with the RTX 3090. Snapshot 2026-11-01.*

## Where the 5% run stands
- Extraction run folders (same selection, same cache): `runs/llm/20261031T232220_tier5pct` (stopped by an HTTP 500 from Ollama at about 98,050 pairs) and `runs/llm/20261101T092612_tier5pct` (resumed from the cache; stopped on purpose, state `partial`).
- The store `data/llm/extractions.sqlite` holds about **135,600 unique pairs** (about 22% of the 617,041 pairs in the 5% shared-venue tier), 0 failed. Everything is stored per pair, so no progress is lost when the process stops.
- The sample is thread-prefix ordered: any stopping point is a set of whole threads covering all 53 weeks. It is a thinner sample than a full 5%, not a biased one.
- Both folders contain `selected_edges.parquet`, which is the same selection. Pipelines A and B read the shared store, so evaluating on the newest folder uses all pairs from both.

## Resume the extraction
Run from the repo root in PowerShell (the venv is `.venv`, Python 3.12).
1. Start the server (do not wait on it, and do not pipe it; it holds the pipe open):
   `Start-Process powershell -ArgumentList "-NoProfile","-File","scripts/llm/start_ollama.ps1","-Parallel","4" -WindowStyle Hidden`
   It sets `OLLAMA_NUM_PARALLEL=4`, `MAX_LOADED_MODELS=1`, `FLASH_ATTENTION=1`, `CONTEXT_LENGTH=4096`, `KEEP_ALIVE=30m`.
2. Start the extraction detached:
   `Start-Process .\.venv\Scripts\python.exe -ArgumentList "scripts/llm/run_extraction.py","--rate","0.05","--budget","1000000000","--name","tier5pct" -RedirectStandardOutput logs_llm/tier5pct.log -RedirectStandardError logs_llm/tier5pct.err -WindowStyle Hidden`
   The progress bar goes to `logs_llm/tier5pct.err`. `status.json` in the newest run folder shows `done`, `failed`, `eta_h`.
3. Stop safely: create a file named `STOP` in the run folder (`New-Item runs\llm\<run>\STOP`), or Ctrl-C if it runs in a terminal.
4. Unattended: `python scripts/llm/overnight.py --stop-at 23:59` restarts the extraction if it dies (this happened once), runs checkpoint evaluations (first at 30,000 pairs, then every 2 h), refreshes a draft report, and commits. Create `logs_llm/FINISH` to make it wrap up early. It does not run the gemma robustness model's 5% run.
5. Keep the PC awake: AC sleep/hibernate/monitor timeouts were set to 0 with `powercfg /change`. Undo with `powercfg /change standby-timeout-ac 30` when done.

Speed: about 2.5 to 2.9 pairs/s, so 100k pairs is about 10 hours and the full 5% would need about 55 to 60 hours. More server slots do not help (see `01_RQ1_findings_report.md`, "Throughput").

## Evaluate whatever is in the store
```
RUN=$(ls -d runs/llm/*_tier5pct | tail -1)
python scripts/llm/run_pipeline_a.py $RUN --name pipeA_5pct_final
python scripts/llm/run_pipeline_b.py $RUN --name pipeB_5pct_final          # slowest step: about an hour at 135k pairs
python scripts/llm/eval_camp_recovery.py $(ls -d runs/llm/*_pipeB_5pct_final | tail -1)
python scripts/llm/rq1_compare.py $(ls -d runs/llm/*_pipeA_5pct_final | tail -1) $(ls -d runs/llm/*_pipeB_5pct_final | tail -1) --out results/llm/rq1_5pct/5pct_final --label 5pct_final
```
(`rq1_compare.py` implements `docs/eval_spec.md` exactly; the rule clarification D-E3 was logged before any 5% number existed.)

## Robustness model (not run yet)
`gemma3:12b` is pulled. Step 1: score it on the 150-edge pilot (needs the GPU, which is free now): `python scripts/llm/pilot_run.py gemma3:12b --prompt v2 --split dev` then `--split heldout --final`. The primary scored 0.733 (dev), 0.707 (held-out), 0.72 combined. Only if gemma is not clearly worse, run the same 5% ordering for about 2 hours and report delta rho per model plus inter-model agreement. It is a sensitivity analysis, never a replacement for the primary (D-M7). Its speed in Ollama is unknown; qwen3.5 is served on one slot only.

## Files that are NOT in git (back them up)
`data/llm/extractions.sqlite` (the whole extraction cache, crown jewel), `runs/llm/*`, `data/llm/pilot/` (labels and private key). Only docs, configs, scripts and small CSVs in `results/llm/` are committed. Branch: `llm_implementation`.

## Windows gotchas met
- `Start-Process -Wait` on `start_ollama.ps1` hangs (the server inherits the pipe).
- `ollama pull` resumes partial downloads after an interruption.
- `tests/test_llm_engine.py::test_label_tool_resume_back_skip` fails on Windows only because `termios` is Unix-only (irrelevant to the run).
- macOS commands (`caffeinate`, `nice`) do not apply.
- Another project (Open-Jev) used the GPU earlier; it needed to be closed before the 3090 had free VRAM.
