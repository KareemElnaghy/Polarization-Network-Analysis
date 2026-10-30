# Lab notebook (Kareem, LLM side)

## 2026-10-30
- Phase R recon done and approved. Findings in the Phase R message; samples are not nested.
- Phase 1 started. Ollama 0.40.11 had no `~/.ollama/id_ed25519`, so `ollama pull` failed with "open id_ed25519: no such file". Generated a key with `ssh-keygen -t ed25519` and the pulls worked (3-4 MB/s: llama3.1:8b ~25 min, qwen3:8b ~25 min, gemma3:12b ~40 min).
- Built `src/algorithms/llm/{schema,backends,store,extract,data}.py`, `configs/llm/llm.yaml`; 10 mock-backend tests pass.
- Pilot sheet drawn: 150 edges, 50 per lexical-cue stratum (agree-ish / disagree-ish / other), all 53 weeks.
- Pulled llama3.1:8b (qwen3/gemma3 pulls cancelled). Logprob validation OK (see Phase 1 report). Bench: 0.77 edges/s regardless of client threads, so server is serial; 5% shared tier projects to ~223 h.
- Started a second `ollama serve` on :11435 with OLLAMA_NUM_PARALLEL=4 to test; two models loaded (12.6 GB) during user memory pressure, so that run is invalid. Killed it (`pkill -f` also killed its own shell, exit 137). Do not use `pkill -f` with a pattern that appears in the command itself. Ollama app server has a new pid after this; no model loaded, 84% memory free.
- bench.py: fixed peak-memory process matching (runner is `llama-server`); added --base-url and --tag.
- Pilot labels (Kareem, 150, from the Markdown sheet; old 14 terminal labels backed up, 13/14 identical). llama3.1:8b v1 prompt: acc 0.587, kappa 0.205 (single rater); errors = third-party negativity read as disagreement. Dev/held-out split fixed (75/75) before editing the prompt. Yasmina's labels still missing.
- Prompt v2 on dev: acc 0.627 kappa 0.307 (v1 0.587/0.234). Agree-vs-disagree AUC ~0.75 either way. Tokens/edge 36->29.5 (+11% speed). Next: compare other model families on the same dev split before choosing primary (D-M1).
- qwen3.5:9b-q4_K_M dev v2: acc 0.733 kappa 0.548 AUC 0.87 (llama 0.627/0.307/0.75). 0.81 edges/s. Stopped model afterwards to free memory.
- Held-out (scored once, --final): qwen3.5:9b v2 acc 0.707 kappa 0.439. Primary frozen (D-M7); config updated. Held-out run speed 0.64 edges/s.
- 2026-10-30 23:31 launched Tier 0 (run runs/llm/20261030T233105_tier0_b60): 1% shared sample, whole threads, budget 60 edges/week -> 12,733 edges / 12,720 unique pairs / 220 threads, qwen3.5:9b-q4_K_M prompt v2, ETA ~5.7 h, truncation 3.5%. Budget 230 would have been 24k edges / 9.5 h, so 60 was used. Stop: touch <run>/STOP. Log: logs_llm/tier0_b60.log. Gate approved by Kareem.
- Phase 2 built: Pipeline A rules (A-rules-v1), aggregation, run_pipeline_a.py, test (13 engine tests pass). Plumbing run on 88 edges OK (runs/llm/20261030T233322_pipeA_partial). Re-run when Tier 0 has more coverage.
- Phase 3 built and tested (9 tests). Heuristic: annealing budget raised after 38%-optimal first result; now 88% optimal, mean gap 0.08%. First benchmark bug: duplicate pairs made ILP and sparse matrix disagree (benchmark generator only). Plumbing run on 854 partial edges OK; no camp-recovery number yet.
