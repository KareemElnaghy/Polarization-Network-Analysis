# Phase 1 report (interim): LLM engine and shared extraction pass

**Status: gate reached for `llama3.1:8b` only.** Engine tested; logprob validation and a throughput baseline done; pilot kappa and model-vs-human agreement pending labels; qwen/gemma not pulled (D-M5).

## Built
- `src/algorithms/llm/`: `schema.py` (static system prompt, JSON schema, `agreement` first), `backends.py` (`OllamaBackend` over HTTP with constrained decoding, `think:false`, logprobs; `MockBackend`), `store.py` (append-only SQLite, Parquet export with sha256 manifest), `extract.py` (dedup, cache skip, threaded, retry on invalid JSON, week interleaving), `data.py` (shared-venue loader), `provenance.py` (immutable run folders, `results/llm/INDEX.md`).
- Config `configs/llm/llm.yaml`; scripts in `scripts/llm/` (pilot sheet, logprob validation, benchmark with prep-job guard, pilot evaluation).
- Tests: 11 pass (`tests/test_llm_engine.py`): caching, retry, resume after kill, dedup, cache-key sensitivity (model, prompt version, temperature, seed), probability normalisation, logprob parsing, schema validation, no forbidden fields in the prompt, Parquet export, run folder/registry.
- Existing suite: 85 pass before my additions.

## Blocked / not done
- `ollama pull` initially failed (`~/.ollama/id_ed25519` missing); I generated the key. Downloads are now very slow (about 0.3 MB/s; llama3.1:8b 9% after 25 min), so no model is available yet.
- Without a model: no logprob validation (D-P1 open), no throughput projection, no JSON-validity numbers.
- Pilot kappa needs your and Yasmina's labels on `data/llm/pilot/pilot_sheet_blank.csv` (150 rows, columns `parent_comment`, `reply`, fill `label_agree_neutral_disagree` with agree/neutral/disagree). Save as `labels_kareem.csv` and `labels_yasmina.csv` with columns `pilot_id,label`.

## Raw counts for the run plan (shared venue, before dedup)
1%: 152,496 edges; 5%: 617,633; 20%: 2,736,934. Unique pairs and hours per model come from `bench.py` once a model exists.

## Limitations / open questions
See `docs/limitations.md`, `docs/llm_decisions.md` (D-S5 non-nested samples, D-P1 probability method, D-M3 candidate tags, D-E2 attribute set TODO(verify)).


## Results (llama3.1:8b, Q4_K_M, Ollama 0.40.11, 100% GPU, 5.3 GB loaded)

**Logprob probabilities (D-P1), 100 pilot edges, logprobs at T=0 vs 10 samples at T=0.7** (`results/llm/bench/probs_llama3.1_8b.json`):
logprobs returned and parsed for 100% of edges; argmax equals the greedy label 100%; mean max-probability 0.90; MAE vs sample frequencies 0.059; Spearman with sample frequencies 0.54 (agree), 0.56 (neutral), 0.73 (disagree). Usable, but the model is confident (mean max p 0.90) so the probabilities are close to hard labels; the weaker agree/neutral correlations are partly sampling noise with k=10.

**Validity** (150 pilot edges): 100% valid JSON on the first try, 0 failures, 0% truncation.

**Throughput.** Clean baseline (single server, nothing else heavy running), `bench.py --parallel 1,2,4`:

| client threads | edges/s | tokens/s |
|---|---|---|
| 1 | 0.77 | 27.6 |
| 2 | 0.76 | 27.4 |
| 4 | 0.76 | 27.3 |

No gain from client parallelism: the server handles one request at a time. Projected hours at 0.77 edges/s (unique shared-venue pairs): **1% = 152k pairs, 55 h; 5% = 617k, 223 h; 20% = 2.73M, 988 h.** On this setting 5% is infeasible.

A second test (a separate server with `OLLAMA_NUM_PARALLEL=4`, clients 4 and 8) gave 0.68 and 1.29 edges/s, but it ran with two copies of the model loaded (peak Ollama RSS 12.6 GB) and while you reported heavy memory pressure, so it is **not a valid measurement** and I stopped further runs. The one hint: 8 clients may give up to about 1.7x, which still leaves 5% at 130 h or more. `peak_ollama_gb` in the first bench (0.05) is wrong: the monitor matched the wrong process name (fixed in `bench.py`).

**Pilot kappa / model accuracy:** pending. `labels_kareem.csv` has 14 labels, none from Yasmina yet.

## Options for the run plan (needs your decision)
1. Cut the output tokens (about 35 per edge): compact one-line JSON, or drop fields that Pipeline B does not need. Decode time dominates (about 1.3 s/edge).
2. Re-measure with `OLLAMA_NUM_PARALLEL=4` on the main server when memory is free (needs an Ollama restart with that env var).
3. Try an MLX backend on this Mac.
4. 1% only on this machine (55 h), 5% on a rented GPU with vLLM.
5. Process fewer weeks or a smaller share of the 5% tier.
