# Phase 1 report (interim): LLM engine and shared extraction pass

**Status: ready for gate approval.** Primary model chosen and frozen (D-M7): `qwen3.5:9b-q4_K_M`, prompt v2. See 'Final gate summary' at the end; the llama3.1:8b sections below are the earlier baseline.

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


## Final gate summary

**Pilot** (150 edges; labels = Kareem and Yasmina's agreed consensus, no inter-rater kappa; 83 disagree / 55 agree / 12 neutral; dev/held-out split fixed before any prompt edit):

| Model | Prompt | Split | Accuracy | kappa | Agree-vs-disagree AUC |
|---|---|---|---|---|---|
| llama3.1:8b | v1 | all 150 | 0.587 | 0.205 | 0.76 (dev) |
| llama3.1:8b | v2 | dev 75 | 0.627 | 0.307 | 0.75 |
| qwen3.5:9b-q4_K_M | v2 | dev 75 | 0.733 | 0.548 | 0.87 |
| **qwen3.5:9b-q4_K_M** | **v2** | **held-out 75 (once)** | **0.707** | **0.439** | n/a |

Held-out confusion (rows human, columns model): agree 16/10/2 (agree/disagree/neutral); disagree 7/35/1; neutral 0/2/2. Neutral is rare (12 of 150) so its accuracy is poorly measured. gemma3:12b and qwen3:8b were not tested (D-M3 open; planned as robustness models).

**Probability method:** token logprobs of the first `agreement` value token under constrained decoding, normalised over three labels (D-P1). Validated on llama (argmax = greedy 100%; Spearman 0.54-0.73 vs 10 samples). qwen3.5:9b returns logprobs for 100% of edges with mean top probability 0.73. Not separately validated against k-samples for qwen; do that when the larger machine is available.

**Throughput (qwen3.5:9b, one server, one stream):** 0.64-0.81 edges/s, 23.6 output tokens per edge, 100% GPU, 4.8 GB loaded. Unique shared-venue pairs: 1% 152k, 5% 617k, 20% 2.73M. Projected hours: 1% = 52-66 h; 5% = 210-270 h; 20% = 940-1200 h. Client parallelism did not help on the default server (llama test). 5% and 20% are out of reach on this Mac; they belong to the larger-hardware milestone.

**Proposed run plan (needs your approval; nothing is running):**
1. **Tier 0 (plumbing, about 5-6 h):** the 1% shared sample, whole threads in a fixed random order (seed 0), taking threads week by week until about 230 edges per week (about 12k edges). Threads stay intact so signed graphs keep their structure. Compared only with the full-week GT (the Yasmina 1% GT covers all 1% threads, not this subset), and labelled weak.
2. **Extend in place:** more threads from the same ordering fill up to the full 1% tier (52-66 h) without redoing anything, because the store is keyed by text pair.
3. Then, on the faster machine, the 5% tier and the robustness models.
Detached, resumable, `caffeinate`, status file readable, low priority. I will not start without your approval.

**Limitations to log:** no inter-rater kappa; neutral class is rare; model is mid-accuracy (about 0.72 on the pilot), so downstream signs are noisy; single primary model; llama baseline and qwen differ in more than size (family and generation).
