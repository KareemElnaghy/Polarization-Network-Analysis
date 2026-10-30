# Phase 1 report (interim): LLM engine and shared extraction pass

**Status: engine built and tested on the mock backend; model comparison, logprob validation, pilot kappa and throughput are NOT done yet** (blocked, see below).

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
