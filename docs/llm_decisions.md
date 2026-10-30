# LLM decisions log

Every methodological choice, at the moment it is made. Status: `open` (revisit), `frozen` (pre-registered), `final`.
`TODO(verify)` marks a point where the cited method is ambiguous and must be checked against the paper.

| ID | Date | Decision | Alternatives | Justification | Status |
|---|---|---|---|---|---|
| D-S1 | 2026-10-30 | One shared LLM call per edge returns the Pipeline A attributes and the agreement label for B. A and B differ only in aggregation. | Separate prompts per pipeline | Halves cost; isolates "is the network worth building?" from prompt differences (the controlled-comparison gap in the proposal). **Cost:** A and B are not independent replications of Chowdhury et al. 2026 and Guo et al. 2026. A robustness check with separate prompts on a small subset is planned. | frozen |
| D-S2 | 2026-10-30 | Scope: shared-venue edges (`politics`, `PoliticalDiscussion`, per `config/gt_runs.yaml` venue `shared`) from the sampled files. No conflict with Yasmina's "same sampled edges" decision: the sampled files are the input, filtered to the venue where GT is measured. All-subreddit run is secondary, if time allows. | All sampled edges | GT is defined in the shared venue; about halves the cost (51-57% of sampled edges are shared-venue). | frozen |
| D-S3 | 2026-10-30 | LLM input is `parent_body_clean` and `body_clean`, truncated to 1200 chars each (rate logged per run). No usernames, subreddits, labels or GT fields. Short replies kept. | Raw `body` | Cleaned columns are what Yasmina's pipeline uses; raw has HTML entities and markdown. | open (check truncation rate) |
| D-S4 | 2026-10-30 | Drop edges with `reply_removed` or `parent_removed`, and empty cleaned text, before the LLM. | Keep and let the model see "[removed]" | No information in the text; wasted calls. Affects P_A and P_B n per week; the same filter applies to both. | frozen |
| D-S5 | 2026-10-30 | Samples are **not nested** (570 of 10,356 1% threads are in the 5% sample; checked on sampled edge `link_id`s). Each rate is processed on its own, weeks interleaved; the extraction cache dedups identical text pairs across rates. | Tiered 1% then 5% remainder | Nested tiering is impossible with independent draws. Hand-off item: ask Yasmina whether independence is intended. | open |
| D-S6 | 2026-10-30 | Camp labels (`data/processed/labels`) are provisional (rule not locked). They enter only the camp-recovery evaluation, never a pipeline. Labelled-user population for camp recovery is the 100,108 dem+rep users of the `main` set, not the 107,705 of `results/label_versions` (that figure includes 7,597 neither-side users). | | Per Yasmina's hand-off and recon check. | open |
| D-M1 | 2026-10-30 | Primary model selection rule (pre-specified before any GT correlation is computed): best agreement with human consensus labels on the 150-edge pilot, subject to first-try JSON validity >= 99% and a throughput budget (5% shared tier in <= 24 h). Never chosen or swapped on agreement with GT. Two robustness models from two other families; same prompt, schema, sample, weeks and code. | Choose by GT correlation | Avoids forking paths. | frozen |
| D-M2 | 2026-10-30 | Thinking/extended reasoning disabled (`think: false` in the Ollama request); reasoning-style models (e.g. `gpt-oss`) are skipped for the main run. | | Thinking traces distort throughput and complicate logprobs. | frozen |
| D-M3 | 2026-10-30 | Candidates: `llama3.1:8b` (anchor to Chowdhury et al., who used Llama-3.1-70B), `qwen3:8b`, `gemma3:12b`. Tags verified by successful `ollama pull` on Ollama 0.40.11. Upper-bound reference (`qwen3:30b-a3b-instruct-2507`, ~18 GB) only on a small subsample, if memory allows. | qwen3:14b, gemma4 | 24 GB unified memory; weights <= ~16 GB. Newer tags (Qwen 3.5+/Gemma 4) not checked yet. | open |
| D-P1 | 2026-10-30 | Agreement probabilities: token logprobs of the first value token of `agreement` under constrained decoding, normalised over the three labels (prefix match: "dis"->disagree, "ag"->agree, "neu"->neutral). Validated against k=10 samples at T=0.7 on 100 pilot edges (`scripts/llm/validate_probs.py`); fallback is k samples for `agreement` only (`prob_method: samples`). | k-sample frequencies | One call instead of k. Result recorded in the Phase 1 report. | open (pending validation) |
| D-E1 | 2026-10-30 | Extraction store is SQLite, keyed by (pair_hash, model, quantization, prompt_version, temperature, seed) rather than comment_id; edges map to rows through the pair hash of the visible (truncated) text. Identical pairs share one call. | Key by comment_id | Dedup is free; comment-level results are a join. Append-only (INSERT OR IGNORE), exported to Parquet shards with a sha256 manifest. | frozen |
| D-E2 | 2026-10-30 | Attribute set: agreement, emotional_charge, hostility, personal_attack. `TODO(verify)` against Chowdhury et al. (stance, affective tone, agreement) and justify deviations. | | | open |
| D-R1 | 2026-10-30 | Environment: `.venv` (Python 3.14.2), `uv` not installed; `pytest` collects 85 existing tests cleanly. Imports use the `src.` prefix like existing tests. `.gitignore` gets `!results/llm/` so small result files can be committed. | | | final |
| D-M4 | 2026-10-30 | **Fallback rule (pre-registered, git tag `fallback-rule-v1`).** If fewer than two candidate models are *available* by **2026-11-02** (available = pulled in Ollama and returning >= 99% valid JSON on the pilot slice), `llama3.1:8b` is the primary model (literature anchor to Chowdhury et al., who used Llama-3.1-70B). Any other candidates that arrive later become robustness runs. Otherwise D-M1 applies (pilot agreement with consensus, JSON validity, throughput). Neither branch may use agreement with GT. | Wait indefinitely; pick by GT | Downloads on this network are about 0.3 MB/s; a fixed date stops the choice drifting. | frozen |
| D-M5 | 2026-10-30 | Download queue: only `llama3.1:8b` is downloading; `qwen3:8b` and `gemma3:12b` pulls paused (not started). Nothing pulled beyond that without approval. | | Network bandwidth. | open |

### Candidate tag survey (ollama.com/library/<model>/tags, 2026-10-30; sizes are Ollama's, Q4_K_M unless noted)
Memory budget: weights <= ~14-16 GB. Thinking must be off (D-M2); newer families may think by default, so each needs a `think:false` smoke test before use.

| Family | Tag | Size | Note |
|---|---|---|---|
| Llama | `llama3.1:8b` | 4.9 GB | primary anchor; downloading |
| Qwen3 | `qwen3:8b` | 5.2 GB | hybrid thinking model; `think:false` |
| Qwen3 | `qwen3:14b` | 9.3 GB | same |
| Qwen3 | `qwen3:4b-instruct-2507-q4_K_M` | 2.5 GB | non-thinking instruct (cheapest Qwen option) |
| Qwen3 | `qwen3:30b-a3b-instruct-2507-q4_K_M` | 19 GB | MoE, ~3B active; **over budget**, small-subsample reference only |
| Qwen3.5 | `qwen3.5:9b-q4_K_M` (`qwen3.5:latest` is 6.6 GB) | 6.6 GB | newer; also `qwen3.5:4b` 3.3 GB, `qwen3.5:27b` 17 GB (tight) |
| Qwen3.6 | `qwen3.6:27b` | 18 GB | no small tag listed (27b/35b-a3b only); over budget |
| Gemma3 | `gemma3:12b` | 8.2 GB | |
| Gemma3 | `gemma3:27b` | 17 GB | tight, subsample reference |
| Gemma4 | `gemma4:12b` | 7.7 GB | newer |
| Gemma4 | `gemma4:e4b-it-q4_K_M` | 6.6 GB | (`gemma4:latest` 6.6 GB) |
| Gemma4 | `gemma4:26b` (MoE `26b-a4b`) | 16 GB | tight |

Proposed robustness pair if both arrive (two different families, one newer each, under 9 GB): `qwen3.5:9b-q4_K_M` (or `qwen3:8b`) and `gemma4:12b` (or `gemma3:12b`). Not decided; D-M3 stays open.
