# Distilled encoder: the speed lever

## 1. Purpose

The LLM costs about 2 pairs/s (61.5 h for the 5% tier). The full 2016 shared venue has about 13M edges (≈ 52% of 25.3M cleaned edges; P2.1 measures the exact number), which would take 2.5 months on the LLM. A cross-encoder trained to reproduce qwen3.5's per-edge outputs can score hundreds of pairs per second. That makes the density ladder (D-E6 §D) and other years feasible.

The encoder **imitates the frozen LLM**. It is selected and accepted only on agreement with the LLM and with the human pilot, never on GT (D-M1 logic). Every pipeline at the encoder densities reads encoder outputs only (D-S1).

## 2. Distillation dataset (G1.1): `scripts/final/export_distill.py`

- Rows: every pair in the cache for qwen3.5:9b-q4_K_M / prompt v2, joined to the 5% edge file to get the week and `link_id` (thread). A pair that appears in several edges keeps its first occurrence, and the full list is recorded.
- Columns: `pair_hash`, `parent_text`, `reply_text` (the *exact truncated* strings the LLM saw, 1200 characters each, rebuilt with the same code path as `data.py`/`extract.py`; assert that the hash matches), `p_agree`, `p_neutral`, `p_disagree`, `agreement`, `emotional_charge`, `hostility`, `personal_attack`, `week`, `thread`, `split`.
- **Split by thread hash:** 90% train, 5% val, 5% test (`int(sha1(thread)) % 20`: 0 → test, 1 → val). Remove the 150 pilot pairs entirely (match on `pair_hash`, and also on the text pair to be safe).
- Output: `data/llm/distill/distill_v1.parquet` plus a `manifest.json` with the sha256, row counts per split, and the label distribution per split.

## 3. Model and training (G1.3 on WIN, G1.4 on UBU): `src/algorithms/encoder/`, `scripts/final/train_encoder.py`

| | E1 | E2 |
|---|---|---|
| Backbone | `microsoft/deberta-v3-base` | `answerdotai/ModernBERT-base` |
| Machine | WIN (3090) | UBU |
| Learning rate | 2e-5 | 5e-5 |

Settings shared by both candidates:
- **Input:** a tokenizer pair (parent, reply) in that order (as in the prompt), `truncation="longest_first"`. `max_length` = 384, unless more than 10% of training pairs would be truncated, in which case use 512 (decide from the token-length histogram *before* training and log it).
- **Heads** on the pooled representation:
  - agreement: 3-way, trained with KL divergence against the LLM probabilities (soft targets),
  - emotional_charge: 3-way cross-entropy on the LLM label,
  - hostility: 3-way cross-entropy,
  - personal_attack: binary cross-entropy.
  - Loss = 1.0·agreement + 0.5·(the others).
- **Optimiser:** AdamW, weight decay 0.01, 6% linear warmup, then linear decay. bf16 if supported, else fp16. Effective batch 32, sequences sorted into length buckets.
- **Duration:** 1 epoch. Evaluate on val every 5k steps and keep the best checkpoint by val agreement accuracy. A second epoch is allowed only if the epoch-1 val curve is still rising and the time box allows.
- **Seed:** 0. Log the curves to `runs/llm/<ts>_encoder_<name>/`.
- **Memory:** both backbones fit easily in 16 GB at batch 32 / length 384. If you hit OOM, use gradient accumulation; don't shorten `max_length`.

## 4. Outputs per edge (inference)

The outputs are p_agree, p_neutral and p_disagree (softmax of the agreement head), plus the argmax label and probabilities for each other head. Store them in `data/llm/encoder_outputs.sqlite`, using **the same table schema** as `extractions.sqlite` (columns and fields JSON), with:

- `model = "enc-<backbone>-distill-v1"`
- `quant = "fp16"`
- `prompt_version = "enc-v1"`
- `temperature = 0`, `seed = 0`

Then `run_pipeline_a.py`, `run_pipeline_b.py` and `run_pipeline_c.py` read it through new `--store` and `--model` arguments. The defaults stay as they are, so old commands still reproduce old runs. Add a test that the encoder rows parse through the existing Pipeline A rule table and the sign-inference functions.

## 5. Selection rule (G1.5): fixed now

Choose the candidate with the higher **val** agreement-argmax accuracy against the LLM. If the difference is under 0.5 percentage points, choose the faster one (pairs/s measured on the same GPU). Never look at the test split, the pilot or GT when choosing.

## 6. Acceptance checks for gate G-ENC (on the selected model, run once, logged in addendum D-E6a)

| # | Check | Threshold |
|---|---|---|
| 1 | Test-split agreement argmax accuracy vs the LLM | ≥ 0.85 |
| 2 | Test-split Spearman of the Guo weight vs the LLM's | ≥ 0.80 |
| 3 | Accuracy on the 150 human pilot pairs, 3-class, against consensus labels (the LLM scores 0.72) | ≥ 0.65 |
| 4 | Weekly P_A from the encoder vs from the LLM on the 5% tier (same edges, Spearman over 53 weeks) | ≥ 0.90 |
| 5 | Throughput on the 3090 (fp16, batch 256, length buckets) | ≥ 200 pairs/s |

Reported but not gating: the weekly P_B fidelity (encoder vs LLM, continuous, 5%), the per-attribute test accuracies, the confusion matrices, and the C1 weight Spearman.

Write D-E6a **before** any encoder-based GT comparison. It records the selected checkpoint's sha256, the five check values, and the statement that the encoder and all pipeline rules are now frozen for Phase 2 and Phase 3.

## 7. If a check fails

- **Checks 1, 2 or 4 fail:** one retry, time-boxed to 6 h of wall time: `microsoft/deberta-v3-large` on WIN, 1 epoch, lr 1e-5, batch 16 with accumulation 2. Re-select between all three candidates with the §5 rule.
- **Check 3 fails, but 1, 2 and 4 pass:** the encoder imitates the LLM well but disagrees with humans more than the LLM does. Accept it, with a written caveat in D-E6a: Phase 2 is then about structure at density *given LLM-like signs*.
- **The retry also fails:** gate G-ENC fails. Fallback for Phase 2: run the 20% tier on the LLM for **12 weeks** (ISO weeks 2, 6, 10, …, 46: every 4th week, fixed now). That is about 620k edges, using G1.6's faster server if it passed, or Ollama otherwise (about 3.5 days on Ollama: start immediately). The ladder becomes 1% / 5% / 20% on those 12 weeks, all LLM-signed. Report H1 on those weeks only, and say so.

## 8. LLM server benchmark (G1.6, hard limit 2 h)

The goal is a faster qwen3.5 server for the remaining LLM work: validation samples, the 2017 check, and optionally `gemma3:12b`.

1. Find the GGUF that Ollama uses for `qwen3.5:9b-q4_K_M` (`ollama show --modelfile qwen3.5:9b-q4_K_M`, then the blob path under `%USERPROFILE%\.ollama\models\blobs`).
2. Get a recent CUDA build of llama.cpp (a prebuilt Windows release). Run it with `llama-server -m <blob> -ngl 999 -c 32768 -np 16 --port 8081`, adding flash attention (`-fa`, or `--flash-attn on` on newer builds; check `--help`). Continuous batching is on by default. This gives 16 slots of 2,048 context each. Our prompts are under 1,200 tokens.
3. If the blob does not load (Ollama may use its own engine for this architecture), stop. Downloading a different GGUF is a model change (D-M1 would require re-scoring the pilot), so it is not worth it in this time box. Record the result and move on.
4. Add a `LlamaServerBackend` to `src/algorithms/llm/backends.py` that uses the OpenAI-compatible endpoint, a JSON-schema response format, `temperature 0`, `seed 0`, and logprobs with top 10. Map the probabilities exactly as in D-P1.
5. **Parity check** on 1,000 random cached pairs: the agreement label matches the cache for at least 99%, and the mean |Δp_agree| ≤ 0.02. Throughput is measured with 16 concurrent client threads.
6. **Pass:** ≥ 99% parity and ≥ 3× 1.94 pairs/s. Pass means it is registered as a backend with a cache key that is **identical** in model, prompt, temperature and seed, so it shares the existing cache. Record this as a decision (D-T3), because it changes the server and not the model, and parity proves equivalence. **Fail:** keep Ollama.

## 9. Inference at scale (P2.2): `scripts/final/encode_edges.py`

- Input: `--rate {0.01, 0.05, 0.2, 1.0}`, `--weeks` (a list or a shard spec such as `1-27`), `--out-store`.
- Deduplicate on the truncated-text pair hash. Skip hashes that are already in the target store, so runs are resumable and a STOP file works.
- Sort by token length, batch 256, fp16, run on GPU. Write every 10k rows.
- Each machine writes its own shard store (`encoder_outputs_<machine>.sqlite`). Merge them with `scripts/final/merge_stores.py` (`INSERT OR IGNORE` on the primary key, then verify that row counts equal the sum of unique hashes). The merge protocol is in MACHINES.md.
- Log pairs/s every minute. The final log line gives the total count, failures (expected 0) and the elapsed time.
