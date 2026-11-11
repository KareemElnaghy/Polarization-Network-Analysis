# Pipeline C: a signed network whose edges carry more than a sign

## Motivation (the professor's suggestion)

Pipeline B reduces each reply to one number, the Guo weight w = (p_agree − p_disagree)(1 − p_neutral). It discards the hostility, emotional charge and personal-attack attributes that the same LLM call already produced, and it discards the text itself. Pipeline C keeps B's whole structural machinery (graph, frustration, nulls, camp recovery) and changes **only the edge weight**, so that any difference from B comes from the edge information alone. This keeps the D-S1 logic: one change at a time.

Two constraints decide the design.

1. **Frustration needs a scalar per edge.** A vector-valued edge has no established frustration or balance measure, and inventing and validating one (or a signed GNN) does not fit the time left. Each variant therefore defines a **pre-specified mapping from edge information to one signed weight**. The richer information enters through that mapping. C3 is the one exploratory exception.
2. **Circularity with GT#2.** GT#2 measures partisan *language*. A text-embedding edge weight might correlate with GT#2 because embeddings encode partisan vocabulary, not because of structure. So "C beats B" is judged on **GT#1** (D-E6 §C). We also report how much C's weights correlate with the partisan-phrase leaning of the reply text (a leakage diagnostic: the Spearman correlation between |w_C − w_B| and the reply's GT#2 phrase-leaning extremity, where available).

## Package layout

`src/algorithms/pipeline_c/` contains `weights.py` (C1, C2), `embed.py` (the C2 embedding features and head) and `coherence.py` (C3, exploratory). The runner is `scripts/llm/run_pipeline_c.py`: a thin wrapper that computes the weights, then calls the **existing** Pipeline B graph, frustration and null functions unchanged. Pipeline C must not import labels, GT or eval (add it to the firewall test).

---

## C1: intensity-weighted sign (free, no new model)

For each edge, using the same per-edge output that B reads:

```
w_B  = (p_agree − p_disagree) · (1 − p_neutral)          # Guo weight, as in B
h    = {none: 0, mild: 0.5, strong: 1}[hostility]
a    = 1 if personal_attack else 0
e    = {low: 0, medium: 0.5, high: 1}[emotional_charge]
m    = 1 + h + a + 0.5·e                                  # intensity multiplier, range [1, 3.5]
w_C1 = w_B · m
```

- **Why this form.** The sign still comes from agreement, exactly as in B. Hostile, attacking and emotionally charged exchanges carry more weight, in both directions, so a hostile disagreement counts more than a mild one. The coefficients are fixed here (1, 1, 0.5) and are never tuned.
- **Normalisation.** P_C1 = 1 − F_obs / mean F_null, with sign-shuffle nulls (R = 50) that permute w_C1 among edges, as in B. Because P is a ratio to its own null, the overall scale of m cancels; only the relative weights matter.
- **Pair aggregation and giant component:** the same rules as B (D-B2).
- **Robustness (not headline):** C1-bin, with w = sign(argmax agreement) · m.
- **Tests:** totality over all 54 attribute combinations; m ∈ [1, 3.5]; when all intensities are 0, w_C1 = w_B.

## C2: text-embedding edge weight (the professor's embedding version)

The edge carries a text embedding, and a pre-specified learned projection turns it into a signed weight.

1. **Embeddings.** Use a frozen general-purpose sentence encoder, `BAAI/bge-base-en-v1.5` (fallback: `sentence-transformers/all-mpnet-base-v2` if bge fails to load). Embed the parent text p and the reply text r separately, with the same 1200-character truncation. Cache embeddings **per unique comment text hash**, because a parent appears in many edges.
2. **Edge features:** x = [u_r, u_p, u_r ⊙ u_p, |u_r − u_p|], a 3,072-dimensional vector for a 768-dimensional encoder.
3. **Target:** the C1 weight w_C1 from the LLM cache. The head therefore learns "signed, intensity-weighted interaction" from text alone.
4. **Head:** ridge regression (sklearn `RidgeCV`, alphas = logspace(−2, 3, 12), chosen by internal CV on the training folds only). The training set is the 617k cached pairs excluding the 150 pilot pairs.
5. **No in-sample predictions.** On the 5% tier, C2 weights are **out-of-fold** predictions from 5-fold GroupKFold by thread. On every other density, a head fit on all 617k pairs is used, and edges that are in the 5% training set keep their out-of-fold prediction.
6. **Signed weight:** w_C2 = clip(ŷ, −3.5, 3.5). Normalisation, aggregation and nulls are the same as for C1.
7. **Quality report (selection-free, logged before any GT comparison):** the R² and Spearman of ŷ against w_C1 out of fold; the sign agreement with the LLM argmax; and accuracy on the 150 human pilot pairs, scored by the sign of ŷ.
8. **What C2 answers.** Can a cheap embedding plus a linear head reproduce what the 9B LLM contributes to the network? It also forms the C2 rung of the RQ3 ladder.

> Do not confuse C2 with the distilled cross-encoder (SPEC_ENCODER). The cross-encoder is a fine-tuned stand-in for the LLM, used to supply A, B and C1 at high density. C2 is a *different, simpler* edge representation that is part of what is being compared. At the encoder densities, C2 is computed from the same embeddings and head and run beside A-enc, B-enc and C1-enc.

## C3: edge-embedding coherence (EXPLORATORY: report it but never use it as a headline)

This is a vector-edge measure that does not reduce edges to scalars:

1. Take B's frustration partition for the week. It is label-free (`camp_assignments.parquet`).
2. Label each edge in the giant component as within-partition or cross-partition.
3. Fit an L2 logistic regression on the C2 edge features x to predict cross versus within, with 5-fold GroupKFold by thread. The weekly score is the cross-validated AUC.
4. Null: shuffle the partition labels across nodes (preserving the sizes) 20 times, giving the mean null AUC. P_C3 = AUC − null AUC.
5. Interpretation: do the edges that the structure places across the divide also *read* differently? This is the "discursive alignment" construct from the proposal's §2.1, and it is circular with GT#2 by design. Report rho with GT#1 only as a headline-free diagnostic.

## Order of work

| Step | When | Cost |
|---|---|---|
| C1 at 5% (LLM) | Phase 1, P1.5 | ~3 h (B machinery, R = 50) |
| C2 embeddings for the 617k pairs, head with out-of-fold predictions, C2 at 5% | Phase 1-GPU (G1.7), after G1.3 or while E2 trains | embedding about 1.2M unique texts: under 1 h on a 3090 |
| C1-enc and C2 at 1%, 20% and full density | Phase 2, P2.3 | embedding the unique texts of the full venue: a few hours, shardable |
| C3 | Phase 3 or Phase 4 if time allows | 1–2 h |

## How C is reported

There is one table per density, with rows A, B, C1, C2 and columns rho(GT#1), rho(GT#2), the deltas against B and A with CIs, camp recovery and Δw alignment. Below the table, one sentence answers the professor's question: *does carrying more than a sign on each edge make the network track polarization better?* Use the wording in `CLAIMS.md`.
