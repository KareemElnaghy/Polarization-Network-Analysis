# Limitations (running list, written as the paper's section would be)

- **Shared extraction.** Pipelines A and B read the same LLM call per edge (D-S1), so they are not independent replications of the cited methods; differences reflect aggregation only.
- **Single primary model**, open weights at 4-bit-class quantization on consumer hardware; robustness models are sensitivity analyses.
- **Ground truths are provisional.** Camp labels are a proxy (about 20% disagreement with self-declared votes; rule not locked); GT#2 effect is small (about 0.514 vs null 0.500); GT#1 does not track events and is largely explained by composition.
- **1% samples are weak** against both truths (GT#1 full-vs-1% Spearman 0.38); 5% is the first credible rate.
- **Samples are independent draws**, and the sample-matched GT shares sampling noise with the pipelines.
- **Truncation** of long comments to 1200 characters (rate reported per run).
- **Sign-inference quality** on Reddit is measured only by a 150-edge pilot labelled by two people.
