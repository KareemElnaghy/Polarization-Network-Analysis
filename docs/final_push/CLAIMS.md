# What we may claim, for each outcome

The goal is the strongest claim the evidence supports, and nothing beyond it. Every claim in the paper and slides comes from this file or from a test result that satisfies the D-E6 definitions. The final report quotes the test ID behind each claim.

## The claim ladder for "does network structure help?"

From strongest to weakest. Use the highest rung whose condition is met.

| Rung | Claim | Condition (all from D-E6) |
|---|---|---|
| 5 | "Routing through a signed network tracks polarization better than direct LLM scoring, and the advantage replicates on an unseen year." | H1b and H1c pass at full density in 2016 **and** in 2017 (§H) |
| 4 | "Once the interaction graph is dense enough, the signed network tracks polarization and outperforms direct scoring; at sampled densities it is limited by noise." | H1a (trend) **and** H1b (tracks at 100%) **and** H1c (beats A at 100%) pass; H-NC shows low reliability of B at 5% |
| 3 | "The signed network's ability to track polarization grows with density, while direct scoring's does not." | H1a passes, H1b does not (rho_B rises with density but is not significant at 100%) |
| 2 | "The signed network is never worse than direct scoring and passes our pre-registered comparison rule, but neither measure demonstrably tracks the ground truths at the densities we could afford." | The current status (D-E5), with H1 not passing |
| 1 | "In this setting, structure adds nothing that the text does not already carry." | H1a fails, B ≈ A at every density, and H-AL fails |

Whichever rung we reach, the paper also reports the mechanism results below, which stand on their own.

## Mechanism results (each one is a finding whatever the rung)

| Result | If it passes | If it fails |
|---|---|---|
| **H-AL (alignment)** | "Disagreement concentrates across the partisan boundary: cross-camp replies have Δw = … lower Guo weight than same-camp replies (CI …). This is the condition that structural balance needs." | "LLM-detected disagreement is not aligned with the partisan boundary in the shared venue (Δw = …, CI includes 0). In Fraxanet's terms, this is antagonism without alignment, so no frustration-based measure can recover the camps. This explains the near-chance camp recovery." |
| **H-AL by sign source** | "LLM signs carry more partisan alignment than sentiment (Δw_LLM vs Δw_VADER)." | "LLM signs and sentiment are equally (un)aligned, so the LLM's contextual reading adds little *partisan* information at the edge level." |
| **H-AN (why A runs negative)** | "Text scoring runs opposite to segregation because, as the camps mixed more after June 2016 (GT#1 fell), more replies crossed camp lines and cross-camp replies are more hostile. Segregation and antagonism move in opposite directions." | "Pipeline A's negative association is not explained by the cross-camp share. We report it as an open problem." |
| **H-NC (noise ceiling)** | "P_B's split-half reliability at 5% is only …; the observable rho_B is bounded at about …." | "P_B is reliable at 5% (…), so its weak association with GT is not a noise artefact." |
| **Pipeline C (C1/C2 vs B)** | "Carrying hostility and intensity (C1) or a text embedding (C2) on each edge improves tracking of the structural truth GT#1 beyond a bare sign." | "Richer edge information does not improve on the sign alone. The bottleneck is not what each edge carries." |
| **H3 (noise curve)** | "Sign errors degrade the structural measure: at the LLM's measured error rate (p ≈ …), B loses … of its noise-free association." | "B is insensitive to sign noise in this range, which suggests the signs are not what drives P_B here." |
| **RQ4** | "Adding the network score improves one-week-ahead forecasts of change in polarization (DM p = …)." | "No score beats persistence for forecasting weekly change, consistent with the near-zero first-difference correlations." This is the expected outcome, and it is a finding. |

## Language rules

- Write "tracks" or "is associated with", never "measures polarization better", unless rung ≥ 4.
- Always give rho with its CI and its Holm-adjusted p. Never give a p value without the test name.
- Name the truth every time: "on the structural truth (GT#1)" or "on the linguistic truth (GT#2)".
- Encoder-based results: "using a distilled encoder that reproduces the LLM's edge labels with … accuracy". Never present them as LLM results.
- Confirmatory and exploratory results go in separate subsections, and exploratory ones are labelled in the text.
- Never "the network is better" as a bare sentence, in the paper or on a slide.

## The one-sentence framing for the presentation (fill in the brackets from the results)

> "We ran the first controlled comparison of LLM text scoring against LLM-signed networks on the same 2016 Reddit data and two LLM-free ground truths. [Rung sentence.] [The strongest mechanism sentence.] The practical answer to 'is the network worth building?' is: [yes, once the graph is dense enough / only when disagreement aligns with the group boundary / not in this setting], and we show why."
