# Results and Analysis Draft

## Overview

Our experiments support an evidence-first framing of wearable sensor-language understanding. The structured verifier is responsible for verifiable support and counterfactual rejection, while caption-order self-consistency provides the strongest caption backbone. The forced-choice reranker is therefore not treated as a general grounding module. Instead, it is used as a selective ambiguity resolver for cases where the caption candidates remain difficult after the order-vote backbone.

This distinction is important for the claims in the paper. Improvements in caption selection should not be interpreted as improvements in counterfactual support F1. Conversely, strong support/rejection results should be attributed to the structured support backbone, not to the reranker. The experiments below are organized around this separation: main caption selection, selective reranking, evidence controls, cross-model robustness, and qualitative failure analysis.

## Main Caption Selection Results

The strongest caption backbone before reranking is the order-vote5 variant. A fixed margin-2 gate improves over this backbone on two of the three datasets: UCI HAR improves from 0.7361 to 0.7607, and MHEALTH improves from 0.9329 to 0.9512. Both improvements are significant under paired McNemar testing. WISDM shows a small positive change from 0.7925 to 0.7956, but the effect is not statistically significant. This supports a conservative conclusion: gated forced-choice reranking can help residual caption ambiguity, but the gain is dataset-dependent.

To avoid threshold tuning on the test set, we also evaluate a dev-calibrated gate. The selected thresholds are 1.5 for UCI HAR and 1.0 for both WISDM and MHEALTH. On the held-out split, the calibrated gate reaches 0.7706 on UCI HAR, 0.8147 on WISDM, and 0.9855 on MHEALTH. These results are stronger than the fixed-margin summary and provide a safer main-table protocol. They suggest that the reranker is most useful when treated as a calibrated selective decision rule rather than a universally applied replacement for the vote backbone.

| Dataset | Vote5 Acc. | Fixed m=2 Acc. | Fixed Delta | Fixed mid-p | Dev Threshold | Heldout Acc. | Heldout Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| UCI HAR | 0.7361 | 0.7607 | 0.0246 | 0.0000 | 1.5 | 0.7706 | 0.5832 |
| WISDM | 0.7925 | 0.7956 | 0.0031 | 0.3593 | 1.0 | 0.8147 | 0.6026 |
| MHEALTH | 0.9329 | 0.9512 | 0.0183 | 0.0156 | 1.0 | 0.9855 | 0.9564 |

## Selective Gating and Efficiency

The gate is not merely a reporting device. It changes only a subset of decisions, which makes it qualitatively different from always overriding the vote backbone. Under the fixed margin-2 setting, the override coverage is 35.3% for UCI HAR, 19.2% for WISDM, and 14.6% for MHEALTH. This matters for two reasons. First, selective intervention limits the damage from forced-choice errors. Second, it gives the method a natural cost interpretation: the offline analysis scores every example to estimate the gate, but a deployed version should be discussed in terms of the proportion of examples whose decision would actually be changed.

No-gate override is weaker than selective gating on UCI HAR and WISDM, which supports the need for selective intervention. MHEALTH is an exception because the forced-choice scorer is already extremely strong on that candidate set. This exception should be discussed as a dataset-specific boundary rather than as evidence that gating is unnecessary in general.

## Evidence-Control Analysis

The evidence controls are central to the soundness of the paper. They show that the reranker is not simply a clean sensor-evidence interpreter. With the Qwen2.5-Coder reranker, hidden-evidence performance is surprisingly high, reaching 0.8512 on UCI HAR, 0.8853 on WISDM, and 0.9817 on MHEALTH under the choice scorer. This indicates that candidate-language priors and answer/candidate structure contain substantial signal. Numeric-mask controls are also close to visible-evidence performance on several datasets. These findings prevent an over-strong grounding claim.

At the same time, the controls do not reduce the method to a pure prior. Shuffled evidence and visible evidence diverge in important settings, especially in the Qwen3 control runs. For Qwen3, shuffled evidence drops by 0.1100 on UCI HAR, 0.1170 on WISDM, and 0.3811 on MHEALTH relative to visible evidence. This indicates that the evidence-to-window mapping can matter, even though the model also exploits candidate priors. The correct interpretation is therefore bounded: the reranker uses evidence, but its decisions remain entangled with language and candidate-set biases.

| Dataset | Visible Choice | Shuffled Choice | Numeric-Mask Choice | Hidden Choice | Main Risk |
|---|---:|---:|---:|---:|---|
| UCI HAR | 0.6843 | 0.5744 | 0.6649 | 0.6455 | numeric evidence contributes only modestly |
| WISDM | 0.6708 | 0.5538 | 0.6474 | 0.7090 | hidden evidence exceeds visible evidence |
| MHEALTH | 0.9848 | 0.6037 | 0.9726 | 0.7348 | visible evidence helps, but numeric-mask remains high |

## Cross-Model Robustness and Calibration Boundary

The Qwen3 cross-model experiment is a useful boundary check. Using the same closed-candidate setting, Qwen3 underperforms order-vote5 on UCI HAR and WISDM, with gated accuracies of 0.6856 and 0.6841, respectively. On MHEALTH, however, Qwen3 reaches 0.9848 and significantly exceeds the vote5 baseline. This mixed behavior shows that forced-choice reranking is not a model-independent improvement mechanism. It should be described as a calibrated module whose reliability depends on the model, dataset, and candidate distribution.

The calibration results strengthen this point. Qwen3 requires much larger dev-selected thresholds: 5.0 for UCI HAR and 10.0 for WISDM and MHEALTH. In contrast, the main Qwen2.5-Coder setting uses lower thresholds. Thus, a fixed margin cannot be presented as a universal rule. The paper should use dev calibration as the primary protocol and report fixed-margin results as an analysis setting.

## Failure Taxonomy

The qualitative failure analysis further supports a selective ambiguity-resolution interpretation. The fixed-margin gate produces 20 vote5-wrong-to-gated-right corrections on UCI HAR, 11 on WISDM, and 6 on MHEALTH. Regressions are much rarer on UCI HAR and MHEALTH, with 1 and 0 cases respectively. WISDM has 7 regressions, which helps explain why its aggregate gain is small and non-significant.

This pattern suggests that the reranker is most helpful when the vote backbone is uncertain among semantically close candidate captions. It is less reliable when dataset-specific candidate priors or ambiguous activity patterns dominate the forced-choice comparison. In the paper, the corrected cases should be used to illustrate residual ambiguity resolution, while the WISDM regressions should be used to motivate the selective gate and the bounded grounding claim.

| Dataset | Vote5 Wrong -> Gated Right | Gated Wrong -> Vote5 Right | Interpretation |
|---|---:|---:|---|
| UCI HAR | 20 | 1 | selective reranking is mostly beneficial |
| WISDM | 11 | 7 | benefits are offset by regressions |
| MHEALTH | 6 | 0 | gate is conservative and rarely harmful |

## Recommended Claims

The safe main claim is that structured verification provides the counterfactual grounding backbone, while dev-calibrated gated reranking improves residual caption selection. The experiments support this claim across multiple datasets, but with varying strength. UCI HAR and MHEALTH provide the clearest evidence. WISDM should be described as a weaker or boundary case.

The paper should also explicitly report the evidence controls. These controls are not a weakness if framed correctly. They show that language priors are measurable and that the method does not rely on an unverified assumption of pure sensor grounding. This makes the evidence-first framing more credible: the structured verifier supplies the verifiable component, and the reranker is constrained to a narrower selective role.

A suitable concise claim for the Results section is:

> Dev-calibrated forced-choice reranking improves residual caption ambiguity when used selectively on top of order-vote self-consistency, but evidence controls reveal substantial candidate-prior effects. We therefore attribute counterfactual support and rejection to the structured verifier, and treat reranking as a bounded caption-selection module rather than a general grounding mechanism.

## What Should Go in the Main Paper

The main paper should include the dev-calibrated caption table, the fixed-margin paired significance result, and a compact evidence-control table. The Qwen3 cross-model results, position-prior baseline, balanced candidate subset, and detailed failure cases can go to the appendix unless space permits a short robustness subsection. At least one corrected case and one WISDM regression should appear in the qualitative analysis to make the selective nature of the method concrete.

