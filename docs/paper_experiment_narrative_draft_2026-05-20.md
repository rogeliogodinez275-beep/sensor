# 论文实验叙事草稿：Evidence-First Grounding

## 实验主线

本阶段实验建议将论文主线固定为 evidence-first / verifiable counterfactual grounding，而不是围绕 caption accuracy 的小幅提升展开。具体而言，structured verifier 负责可验证的 support 与 counterfactual rejection；caption-order self-consistency 的 order-vote5 作为最稳 caption backbone；confidence-gated forced-choice reranking 只作为 residual ambiguity resolver，用于在高置信边界下修正少量 caption 选择。

这一叙事有两个优点。第一，它避免把 reranker 的 caption 改进错误归因到 CF F1 或 support rejection 上；第二，它把 hidden-evidence、numeric-mask、position-prior 等控制实验转化为 soundness boundary，而不是被动防御。

## 主结果：dev-calibrated caption gate

# Dev-Calibrated Main Caption Table

This table separates the current fixed margin=2 result from the safer dev-calibrated threshold protocol. The fixed-margin result is useful as a locked finding; the dev-calibrated row is the safer candidate for the main paper table if it remains competitive.

| Dataset | Vote5 Acc. | Fixed m=2 Acc. | Fixed Delta | Fixed mid-p | Dev Threshold | Dev Acc. | Dev Coverage | Heldout Acc. | Heldout Coverage | Heldout N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UCI HAR | 0.7361 | 0.7607 | 0.0246 | 0.0000 | 1.5 | 0.7532 | 0.5519 | 0.7706 | 0.5832 | 619 |
| WISDM | 0.7925 | 0.7956 | 0.0031 | 0.3593 | 1.0 | 0.8168 | 0.6081 | 0.8147 | 0.6026 | 1009 |
| MHEALTH | 0.9329 | 0.9512 | 0.0183 | 0.0156 | 1.0 | 1.0000 | 0.9623 | 0.9855 | 0.9564 | 275 |

Interpretation:

- The fixed margin=2 result remains the locked comparison against order-vote5: UCI and MHEALTH are significant; WISDM is small and non-significant.
- The dev-calibrated protocol should be used to defuse threshold-tuning concerns. If heldout accuracy is lower, report it honestly as the more conservative main result and keep fixed margin=2 as an analysis result.
- This table concerns caption ambiguity only. It does not attribute CF F1 or support rejection gains to the reranker.


写作建议：主文优先报告 dev-calibrated protocol，用来规避 margin=2 被认为 test tuning 的风险。fixed margin=2 可以作为 locked analysis result：UCI 与 MHEALTH 显著，WISDM 正向但非显著。若篇幅紧张，主表保留 Vote5、Fixed m=2、Dev-calibrated heldout 三列即可。

## Soundness controls

Qwen2.5-Coder controls:

# Evidence-Control Summary

## Control Table

| Dataset | Condition | Choice Acc | Gated Acc | Gated Coverage |
|---|---|---:|---:|---:|
| UCI HAR | visible | 0.7193 | 0.7607 | 0.3532 |
| UCI HAR | shuffled-evidence | 0.7309 | 0.7426 | 0.1138 |
| UCI HAR | numeric-mask | 0.7063 | 0.7490 | 0.2147 |
| UCI HAR | hidden-evidence | 0.8512 | 0.7413 | 0.0285 |
| WISDM | visible | 0.7683 | 0.7956 | 0.1919 |
| WISDM | shuffled-evidence | 0.7652 | 0.7855 | 0.0920 |
| WISDM | numeric-mask | 0.7582 | 0.7949 | 0.1825 |
| WISDM | hidden-evidence | 0.8853 | 0.8081 | 0.1178 |
| MHEALTH | visible | 0.9939 | 0.9512 | 0.1463 |
| MHEALTH | shuffled-evidence | 0.9207 | 0.9299 | 0.1159 |
| MHEALTH | numeric-mask | 0.9939 | 0.9451 | 0.1707 |
| MHEALTH | hidden-evidence | 0.9817 | 0.9360 | 0.0823 |

## Interpretation Guardrails

- UCI HAR 的 shuffled-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- UCI HAR 的 shuffled-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- UCI HAR 的 numeric-mask choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- UCI HAR 的 numeric-mask gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- UCI HAR 的 hidden-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- UCI HAR 的 hidden-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- WISDM 的 shuffled-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- WISDM 的 shuffled-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- WISDM 的 numeric-mask choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- WISDM 的 numeric-mask gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- WISDM 的 hidden-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- WISDM 的 hidden-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- MHEALTH 的 shuffled-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- MHEALTH 的 numeric-mask choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- MHEALTH 的 numeric-mask gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- MHEALTH 的 hidden-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- MHEALTH 的 hidden-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。

## Paper Use

- visible > control: can support partial evidence dependence.
- control close to visible: write as language-prior / candidate-bias risk, not as grounded understanding.
- CF F1 remains structured-only and should be narrated separately from caption controls.


Qwen3 controls:

# External Evidence-Control Summary

Model tag: `qwen3_4b`

| Dataset | Condition | Choice Acc. | Gated Acc. | No-Gate Acc. | Choice Delta vs Visible | N |
|---|---|---:|---:|---:|---:|---:|
| ucihar | visible | 0.6843 | 0.6856 | 0.6843 | 0.0000 | 773 |
| ucihar | shuffled | 0.5744 | 0.6404 | 0.5744 | -0.1100 | 773 |
| ucihar | numeric-mask | 0.6649 | 0.6611 | 0.6649 | -0.0194 | 773 |
| ucihar | hidden | 0.6455 | 0.6598 | 0.6455 | -0.0388 | 773 |
| wisdm | visible | 0.6708 | 0.6841 | 0.6708 | 0.0000 | 1282 |
| wisdm | shuffled | 0.5538 | 0.5905 | 0.5538 | -0.1170 | 1282 |
| wisdm | numeric-mask | 0.6474 | 0.6498 | 0.6474 | -0.0234 | 1282 |
| wisdm | hidden | 0.7090 | 0.7402 | 0.7090 | 0.0382 | 1282 |
| mhealth | visible | 0.9848 | 0.9848 | 0.9848 | 0.0000 | 328 |
| mhealth | shuffled | 0.6037 | 0.6951 | 0.6037 | -0.3811 | 328 |
| mhealth | numeric-mask | 0.9726 | 0.9787 | 0.9726 | -0.0122 | 328 |
| mhealth | hidden | 0.7348 | 0.7378 | 0.7348 | -0.2500 | 328 |

Interpretation: visible should be read against shuffled, numeric-mask, and hidden controls. Small visible-control gaps indicate language or candidate priors and require a bounded grounding claim.


写作建议：visible 与 shuffled 的差距可以支持模型确实使用了 evidence-to-window mapping；numeric-mask 与 visible 接近、以及 WISDM hidden 高于 visible，则必须写成 language/candidate-prior boundary。推荐主文用一句话概括：evidence controls reveal both evidence dependence and residual language-prior risk。

## Cross-model boundary

# External Cross-Model Reranker Summary

Model tag: `qwen3_4b`

| Dataset | Choice Acc. | Gated Acc. | No-Gate Acc. | Delta vs Vote5 | McNemar mid-p | Gate Count | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| ucihar | 0.6843 | 0.6856 | 0.6843 | -0.0505 | 0.0020 | 770 | 773 |
| wisdm | 0.6708 | 0.6841 | 0.6708 | -0.1084 | 0.0000 | 1248 | 1282 |
| mhealth | 0.9848 | 0.9848 | 0.9848 | 0.0518 | 0.0003 | 327 | 328 |

Interpretation: this table is a cross-model robustness check for the selective reranking claim. It should not be used to attribute CF support or rejection gains to the reranker; those remain owned by the structured verifier.


Qwen3 在 UCI/WISDM 上低于 order-vote5，而在 MHEALTH 上显著更高，说明 forced-choice reranking 不是模型无关的主贡献。论文中应将其定位为 calibrated selective module，而不是 universal reranker。Qwen3 dev threshold 需要 5/10/10，也进一步说明 gate margin 需要 dev calibration。

## Failure taxonomy and qualitative analysis

Aggregate counts from case study:

## Aggregate Counts

| Dataset | Fixes | Regressions |
|---|---:|---:|
| ucihar | 20 | 1 |
| wisdm | 11 | 7 |
| mhealth | 6 | 0 |

Writing note: fixes are best framed as ambiguity-resolution cases; regressions should be used to motivate selective gating and bounded claims.


写作建议：UCI 与 MHEALTH 的 regressions 很少，适合展示 gated reranking 解决 residual ambiguity；WISDM 同时有 fixes 和 regressions，适合作为 boundary case，解释为什么 WISDM 的总体提升非显著。案例分析不要写成 anecdotal proof，而是用来解释收益来源和失败模式。

## 推荐论文措辞

- Safe claim: The structured verifier provides verifiable support and counterfactual rejection, while the gated forced-choice module selectively improves residual caption ambiguity.
- Safe claim: Dev-calibrated gating improves or preserves caption selection on heldout splits across the three datasets, but the magnitude and reliability vary by dataset.
- Boundary claim: Evidence controls show that the reranker is not purely evidence-grounded; candidate language priors and answer-index priors remain measurable and must be bounded.
- Do not claim: The reranker improves CF F1, support F1, or general grounding quality beyond caption selection.

## 下一步写作动作

1. 将 dev-calibrated caption table 放入主文实验表或主文补充表。
2. 将 visible/shuffled/numeric-mask/hidden controls 放入 soundness analysis。
3. 将 Qwen3 cross-model 与 position-prior 放入 appendix 或 reviewer-risk analysis。
4. 从 failure case 文档中挑 2 个 fix、1 个 regression，压缩成论文 qualitative examples。
