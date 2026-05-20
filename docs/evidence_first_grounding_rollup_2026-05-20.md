# Evidence-First Grounding 实验进展总览

生成时间：2026-05-20 14:20 左右。

## 1. 当前安全主张

- 主线应写成 verifiable counterfactual grounding，而不是单纯 caption accuracy 小幅提升。
- structured verifier 负责 support / counterfactual rejection；caption reranker 只解决 residual caption ambiguity。
- Qwen2.5-Coder 的 gated forced-choice 相对 order-vote5 在 UCI 与 MHEALTH 显著提升，WISDM 为小幅非显著正向。
- Qwen3 外部模型结果显示 reranker 和 gate calibration 不具备模型无关稳定性，因此论文中必须把 evidence controls 和 calibration boundary 写清楚。

## 2. Qwen2.5-Coder 主结果锁定

| Dataset | Vote5 Acc. | Gated Acc. | Delta | McNemar mid-p | 结论 |
|---|---:|---:|---:|---:|---|
| ucihar | 0.7361 | 0.7607 | 0.0246 | 0.0000 | 显著正向 |
| wisdm | 0.7925 | 0.7956 | 0.0031 | 0.3593 | 非显著/需谨慎 |
| mhealth | 0.9329 | 0.9512 | 0.0183 | 0.0156 | 显著正向 |

## 3. Qwen3 Cross-Model 边界


| Dataset | Choice Acc. | Gated Acc. | No-Gate Acc. | Delta vs Vote5 | McNemar mid-p | Gate Count | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| ucihar | 0.6843 | 0.6856 | 0.6843 | -0.0505 | 0.0020 | 770 | 773 |
| wisdm | 0.6708 | 0.6841 | 0.6708 | -0.1084 | 0.0000 | 1248 | 1282 |
| mhealth | 0.9848 | 0.9848 | 0.9848 | 0.0518 | 0.0003 | 327 | 328 |

Interpretation: this table is a cross-model robustness check for the selective reranking claim. It should not be used to attribute CF support or rejection gains to the reranker; those remain owned by the structured verifier.


解释：Qwen3 在 UCI/WISDM 明显低于 vote5，而 MHEALTH 显著更高。这说明 forced-choice reranking 不是可直接跨模型迁移的主贡献，最多作为 calibrated residual resolver。

## 4. Evidence-Control 风险

### Qwen3 controls


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


关键解读：shuffled evidence 通常显著下降，说明 evidence order/mapping 有贡献；但 numeric-mask 与 visible 很接近，WISDM hidden 高于 visible，说明候选语言先验仍强，grounding claim 必须 bounded。

## 5. Calibration 与效率

| Dataset | Best Dev Threshold | Dev Acc. | Heldout Acc. | Heldout Coverage |
|---|---:|---:|---:|---:|
| ucihar | 5.0 | 0.6948 | 0.6850 | 0.9822 |
| wisdm | 10.0 | 0.7143 | 0.7304 | 0.8741 |
| mhealth | 10.0 | 1.0000 | 0.9636 | 0.8145 |

Interpretation: this checks whether the fixed margin=2 gate transfers to the external Qwen3 reranker. Large threshold shifts or low heldout accuracy indicate model-specific calibration risk.


## Reranker Cost Proxy

| Dataset | N | Full Choice Prompts | Override Prompts | Gate Coverage |
|---|---:|---:|---:|---:|
| UCI HAR | 773.0000 | 773.0000 | 273.0000 | 0.3532 |
| WISDM | 1282.0000 | 1282.0000 | 246.0000 | 0.1919 |
| MHEALTH | 328.0000 | 328.0000 | 48.0000 | 0.1463 |

## Interpretation

The full forced-choice scorer requires one prompt per evaluated window. The deployed gated method should be discussed by override coverage: only rows above the margin threshold change the vote5 decision, even though the current offline analysis scores all rows to estimate the gate.


Qwen3 需要 5/10/10 的 dev threshold，进一步证明 margin=2 不能写成通用阈值；主文应使用 dev-calibrated protocol 或把 margin=2 标注为主模型上的 conservative gate。

## 6. Balanced Candidate Subset

- 已生成 docs/balanced_candidate_subset_ucihar.md、docs/balanced_candidate_subset_wisdm.md、docs/balanced_candidate_subset_mhealth.md。
- 该分析把候选长度和 evidence-overlap 偏差最低的 50% 样本单独评估；适合作为附录/风险分析。
- 若 hidden/position-prior 在 balanced subset 仍高，必须正面讨论 candidate prior，而不是把全部收益解释为 sensor evidence grounding。

## 7. 写作建议

- 主表：放 Qwen2.5 order-vote5 vs gated + paired significance + CI。
- 控制表：放 visible/shuffled/numeric-mask/hidden，突出 evidence dependence 与 language-prior boundary。
- 附录：放 Qwen3 cross-model、position-prior、balanced subset、per-fact breakdown、efficiency。
- 方法叙事：structured support backbone 与 caption reranking 分开；不要声称 reranker 提升 CF F1。
- 局限：threshold calibration、hidden-evidence 高分、numeric-mask 接近 visible、Qwen3 UCI/WISDM 负迁移。

## 8. 下一步

1. 用 dev-calibrated threshold 重新确定最终主表阈值，避免 test tuning 质疑。
2. 把 failure taxonomy 中 vote5 wrong -> gated right 的典型样本转成论文 case study。
3. 若还有时间，补 approximate randomization 或更正式的 paired bootstrap CI 表。
4. 开始按 Evidence-First Grounding 叙事写实验 section，不再追 vote9/prompt variant。
