# 专家讨论文档：Evidence-First Grounding for Wearable Sensor-Language Understanding

生成日期：2026-05-20

## 1. 研究定位

本项目研究 wearable sensor-language understanding 中的可验证 grounding 问题：给定可穿戴传感器窗口及其结构化统计证据，系统不仅需要选择正确的自然语言 activity/caption，还需要对支持证据与反事实拒绝给出可核查判断。当前论文主线建议从“小幅提升 caption accuracy”升级为 **evidence-first / verifiable counterfactual grounding**。

该定位的核心价值在于：传感器-语言任务中的语言输出容易被候选文本先验、类别共现和模板偏差影响。仅报告 caption selection accuracy 难以证明模型真正使用了 sensor evidence。因此，本研究将 grounding soundness 作为主问题，通过结构化 verifier 与 evidence-control 实验共同限定模型能力边界。

## 2. 方法框架

当前系统由三个互相分工的部分组成。

| 组件 | 作用 | 当前论文中应承担的 claim |
|---|---|---|
| Structured verifier | 基于结构化 sensor evidence 做 support 判断与 counterfactual rejection | 负责 verifiable support / CF rejection 的主贡献 |
| Caption order-vote5 | 对 caption-order self-consistency 进行投票聚合 | 当前最稳的 caption backbone |
| Confidence-gated forced-choice reranker | 只在高置信 margin 下覆盖 vote5 的 caption 选择 | residual caption ambiguity resolver |
| Evidence controls | visible / shuffled / numeric-mask / hidden 条件对照 | 证明或限制 evidence-dependence claim |

必须强调：reranker 的改进只对应 caption ambiguity，不应写成 CF F1 或 support F1 的提升来源。CF F1 与 support/rejection 叙事应保留给 structured-only backbone。

## 3. 当前安全主张

1. Structured verifier 提供可验证 support 与 counterfactual rejection，是 grounding 论文主线的支撑模块。
2. order-vote5 是目前最稳的 caption baseline。vote7 平均可能略涨，但不稳定；contrastive 与 axis-ledger 方向已判负，不宜继续作为主线。
3. Qwen2.5-Coder 的 confidence-gated forced-choice reranking 相比 order-vote5 在 UCI HAR 和 MHEALTH 上显著提升，在 WISDM 上仅小幅正向且非显著。
4. Evidence-control 结果显示 reranker 部分使用 evidence mapping，但候选语言先验仍然很强，因此 grounding claim 必须 bounded，而不能夸大为“模型可靠理解所有 sensor evidence”。
5. Qwen3 cross-model 结果暴露 calibration/model-specific 风险：reranking 不是稳定跨模型迁移的主贡献，更适合作为选择性 ambiguity resolver。

## 4. 主实验结果

### 4.1 固定 margin=2 的锁定结果

| Dataset | Vote5 Acc. | Gated Acc. | Delta | McNemar mid-p | 结论 |
|---|---:|---:|---:|---:|---|
| UCI HAR | 0.7361 | 0.7607 | +0.0246 | 0.000011 | 显著正向 |
| WISDM | 0.7925 | 0.7956 | +0.0031 | 0.3593 | 正向但非显著 |
| MHEALTH | 0.9329 | 0.9512 | +0.0183 | 0.0156 | 显著正向 |

解释：该表可以作为 locked analysis result。它说明 selective reranking 确实能修正一部分 residual ambiguity，但 WISDM 的结果不足以支撑稳定普适提升。

### 4.2 Dev-calibrated threshold 协议

| Dataset | Dev Threshold | Dev Acc. | Dev Coverage | Heldout Acc. | Heldout Coverage | Heldout N |
|---|---:|---:|---:|---:|---:|---:|
| UCI HAR | 1.5 | 0.7532 | 0.5519 | 0.7706 | 0.5832 | 619 |
| WISDM | 1.0 | 0.8168 | 0.6081 | 0.8147 | 0.6026 | 1009 |
| MHEALTH | 1.0 | 1.0000 | 0.9623 | 0.9855 | 0.9564 | 275 |

解释：该协议用于回应 margin=2 可能被质疑为 test tuning 的问题。论文主表更适合优先报告 dev-calibrated heldout 结果，固定 margin=2 结果放入分析或附录。

## 5. Soundness Controls

### 5.1 Qwen2.5-Coder evidence controls

| Dataset | Condition | Choice Acc. | Gated Acc. | Gated Coverage |
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

结论：MHEALTH 的 shuffled-evidence 明显下降，支持 evidence mapping 的作用；但 UCI/WISDM 中 hidden-evidence 和 numeric-mask 结果偏高，说明候选语言先验很强。这个结果不应被隐藏，而应作为论文中的 reviewer-risk finding：本方法提供可验证边界，而不是声称 reranker 完全依赖数值传感器证据。

### 5.2 Qwen3 external controls

| Dataset | Visible Choice | Shuffled Choice | Numeric-mask Choice | Hidden Choice | 主要解读 |
|---|---:|---:|---:|---:|---|
| UCI HAR | 0.6843 | 0.5744 | 0.6649 | 0.6455 | shuffled 明显下降，但 numeric-mask 接近 visible |
| WISDM | 0.6708 | 0.5538 | 0.6474 | 0.7090 | hidden 高于 visible，语言先验风险突出 |
| MHEALTH | 0.9848 | 0.6037 | 0.9726 | 0.7348 | evidence mapping 很关键，但 numeric-mask 仍较高 |

结论：Qwen3 controls 更清楚地显示 shuffled evidence 会降低性能，说明 evidence-window mapping 并非完全无关；但 numeric-mask 接近 visible 与 WISDM hidden 高分意味着数值 evidence 的独立贡献仍需谨慎表述。

## 6. Cross-Model Boundary

| Dataset | Qwen3 Choice Acc. | Qwen3 Gated Acc. | Delta vs Vote5 | McNemar mid-p | 结论 |
|---|---:|---:|---:|---:|---|
| UCI HAR | 0.6843 | 0.6856 | -0.0505 | 0.0020 | 显著低于 vote5 |
| WISDM | 0.6708 | 0.6841 | -0.1084 | 0.0000 | 显著低于 vote5 |
| MHEALTH | 0.9848 | 0.9848 | +0.0518 | 0.0003 | 显著高于 vote5 |

解释：该结果应作为边界分析，而非主结果。它说明 forced-choice reranking 的有效性依赖模型和阈值校准，不能被写成模型无关的通用提升机制。

## 7. 效率与选择性调用

| Dataset | N | Full Choice Prompts | Effective Override Prompts | Gate Coverage |
|---|---:|---:|---:|---:|
| UCI HAR | 773 | 773 | 273 | 0.3532 |
| WISDM | 1282 | 1282 | 246 | 0.1919 |
| MHEALTH | 328 | 328 | 48 | 0.1463 |

解释：实际覆盖率较低，说明 gate 并非对所有样本盲目覆盖，而是在少数高置信样本上选择性干预。这可以回应“小收益是否值得成本”的质疑，但仍需要在论文中说明 forced-choice inference 的额外计算开销。

## 8. Failure Taxonomy 初步结论

| Dataset | Vote5 wrong -> Gated right | Gated wrong -> Vote5 right | 解读 |
|---|---:|---:|---|
| UCI HAR | 20 | 1 | gate 主要修正 vote5 错误，风险较低 |
| WISDM | 11 | 7 | 收益和回退并存，解释了非显著结果 |
| MHEALTH | 6 | 0 | gate 在该数据集上几乎只带来修正 |

该分析适合放入 results analysis，用来解释为什么 UCI/MHEALTH 更稳定，而 WISDM 需要谨慎。

## 9. 论文主结果建议

主文建议放入：

1. Structured verifier 的 support / CF rejection 主结果。
2. Dev-calibrated caption gate 主表。
3. 固定 margin=2 的 locked comparison，作为与 order-vote5 的直接 paired 对比。
4. Evidence-control 表，尤其是 visible vs shuffled / numeric-mask / hidden。
5. Failure taxonomy 的简表或代表案例。

附录建议放入：

1. Qwen3 cross-model boundary。
2. Margin sweep 与 coverage-accuracy curve。
3. Per-class / per-fact breakdown。
4. Position-prior 与 balanced candidate subset 结果。
5. 完整脚本和结果 JSON 路径说明。

## 10. 当前不足与专家讨论问题

1. **语言先验风险仍然明显。** Hidden-evidence 在 Qwen2.5-Coder 上偏高，WISDM 在 Qwen3 hidden 条件下甚至高于 visible。这会被审稿人追问“模型是否真的看 sensor evidence”。
2. **numeric evidence 的独立贡献不足够强。** Numeric-mask 与 visible 接近，说明模板/标签文本可能贡献很大。需要讨论是否加入更强的 numeric perturbation 或 counterfactual numeric swap。
3. **cross-model 稳定性不足。** Qwen3 在 UCI/WISDM 上明显弱于 vote5，说明 reranker 不能作为主贡献，只能作为 calibrated auxiliary resolver。
4. **WISDM 改进不显著。** 当前证据不足以支持三数据集稳定提升，只能写“两显著一非显著”。
5. **外部 baseline 仍需进一步选择。** 若投稿 EMNLP，需要找到最接近 sensor-language / time-series LLM 的公平 baseline，避免“只和自己比”的质疑。

## 11. Recommended Next Steps

Priority 1：

- 将 paper claim 固定为 evidence-first grounding，避免把 reranker 写成 CF F1 提升来源。
- 在主文中正面报告 hidden/numeric-mask 风险，并将其解释为 bounded grounding finding。
- 补充或整理 structured verifier 的 support/CF rejection 主表，使 caption 改进与 CF F1 叙事分离。

Priority 2：

- 设计 numeric perturbation / sensor-stat swap control，用于更强地验证数值 evidence 是否被使用。
- 选择一个外部 time-series / sensor-language LLM baseline，并统一 closed-candidate candidate space。
- 把 failure taxonomy 扩展成 4-6 类错误类型及代表案例。

Priority 3：

- 加入 per-class / hardness breakdown，解释 WISDM 失效类别。
- 汇总 efficiency/cost，更清楚地呈现 gate 覆盖率、调用成本和收益的权衡。

## 12. 仓库文件索引

| 路径 | 内容 |
|---|---|
| sensorfact/ | 核心库：benchmark、evidence、metrics、structured verifier、reranker |
| scripts/ | 数据构造、reranking、gate calibration、evidence controls、significance、failure taxonomy |
| tests/ | 对关键脚本和数据映射逻辑的单元测试 |
| docs/evidence_first_grounding_rollup_2026-05-20.md | 当前实验总览与安全主张 |
| docs/dev_calibrated_main_caption_table_2026-05-20.md | dev-calibrated 主结果表 |
| docs/evidence_control_summary_2026-05-20.md | evidence-control 结果 |
| docs/failure_taxonomy_case_studies_2026-05-20.md | failure taxonomy 与案例 |
| outputs/ | 轻量 summary JSON/CSV，不含完整 row-level outputs |
