# Risk-Bounded Harder Evidence Controls 实验总结

- 生成时间：2026-05-21 00:00 CST
- 远程项目路径：`/root/autodl-tmp/emnlp2026`
- 队列日志：`outputs/queue_logs/emnlp_harder_controls_status.tsv`
- 本轮状态：harder-controls 队列已完成，`failures=0`
- GPU 状态：RTX 4090 当前空闲，说明本轮 harder-control LLM 推理已经结束

## 1. 实验目的

本轮实验用于支撑 SensorFact 的 **risk-bounded benchmark / verifiable sensor-language grounding** 叙事。目标不是继续追逐小幅 caption accuracy，而是检验 confidence-gated forced-choice reranker 是否真正依赖可验证 sensor evidence，并明确其 grounding 边界。

本轮新增三个 harder evidence controls：

- `numeric_swap`：扰动数值统计量，检查模型是否依赖原始数值幅度。
- `axis_permutation`：置换传感器轴或通道，检查模型是否依赖轴向/通道语义。
- `trend_flip`：翻转趋势描述，检查趋势字段是否实质影响决策。

论文主张应保持保守：structured verifier 负责 support 与 counterfactual rejection；`order-vote5` 是 caption backbone；confidence-gated forced-choice 只作为 residual ambiguity resolver；evidence controls 用于证明并限制 grounding claim。

## 2. 主结果表

| Dataset | Control | Forced-choice Acc | Gated Acc | No-gate Acc | Gate overrides | N |
|---|---:|---:|---:|---:|---:|---:|
| UCI HAR | `numeric_swap` | 0.7141 | 0.7594 | 0.7141 | 265 | 773 |
| UCI HAR | `axis_permutation` | 0.4825 | 0.5110 | 0.4825 | 278 | 773 |
| UCI HAR | `trend_flip` | 0.7219 | 0.7581 | 0.7219 | 272 | 773 |
| WISDM | `numeric_swap` | 0.7683 | 0.7972 | 0.7683 | 250 | 1282 |
| WISDM | `axis_permutation` | 0.4836 | 0.5523 | 0.4836 | 467 | 1282 |
| WISDM | `trend_flip` | 0.7707 | 0.7956 | 0.7707 | 247 | 1282 |
| MHEALTH | `numeric_swap` | 0.9939 | 0.9543 | 0.9939 | 70 | 328 |
| MHEALTH | `axis_permutation` | 0.5396 | 0.5793 | 0.5396 | 123 | 328 |
| MHEALTH | `trend_flip` | 0.9939 | 0.9543 | 0.9939 | 73 | 328 |

## 3. 结果解读

### 3.1 Axis permutation 是最强的 evidence-dependence 信号

`axis_permutation` 在三个数据集上都显著破坏 forced-choice 表现：UCI HAR 降至 0.4825，WISDM 降至 0.4836，MHEALTH 降至 0.5396。gated 版本因为保留 vote5 backbone，性能有所恢复，但仍明显低于 numeric/trend control 条件。

这说明 reranker 并非完全依赖候选文本先验；至少在轴向/通道级 evidence 被破坏时，模型决策会明显受损。这是当前最强的 grounding-positive 证据。

### 3.2 Numeric swap 与 trend flip 暴露 grounding 边界

`numeric_swap` 与 `trend_flip` 对 UCI/WISDM 的影响较弱，对 MHEALTH 几乎不降低 forced-choice accuracy。特别是 MHEALTH 在 `numeric_swap` 与 `trend_flip` 下仍达到 0.9939，说明模型很可能更多依赖活动标签、轴向结构或候选模板，而不是稳定读取细粒度数值幅度与趋势字段。

因此，当前结果支持 **bounded grounding**，但不足以支持“模型充分理解 sensor 数值证据”的强 claim。这个风险应进入论文主文或 limitations。

### 3.3 Gate 的作用是风险约束，而不是万能提升器

在 UCI/WISDM 上，gated 通常高于 no-gate/forced-choice，说明 selective intervention 能减少直接覆盖 vote5 的错误传播。但在 MHEALTH 的 `numeric_swap` 和 `trend_flip` 下，forced-choice/no-gate 为 0.9939，而 gated 为 0.9543，说明固定 `margin=2.0` 不是普适最优阈值。

后续论文主表应优先使用 dev-calibrated threshold；`margin=2.0` 只能作为 locked analysis 或 conservative gate 设置，而不应被写成最终最优策略。

## 4. 对论文主张的影响

可以支持的表述：

- SensorFact 提供了一个能显式暴露 evidence dependence 与 language-prior risk 的 benchmark/control framework。
- Axis/channel evidence 对当前 reranker 决策有明显影响。
- Confidence-gated forced-choice 可以作为 residual caption ambiguity resolver，但其作用应被风险控制实验约束。

需要避免的表述：

- 不应声称 reranker 提升了 support F1 或 counterfactual rejection F1。
- 不应声称 reranker 已充分理解所有 sensor 数值证据。
- 不应把 `numeric_swap` / `trend_flip` 的高分解释为 grounding 成功；它们更像是 language-prior 或模板依赖风险信号。

## 5. 当前完成状态

| 项目 | 状态 | 说明 |
|---|---|---|
| Harder controls queue | 完成 | `failures=0` |
| Metrics files | 完成 | 3 datasets × 3 controls × 3 eval modes = 27 files，全齐 |
| Control JSONL | 完成 | 9 个 constrained harder-control 文件 |
| Dev-calibrated threshold | 完成 | UCI HAR best=1.5；WISDM best=1.0；MHEALTH best=1.0 |
| Dev-calibrated gated rows | 完成 | 三个数据集均已生成逐行 rows 与 metrics |
| Margin curve | 完成 | 三个数据集均已生成 threshold curve 与 margin bucket CSV |
| McNemar significance | 完成 | UCI HAR 与 MHEALTH 显著；WISDM 不显著 |
| Dev-calibrated failure taxonomy | 完成 | UCI 修正30/回退6；WISDM 修正49/回退20；MHEALTH 修正18/回退0 |
| Qwen3 harder-control confirmation | 完成 | 9/9 指标齐全，`missing=0`，队列 `failures=0` |
| Annotation subset | 完成 | 3 个 CSV 子集 |
| Lightweight alignment smoke | 完成 | 仅作接口/流程 smoke，不作为主结论 |
| GPU | 空闲 | 当前没有未完成的 harder-controls GPU 任务 |

## 6. Dev Calibration 与显著性补充

### 6.1 Dev-calibrated threshold

| Dataset | Dev-selected threshold | Dev accuracy | Dev coverage |
|---|---:|---:|---:|
| UCI HAR | 1.5 | 0.7532 | 0.5519 |
| WISDM | 1.0 | 0.8168 | 0.6081 |
| MHEALTH | 1.0 | 1.0000 | 0.9623 |

这些结果说明固定 `margin=2.0` 不应作为最终主协议。论文主结果应优先切到 dev-calibrated threshold，并在方法中明确：阈值只在 dev split 上选择，test split 只评估一次。

### 6.2 Paired McNemar significance：vote5 vs gated margin2

| Dataset | Vote5 Acc | Gated Acc | Delta | McNemar mid-p | 结论 |
|---|---:|---:|---:|---:|---|
| UCI HAR | 0.7361 | 0.7607 | +0.0246 | 1.10e-05 | 显著 |
| WISDM | 0.7925 | 0.7956 | +0.0031 | 0.3593 | 不显著 |
| MHEALTH | 0.9329 | 0.9512 | +0.0183 | 0.0156 | 显著 |

该结果与当前 safe claim 一致：gated forced-choice 在 UCI HAR 和 MHEALTH 上有统计支持；WISDM 只有小幅提升，当前证据不足以声称显著改进。

### 6.3 Dev-calibrated gated rows：更适合作为主结果候选

| Dataset | Threshold | Vote5 Acc | Dev-calibrated Gated Acc | Delta | McNemar mid-p | Gate overrides |
|---|---:|---:|---:|---:|---:|---:|
| UCI HAR | 1.5 | 0.7361 | 0.7671 | +0.0310 | 4.13e-05 | 446 |
| WISDM | 1.0 | 0.7925 | 0.8151 | +0.0226 | 4.40e-04 | 774 |
| MHEALTH | 1.0 | 0.9329 | 0.9878 | +0.0549 | 3.81e-06 | 314 |

这组结果比固定 `margin=2.0` 更适合作为论文主结果候选，因为阈值来自 dev split，而不是直接用 test 表现解释。需要注意的是，当前 dev-calibrated rows 是在完整评估集上按 dev 选出的阈值重新生成的逐行结果；最终论文应进一步明确 dev/test split protocol，避免被质疑为同一集合内的后验选择。

### 6.4 Failure taxonomy：收益来源与回退风险

| Dataset | Corrected | Regressed | Unchanged correct | Unchanged wrong |
|---|---:|---:|---:|---:|
| UCI HAR | 30 | 6 | 563 | 174 |
| WISDM | 49 | 20 | 996 | 217 |
| MHEALTH | 18 | 0 | 306 | 4 |

这说明 dev-calibrated gate 的收益不是平均噪声：三个数据集均有净修正，尤其 MHEALTH 没有观察到回退。但 WISDM 的回退数较高，论文需要用 case study 解释哪些活动或候选类型容易被 reranker 错误覆盖。

## 7. Cross-model Confirmation：Qwen3 Harder Controls

| Dataset | Control | Qwen3 Forced-choice Acc | Qwen3 Gated Acc | Qwen3 No-gate Acc | Gate overrides | N |
|---|---:|---:|---:|---:|---:|---:|
| UCI HAR | `numeric_swap` | 0.6831 | 0.6831 | 0.6831 | 765 | 773 |
| UCI HAR | `axis_permutation` | 0.3842 | 0.4062 | 0.3842 | 704 | 773 |
| UCI HAR | `trend_flip` | 0.6856 | 0.6856 | 0.6856 | 769 | 773 |
| WISDM | `numeric_swap` | 0.6654 | 0.6708 | 0.6654 | 1256 | 1282 |
| WISDM | `axis_permutation` | 0.3565 | 0.4275 | 0.3565 | 1094 | 1282 |
| WISDM | `trend_flip` | 0.6685 | 0.6802 | 0.6685 | 1248 | 1282 |
| MHEALTH | `numeric_swap` | 0.9756 | 0.9848 | 0.9756 | 314 | 328 |
| MHEALTH | `axis_permutation` | 0.3415 | 0.3811 | 0.3415 | 278 | 328 |
| MHEALTH | `trend_flip` | 0.9817 | 0.9848 | 0.9817 | 327 | 328 |

Qwen3 复核加强了核心发现：`axis_permutation` 在三个数据集上都是破坏最大的 control，说明 axis/channel evidence dependence 不是单一模型偶然现象。与此同时，`numeric_swap` 与 `trend_flip` 仍未造成同等破坏，尤其 MHEALTH 仍接近满分，因此论文应写成“axis/channel-grounded but numerically risk-bounded”，而不是泛化为 fully sensor-grounded reasoning。

## 8. 下一步建议

### Priority 1：结果锁定与文档同步

把本轮结果写入 discussion package 与 GitHub summary，固定 bounded-grounding 叙事。主文应该把 `axis_permutation` 作为 evidence-dependence 的正面证据，同时把 `numeric_swap` 与 `trend_flip` 作为 grounding boundary/control finding。

### Priority 2：Dev-calibrated test rows 与主表回填

当前已经产出 dev threshold、dev-calibrated gated rows 与 paired significance。下一步应把这组结果回填进主表，并补 failure taxonomy 与 efficiency analysis，用于解释收益来源与成本。

### Priority 3：Cross-model harder-controls confirmation

如果继续使用 GPU，最有价值的下一步是用 Qwen3 或另一个强模型复核 `numeric_swap`、`axis_permutation`、`trend_flip`，检查 evidence dependence 是否跨模型稳定。当前不建议继续跑随机 prompt variants、vote9 或盲目换 prompt。
