# SensorFact 纠偏实验说明：从 Oracle Evidence 到 Raw-Sensor-Only Baseline

## 一句话结论

之前表中接近 `1.0` 的 supervised / aligner 结果不能作为公平主结果，因为它们在 test-time 读取了 structured evidence card 或其中的 numeric summary。我们已补做一个只读 raw sensor window 的 GPU baseline，三库结果回到合理区间，说明数据泄露/上界风险已经被识别并隔离。

## 核心 Idea

SensorFact 的论文主线应写成 **verifiable sensor-language grounding under bounded risk**：

- `structured verifier` 负责可验证 support / counterfactual rejection。
- `order-vote5` 和 confidence-gated reranker 只用于 caption ambiguity resolution。
- evidence controls 用来证明和限制模型是否真的依赖 sensor evidence。
- 新增 `raw_sensor_field_aligner` 作为公平强 baseline：训练时用 train split evidence fields 做监督，测试时只输入 raw sensor window 和 candidate text。

## 泄露审计结论

以下结果只能标为 oracle upper bound / ablation，不能进入公平主表：

- `supervised oracle_fields`
- `supervised axis_drop`
- `supervised numeric_drop`
- `aligner base_full / distill_full / riskcal_full`
- `aligner distill_axis_drop / distill_numeric_drop`

原因是这些路径在 test-time scoring 中读取 `record["evidence"]` 的离散字段，并直接与候选文本做字段匹配。`numeric_only` 虽然不读离散字段，但仍从 eval evidence card 中读取 numeric summary，因此也只能写成 numeric-summary diagnostic，而不是 raw-sensor-only baseline。

## Corrected GPU Result

| Dataset | Caption Acc. | Caption Macro-F1 | CF Reject F1 | Support ECE | Support Brier | Evidence Field Acc. | Device | Train N | Eval N |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| UCI HAR | 0.8286 | 0.8287 | 0.4342 | 0.3762 | 0.2426 | 0.7741 | cuda | 7352 | 2947 |
| WISDM | 0.9098 | 0.9096 | 0.6590 | 0.4372 | 0.2774 | 0.8401 | cuda | 11067 | 3094 |
| MHEALTH | 0.9446 | 0.9445 | 0.6878 | 0.4071 | 0.2447 | 0.8424 | cuda | 4182 | 1047 |

这些结果不再是异常的 `1.0`，更适合作为专家讨论中的公平 baseline。注意：CF Reject F1 在这里来自 raw-sensor field predictor 对候选字段的 compatibility score，不应与 structured verifier 的 CF backbone 混写。

## 写作边界

可以主张：

- structured verifier 提供可验证 support / rejection。
- gated forced-choice reranker 改善 residual caption ambiguity。
- harder controls 和 raw-sensor baseline 暴露并限制 language-prior / oracle-evidence 风险。

不要主张：

- reranker 提升了 support F1 或 CF F1。
- oracle evidence-card 结果是公平 supervised baseline。
- numeric-only evidence-card summary 等同 raw sensor grounding。

## GitHub 轻量包内容

上传内容只包含代码、脚本、测试、summary 和 metrics JSON；不上传 raw rows 或 model weights。完整 rows/model 保留在本地和远程服务器归档。
