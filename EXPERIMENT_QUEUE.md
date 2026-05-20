# EXPERIMENT_QUEUE

## Prioritized Queue

| Priority | Experiment | Hypothesis | Expected GPU Cost | Status | Reason |
| --- | --- | --- | --- | --- | --- |
| P0 | `emnlp_coder_hybrid_prompt_queue` | hybrid 对 direct coder 的 prompt surface 变化保持 caption 增益，同时 support 不掉 | 中 | Running | 直接支撑主方法鲁棒性；当前已完成 UCI `chain_then_json`，正在跑 WISDM |
| P1 | `emnlp_coder_fewshot_queue` | `fewshot_json` 能抬 direct coder，且 hybrid 可进一步吸收这部分增益 | 中 | Approved | 直接测试是否还能拉高主线表现 |
| P2 | `emnlp_coder_fewshot_stress_queue` | few-shot 增益在 caption 顺序、support 顺序、负样本平衡下不脆弱 | 中-高 | Approved | 关键 reviewer 风险控制 |
| P3 | `emnlp_qwen_fewshot_queue` | 若 `fewshot_json` 只是 prompt recipe，则 Qwen 4B 也应显著受益；若增益有限，则更能说明当前主线不是纯 prompt hack | 中 | Approved | 高价值 prompt-only 强对照；已挂在 `fewshot_stress` 之后自动接力 |
| P4 | `Hybrid Threshold Sweep` (CPU) | hybrid 固定阈值 `0.5` 已处于最优平台，无需 adaptive gate | 低 / CPU-only | Finished | 已写入正式报告 |
| P5 | 更强 baseline / external comparison | 需要 reviewer 更熟悉的 baseline 或更锋利的对比 | 未定 | Pending | 仅在 `qwen_fewshot` 之后仍有高价值证据缺口时批准 |

## Rejected / Deferred

| Experiment | Status | Reason |
| --- | --- | --- |
| Adaptive hybrid gate GPU run | Rejected for now | threshold sweep 显示 `0.05~1.0` 基本同平台，继续烧 GPU 价值低 |
| 随机新 prompt / 随机大模型下载 | Rejected for now | 缺少明确假设，且网络下载不稳定，容易浪费 GPU 时间 |
| 继续扩展 direct structured generation | Rejected for now | 已经是足够强的负例，再烧卡很难转化为更强 paper claim |

## Queue Discipline

1. 当前 GPU 只跑已经验证不会空转、且与 paper claims 直接相关的队列。
2. 下一个任务必须在当前任务结束前准备好并挂好 relay 或 heartbeat 接力。
3. 不重复跑无价值重复实验；重复只在 seed / variance 有明确论文价值时进行。


<!-- update 2026-05-20T11:09:40 -->

## 2026-05-20 Confidence-Gated Forced-Choice Reranking

Decision: promote to active candidate method, pending paired/bootstrap confirmation. The method keeps support predictions on the structured verifier and improves only ambiguous caption fallback by using a forced-choice next-token log-prob reranker. A fixed conservative gate uses the alternate forced-choice result only when its top-2 log-prob margin is > 2.0.

Main comparison against constrained order-vote5 hybrid:

| Dataset | Vote5 caption acc | Gated forced-choice caption acc | Delta | Gated fallback acc | CF F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCI HAR hard v3 | 0.8565 | 0.8629 | +0.0064 | 0.7607 | 0.7635 |
| WISDM hard v3 | 0.8859 | 0.8872 | +0.0013 | 0.7956 | 0.6918 |
| MHEALTH hard v3 | 0.9351 | 0.9408 | +0.0057 | 0.9512 | 0.7613 |

Interpretation: stronger than vote7/prompt variants because it is a different scoring mechanism, not just candidate-order self-consistency. It gives consistent positive caption gains on all three datasets while preserving structured-only support/CF F1.

Risk: threshold selection must be presented carefully. Threshold 2.0 is a single conservative rule, while margin sweep shows higher dataset-specific thresholds. Need paired/bootstrap significance and a no-label calibration or train/dev threshold variant before making the final main-table claim.


## Follow-up Evidence Added

- Paired bootstrap vs vote5: UCI delta +0.00645, 95% CI [0.00373, 0.00984]; MHEALTH delta +0.00573, 95% CI [0.00191, 0.01051]; WISDM delta +0.00129, 95% CI [-0.00129, 0.00388]. Interpretation: two datasets significantly improve, WISDM is a non-significant small gain.
- Full-candidate forced-choice baseline is much weaker than constrained/gated hybrid: UCI 0.6725, WISDM 0.6949, MHEALTH 0.7383. This supports the selective ambiguous-window framing rather than a generic direct LLM baseline claim.
- Remote safety tests passed: 	ests/test_logprob_reranker.py and 	ests/test_gate_caption_rows.py have 8 passing tests.
