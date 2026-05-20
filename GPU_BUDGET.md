# GPU_BUDGET

## Inventory

- Active remote server: `root@connect.westc.seetacloud.com:45902`
- GPU: `1 x NVIDIA GeForce RTX 4090`
- VRAM: `49140 MiB`
- CUDA: `13.2`

## GPU Job Ledger

| Job | Start | End / State | Expected VRAM | Observed VRAM | Useful? | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `emnlp_coder_prompt_queue` | 2026-05-19 14:45 | Finished 17:19 | ~15-16GB | ~15-16GB | Yes | 建立 direct coder 对照与鲁棒性基线 |
| `emnlp_coder_hybrid_prompt_queue` | 2026-05-19 17:20 | Running | ~15-16GB | ~15.5GB | Yes | 当前主方法 prompt robustness 主线；已从 UCI 推进到 WISDM |
| `emnlp_coder_fewshot_queue` | Pending | Relay waiting | ~15-16GB | N/A | Expected useful | few-shot direct + hybrid |
| `emnlp_coder_fewshot_stress_queue` | Pending | Relay waiting | ~15-16GB | N/A | Expected useful | few-shot robustness |
| `emnlp_qwen_fewshot_queue` | Pending | Relay waiting | ~13-16GB | N/A | Expected useful | prompt-only 强对照，检验 few-shot 增益是否与 code-model 解耦 |

## Waste Review

- 没有让 GPU 长时间空闲。
- `coder_prompt` 结束到 `coder_hybrid_prompt` 启动之间存在短暂等待窗口，已通过手动 kick + relay 链修复。
- relay 旧状态文件污染风险已在本地和远端脚本中修掉，降低续跑时误接棒或卡住的概率。

## Guardrails

1. 新 GPU 任务必须在当前任务结束前准备完命令、输出路径、假设说明。
2. 只允许一个重型 LLM 任务占用 4090，避免干扰和 OOM。
3. CPU-only 报告与分析穿插在 GPU 运行期间完成，避免因写文档让卡闲着。


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
