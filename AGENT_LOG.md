# AGENT_LOG

## Mission

目标：把 `SensorFact` 推到 EMNLP 2026 main 投稿强度，在远端 `/root/autodl-tmp/emnlp2026` 保持高价值 GPU 利用率，同时持续补齐对比、消融、鲁棒性分析和 paper claim 证据链。

## Audit Plan

1. 仓库与环境审计：确认 repo 结构、入口脚本、Python 环境、GPU、数据与输出路径。
2. 正确性门槛：确认 Stage 0/1/2 已完成，有测试、队列日志、最小 GPU 任务与输出作为证据。
3. 主结果主线：围绕 `structured verifier + direct fallback` 推进，优先保证主方法、baseline、关键消融和 reviewer 风险分析并行开展。
4. 运行管理：GPU 跑当前队列时，其他工位并行做代码审计、实验排队、方法批判和 paper claim 补强。

## Multi-Agent Split

| Role | Current Owner | Current Focus | Status |
| --- | --- | --- | --- |
| Project Manager / Coordinator | 主线程 | 全局任务板、队列接力、管理文件、自动化汇报 | Running |
| Code Engineer | 主线程 | 队列脚本、relay、few-shot prompt、报告与聚合逻辑 | Running |
| Experiment Manager | 主线程 | 维护 `coder_hybrid_prompt -> coder_fewshot -> coder_fewshot_stress -> qwen_fewshot` 队列链 | Running |
| GPU Scheduler / Cost Controller | 主线程 + automation | 监控 `nvidia-smi`、检查空闲、挂好下一棒 | Running |
| Research Critic | 并行工位 `Hume` | 审查主线强弱、负例价值、缺失对比/分析 | Running |
| Novelty & Strongness Reviewer | 并行工位 `Hume` | 审视 novelty / EMNLP main 风险 | Running |
| Paper Strategy Agent | 主线程 + `Hume` | paper claim、证据缺口、reviewer attack 面 | Running |
| Repo / Env Auditor | 并行工位 `Carver` | Stage 0/1/2 审计与隐性故障点 | Running |

## Active GPU Strategy

- 当前策略：GPU 不因写文档、思考、代码审计而空闲。
- 当前链路：
  1. `run_emnlp_coder_hybrid_prompt_queue.sh`: Running
  2. `run_emnlp_coder_fewshot_queue.sh`: Approved + relay 已挂
  3. `run_emnlp_coder_fewshot_stress_queue.sh`: Approved + relay 已挂
  4. `run_emnlp_qwen_fewshot_queue.sh`: Approved + relay 已挂，作为 prompt-only 强对照
- 自动汇报：`nash-30min-mhealth-progress` 已更新到当前队列链，30 分钟中文汇报。

## Pre-Active Checklist

| Gate | Evidence | Status |
| --- | --- | --- |
| Repo entry points identified | `README.md`, `scripts/run_emnlp_*`, `sensorfact/*` | Pass |
| CPU / import / unit checks | 本地与远端相关回归通过 | Pass |
| Minimal GPU smoke / real eval path | 多条真实队列已跑通，当前 `run_qwen_llm_eval.py` 正常占用 GPU | Pass |
| Logging / outputs / queue status | `outputs/queue_logs/*.tsv`, `launcher.log`, `outputs/emnlp_experiment_report.md` | Pass |
| Next job ready before current job ends | relay 已挂到 `qwen_fewshot` | Pass |

## Completed This Cycle

1. 新增并验证 `fewshot_json` direct coder prompt。
2. 新增并验证 `run_emnlp_coder_fewshot_queue.sh`。
3. 新增并验证 `relay_after_coder_hybrid_prompt.sh`。
4. 新增并验证 `run_emnlp_coder_fewshot_stress_queue.sh`。
5. 新增并验证 `relay_after_coder_fewshot.sh`。
6. 新增并验证 `scripts/sweep_hybrid_threshold.py`。
7. `Hybrid Threshold Sweep` 已写入远端正式报告。
8. 修复 `relay_after_coder_hybrid_prompt.sh` 与 `relay_after_coder_fewshot.sh` 的旧状态文件污染风险。
9. 新增并验证 `run_emnlp_qwen_fewshot_queue.sh`。
10. 新增并验证 `relay_after_coder_fewshot_stress.sh`。
11. 将新的 `qwen_fewshot` 队列和 relay 同步到远端，并在远端启动等待进程。
12. 自动汇报已更新为跟踪 `coder_hybrid_prompt -> coder_fewshot -> coder_fewshot_stress -> qwen_fewshot`。

## Current Risks / Failures

1. novelty 风险仍在：现阶段更像“结构化 verifier + selective fallback”的强工程/方法论文，需要更锋利的 claim 和更清楚的 reviewer 防守。
2. support F1 主要由 structured 分支托底，解释时要避免夸大 direct branch 的贡献。
3. 若 `fewshot_json` 对 Coder 和 Qwen 都显著提分，则 paper 叙事需要更谨慎地区分“prompt recipe”与“code-model / hybrid design”的贡献。
4. 本地工作区和远端运行态不能混为一谈，论文强度判断必须以远端实跑证据为准。

## Next Actions

1. 继续监控远端 `coder_hybrid_prompt`，确认它已从 UCI 推进到 WISDM，再继续接到 `fewshot`、`fewshot_stress` 和 `qwen_fewshot`。
2. 依据 `Hume` 的审计摘要继续收紧 reviewer 风险、主线叙事和 baseline 缺口。
3. 若 `qwen_fewshot` 结束后仍未提供新证据，就停止继续烧 prompt 变体，把 GPU 转向更强 baseline 或 paper-facing 分析。


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
