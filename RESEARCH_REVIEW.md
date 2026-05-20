# RESEARCH_REVIEW

## A. Can It Work?

- 结论：能工作，而且已经工作。
- 证据：
  - 多条 `run_emnlp_*` 队列脚本与 relay 链已能在远端真实占用 GPU 并产出结果。
  - 本地与远端相关回归通过。
  - 当前 `coder_hybrid_prompt` 已从 UCI `chain_then_json` 推进到 WISDM，GPU 利用率约 96%。
- 主要隐患：
  1. 结果汇总和 paper claim 仍需持续对齐，避免脚本存在先于远端证据。
  2. 当前主线的解释风险大于代码能不能跑的风险。

## B. Is It Strong?

- 当前最强主线：`structured regex verifier + direct fallback`
  - UCI caption `0.8069`, support F1 `0.7635`
  - WISDM caption `0.7986`, support F1 `0.6918`
  - MHEALTH caption `0.8138`, support F1 `0.7613`
- paired comparison 已表明：
  - hybrid 明显优于 structured-only caption
  - hybrid 明显优于 coder direct caption / support
- 但还不够“稳到 main”：
  - 需要把 `fewshot` 与 `fewshot_stress` 跑完，证明主线不是靠单一 prompt 偶然成立。
  - 需要把 `qwen_fewshot` 跑完，证明收益不只是 prompt recipe 对任意 direct model 的普遍加成。
  - 需要更明确地说清 direct branch 的作用边界，避免 reviewer 认为只是 verifier 在工作。

## C. Is It Novel?

- 目前 novelty 级别：中等偏弱到中等。
- 真正能讲的不是“让 LLM 做 sensor grounding”，而是：
  - 结构化证据抽取 / 验证器负责 support rejection，
  - direct branch 只在 structured 底座不够时介入 caption decision，
  - 整体形成可解释、可验证、鲁棒的 fallback 机制。
- 近期最相关的公开方向需要正面定位：
  - `IMU2CLIP` (Findings EMNLP 2023): 更偏 language-grounded motion-sensor alignment / retrieval。
  - `SensorLLM` (arXiv 2024): 更偏 sensor-language alignment + task-aware tuning 做 HAR。
  - `SensorLM` (arXiv 2025): 更偏大规模 sensor-language foundation model 与预训练。
  - `TSVer` (EMNLP 2025): 更偏 time-series fact verification benchmark。
  - `ActivityNarrated` (arXiv 2026): 更偏 open-ended wearable narrative understanding。
- 风险：
  - 如果写成“又一个 prompt / ensemble / fallback 工程”，会显得增量。
  - 如果把自己写成“又一个 sensor-language alignment / caption model”，会直接撞上 `SensorLLM` / `SensorLM` / `ActivityNarrated` 的主叙事。
  - 如果 `qwen_fewshot` 也显著抬分，则“code-model 特别有效”的叙事要进一步收缩。

## D. Workload And Execution Risk

- 当前 workload 状态：中高，但可控。
- 已完成：
  - axisfix
  - hybrid baseline
  - coder prompt robustness
  - threshold sweep
  - report aggregation
- 当前进行中：
  - `coder_hybrid_prompt`
- 已挂好下一棒：
  - `coder_fewshot`
  - `coder_fewshot_stress`
  - `qwen_fewshot`

## E. Innovation Assessment

- 现阶段最像论文贡献的部分：
  1. hard v3 benchmark + counterfactual grounding setting
  2. structured verifier 的高 support reliability
  3. selective fallback 在 caption 上稳定获益，且 support 不退化
- 不够像贡献的部分：
  - direct branch 自己并不强，单独拿出去很难讲
  - adaptive threshold 没有新增收益，不能作为新亮点

## Negative Results Worth Keeping

1. `coder structured model` 直接输出结构化字段，caption 很差、support F1 极低，但 parse 高。
2. adaptive threshold sweep 几乎全平台，说明当前 hybrid gate 不靠精细调参吃饭。
3. direct coder 在 support-balance / order 控制下总体仍弱于 structured / hybrid。

## Reviewer-Risk Analysis

1. “只是工程组合，没有方法新意。”
2. “direct branch 太弱，贡献几乎都来自 structured verifier。”
3. “few-shot 可能只是 prompt trick，不一定是方法收益。”
4. “benchmark contribution 和 method contribution 混在一起讲了。”
5. “prior art 对齐和更熟悉 baseline 还不够清楚，尤其是和 `SensorLLM` / `SensorLM` / `TSVer` 的差别。”

## Missing Baselines / Ablations / Analyses

1. 跑完 `fewshot` 与 `fewshot_stress`。
2. 跑完 `qwen_fewshot` 这个 prompt-only 强对照。
3. 更明确的 baseline 与 reviewer 熟悉 prior art 映射。
4. 更清楚的 error taxonomy / failure-case narrative。

## Recommended Next Move

1. 先跑完当前 GPU 链，不中断。
2. 当前链结束后先看 `qwen_fewshot` 对照，而不是继续随机挖更多 prompt 变体。
3. 若 `qwen_fewshot` 也显著增益，则收紧“code-model / hybrid design”表述；若增益有限，则当前主线论证更稳。
4. 所有“论文是否站得住”的判断以远端实跑主表和 `outputs/queue_logs` 为准。


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
