# PAPER_CLAIMS

## Main Claim

在 wearable sensor-language grounding 的 hard counterfactual setting 中，`structured verifier + selective direct-model fallback` 可以显著提升 caption selection，同时保持高 counterfactual rejection 质量与较强的鲁棒性。

## Evidence Needed

| Claim | Required Evidence | Current Evidence | Status |
| --- | --- | --- | --- |
| Structured verifier 在 support rejection 上可靠 | support F1 / balanced accuracy / sanity analysis | 已有 axisfix structured regex + metric sanity | Strong |
| Direct coder 单独并不够好 | direct coder baseline across prompt/order/support controls | `emnlp_coder_prompt_queue` 已完成 | Strong |
| Hybrid 在 caption 上优于 structured-only | paired comparisons + CI | 已有 `hybrid_regex_coder_*` paired deltas | Strong |
| Hybrid 不靠单一 prompt 表面成立 | `coder_hybrid_prompt` + `coder_fewshot` + `fewshot_stress` | 部分完成，当前仍在跑 | In progress |
| few-shot 增益不只是“任何 direct model 都吃到的 prompt trick” | `qwen_fewshot` 对照 | 队列已准备并已挂在当前链后 | In progress |
| 主方法不依赖精细阈值调参 | threshold sweep | JSON 已生成且已写入正式报告 | Strong |

## Current Evidence

1. UCI / WISDM / MHEALTH 三套 hard v3 结果已有强正例。
2. coder structured model 是强负例，能支撑“直接结构化生成并不可靠”。
3. hybrid 对 structured-only 的 caption 增益已在 paired comparison 中显著。

## Missing Evidence

1. `fewshot` 与 `fewshot_stress` 还没跑完。
2. `qwen_fewshot` 还没跑完。
3. 需要更清楚地把 benchmark contribution 和 method contribution 区分开。
4. 需要 reviewer-friendly 的 baseline positioning 说明。

## Weak Points Reviewers May Attack

1. 新意不足，像 heuristic ensemble。
2. 可能被质疑 few-shot 只是 prompt trick，而不是 code-model / hybrid 特有收益。
3. direct branch 太弱，像 verifier 单兵作战。
4. 数据集范围与 prior work 对齐不够清楚。
5. 消融虽然在补，但 narrative 还没完全收束。

## Strongest Safe Story Right Now

1. 这不是又一个通用 sensor-language alignment / captioning 模型，而是一个更可验证的 `counterfactual grounding` 设定。
2. 结构化 verifier 在 counterfactual rejection 上提供高可靠底座。
3. selective fallback 只在 structured 不确定时介入，显著改善 caption selection。
4. 多组 prompt / order / support controls 证明结论不是脆弱偶然值。
5. 与 `SensorLLM` / `SensorLM` / `ActivityNarrated` 这类“学对齐、学叙述”的路线相比，我们的安全叙事应强调“可验证判断”而不是“大模型会讲传感器语言”。


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
