# EXPERIMENTS

## `emnlp_coder_prompt_queue`

- Hypothesis: direct coder baseline 在 prompt、candidate order、support order、support balance 控制下的行为可以作为 hybrid 的对照与鲁棒性参照。
- Command: `bash scripts/run_emnlp_coder_prompt_queue.sh`
- GPU: 48GB RTX 4090
- Runtime: 约 `2026-05-19 14:45` 到 `17:19`
- Outputs:
  - `outputs/queue_logs/emnlp_coder_prompt_status.tsv`
  - `outputs/coder_llm_*_prompt_terse_*`
  - `outputs/coder_llm_*_caption_order_seed5153_*`
  - `outputs/coder_llm_*_support_order_seed7101_*`
  - `outputs/coder_llm_*_support_balanced_neg3_*`
- Result summary:
  - direct coder caption 约在 `0.51 ~ 0.59`
  - support F1 约在 `0.54 ~ 0.61`，但仍显著弱于 structured / hybrid 主线
- Decision: 保留。它提供了足够强的 direct baseline 与鲁棒性对照。

## `emnlp_coder_hybrid_prompt_queue`

- Hypothesis: `structured regex + coder fallback` 的 caption 增益不依赖单一 direct prompt surface。
- Command: `bash scripts/run_emnlp_coder_hybrid_prompt_queue.sh`
- GPU: 48GB RTX 4090
- Current status: Running
- Outputs:
  - `outputs/queue_logs/emnlp_coder_hybrid_prompt_status.tsv`
  - `outputs/hybrid_regex_coder_*_prompt_terse_*`
  - `outputs/coder_llm_*_prompt_chain_then_json_*`
  - `outputs/hybrid_regex_coder_*_prompt_chain_then_json_*`
- Current evidence:
  - `coder_llm_ucihar_hard_v3_prompt_chain_then_json` 已在远端完成。
  - `hybrid_regex_coder_ucihar_hard_v3_prompt_chain_then_json` 已完成。
  - 当前已推进到 `coder_llm_wisdm_hard_v3_prompt_chain_then_json`，GPU 利用率约 96%，显存约 15.5GB。
- Decision: Continue。它是当前最核心的 EMNLP 主线支撑实验。

## `emnlp_coder_fewshot_queue`

- Hypothesis: `fewshot_json` 会显著抬 direct coder；若抬分成立，hybrid 可能进一步吸收这部分增益。
- Command: `bash scripts/run_emnlp_coder_fewshot_queue.sh`
- GPU: 48GB RTX 4090
- Current status: Approved，`relay_after_coder_hybrid_prompt.sh` 等待中。
- Outputs:
  - `outputs/coder_llm_*_prompt_fewshot_json_*`
  - `outputs/hybrid_regex_coder_*_prompt_fewshot_json_*`
- Decision: 紧接 `emnlp_coder_hybrid_prompt_queue`。

## `emnlp_coder_fewshot_stress_queue`

- Hypothesis: few-shot 提升若真实存在，应能经受 candidate order、support order、negative balance 压力测试。
- Command: `bash scripts/run_emnlp_coder_fewshot_stress_queue.sh`
- GPU: 48GB RTX 4090
- Current status: Approved，`relay_after_coder_fewshot.sh` 等待中。
- Outputs:
  - `outputs/coder_llm_*_caption_order_seed5153_prompt_fewshot_json_*`
  - `outputs/coder_llm_*_support_order_seed7101_prompt_fewshot_json_*`
  - `outputs/coder_llm_*_support_balanced_neg3_prompt_fewshot_json_*`
- Decision: 紧接 `emnlp_coder_fewshot_queue`。

## `emnlp_qwen_fewshot_queue`

- Hypothesis: 如果 `fewshot_json` 的收益主要来自 prompt recipe 而不是 code-model / hybrid 设计，那么基础 Qwen 4B 也应该在 hard v3 上获得明显提升。
- Command: `bash scripts/run_emnlp_qwen_fewshot_queue.sh`
- GPU: 48GB RTX 4090
- Current status: Approved，`relay_after_coder_fewshot_stress.sh` 已在远端等待。
- Outputs:
  - `outputs/qwen_llm_ucihar_hard_v3_prompt_fewshot_json_*`
  - `outputs/qwen_llm_wisdm_hard_v3_prompt_fewshot_json_*`
  - `outputs/qwen_llm_mhealth_hard_v3_prompt_fewshot_json_*`
- Decision: 作为当前链尾的强对照保留。它能帮助区分“prompt-only 提升”和“当前主线方法提升”。

## CPU Analysis: `Hybrid Threshold Sweep`

- Hypothesis: 如果 `structured_threshold=0.5` 不是最优，adaptive gate 可能给主方法带来额外收益。
- Command example:
  - `python scripts/sweep_hybrid_threshold.py --workspace /root/autodl-tmp/emnlp2026 --structured-rows outputs/axisfix_structured_regex_ucihar_hard_v3_rows.jsonl --direct-rows outputs/coder_llm_ucihar_hard_v3_rows.jsonl --step 0.05 --output-json outputs/hybrid_threshold_sweep_ucihar.json`
- Current result:
  - UCI / WISDM / MHEALTH 都显示 `0.05~1.0` 几乎同平台，`0.5` 已在最优平台内。
- Decision: Stop as GPU direction。保留为报告中的阈值鲁棒性分析。


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
