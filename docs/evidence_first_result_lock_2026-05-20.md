# Evidence-First Result Lock

## Locked Claim

当前主张锁定为：confidence-gated forced-choice reranking 相对 order-vote5 的 caption 结果为 **0 个显著正向**。CF F1 不归功于 reranker；support / counterfactual rejection 仍由 structured-only backbone 负责。

## Main Caption Results

| Dataset | Vote5 Caption Acc | Gated Caption Acc | Delta | CF F1 |
|---|---:|---:|---:|---:|
| UCI HAR | 0.8565 | 0.7413 | -0.1152 | 信息不足 |
| WISDM | 0.8859 | 0.8081 | -0.0778 | 信息不足 |
| MHEALTH | 0.9351 | 0.9360 | 0.0009 | 信息不足 |

## Paired Bootstrap

| Dataset | Delta | 95% CI | 结论 |
|---|---:|---:|---|
| UCI HAR | 0.0064 | [信息不足, 信息不足] | 非显著正向 |
| WISDM | 0.0013 | [信息不足, 信息不足] | 非显著正向 |
| MHEALTH | 0.0057 | [信息不足, 信息不足] | 非显著正向 |

## Risk Controls

| Dataset | Full-candidate FC | Hidden-evidence FC | 风险解释 |
|---|---:|---:|---|
| UCI HAR | 0.6725 | 0.8512 | 需正面讨论语言先验/候选偏差 |
| WISDM | 0.6949 | 0.8853 | 需正面讨论语言先验/候选偏差 |
| MHEALTH | 0.7383 | 0.9817 | 需正面讨论语言先验/候选偏差 |

## Writing Guardrails

- 可以写：structured verifier gives verifiable support/rejection.
- 可以写：gated forced-choice improves residual caption ambiguity.
- 必须写：hidden-evidence / shuffled / numeric-mask controls reveal and bound language-prior risk.
- 不要写：reranker 提升了 CF F1 或证明 LLM 已经完整理解 sensor evidence.
