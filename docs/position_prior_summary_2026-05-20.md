# Position-Prior Baseline Summary

## Results

| Dataset | Position-Prior Acc | Full Choice Acc | Choice - Position | N |
|---|---:|---:|---:|---:|
| UCI HAR | 0.4838 | 0.7193 | 0.2354 | 773.0000 |
| WISDM | 0.4641 | 0.7683 | 0.3042 | 1282.0000 |
| MHEALTH | 0.6250 | 0.9939 | 0.3689 | 328.0000 |

## Interpretation

- UCI HAR 的位置先验较高，候选顺序/答案索引偏置必须在论文中控制。
- WISDM 的位置先验较高，候选顺序/答案索引偏置必须在论文中控制。
- MHEALTH 的位置先验较高，候选顺序/答案索引偏置必须在论文中控制。

This baseline sees only option labels, not evidence or candidate text. It should be reported as an answer-index / position-prior control, not as a sensor-language model.
