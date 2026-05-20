# Balanced Candidate Subset Analysis

Benchmark: `outputs/constrained_wisdm_hard_v3_subset.jsonl`

| Total | Selected | Target Fraction | Mean Length Range | Mean Evidence-Overlap Range |
|---:|---:|---:|---:|---:|
| 1282 | 641 | 0.50 | 0.8050 | 0.0321 |

| System | Full Acc. | Balanced Acc. | Delta | Balanced N | Full N |
|---|---:|---:|---:|---:|---:|
| vote5 | 0.7925 | 0.8830 | 0.0905 | 641 | 1282 |
| visible_choice | 0.7683 | 0.8549 | 0.0866 | 641 | 1282 |
| visible_gated | 0.7956 | 0.9002 | 0.1045 | 641 | 1282 |
| hidden_choice | 0.8853 | 0.9813 | 0.0959 | 641 | 1282 |
| hidden_gated | 0.8081 | 0.9002 | 0.0920 | 641 | 1282 |
| position_prior | 0.4641 | 0.4431 | -0.0211 | 641 | 1282 |
| qwen3_choice | 0.6708 | 0.6864 | 0.0156 | 641 | 1282 |
| qwen3_gated | 0.6841 | 0.7114 | 0.0273 | 641 | 1282 |
| qwen3_position_prior | 0.4641 | 0.4431 | -0.0211 | 641 | 1282 |

Interpretation: this subset reduces obvious candidate-length and evidence-overlap imbalance. It is a diagnostic control, not a replacement for the main benchmark.
