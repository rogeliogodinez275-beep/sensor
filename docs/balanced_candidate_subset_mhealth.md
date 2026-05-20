# Balanced Candidate Subset Analysis

Benchmark: `outputs/constrained_mhealth_hard_v3_subset.jsonl`

| Total | Selected | Target Fraction | Mean Length Range | Mean Evidence-Overlap Range |
|---:|---:|---:|---:|---:|
| 328 | 164 | 0.50 | 1.0183 | 0.0471 |

| System | Full Acc. | Balanced Acc. | Delta | Balanced N | Full N |
|---|---:|---:|---:|---:|---:|
| vote5 | 0.9329 | 0.8902 | -0.0427 | 164 | 328 |
| visible_choice | 0.9939 | 1.0000 | 0.0061 | 164 | 328 |
| visible_gated | 0.9512 | 0.9268 | -0.0244 | 164 | 328 |
| hidden_choice | 0.9817 | 0.9939 | 0.0122 | 164 | 328 |
| hidden_gated | 0.9360 | 0.8963 | -0.0396 | 164 | 328 |
| position_prior | 0.6250 | 0.5793 | -0.0457 | 164 | 328 |
| qwen3_choice | 0.9848 | 0.9756 | -0.0091 | 164 | 328 |
| qwen3_gated | 0.9848 | 0.9756 | -0.0091 | 164 | 328 |
| qwen3_position_prior | 0.6250 | 0.5793 | -0.0457 | 164 | 328 |

Interpretation: this subset reduces obvious candidate-length and evidence-overlap imbalance. It is a diagnostic control, not a replacement for the main benchmark.
