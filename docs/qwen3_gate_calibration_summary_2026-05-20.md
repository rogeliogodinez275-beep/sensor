# Qwen3 Gate Calibration Summary

| Dataset | Best Dev Threshold | Dev Acc. | Heldout Acc. | Heldout Coverage |
|---|---:|---:|---:|---:|
| ucihar | 5.0 | 0.6948 | 0.6850 | 0.9822 |
| wisdm | 10.0 | 0.7143 | 0.7304 | 0.8741 |
| mhealth | 10.0 | 1.0000 | 0.9636 | 0.8145 |

Interpretation: this checks whether the fixed margin=2 gate transfers to the external Qwen3 reranker. Large threshold shifts or low heldout accuracy indicate model-specific calibration risk.
