# Qwen3 Harder Evidence Controls Summary

This table reports Qwen3-4B cross-model confirmation for the harder evidence controls.

| Dataset | Control | Forced-choice Acc | Gated Acc | No-gate Acc | Gate overrides | N |
|---|---:|---:|---:|---:|---:|---:|
| UCI HAR | `numeric_swap` | 0.6831 | 0.6831 | 0.6831 | 765 | 773 |
| UCI HAR | `axis_permutation` | 0.3842 | 0.4062 | 0.3842 | 704 | 773 |
| UCI HAR | `trend_flip` | 0.6856 | 0.6856 | 0.6856 | 769 | 773 |
| WISDM | `numeric_swap` | 0.6654 | 0.6708 | 0.6654 | 1256 | 1282 |
| WISDM | `axis_permutation` | 0.3565 | 0.4275 | 0.3565 | 1094 | 1282 |
| WISDM | `trend_flip` | 0.6685 | 0.6802 | 0.6685 | 1248 | 1282 |
| MHEALTH | `numeric_swap` | 0.9756 | 0.9848 | 0.9756 | 314 | 328 |
| MHEALTH | `axis_permutation` | 0.3415 | 0.3811 | 0.3415 | 278 | 328 |
| MHEALTH | `trend_flip` | 0.9817 | 0.9848 | 0.9817 | 327 | 328 |

## Interpretation

- `axis_permutation` produces the largest degradation across all datasets, which independently confirms that axis/channel evidence is a major decision signal.
- `numeric_swap` and `trend_flip` preserve high accuracy on MHEALTH and remain much less damaging than axis permutation, so the grounding claim should remain bounded.
- This supports cross-model evidence-dependence for axis/channel fields, while still exposing language-prior or template-dependence risk for numeric magnitude and trend fields.
