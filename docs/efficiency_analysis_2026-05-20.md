# Efficiency Analysis

## Reranker Cost Proxy

| Dataset | N | Full Choice Prompts | Override Prompts | Gate Coverage |
|---|---:|---:|---:|---:|
| UCI HAR | 773.0000 | 773.0000 | 273.0000 | 0.3532 |
| WISDM | 1282.0000 | 1282.0000 | 246.0000 | 0.1919 |
| MHEALTH | 328.0000 | 328.0000 | 48.0000 | 0.1463 |

## Interpretation

The full forced-choice scorer requires one prompt per evaluated window. The deployed gated method should be discussed by override coverage: only rows above the margin threshold change the vote5 decision, even though the current offline analysis scores all rows to estimate the gate.
