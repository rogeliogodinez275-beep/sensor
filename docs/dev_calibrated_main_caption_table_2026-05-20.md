# Dev-Calibrated Main Caption Table

This table separates the current fixed margin=2 result from the safer dev-calibrated threshold protocol. The fixed-margin result is useful as a locked finding; the dev-calibrated row is the safer candidate for the main paper table if it remains competitive.

| Dataset | Vote5 Acc. | Fixed m=2 Acc. | Fixed Delta | Fixed mid-p | Dev Threshold | Dev Acc. | Dev Coverage | Heldout Acc. | Heldout Coverage | Heldout N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UCI HAR | 0.7361 | 0.7607 | 0.0246 | 0.0000 | 1.5 | 0.7532 | 0.5519 | 0.7706 | 0.5832 | 619 |
| WISDM | 0.7925 | 0.7956 | 0.0031 | 0.3593 | 1.0 | 0.8168 | 0.6081 | 0.8147 | 0.6026 | 1009 |
| MHEALTH | 0.9329 | 0.9512 | 0.0183 | 0.0156 | 1.0 | 1.0000 | 0.9623 | 0.9855 | 0.9564 | 275 |

Interpretation:

- The fixed margin=2 result remains the locked comparison against order-vote5: UCI and MHEALTH are significant; WISDM is small and non-significant.
- The dev-calibrated protocol should be used to defuse threshold-tuning concerns. If heldout accuracy is lower, report it honestly as the more conservative main result and keep fixed margin=2 as an analysis result.
- This table concerns caption ambiguity only. It does not attribute CF F1 or support rejection gains to the reranker.
