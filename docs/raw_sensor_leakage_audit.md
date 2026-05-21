# Raw-Sensor Leakage Audit and Corrected Baseline

## Why We Re-ran

The previous near-1.0 supervised and aligner results are not fair main baselines. They read structured evidence-card fields, or numeric summaries stored inside the eval evidence card, during test-time scoring. Those numbers should be treated as oracle/upper-bound diagnostics, not as evidence that the model learned sensor grounding.

The corrected baseline below trains on raw sensor windows with train evidence fields as supervision. At test time it reads only the raw sensor window and candidate text. Eval evidence is used only after prediction to compute diagnostic evidence-field accuracy.

## Corrected Raw-Sensor Baseline

| Dataset | Caption Acc. | Caption Macro-F1 | CF Reject F1 | Support ECE | Support Brier | Evidence Field Acc. | Device | Train N | Eval N | Threshold | Status |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| UCI HAR | 0.8286 | 0.8287 | 0.4342 | 0.3762 | 0.2426 | 0.7741 | cuda | 7352 | 2947 | 0.7988 | done |
| WISDM | 0.9098 | 0.9096 | 0.6590 | 0.4372 | 0.2774 | 0.8401 | cuda | 11067 | 3094 | 0.8044 | done |
| MHEALTH | 0.9446 | 0.9445 | 0.6878 | 0.4071 | 0.2447 | 0.8424 | cuda | 4182 | 1047 | 0.8028 | done |

## Claim Boundary

- Fair main baseline candidate: `raw_sensor_field_aligner`.
- `oracle_fields`, `axis_drop`, `numeric_drop`, and full/distilled aligners are upper bounds or ablations because they read structured evidence at test time.
- Numeric-only modes that read numeric summaries from eval evidence cards are not clean enough for the main result; they are at best semi-oracle diagnostics.
- Reranker results should remain caption-only residual ambiguity results and must not be credited for support F1 or CF rejection F1.

## Files

- UCI HAR: metrics `outputs/emnlp_raw_sensor_clean/raw_sensor_ucihar_metrics.json`; rows/model are archived locally or on the remote server, not in the lightweight GitHub package.
- WISDM: metrics `outputs/emnlp_raw_sensor_clean/raw_sensor_wisdm_metrics.json`; rows/model are archived locally or on the remote server, not in the lightweight GitHub package.
- MHEALTH: metrics `outputs/emnlp_raw_sensor_clean/raw_sensor_mhealth_metrics.json`; rows/model are archived locally or on the remote server, not in the lightweight GitHub package.
