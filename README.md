# SensorFact: Evidence-First Grounding for Wearable Sensor-Language Understanding

This repository is a lightweight discussion package for an EMNLP-oriented project on verifiable grounding in wearable sensor-language understanding. It contains the core Python package, experiment scripts, targeted tests, locked result summaries, and a Chinese expert-discussion memo.

## Core Idea

The current research story is **evidence-first verifiable counterfactual grounding**, not simply a small caption-accuracy improvement. The system separates responsibilities:

- **Structured verifier**: owns verifiable support and counterfactual rejection.
- **Caption order-vote5**: current strongest caption backbone.
- **Confidence-gated forced-choice reranking**: a selective residual ambiguity resolver for caption choices only.
- **Evidence controls**: expose and bound language-prior risk through visible, shuffled, numeric-mask, and hidden-evidence conditions.

The safe claim is that gated forced-choice reranking improves residual caption ambiguity on UCI HAR and MHEALTH significantly, while WISDM is positive but non-significant. Counterfactual F1/support claims remain attributed to the structured verifier, not to the reranker.

## Key Results Snapshot

| Dataset | Vote5 Acc. | Fixed m=2 Gated Acc. | Delta | McNemar mid-p | Interpretation |
|---|---:|---:|---:|---:|---|
| UCI HAR | 0.7361 | 0.7607 | +0.0246 | 0.000011 | Significant positive caption improvement |
| WISDM | 0.7925 | 0.7956 | +0.0031 | 0.3593 | Positive but non-significant |
| MHEALTH | 0.9329 | 0.9512 | +0.0183 | 0.0156 | Significant positive caption improvement |

Dev-calibrated thresholding gives the safer paper-facing protocol:

| Dataset | Dev Threshold | Dev Acc. | Heldout Acc. | Heldout Coverage | Heldout N |
|---|---:|---:|---:|---:|---:|
| UCI HAR | 1.5 | 0.7532 | 0.7706 | 0.5832 | 619 |
| WISDM | 1.0 | 0.8168 | 0.8147 | 0.6026 | 1009 |
| MHEALTH | 1.0 | 1.0000 | 0.9855 | 0.9564 | 275 |

## Repository Map

- [EXPERT_DISCUSSION.md](EXPERT_DISCUSSION.md): Chinese research memo for discussion with advisors/experts.
- [sensorfact/](sensorfact/): core benchmark, evidence, metrics, verifier, and reranker code.
- [scripts/](scripts/): experiment construction, reranking, gating, controls, calibration, significance, and analysis scripts.
- [tests/](tests/): focused tests for benchmark construction, gating, controls, summaries, and queue scripts.
- [docs/](docs/): locked experiment summaries and paper-writing drafts generated on 2026-05-20.
- [outputs/](outputs/): lightweight summary JSON/CSV files only; full row-level outputs are intentionally excluded.
- [configs/](configs/) and [checkpoints/](checkpoints/): small pilot/config artifacts used by the experiments.

## What Is Not Included

This upload intentionally excludes model weights, raw processed datasets, full LLM row dumps, and local scratch directories. The goal is to support expert review of the idea, evidence, and code path without creating a heavyweight artifact repository.

## Most Useful Discussion Files

- [docs/evidence_first_grounding_rollup_2026-05-20.md](docs/evidence_first_grounding_rollup_2026-05-20.md)
- [docs/dev_calibrated_main_caption_table_2026-05-20.md](docs/dev_calibrated_main_caption_table_2026-05-20.md)
- [docs/evidence_control_summary_2026-05-20.md](docs/evidence_control_summary_2026-05-20.md)
- [docs/failure_taxonomy_case_studies_2026-05-20.md](docs/failure_taxonomy_case_studies_2026-05-20.md)
- [docs/paper_experiment_narrative_draft_2026-05-20.md](docs/paper_experiment_narrative_draft_2026-05-20.md)
