from __future__ import annotations

import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


SUMMARY = """

## 2026-05-20 Confidence-Gated Forced-Choice Reranking

Decision: promote to active candidate method, pending paired/bootstrap confirmation. The method keeps support predictions on the structured verifier and improves only ambiguous caption fallback by using a forced-choice next-token log-prob reranker. A fixed conservative gate uses the alternate forced-choice result only when its top-2 log-prob margin is > 2.0.

Main comparison against constrained order-vote5 hybrid:

| Dataset | Vote5 caption acc | Gated forced-choice caption acc | Delta | Gated fallback acc | CF F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCI HAR hard v3 | 0.8565 | 0.8629 | +0.0064 | 0.7607 | 0.7635 |
| WISDM hard v3 | 0.8859 | 0.8872 | +0.0013 | 0.7956 | 0.6918 |
| MHEALTH hard v3 | 0.9351 | 0.9408 | +0.0057 | 0.9512 | 0.7613 |

Interpretation: stronger than vote7/prompt variants because it is a different scoring mechanism, not just candidate-order self-consistency. It gives consistent positive caption gains on all three datasets while preserving structured-only support/CF F1.

Risk: threshold selection must be presented carefully. Threshold 2.0 is a single conservative rule, while margin sweep shows higher dataset-specific thresholds. Need paired/bootstrap significance and a no-label calibration or train/dev threshold variant before making the final main-table claim.
"""


METHOD_DOC = """# Confidence-Gated Forced-Choice Reranking (2026-05-20)

## Method
Use the structured verifier as the support/counterfactual backbone. On ambiguous caption windows, retain order-vote5 as the primary fallback and add a forced-choice log-prob reranker as an alternate. The reranker feeds all candidate captions in one prompt and scores the next-token log-probability of A/B/C/... choices. A conservative gate replaces vote5 only when the forced-choice top-2 log-prob margin is greater than 2.0.

## Why This Is Stronger Than Another Prompt Variant
This changes the decision mechanism from generated JSON selection or order voting to calibrated forced-choice scoring. It directly targets prompt/order instability and gives an interpretable confidence signal for selective intervention. Support-side CF F1 remains structured-only, so caption gains are not confused with support rejection gains.

## Main Results

| Dataset | Vote5 caption acc | Gated forced-choice caption acc | Delta | Gated fallback acc | CF F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UCI HAR hard v3 | 0.8565 | 0.8629 | +0.0064 | 0.7607 | 0.7635 |
| WISDM hard v3 | 0.8859 | 0.8872 | +0.0013 | 0.7956 | 0.6918 |
| MHEALTH hard v3 | 0.9351 | 0.9408 | +0.0057 | 0.9512 | 0.7613 |

## Current Caveats
Threshold 2.0 is a fixed conservative rule and should be justified as a no-label confidence gate. The margin sweep shows additional headroom but should be treated as analysis unless a proper dev-calibration protocol is added. Next required checks are paired/bootstrap significance, threshold robustness, and failure-case analysis.

## Files
- `sensorfact/logprob_reranker.py`
- `scripts/run_logprob_reranker.py`
- `scripts/gate_caption_rows.py`
- `outputs/choice_logprob_margin_sweep.json`
- `outputs/hybrid_regex_coder_gated_vote5_choice_logprob_*_hard_v3_constrained_margin2_metrics.json`
"""


def main() -> None:
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for name in [
        "AGENT_LOG.md",
        "EXPERIMENTS.md",
        "EXPERIMENT_QUEUE.md",
        "RESEARCH_REVIEW.md",
        "PAPER_CLAIMS.md",
        "GPU_BUDGET.md",
    ]:
        path = ROOT / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if "Confidence-Gated Forced-Choice Reranking" not in old:
            path.write_text(old + f"\n\n<!-- update {now} -->" + SUMMARY, encoding="utf-8")

    method_doc = ROOT / "docs" / "confidence_gated_forced_choice_reranking_2026-05-20.md"
    method_doc.parent.mkdir(parents=True, exist_ok=True)
    method_doc.write_text(METHOD_DOC, encoding="utf-8")
    print(f"updated docs: {method_doc}")


if __name__ == "__main__":
    main()
