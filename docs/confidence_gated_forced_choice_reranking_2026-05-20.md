# Confidence-Gated Forced-Choice Reranking (2026-05-20)

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


## Follow-up Evidence Added

- Paired bootstrap vs vote5: UCI delta +0.00645, 95% CI [0.00373, 0.00984]; MHEALTH delta +0.00573, 95% CI [0.00191, 0.01051]; WISDM delta +0.00129, 95% CI [-0.00129, 0.00388]. Interpretation: two datasets significantly improve, WISDM is a non-significant small gain.
- Full-candidate forced-choice baseline is much weaker than constrained/gated hybrid: UCI 0.6725, WISDM 0.6949, MHEALTH 0.7383. This supports the selective ambiguous-window framing rather than a generic direct LLM baseline claim.
- Remote safety tests passed: 	ests/test_logprob_reranker.py and 	ests/test_gate_caption_rows.py have 8 passing tests.
