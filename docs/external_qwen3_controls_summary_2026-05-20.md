# External Evidence-Control Summary

Model tag: `qwen3_4b`

| Dataset | Condition | Choice Acc. | Gated Acc. | No-Gate Acc. | Choice Delta vs Visible | N |
|---|---|---:|---:|---:|---:|---:|
| ucihar | visible | 0.6843 | 0.6856 | 0.6843 | 0.0000 | 773 |
| ucihar | shuffled | 0.5744 | 0.6404 | 0.5744 | -0.1100 | 773 |
| ucihar | numeric-mask | 0.6649 | 0.6611 | 0.6649 | -0.0194 | 773 |
| ucihar | hidden | 0.6455 | 0.6598 | 0.6455 | -0.0388 | 773 |
| wisdm | visible | 0.6708 | 0.6841 | 0.6708 | 0.0000 | 1282 |
| wisdm | shuffled | 0.5538 | 0.5905 | 0.5538 | -0.1170 | 1282 |
| wisdm | numeric-mask | 0.6474 | 0.6498 | 0.6474 | -0.0234 | 1282 |
| wisdm | hidden | 0.7090 | 0.7402 | 0.7090 | 0.0382 | 1282 |
| mhealth | visible | 0.9848 | 0.9848 | 0.9848 | 0.0000 | 328 |
| mhealth | shuffled | 0.6037 | 0.6951 | 0.6037 | -0.3811 | 328 |
| mhealth | numeric-mask | 0.9726 | 0.9787 | 0.9726 | -0.0122 | 328 |
| mhealth | hidden | 0.7348 | 0.7378 | 0.7348 | -0.2500 | 328 |

Interpretation: visible should be read against shuffled, numeric-mask, and hidden controls. Small visible-control gaps indicate language or candidate priors and require a bounded grounding claim.
