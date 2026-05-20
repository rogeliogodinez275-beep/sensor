# Balanced Candidate Subset Analysis

Benchmark: `outputs/constrained_ucihar_hard_v3_subset.jsonl`

| Total | Selected | Target Fraction | Mean Length Range | Mean Evidence-Overlap Range |
|---:|---:|---:|---:|---:|
| 773 | 386 | 0.50 | 0.9430 | 0.0254 |

| System | Full Acc. | Balanced Acc. | Delta | Balanced N | Full N |
|---|---:|---:|---:|---:|---:|
| vote5 | 0.7361 | 0.8316 | 0.0955 | 386 | 773 |
| visible_choice | 0.7193 | 0.8368 | 0.1175 | 386 | 773 |
| visible_gated | 0.7607 | 0.8808 | 0.1202 | 386 | 773 |
| hidden_choice | 0.8512 | 1.0000 | 0.1488 | 386 | 773 |
| hidden_gated | 0.7413 | 0.8368 | 0.0955 | 386 | 773 |
| position_prior | 0.4838 | 0.4715 | -0.0123 | 386 | 773 |
| qwen3_choice | 0.6843 | 0.6943 | 0.0100 | 386 | 773 |
| qwen3_gated | 0.6856 | 0.6969 | 0.0113 | 386 | 773 |
| qwen3_position_prior | 0.4838 | 0.4715 | -0.0123 | 386 | 773 |

Interpretation: this subset reduces obvious candidate-length and evidence-overlap imbalance. It is a diagnostic control, not a replacement for the main benchmark.
