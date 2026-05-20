# External Cross-Model Reranker Summary

Model tag: `qwen3_4b`

| Dataset | Choice Acc. | Gated Acc. | No-Gate Acc. | Delta vs Vote5 | McNemar mid-p | Gate Count | N |
|---|---:|---:|---:|---:|---:|---:|---:|
| ucihar | 0.6843 | 0.6856 | 0.6843 | -0.0505 | 0.0020 | 770 | 773 |
| wisdm | 0.6708 | 0.6841 | 0.6708 | -0.1084 | 0.0000 | 1248 | 1282 |
| mhealth | 0.9848 | 0.9848 | 0.9848 | 0.0518 | 0.0003 | 327 | 328 |

Interpretation: this table is a cross-model robustness check for the selective reranking claim. It should not be used to attribute CF support or rejection gains to the reranker; those remain owned by the structured verifier.
