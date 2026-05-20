# Evidence-Control Summary

## Control Table

| Dataset | Condition | Choice Acc | Gated Acc | Gated Coverage |
|---|---|---:|---:|---:|
| UCI HAR | visible | 0.7193 | 0.7607 | 0.3532 |
| UCI HAR | shuffled-evidence | 0.7309 | 0.7426 | 0.1138 |
| UCI HAR | numeric-mask | 0.7063 | 0.7490 | 0.2147 |
| UCI HAR | hidden-evidence | 0.8512 | 0.7413 | 0.0285 |
| WISDM | visible | 0.7683 | 0.7956 | 0.1919 |
| WISDM | shuffled-evidence | 0.7652 | 0.7855 | 0.0920 |
| WISDM | numeric-mask | 0.7582 | 0.7949 | 0.1825 |
| WISDM | hidden-evidence | 0.8853 | 0.8081 | 0.1178 |
| MHEALTH | visible | 0.9939 | 0.9512 | 0.1463 |
| MHEALTH | shuffled-evidence | 0.9207 | 0.9299 | 0.1159 |
| MHEALTH | numeric-mask | 0.9939 | 0.9451 | 0.1707 |
| MHEALTH | hidden-evidence | 0.9817 | 0.9360 | 0.0823 |

## Interpretation Guardrails

- UCI HAR 的 shuffled-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- UCI HAR 的 shuffled-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- UCI HAR 的 numeric-mask choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- UCI HAR 的 numeric-mask gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- UCI HAR 的 hidden-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- UCI HAR 的 hidden-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- WISDM 的 shuffled-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- WISDM 的 shuffled-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- WISDM 的 numeric-mask choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- WISDM 的 numeric-mask gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- WISDM 的 hidden-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- WISDM 的 hidden-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- MHEALTH 的 shuffled-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- MHEALTH 的 numeric-mask choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- MHEALTH 的 numeric-mask gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。
- MHEALTH 的 hidden-evidence choice 与 visible 差距小于 0.03，需要收紧 grounding claim。
- MHEALTH 的 hidden-evidence gated 与 visible 差距小于 0.03，需要把结果写成语言先验风险边界。

## Paper Use

- visible > control: can support partial evidence dependence.
- control close to visible: write as language-prior / candidate-bias risk, not as grounded understanding.
- CF F1 remains structured-only and should be narrated separately from caption controls.
