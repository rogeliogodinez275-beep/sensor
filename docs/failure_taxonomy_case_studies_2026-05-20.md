# Failure Taxonomy Case Studies

Purpose: turn existing vote5 wrong -> gated right and gated wrong -> vote5 right rows into paper-ready qualitative evidence. These cases are for analysis only; they should not be used as new quantitative claims.

## ucihar

- vote5 wrong -> gated right: 20
- gated wrong -> vote5 right: 1

### Representative fixes

#### ucihar_test_000139

- gold index: 1; vote5 pred: 1; gated pred: 2; gate source: alternate; margin: 2.0625
- gold fact: gold
- vote5 predicted fact: gold
- gated predicted fact: periodicity+burstiness
- evidence: RMS energy is 0.365, above the upper reference band. The repeatability scores are autocorrelation 0.684 and FFT concentration 0.187, above the repeatability reference. Axis comparison: energy is spread across channels without a clear leader. The dominant spectral peak is 0.391 Hz, below the lower reference band. Peak count is 10, with the profile outside the expected reference band for burstiness. Segment slopes move rising, then falling, then falling. Pairwise channel check: gyro_y and gyro_z m
- gold caption: Overall, the motion shows forceful energy, a repeatable rhythm, energy is spread across channels without a clear leader, slow cadence, and an even profile.
- vote5 caption: Overall, the motion shows forceful energy, a repeatable rhythm, energy is spread across channels without a clear leader, slow cadence, and an even profile.
- gated caption: Overall, the motion shows forceful energy, little repeatable rhythm, energy is spread across channels without a clear leader, slow cadence, and short spikes.

#### ucihar_test_000156

- gold index: 1; vote5 pred: 1; gated pred: 0; gate source: alternate; margin: 2.25
- gold fact: gold
- vote5 predicted fact: gold
- gated predicted fact: intensity+dominant_frequency
- evidence: RMS energy is 0.439, above the upper reference band. The repeatability scores are autocorrelation 0.558 and FFT concentration 0.135, above the repeatability reference. Axis comparison: energy is spread across channels without a clear leader. The dominant spectral peak is 0.391 Hz, below the lower reference band. Peak count is 10, with the profile outside the expected reference band for burstiness. Segment slopes move falling, then falling, then rising. Pairwise channel check: acc_x and acc_z mov
- gold caption: Overall, the motion shows forceful energy, a repeatable rhythm, energy is spread across channels without a clear leader, slow cadence, and an even profile.
- vote5 caption: Overall, the motion shows forceful energy, a repeatable rhythm, energy is spread across channels without a clear leader, slow cadence, and an even profile.
- gated caption: Overall, the motion shows subtle energy, a repeatable rhythm, energy is spread across channels without a clear leader, fast cadence, and an even profile.

#### ucihar_test_000193

- gold index: 0; vote5 pred: 0; gated pred: 1; gate source: alternate; margin: 2.5
- gold fact: gold
- vote5 predicted fact: gold
- gated predicted fact: intensity+dominant_frequency
- evidence: RMS energy is 0.146, inside the middle reference band. The repeatability scores are autocorrelation 0.862 and FFT concentration 0.227, above the repeatability reference. Axis comparison: the largest per-axis energy is on the yaw-rate gyroscope trace. The dominant spectral peak is 0.391 Hz, below the lower reference band. Peak count is 8, with the profile outside the expected reference band for burstiness. Segment slopes move rising, then rising, then rising. Pairwise channel check: acc_x and gyr
- gold caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the yaw-rate gyroscope trace, slow cadence, and an even profile.
- vote5 caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the yaw-rate gyroscope trace, slow cadence, and an even profile.
- gated caption: Overall, the motion shows forceful energy, a repeatable rhythm, the largest per-axis energy is on the yaw-rate gyroscope trace, fast cadence, and an even profile.

### Representative regressions

#### ucihar_test_000949

- gold index: 1; vote5 pred: 3; gated pred: 2; gate source: alternate; margin: 2.4375
- gold fact: gold
- vote5 predicted fact: NA
- gated predicted fact: dominant_axis+periodicity
- evidence: RMS energy is 0.005, below the lower reference band. The repeatability scores are autocorrelation 0.591 and FFT concentration 0.158, above the repeatability reference. Axis comparison: the largest per-axis energy is on the yaw-rate gyroscope trace. The dominant spectral peak is 0.391 Hz, below the lower reference band. Peak count is 10, with the profile outside the expected reference band for burstiness. Segment slopes move falling, then falling, then steady.
- gold caption: Overall, the motion shows subtle energy, a repeatable rhythm, the largest per-axis energy is on the yaw-rate gyroscope trace, slow cadence, and an even profile.
- vote5 caption: NA
- gated caption: Overall, the motion shows subtle energy, little repeatable rhythm, the largest per-axis energy is on the forward-back acceleration trace, slow cadence, and an even profile.

## wisdm

- vote5 wrong -> gated right: 11
- gated wrong -> vote5 right: 7

### Representative fixes

#### wisdm_test_000193

- gold index: 0; vote5 pred: 3; gated pred: 2; gate source: alternate; margin: 2.1875
- gold fact: gold
- vote5 predicted fact: NA
- gated predicted fact: periodicity+burstiness
- evidence: RMS energy is 7.445, inside the middle reference band. The repeatability scores are autocorrelation 0.738 and FFT concentration 0.065, above the repeatability reference. Axis comparison: the largest per-axis energy is on the side-to-side acceleration trace. The dominant spectral peak is 1.875 Hz, below the lower reference band. Peak count is 19, with the profile outside the expected reference band for burstiness. Segment slopes move steady, then rising, then rising. Pairwise channel check: acc_x
- gold caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, slow cadence, and an even profile.
- vote5 caption: NA
- gated caption: Overall, the motion shows moderate energy, little repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, slow cadence, and short spikes.

#### wisdm_test_000194

- gold index: 0; vote5 pred: 3; gated pred: 2; gate source: alternate; margin: 2.1875
- gold fact: gold
- vote5 predicted fact: NA
- gated predicted fact: periodicity+burstiness
- evidence: RMS energy is 7.357, inside the middle reference band. The repeatability scores are autocorrelation 0.698 and FFT concentration 0.073, above the repeatability reference. Axis comparison: the largest per-axis energy is on the side-to-side acceleration trace. The dominant spectral peak is 1.875 Hz, below the lower reference band. Peak count is 19, with the profile outside the expected reference band for burstiness. Segment slopes move rising, then rising, then steady. Pairwise channel check: acc_x
- gold caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, slow cadence, and an even profile.
- vote5 caption: NA
- gated caption: Overall, the motion shows moderate energy, little repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, slow cadence, and short spikes.

#### wisdm_test_000590

- gold index: 0; vote5 pred: 2; gated pred: 0; gate source: alternate; margin: 2.5625
- gold fact: gold
- vote5 predicted fact: periodicity+burstiness
- gated predicted fact: gold
- evidence: RMS energy is 6.905, inside the middle reference band. The repeatability scores are autocorrelation 0.693 and FFT concentration 0.069, above the repeatability reference. Axis comparison: the largest per-axis energy is on the side-to-side acceleration trace. The dominant spectral peak is 1.719 Hz, below the lower reference band. Peak count is 14, with the profile outside the expected reference band for burstiness. Segment slopes move falling, then falling, then falling.
- gold caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, slow cadence, and an even profile.
- vote5 caption: Overall, the motion shows moderate energy, little repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, slow cadence, and short spikes.
- gated caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, slow cadence, and an even profile.

### Representative regressions

#### wisdm_test_001251

- gold index: 1; vote5 pred: 3; gated pred: 2; gate source: alternate; margin: 2.0625
- gold fact: gold
- vote5 predicted fact: NA
- gated predicted fact: dominant_axis+periodicity
- evidence: RMS energy is 9.637, above the upper reference band. The repeatability scores are autocorrelation 0.784 and FFT concentration 0.141, above the repeatability reference. Axis comparison: the largest per-axis energy is on the side-to-side acceleration trace. The dominant spectral peak is 2.500 Hz, above the upper reference band. Peak count is 20, with the profile outside the expected reference band for burstiness. Segment slopes move steady, then falling, then falling. Pairwise channel check: acc_x
- gold caption: Overall, the motion shows forceful energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, fast cadence, and an even profile.
- vote5 caption: NA
- gated caption: Overall, the motion shows forceful energy, little repeatable rhythm, the largest per-axis energy is on the forward-back acceleration trace, fast cadence, and an even profile.

#### wisdm_test_001257

- gold index: 1; vote5 pred: 2; gated pred: 0; gate source: alternate; margin: 2.125
- gold fact: gold
- vote5 predicted fact: dominant_axis+periodicity
- gated predicted fact: periodicity+burstiness
- evidence: RMS energy is 9.617, above the upper reference band. The repeatability scores are autocorrelation 0.776 and FFT concentration 0.111, above the repeatability reference. Axis comparison: the largest per-axis energy is on the side-to-side acceleration trace. The dominant spectral peak is 2.500 Hz, above the upper reference band. Peak count is 21, with the profile outside the expected reference band for burstiness. Segment slopes move rising, then falling, then rising. Pairwise channel check: acc_x 
- gold caption: Overall, the motion shows forceful energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, fast cadence, and an even profile.
- vote5 caption: Overall, the motion shows forceful energy, little repeatable rhythm, the largest per-axis energy is on the forward-back acceleration trace, fast cadence, and an even profile.
- gated caption: Overall, the motion shows forceful energy, little repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, fast cadence, and short spikes.

#### wisdm_test_001445

- gold index: 1; vote5 pred: 3; gated pred: 1; gate source: alternate; margin: 2.1875
- gold fact: gold
- vote5 predicted fact: NA
- gated predicted fact: gold
- evidence: RMS energy is 9.709, above the upper reference band. The repeatability scores are autocorrelation 0.743 and FFT concentration 0.111, above the repeatability reference. Axis comparison: the largest per-axis energy is on the side-to-side acceleration trace. The dominant spectral peak is 2.500 Hz, above the upper reference band. Peak count is 21, with the profile outside the expected reference band for burstiness. Segment slopes move falling, then rising, then falling. Pairwise channel check: acc_x
- gold caption: Overall, the motion shows forceful energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, fast cadence, and an even profile.
- vote5 caption: NA
- gated caption: Overall, the motion shows forceful energy, a repeatable rhythm, the largest per-axis energy is on the side-to-side acceleration trace, fast cadence, and an even profile.

## mhealth

- vote5 wrong -> gated right: 6
- gated wrong -> vote5 right: 0

### Representative fixes

#### mhealth_test_000043

- gold index: 0; vote5 pred: 2; gated pred: 0; gate source: alternate; margin: 2.3125
- gold fact: gold
- vote5 predicted fact: periodicity+burstiness
- gated predicted fact: gold
- evidence: RMS energy is 13.788, inside the middle reference band. The repeatability scores are autocorrelation 0.535 and FFT concentration 0.188, near the repeatability reference. Axis comparison: the largest per-axis energy is on the arm mag z. The dominant spectral peak is 0.391 Hz, below the lower reference band. Peak count is 6, with the profile outside the expected reference band for burstiness. Segment slopes move falling, then rising, then rising. Pairwise channel check: ankle_gyro_x and ankle_gyro
- gold caption: Overall, the motion shows moderate energy, a loose rhythm, the largest per-axis energy is on the arm mag z, slow cadence, and an even profile.
- vote5 caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the arm mag z, slow cadence, and short spikes.
- gated caption: Overall, the motion shows moderate energy, a loose rhythm, the largest per-axis energy is on the arm mag z, slow cadence, and an even profile.

#### mhealth_test_000090

- gold index: 0; vote5 pred: 3; gated pred: 1; gate source: alternate; margin: 2.5625
- gold fact: gold
- vote5 predicted fact: NA
- gated predicted fact: periodicity+burstiness
- evidence: RMS energy is 3.748, below the lower reference band. The repeatability scores are autocorrelation 0.364 and FFT concentration 0.135, near the repeatability reference. Axis comparison: energy is spread across channels without a clear leader. The dominant spectral peak is 0.781 Hz, outside the expected reference band. Peak count is 13, with the profile outside the expected reference band for burstiness. Segment slopes move falling, then falling, then falling. Pairwise channel check: ankle_mag_x an
- gold caption: Overall, the motion shows subtle energy, a loose rhythm, energy is spread across channels without a clear leader, middle-rate cadence, and short spikes.
- vote5 caption: NA
- gated caption: Overall, the motion shows subtle energy, a repeatable rhythm, energy is spread across channels without a clear leader, middle-rate cadence, and an even profile.

#### mhealth_test_000257

- gold index: 0; vote5 pred: 0; gated pred: 2; gate source: alternate; margin: 2.125
- gold fact: gold
- vote5 predicted fact: gold
- gated predicted fact: periodicity+burstiness
- evidence: RMS energy is 8.749, inside the middle reference band. The repeatability scores are autocorrelation 0.404 and FFT concentration 0.119, near the repeatability reference. Axis comparison: the largest per-axis energy is on the arm mag y. The dominant spectral peak is 0.391 Hz, below the lower reference band. Peak count is 4, with the profile outside the expected reference band for burstiness. Segment slopes move rising, then rising, then falling. Pairwise channel check: chest_acc_y and chest_acc_z 
- gold caption: Overall, the motion shows moderate energy, a loose rhythm, the largest per-axis energy is on the arm mag y, slow cadence, and an even profile.
- vote5 caption: Overall, the motion shows moderate energy, a loose rhythm, the largest per-axis energy is on the arm mag y, slow cadence, and an even profile.
- gated caption: Overall, the motion shows moderate energy, a repeatable rhythm, the largest per-axis energy is on the arm mag y, slow cadence, and short spikes.

### Representative regressions

信息不足：没有可展示样本。

## Aggregate Counts

| Dataset | Fixes | Regressions |
|---|---:|---:|
| ucihar | 20 | 1 |
| wisdm | 11 | 7 |
| mhealth | 6 | 0 |

Writing note: fixes are best framed as ambiguity-resolution cases; regressions should be used to motivate selective gating and bounded claims.
