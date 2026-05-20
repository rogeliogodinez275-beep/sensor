# SensorFact Detailed Abstract Draft

## Working Title

**SensorFact: A Verifiable Counterfactual Benchmark and Alignment Method for Wearable Sensor-Language Grounding**

Alternative title: **Do Sensor-Language Models Understand Motion or Match Templates? A Counterfactual Grounding Study**

## One-Sentence Main Claim

Closed-set human activity recognition (HAR) accuracy is insufficient to establish sensor-language grounding, because models may identify activity labels while failing to reject fluent but sensor-inconsistent descriptions.

中文主张：闭集人类活动识别（human activity recognition, HAR）准确率不能证明传感器-语言 grounding；模型可能识别活动标签，却无法判断自然语言描述是否真的被传感器证据支持。

## Detailed English Abstract

Wearable sensors offer a promising interface between physical motion and natural language systems, but the evaluation of sensor-language models remains too close to conventional human activity recognition. A model that correctly predicts labels such as walking, sitting, or running may still fail a more basic grounding test: whether a natural-language statement about the same sensor window is actually supported by the measured signal. This distinction matters for language-centered applications of wearable sensing, including question answering (QA), explanation generation, activity summarization, and assistive monitoring, where the model must reason about evidence rather than merely select a closed-set label.

We propose **SensorFact**, a verifiable counterfactual benchmark and alignment framework for wearable sensor-language grounding. SensorFact starts from a raw multichannel sensor window and converts it into a structured **Sensor Evidence Card** containing auditable motion facts such as intensity, periodicity, dominant axis, dominant frequency, burstiness, trend segments, and selected cross-channel relations. These evidence cards are not intended to replace learned representations; instead, they provide a controlled interface for building and evaluating language supervision. From each card, SensorFact constructs positive captions, paraphrases, minimally edited counterfactual captions, caption-selection examples, and label-obfuscated QA items. The counterfactual captions are designed to be fluent and plausible while changing only one evidence field, making them a direct test of whether a model attends to sensor-supported facts or relies on language templates and activity priors.

On top of this benchmark, we study **fact-contrastive alignment**, a low-compute method that keeps large language models frozen and trains lightweight sensor-language components to bring sensor representations closer to supported descriptions and farther from counterfactual descriptions. The intended comparison is not against HAR state of the art, but against evaluation shortcuts: a HAR classifier, a statistics-only prompt baseline, a trend-only alignment baseline inspired by sensor-language captioning pipelines, a rich-evidence alignment model without counterfactual negatives, and the full SensorFact counterfactual alignment model. The primary metrics are counterfactual rejection, caption selection, label-obfuscated QA, factual consistency, and robustness under held-out templates and paraphrases; HAR accuracy is retained only as an anchor.

The central empirical question is whether closed-set HAR performance can remain high while factual grounding remains weak. Our pilot and main experiments are designed to test the following signal: trend-only alignment should remain competitive on HAR-like recognition but degrade on counterfactual rejection and QA, while fact-level counterfactual supervision should improve grounding-oriented metrics without requiring full large-language-model fine-tuning. Result slots will be filled after the three-day UCI-HAR pilot and then revised after the full multi-dataset main-conference run: [RESULT-SLOT: M1 vs B2 counterfactual rejection], [RESULT-SLOT: QA / caption selection gains], [RESULT-SLOT: robustness under held-out templates].

If supported, SensorFact reframes wearable sensor-language modeling as a factual grounding problem rather than an activity-label classification problem. The contribution is therefore both diagnostic and methodological: it provides a benchmark for exposing when models match templates or labels, and it offers a lightweight alignment strategy for reducing this failure mode under realistic compute constraints.

## Submission-Style English Abstract

Wearable sensor-language models are often evaluated through human activity recognition (HAR), but closed-set label accuracy does not show whether a model can ground natural language in sensor evidence. We introduce **SensorFact**, a verifiable counterfactual benchmark for wearable sensor-language grounding. SensorFact converts each multichannel sensor window into a structured Sensor Evidence Card covering motion intensity, periodicity, dominant axis, frequency, burstiness, and trend facts, then derives positive captions, paraphrases, minimally edited counterfactual captions, caption-selection tasks, and label-obfuscated question answering (QA). We further propose fact-contrastive alignment, a low-compute method that keeps language models frozen while training lightweight sensor-language components to move sensor representations toward supported descriptions and away from fluent but sensor-inconsistent counterfactuals. Our evaluation compares HAR classifiers, statistics-only prompting, trend-only alignment, rich evidence alignment, and full SensorFact counterfactual alignment. The main metrics are counterfactual rejection, caption selection, QA, factual consistency, and robustness, with HAR retained only as an anchor. [RESULT-SLOT: M1 vs B2 counterfactual rejection]. [RESULT-SLOT: QA / caption selection gains]. These experiments test whether strong activity recognition can hide weak language grounding and whether fact-level counterfactual supervision can expose and reduce this failure mode.

**Keywords**: sensor-language grounding, wearable sensing, counterfactual evaluation, factual consistency, human activity recognition, multimodal alignment

## 中文详细摘要

可穿戴传感器为自然语言系统理解人体运动提供了重要入口，但现有评测往往仍停留在人类活动识别（human activity recognition, HAR）的闭集分类范式。一个模型即使能够正确预测 walking、sitting 或 running 等活动标签，也不一定真正理解传感器窗口中的运动证据。对于问答、解释生成、活动摘要和辅助监测等语言任务而言，更关键的问题不是模型能否选择一个类别，而是它能否判断一句自然语言描述是否真的被传感器信号支持。

本文提出 **SensorFact**，一个面向可穿戴传感器-语言 grounding 的可验证反事实 benchmark 与对齐框架。SensorFact 首先将多通道传感器窗口转换为结构化的 Sensor Evidence Card，记录运动强度、周期性、主导轴、主导频率、突发性、趋势片段以及部分跨通道关系等可审计事实。Evidence Card 并不是为了替代学习式表示，而是为语言监督和评测提供一个可控的事实接口。基于这些事实卡，SensorFact 自动构造真实描述、同义改写、最小事实改动的反事实描述、caption selection 样本和不暴露活动标签的问答任务。反事实描述在语言上保持流畅和合理，但只改变一个传感器事实，因此能够直接检验模型是否依赖真实信号，而不是依赖模板、标签先验或常识性猜测。

在方法上，本文研究 fact-contrastive alignment：在冻结大语言模型的低算力设定下，只训练轻量级传感器-语言组件，使传感器表示靠近被证据支持的描述，远离语言上合理但与信号矛盾的反事实描述。本文的目标不是刷新 HAR 分类准确率，而是系统比较不同监督信号对 factual grounding 的影响。实验设计包括 HAR 分类器、statistics-only prompt baseline、类似 SensorLLM 风格的 trend-only alignment、没有反事实负样本的 rich evidence alignment，以及完整的 SensorFact counterfactual alignment。主要评测指标包括 counterfactual rejection、caption selection、label-obfuscated question answering (QA)、事实一致性和模板鲁棒性；HAR 准确率只作为传统 anchor。

本文的核心经验问题是：闭集 HAR 表现是否会掩盖语言 grounding 的失败。三天 UCI-HAR pilot 和后续完整主会实验将检验以下信号：trend-only alignment 可能在 HAR 类任务上表现不差，却在反事实拒绝和 QA 上明显不足；加入事实级反事实监督后，SensorFact 应能提升 grounding 指标，同时不需要对大语言模型进行全参数微调。具体结果将在实验完成后回填：[RESULT-SLOT: M1 vs B2 counterfactual rejection]，[RESULT-SLOT: QA / caption selection gains]，[RESULT-SLOT: robustness under held-out templates]。

如果实验支持这一判断，SensorFact 将可穿戴传感器-语言建模重新定义为事实 grounding 问题，而不是单纯的活动标签分类问题。它的贡献包括两个层面：一方面提供一个能够暴露模板匹配和标签先验的诊断 benchmark；另一方面提供一种在现实算力约束下缓解该失败模式的轻量级对齐方法。

**中文关键词**：传感器-语言对齐，可验证事实，反事实评测，事实一致性，人类活动识别，多模态 grounding

## Result Slots

- `[RESULT-SLOT: M1 vs B2 counterfactual rejection]`: replace with the absolute and relative gain once `outputs/main_results_table.md` is produced on the intended dataset.
- `[RESULT-SLOT: QA / caption selection gains]`: replace with QA and caption-selection changes for B2, B3, and M1.
- `[RESULT-SLOT: robustness under held-out templates]`: replace after held-out template and paraphrase robustness are added to the main-conference run.

## Abstract Quality Checklist

- No fabricated experiment numbers are included.
- Human activity recognition (HAR) and question answering (QA) are defined on first use.
- The abstract explains why this is an EMNLP-style grounding and factuality paper rather than a standard sensing classification paper.
- The current version is conditional and must be revised after real server results are available.
