# Machine-learning detection and classification of stuttered speech

## Overview

State-of-the-art automatic detection and classification of stuttered speech using machine learning is now anchored by deep learning models—especially self-supervised and end-to-end architectures (e.g., wav2vec 2.0, Whisper)—that robustly identify and subtype stutter-like disfluencies at the event level, increasingly using large annotated datasets like SEP-28k and achieving high accuracy, including for co-occurring events and across languages [1,3,4,5,11,13,17,21,43,50].

---

### Core Findings and Technical Landscape

#### **1. Main Advances in ML-Based Stuttering Detection**

- **Self-Supervised and End-to-End Models Dominate**
    - **wav2vec 2.0, Whisper, and other SSL models** now provide the best-performing features for stuttering/disfluency detection, outperforming both classical ML and earlier deep models across benchmarks [1,3,11,13,14,18,22,26,35,37,49,50].
    - **End-to-end architectures** (CNN-BiLSTM-Attention, Transformers, YOLO-style) support both detection and classification jointly at fine temporal granularity [5,11,17,50].

- **Event-, Frame-, and Word-Level Detection**
    - Systems increasingly move beyond utterance/global detection to **frame-level** [2], **event-level** [5,6,11], or even **word-/token-level** ([1,11,17,50]) prediction, aligning better with clinical annotation norms.

- **Multi-Label and Co-Occurrence-Aware Systems**
    - Modern models reflect the reality that stuttering events can overlap: **multi-label architectures** are now standard, significantly improving detection of real-world speech patterns [3,7,11,12,13,17,50].

- **Disfluency Subtype Classification**
    - Reliable automatic distinction among core disfluency types—blocks, prolongations, sound/word repetitions, interjections—is achievable; state-of-the-art models report F1 scores above 0.8 for major categories, with improved identification of rarer co-occurrences [3,4,5,11,13,43,50].

#### **2. Data Foundations and Benchmarking**

- **Major Datasets**
    - **SEP-28k** [43] is the central English-language benchmark, used across nearly all recent research for model development and evaluation [1,2,3,4,5,11,13,14,16,18,19,26,31,32,35,37,40,41,55].
    - **FluencyBank, UCLASS, KSoF, AS-70** (Mandarin) and synthetic datasets (LibriStutter [5], VCTK-Stutter/TTS [17], LLM-Dys [10], Libri-Dys [15,21]) are widely used for additional validation and cross-corpus comparison.
    - **Synthetic data and augmentation** are now key for addressing class imbalance and supporting generalization to rare events or new languages [4,10,11,15,17,21,25,34,50,55].

- **Open Resources, Reproducibility**
    - A significant portion of SOTA methods release code and data, fostering quickly advancing and reproducible research [10,11,15,17,21,25,39,50].

#### **3. Technical and Practical Innovations**

- **Model Efficiency and Transfer**
    - Efficient adaptation strategies (e.g., selective layer freezing in Whisper [12,13,14]) allow effective fine-tuning even with minimal stutter-annotated data.
    - **Few-shot and small-data learning**: Self-supervised representations make robust stutter detection possible with very little labeled training data [9,22,26].

- **Data Augmentation and Class Imbalance**
    - Advanced augmentation (synthetic prolongations, time-stretching, additive noise) and loss weighting (focal loss, balanced cross-entropy) are routinely used to mitigate class imbalance and improve rare-event detection [4,8,10,25,34].

- **Joint and Hierarchical Tasks**
    - Increasing trend toward models outputting not just event labels but also boundaries, subtype, and even ASR transcript in unified frameworks [1,11,17,21,25,50].
    - Some models begin to integrate severity/fluency scoring, though robust event localization remains the primary focus.

- **Cross-Dataset and Multilingual Generalization**
    - Cross-corpus and cross-lingual evaluation demonstrate strong transferability for SSL-based systems when moderate adaptation data is available [3,7,11,15,21,55].

#### **4. Evaluation Practices**

- **Precision, recall, and F1** (often macro/micro and per-event): universal standard metrics.
- **Boundary tolerance** (±200–300 ms) is often applied in event detection evaluation, reflecting clinically meaningful accuracy [3,5,25].
- Increasing adoption of **cross-corpus/speaker splits** to expose possible dataset biases [3,7,31].

---

### **Representative Papers and Themes**

#### **Pivotal and Recent State-of-the-Art Works**

- **Frame/Word-Level & SSL:** [1,2,3,5,13,14,18,22,26,35,37,49]
- **Multi-Label/Token-Based/Object Detection:** [3,7,11,12,13,17,39,50]
- **Data Simulation/Augmentation:** [4,5,10,11,15,17,21,25,50,55]
- **Clinical/Assistive Application & Real-Time:** [11,16,19,46,56]

---

### **Current Challenges and Future Directions**

- **Improved Handling of Co-occurrence and Fine-Temporal Event Boundaries:** Despite progress, further improvements are needed on overlapping or swiftly sequential stuttering events and matching human raters on fine boundary decisions [3,7].
- **Broader Multilingual Data and Cross-lingual Transfer:** Expansion into under-represented languages is ongoing but still limited; advances in synthetic data and transfer strategies hold promise [11,15,21,55].
- **Clinical Integration and Real-World Deployment:** Emerging focus includes real-time inference on devices (“edge” or wearable solutions [56]) and the development of tools directly usable by clinicians.
- **Integration with ASR and Multimodal Sensing:** Increasing work on joint stuttering/ASR frameworks or fusing audio with visual/articulatory data [46].
- **Open Benchmarking and Reproducibility:** Wide availability of code-dataset-model pipelines is accelerating progress, but standardization of evaluation (including boundary criteria) requires consensus [10,11,21,50].

---

### **Summary Table of Highly Influential References**

| **Reference** | **Contribution** |
|---------------|------------------|
| [1] | SSL and word-level stutter detection, SOTA ablation |
| [3,7] | Multi-label (co-occurrence) detection, cross-lingual/corpus |
| [4] | Data aug., class balance, contextual deep learning |
| [5] | End-to-end (CNN-BiLSTM-Attn), event-level SOTA, LibriStutter |
| [11,17,21,25,50] | Token/region-based detection, large synthetic datasets, unified frameworks, benchmarking |
| [12,13,14] | Whisper encoder, parameter-efficient multi-label stutter detection |
| [43] | SEP-28k, foundational event-level SLD dataset |
| [55] | Mandarin SLD benchmark and analysis |

---

### **Conclusion for Researchers**

Automatic ML-based stuttering detection is a highly active research area showing rapid empirical gains, methodological convergence toward self-supervised, multi-label, and end-to-end systems, and a strong culture of reproducibility and open data. The field is well-positioned for clinical impact, with strong support for both core detection tasks and emerging directions in real-time and multilingual settings.

## Categories

### Comprehensive Comparison of Machine Learning Methods for Automatic Stuttered Speech Detection and Classification

Below, the most relevant papers ([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56], with main focus on those directly addressing automatic ML-based detection/classification of stuttered speech) are compared across several axes of core interest to expert researchers. Detailed summary tables and notes highlight methodological choices, empirical advances, task granularity, dataset use, and unique contributions.

---

#### 1. **Key Comparison Dimensions**

- **Disfluency Target & Task Granularity** (Event-level? Frame-level? Word-level? Multi-label vs. Multi-class vs. Binary)
- **Modelling Approach** (Classical ML, CNN, LSTM/GRU, Transformers, Self-Supervised Learning, Multi-task, Multi-modal, etc.)
- **Feature Representation** (Hand-engineered, MFCC/PLP/Prosodic, CNN-learned, Pretrained SSL [wav2vec 2.0, Whisper], etc.)
- **Disfluency Subtypes Handled** (Repetition, Prolongation, Block, Interjection, etc.; multi-label/overlap support)
- **Evaluation Dataset(s)** (e.g. SEP-28k, UCLASS, FluencyBank, KSoF, AS-70, etc.)
- **Notable Technical Innovations** (e.g., data augmentation, class-balancing, multi-tasking, alignment strategies, zero-shot, real-time deployment)
- **Public Code/Data**
- **Reported Metrics/Performance**

---

### Summary Comparison Table

| **Reference** | **Disfluency Target** | **Task Granularity** | **Model/ML Approach** | **Features** | **Disfluency Types** | **Main Dataset(s)** | **Innovations/Comments** | **Notable Results** |
|---------------|-----------------------|----------------------|-----------------------|--------------|----------------------|---------------------|--------------------------|---------------------|
| [1] | Stuttered speech | Word-level | Self-supervised (wav2vec 2.0, Hubert) | SSL Repres. | Multi-class | Custom word-annotated corpus | Word-level detection, ablation of SSL tuning | SOTA word-level F1 |
| [2] | Stuttered speech | Frame-level | Feedforward DNN | Acoustic | Core stutter types | SEP-28k | First frame-level DNN, probabilistic outputs | Used as input for downstream ASR |
| [3,7] | Stuttered speech | Event, multi-label | wav2vec 2.0 + attn. | SSL | Multiple, overlapped | SEP-28k(-E), KSoF, FluencyBank | Multi-label, cross-corpus/lingual | SOTA multi-label performance|
| [4] | Stuttered speech | Event-level | StutterNet (SE-ResNet+BiLSTM) | Acoustic + aug | 5 core types | SEP-28k | Data augmentation, class-balancing, multi-context | F1 improvement, robustness gains |
| [5] | Stuttered speech | Event-level | CNN(SE)-BiLSTM-Attn (FluentNet) | Acoustic (log-Mel, etc) | 5 main | UCLASS, LibriStutter | End-to-end arch + LibriStutter sim. corpus | SOTA UCLASS event F1 |
| [6] | Stuttered speech | Event (multi-class) | ResNet-BiLSTM | Acoustic | Multiple | UCLASS | Emphasized spectro-temporal patterns | Outperforms prior state-of-art |
| [8] | Stuttered speech | Event-level | Multi-task/Adv. on top [5] | Acoustic | 5+ classes | SEP-28k | MTL, Adv. learning to improve rare events | Gains in recall for rare types |
| [9] | Dysfluency (5 types) | Event-level | Siamese network pretrain | SSL, acoustic | 5 types | Multiple small sets | Small data pretraining pipeline | Improves with only mins of labeled data |
| [10,11,15,17,21,25,39,50] | Stuttered/dysfluent | Event + word, token | End-to-end (YOLO, Whisper, LLM) | SSL, TTS, Artic.| Multi-label, co-occurrence | Sim. + real corpora | Region-wise detection, token-as-class, large sim. datasets | SOTA token- and time-based performance; open-sourced corpora |
| [12,13,14] | Stuttered speech | Event-level | Whisper encoder | SSL | Multi-label | SEP-28k, FluencyBank | Layer freezing for efficient finetuning | F1 ≈ 0.8–0.85, param reduction |
| [16,23,32] | Stuttered speech | Event, type classification | TDNN, LSTM/Attn, Transformer | Mel-spectrogram | 5+ types | SEP-28k, FluencyBank | Attention, segment duration optim., TranStutter arch | Up to 83% F1, high overall acc. |
| [18,22,26,35,37,49] | Stuttered speech | Event-level | ECAPA-TDNN, Wav2vec2.0, SVM/NN | SSL, MFCC | Single/Multi-class | SEP-28k, KSoF, UCLASS, FluencyBank | Embedding concatenation, data distillation | +10–30% UAR/F1 over baselines |
| [27,34,36,41,45,51,53] | Stuttered speech | Event-level (mostly 4–5 types) | BiLSTM, CNN, SVM, Hybrid | MFCC, SDC, log-Mel | Core types | UCLASS, FluencyBank | Feature selection, oversampling, Kalman filters | Acc: 65–98% depending on type/arch. |
| [28,29,30,33,40,47,54] | Stuttering/Broad disfluency | Event or global | CNN, ResNet, Hybrid, SVM, BERT | log-Mel, MFCC, SSL | Core types, non-SLD | SEP-28k, FluencyBank | Multi-method study, hybrid arch | Variable, F1 up to 0.93 for general type |

(Note: Only a selection of most-cited and representative works included; additional references as per discussions below.)

---

### **Detailed Comparative Notes**

#### **A. Task Granularity and Disfluency Subtype Coverage**

- **Frame- and event-level detection with subtype classification** dominates the field ([2,3,4,5,6,8,9,10,16,23,32]). SOTA works [5,17,3,11,50] support both segment localization and type (repetition, prolongation, block...), often in multi-label settings to reflect co-occurrence.
- **Multi-label detection** (rather than forced multi-class) is now standard, reflecting the real-world overlap of SLDs ([3,7,11,12,13,17,50]).
- **Word-level and token-level annotation/detection** is rare but growing, see [1,50]. This better fits clinical annotation practice.

#### **B. Feature and Model Trends**

- **Self-supervised representations** ([SSL: wav2vec2.0, Hubert, Whisper]): Clear trend toward leveraging these for fine-tuning or as fixed feature extractors ([1,3,5,13,14,17,18,22,26,29,35,37,49,50]).
  - These substantially outperform classical hand-engineered features, especially for low-resource or rare SLDs ([3,5,22,26,35,37]).
- **End-to-end models:** Mix of CNN-BiLSTM-attention ([5,6,4]), pure Transformer ([32]), and hybrid (e.g., Conformer-BiLSTM [24]) dominate over SVM-GMM or older methods ([52]).
- **Innovations in network structure** (e.g., attention for long-context, lightweight Siamese/pretrain [9], multi-modal ([46])) and data inclusion (augmentation, imbalance handling, simulated corpora [4,25,50]).

#### **C. Disfluency Types and Labeling**

- **Core SLD types:** Block, prolongation, (sound/word) repetition, interjection – nearly all major works support these labels ([5,6,43], etc.).
- **Co-occurrence/multi-label:** Important challenge; multi-label set-ups ([3,7,11,12,13,17]) outperform simple one-at-a-time event systems for real recordings.

#### **D. Training/Evaluation Data**

- **SEP-28k** ([43]) is the current de facto English benchmark: Large, diverse, with event and time-aligned annotation. Seen in almost every recent method ([1,2,3,4,12,13,14,16,18,19,22,23,26,29,31,32,35,37,40,41,43,45,46]).
- **FluencyBank, UCLASS, KSoF:** Also frequent for cross-dataset generalisation ([5,6,41]).
- **Simulated/augmented datasets**: LibriStutter ([5]), VCTK-Stutter/TTS ([17]), VCTK++ ([25]), LLM-Dys ([10]), Libri-Dys ([15,21]), AS-70 for Mandarin ([24,55]) are growing for improved diversity and multilingual work.

#### **E. Empirical Advances and Notable Technical Contributions**

- **Multi-task/adversarial learning** ([8,4,24]) and **data augmentation** (noise, time-stretching, synthetic SLDs) provide measurable gains in rare event detection ([4,8,10,25,34]).
- **Efficient large-model adaptation**: Layer freezing to enable high F1 with much fewer trainable parameters – Whisper/SSL approaches ([12,13,14]).
- **Zero-shot/cross-lingual generalization**: Demonstrated to work with token/region detectors ([3,11,39,50,55]) and with simulated data + adapters.
- **Multi-modal architectures**: Early step ([46]), combining acoustic and visual information.
- **Token-based vs. region-based detection**: Empirical comparison in [50] establishes trade-offs in granularity and clinical fit; open-source benchmarks enable direct comparison.
- **Real-time and deployment studies**: On-device inference (microcontrollers) is now demonstrated and benchmarked ([56]), supporting future clinical and therapeutic integration.

#### **F. Metrics and Clinical Evaluation Considerations**

- **Precision, recall, F1:** Main benchmarks; macro or micro F1 reported for both general and per-type SLD detection.
- **Boundary tolerance in scoring:** Most methods incorporate allowed onset/offset jitter, aligning with clinical expectations ([3,5,25]; see notes in background on window tolerances).
- **Correlation with human/SSI-4 ratings:** Sought but rare; when done, Spearman ρ sometimes preferred for severity regression ([5], background above).
- **Cross-corpus/cross-speaker evaluation:** Now increasingly emphasized ([3,7,31,13,50]), identifying overfitting to dominant speakers or domains in SEP-28k.

#### **G. Public Resources and Reproducibility**

- **Open source code/data:** Many of the most recent works ([10,11,17,21,25,50]) provide code and synthetic/cross-lingual datasets for benchmarking and generalization experiments.

---

### **Representative Reference Groupings By Innovation**

- **Self-Supervised Learning/Pretrained Model Approaches:** [1,3,5,11,13,14,17,18,22,26,29,35,37,49,50]
- **Multi-label/Co-occurrence Focus:** [3,7,11,12,13,17,50]
- **Token/Word-Level Detection:** [1,11,17,25,50]
- **Data Augmentation/Simulation:** [4,10,17,25,34,50,55]
- **Multilingual/Cross-lingual Generalization:** [3,11,15,21,55]
- **Efficiency/Deployability:** [9,12,13,14,56]
- **Evaluation Metric Analysis:** [3,5,31,43,42]
- **Clinical/Severity Correlation:** [5,51] (less common than event detection)

---

### Key Takeaways for the Expert

- **SSL Pretraining is now standard practice, with Wav2vec2.0, Hubert, Whisper fine-tuned for stuttering (or more general dysfluency) event detection, leading to large empirical improvements in recall/F1 for both major and rare event types** ([1,3,5,11,13,14,26,29,35,37,49,50]).
- **Co-occurrence of disfluency types is prevalent; therefore, multi-label architectures and evaluation are becoming best practice**, improving detection of real-world SLDs and providing better fit to clinical annotation philosophies ([3,7,12,17,50]).
- **Public benchmarks (SEP-28k, FluencyBank) dominate model development and comparison**, but **progress is driven by advances in data augmentation, synthetic corpus creation, and alignment methods** ([4,10,11,15,17,21,25,34,50,55]).
- **Frameworks are moving beyond classification, towards token/word-level detection and even joint stuttering-ASR models, supporting more integrated downstream clinical applications** ([1,11,17,21,25,50]).
- **Deployment on low-power hardware and real-time systems for clinical settings are starting to be addressed** ([56]).
- **Evaluation rigor now includes boundary tolerance, cross-corpus and cross-lingual transfer, and open-sourcing to foster reproducibility and broader adoption** ([3,7,21,31,50,55]).

---

### **References**

See summary table and groupings above for explicit linkages. For details on each specific approach and dataset, referenced numerals refer to the comprehensive list provided in your initial query. Major survey and benchmarking papers include [1,3,4,5,11,13,15,17,25,43,50,55].

---

**This comparative analysis provides a state-of-the-art snapshot, highlighting both consensus practices and emerging technical directions in ML-based stuttered speech detection/classification.**

## Timeline

### Historical Development of Ideas

#### Pre-Deep Learning and Early ML Methods (Pre-2018)
- **Traditional Machine Learning:** Initial approaches focused on engineered acoustic features (e.g., MFCCs, prosodic features) combined with classical classifiers such as SVMs, GMMs, and random forests for disfluency/stuttering event detection [52].
- **Small, Limited Datasets:** Earliest studies typically used relatively small, homogenous datasets (such as UCLASS) with limited annotation granularity and speaker diversity [27,52].

#### Emergence of Deep Learning and Acoustic-Only Models (2018–2021)
- **Deep CNN and LSTM Architectures:** The field shifted to deep learning models, particularly combinations of convolutional networks and recurrent networks (e.g., Bidirectional LSTM, TDNN), which could capture temporal and spectral patterns of disfluencies [6,19,27,41,45].
- **Frame/Event-level Modelling:** Growing emphasis on fine-grained, frame- or event-level detection (not just utterance-level), motivated by the clinical importance of precise annotation [2,38].
- **Notable Datasets Emerge:** SEP-28k [43], UCLASS, and FluencyBank began seeing widespread adoption as benchmark datasets, enabling better comparison across studies [5,6,19,22,28].
- **First "End-to-End" Models:** Fully end-to-end neural models, such as FluentNet [5], overtook pipeline architectures reliant on external ASR or feature extraction modules.

#### Explosion of Dataset Scale, Multilinguality & Model Complexity (2021–2023)
- **Large-Scale Benchmarking:** SEP-28k [43] provided unprecedented scale (~28k clips with specific event-type labels), enabling substantial advances in both deep learning performance and reproducibility [2,4,22,30,31].
- **Sophisticated Neural Architectures:** Introduction of architectures with attention [5], ResNets [6,30], Conformers [24], Transformer-based models [32], and sequence-to-sequence models for time and token prediction [17,50].
- **Pre-trained Representations:** Major surge in use of large pre-trained models for feature extraction and transfer learning:
   - **Wav2Vec 2.0:** Became a dominant backbone for nearly all SOTA models [3,4,7,12,13,14,18,22,26,28,29,31,35,37].
   - **ECAPA-TDNN:** Also explored in several works for robust speaker- and event-level embeddings [22,26].
   - Querying which fine-tuning and feature selection strategies offered the best transfer [12,13,14,22,26,35,37].

- **Data Augmentation & Class Imbalance:** Recognized as critical challenges; numerous methods (oversampling, synthetic data, focused loss functions) were systematically studied [4,34,36].
- **Multilingual & Cross-corpus Generalization:** Increasing focus on cross-lingual and cross-corpus robustness (English, German, Mandarin) and transfer learning between corpora [3,7,11,21,55].

#### Advanced Frameworks, Multi-task, and Foundation Models (2023–2025)
- **Multi-label & Co-occurrence Modelling:** Recognition that stuttering types co-occur; shift toward multi-label classification (joint event type prediction) rather than isolated binary/multiclass tasks [3,7,12,23].
- **Self-Supervised and Few-shot Learning:** Emphasis on self-supervised learning, with models pre-trained on large volumes of unlabeled speech and fine-tuned with minimal stuttered data, achieving surprising gains [1,9,22,26].
- **Synthetic Data & Simulators:** Growth in the use of large-scale, realistic synthetic dysfluency corpora to mitigate data scarcity (LibriStutter, VCTK-Stutter/TTS, LLM-Dys, Libri-Dys) [5,10,11,15,17,21,25,39,50,55].
- **Token-Based and Detection Paradigms:** New frameworks recast dysfluency detection as either token-based sequence modeling [50] or region-based object detection in time (YOLO-Stutter) [17,11,39,50].
- **Multi-modal and Large Language Models:** Recent proposals integrate visual modalities (multi-modal stuttering detection) [46] or combine acoustic and textual features with LLMs for enhanced detection [20,21,39,50].

### Key Research Trends and Clusters

#### Data and Benchmarking
- **SEP-28k and FluencyBank:** Central to almost every major recent study, facilitating cross-comparisons and robust evaluation [4,5,6,19,22,43]. Their broad adoption has driven reproducibility and accelerated model development.
- **Synthetic Data Generators:** Open-source release of synthetic dysfluency datasets (LibriStutter, VCTK, LLM-Dys, Libri-Dys) has dramatically broadened training options and allowed exploration of model transfer to rare stutter event types [10,11,15,17,21,25,50].

#### Model Development Trends
- **From Manual Features to Universal Speech Representations:** Transition from hand-crafted features (MFCC, SDC, prosody) to deep-learned representations—especially via transfer/self-supervised learning (wav2vec 2.0, ECAPA-TDNN, Whisper, WavLM) [4,12,13,14,18,26,31,35,37].
- **Hierarchical and Multi-task Learning:** Incorporation of multi-task objectives and simultaneous event-type, time-boundary, and severity prediction [3,4,11,15,24,25,32].
- **Emphasis on Cross-lingual/Domain Generalization:** Shift toward robustness across languages, microphones, and spontaneous vs. read speech [3,7,11,21,55].

#### Analysis and Evaluation
- **Fine-grained Evaluation Metrics:** Adoption of event-wise (type-specific) precision/recall, segment-level accuracy, and matching clinical relevance in boundary error tolerance [2,4,5,15,43].
- **Real-time and Wearable Solutions:** Emerging focus on latency and embedded inference for clinical deployment, including wearable applications [36,56].

### Key Collaborators & Influential Authors

#### S. A. Sheikh & Slim Ouni Group
- **Sustained output on stuttering detection leveraging both classical and deep learning methods, with special focus on class-imbalance and robustness:** [4,8,19,22,26]

#### Gopala Anumanchipalli Lab (Berkeley Speech Group)
- **Major drivers of innovation for large-scale simulated datasets and end-to-end, scalable modeling (SSDM, UDM, YOLO-Stutter, Stutter-Solver, LLM-Dys, Dysfluent-WFST) [10,11,15,17,21,25,39,50]**
- **Consistent focus on time-accurate modeling, dataset curation, open-source toolkits, and evaluation of clinical relevance.**

#### Tedd Kourkounakis & Arash Etemad Team
- **Notable for development of the FluentNet and early deep CNN/BiLSTM baselines, as well as dataset creation (LibriStutter) [5,6,19]**
- **Pioneered use of attention in deep acoustic-only stuttering models.**

#### S. Bayerl & K. Riedhammer
- **Research on cross-dataset/multilingual transfer and multi-label detection [3,7,31], highlighting corpus and evaluation pitfalls.**

#### Frequent Use and Citation of Key Models
- **wav2vec 2.0:** Central to almost all SOTA detection studies (either as feature extractor or fine-tuned model), reflecting a clear convergence toward large self-supervised representations as foundational technology.
- **Whisper and WavLM:** The latest generation of universal speech encoders being adapted for stuttering detection and multi-task fluency/disfluency modeling [12,13,14,39].

### Significant Developments and Milestones

- **Release of SEP-28k [43] and follow-up annotated speech corpora [10,17,21]:** Marked the transition to large, multi-speaker, open datasets enabling robust model evaluation and generalization analysis.
- **First End-to-End Deep Models for Event-Level Detection [5,6]:** Established baseline architectures for SLD detection beyond utterance-level scoring.
- **Self-supervised and Pre-trained Representations (2021–):** Dramatically improved F1 scores and cross-corpus transfer, enabling robust stutter detection even with limited stutter-annotated data [1,3,4,7,12,13,14,22,26,35].
- **Frameworks for Multi-label and Region-wise Detection (2022–2024):** Addressed real clinical structure of co-occurring disfluencies and time-accurate detection [3,7,12,23,17,50].
- **Open Sourcing of Models and Data (2023–2025):** The release of reproducible code, benchmark datasets, and model weights has accelerated progress and enabled broad participation [10,11,15,21,39,50].
- **Mandarin and Multilingual Benchmarks [11,21,55]:** Helped expand research scope and test generalization beyond English.

### Patterns, Significance, and Future Directions

- **Rapid Progress via Transfer Learning:** The consistent superiority of transfer/self-supervised models (wav2vec 2.0, Whisper, WavLM) highlights the field’s shift from reliance on domain-specific feature engineering to adaptation of universal representation learners, mirroring broader speech technology trends.
- **From Pure Detection to Rich Transcription and Diagnosis:** Increasingly, models output not only event classes but also precise boundaries and even severity gradations, bringing systems closer to clinical utility [2,11,15,21,39].
- **Synthetic Speech & Simulation:** Synthetic corpora enable model training and benchmarking beyond the (still scarce) real clinical data, allowing for data balance in rare SLD classes and language coverage [5,10,11,15,17,21,25,39,50,55].
- **Open, Reproducible Science:** Widespread sharing of code/datasets is catalyzing rapid iteration, broader validation, and reducing barriers to field entry.
- **Potential for Multimodal Expansion:** Very recent work exploring multi-modal and sensor-fusion approaches (audio+video) and wearable deployment suggests exciting future directions [46,56].

---

**In summary:**
The automatic detection and classification of stuttered speech has evolved from small-feature engineering studies to a mature, rapidly advancing discipline dominated by deep neural architectures and transfer/self-supervised learning. Progress has been driven primarily by shared, large-scale datasets and a handful of influential collaborative teams, especially around self-supervised learning, synthetic data, and universal speech representations. The next frontiers likely include multimodal sensing, further scaling via synthetic/augmented data, and embedding within real-world clinical and assistive applications.

## Foundational Work

### Which papers form the foundational references on this topic?

The below table shows the resources that are most often cited by the relevant papers on this topic. This is measured by the **reference rate**, which is the fraction of relevant papers that cite a resource. Use this table to determine the most important core papers to be familiar with if you want to deeply understand this topic. Some of these core papers may not be directly relevant to the topic, but provide important context.

| Ref. | Reference Rate | Topic Match | Title | Authors | Journal | Year | Total Citations | Cited By These Relevant Papers |
|---|---|---|---|---|---|---|---|---|
| [43] | 0.65 | 100% | SEP-28k: A Dataset for Stuttering Event Detection from Podcasts with People Who Stutter | Colin S. Lea, ..., and Jeffrey P. Bigham | ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2021 | 103 | [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 22, 23, 26, 29, 31, 32, 35, 37, 40, 41, 42, 44, 45, 46, 47, 51, 55, 57, 62, 65, 66, 74, 78, 80, 82, 84, 87, 90, 91, 92, 94, 96, 97, 99, 100, 104, 107, 108] |
| [6] | 0.44 | 100% | Detecting Multiple Speech Disfluencies Using a Deep Residual Network with Bidirectional Long Short-Term Memory | Tedd Kourkounakis, ..., and A. Etemad | ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2019 | 78 | [3, 4, 7, 8, 9, 11, 13, 14, 15, 19, 22, 26, 29, 32, 35, 37, 40, 41, 43, 44, 45, 46, 47, 48, 50, 51, 55, 57, 62, 65, 77, 80, 83, 84, 86, 88, 90, 99, 100] |
| [19] | 0.34 | 100% | StutterNet: Stuttering Detection Using Time Delay Neural Network | S. A. Sheikh, ..., and Slim Ouni | 2021 29th European Signal Processing Conference (EUSIPCO) | 2021 | 42 | [1, 2, 4, 8, 12, 14, 16, 22, 26, 29, 32, 35, 37, 40, 44, 45, 46, 57, 65, 70, 80, 87, 90, 96, 97, 99] |
| [17] | 0.23 | 100% | YOLO-Stutter: End-to-end Region-Wise Speech Dysfluency Detection | Xuanru Zhou, ..., and G. Anumanchipalli | Interspeech | 2024 | 9 | [10, 11, 15, 21, 39, 50, 152, 156, 157, 158, 160, 162, 164, 165, 166, 167, 169] |
| [112] | 0.22 | 13% | Towards Hierarchical Spoken Language Disfluency Modeling | Jiachen Lian and G. Anumanchipalli | ArXiv | 2024 | 12 | [1, 10, 11, 15, 17, 21, 39, 50] |
| [49] | 0.20 | 100% | Dysfluency Classification in Stuttered Speech Using Deep Learning for Real-Time Applications | Mélanie Jouaiti and K. Dautenhahn | ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2022 | 25 | [3, 4, 7, 8, 11, 15, 21, 22, 25, 26, 37, 40, 45, 50, 62, 161] |
| [4] | 0.20 | 100% | Advancing Stuttering Detection via Data Augmentation, Class-Balanced Loss and Multi-Contextual Deep Learning | S. A. Sheikh, ..., and Slim Ouni | IEEE Journal of Biomedical and Health Informatics | 2023 | 18 | [13, 22, 30, 36, 40, 47, 55, 62, 78, 96, 99] |
| [123] | 0.15 | 1% | Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0 | S. Bayerl, ..., and K. Riedhammer | ArXiv | 2022 | 51 | [1, 3, 7, 12, 14, 20, 29, 39, 46, 50, 108] |
| [141] | 0.15 | Not measured | Fluency Bank: A new resource for fluency research and practice. | N. Bernstein Ratner and B. MacWhinney | Journal of fluency disorders | 2018 | 99 | [1, 3, 4, 7, 12, 14, 20, 29, 32, 43, 45, 46, 51, 60, 104] |
| [50] | 0.15 | 100% | Time and Tokens: Benchmarking End-to-End Speech Dysfluency Detection | Xuanru Zhou, ..., and G. Anumanchipalli | ArXiv | 2024 | 6 | [10, 15, 21, 39, 152, 156, 157, 158, 160] |
| [3] | 0.15 | 100% | A Stutter Seldom Comes Alone - Cross-Corpus Stuttering Detection as a Multi-label Problem | S. Bayerl, ..., and K. Riedhammer | ArXiv | 2023 | 12 | [1, 12, 15, 20, 21, 55, 78] |
| [142] | 0.14 | Not measured | Robust Speech Recognition via Large-Scale Weak Supervision | Alec Radford, ..., and I. Sutskever | N/A | 2022 | 3770 | [1, 11, 12, 14, 17, 20, 50, 104, 108] |
| [25] | 0.14 | 100% | Unconstrained Dysfluency Modeling for Dysfluent Speech Transcription and Detection | Jiachen Lian, ..., and G. Anumanchipalli | 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU) | 2023 | 16 | [1, 11, 17, 39, 50] |
| [143] | 0.14 | Not measured | The University College London Archive of Stuttered Speech (UCLASS). | P. Howell, ..., and J. Bartrip | Journal of speech, language, and hearing research : JSLHR | 2009 | 99 | [1, 3, 6, 7, 11, 19, 30, 32, 45, 51, 58, 60, 75, 81, 83] |
| [20] | 0.14 | 100% | Large Language Models for Dysfluency Detection in Stuttered Speech | Dominik Wagner, ..., and T. Bocklet | ArXiv | 2024 | 6 | [10, 12, 16, 21, 39] |
| [11] | 0.14 | 100% | Stutter-Solver: End-To-End Multi-Lingual Dysfluency Detection | Xuanru Zhou, ..., and G. Anumanchipalli | 2024 IEEE Spoken Language Technology Workshop (SLT) | 2024 | 6 | [10, 15, 21, 39, 152, 156, 157, 158, 160, 162, 164, 165, 167] |
| [2] | 0.14 | 100% | Frame-Level Stutter Detection | John Harvill, ..., and C. Yoo | N/A | 2022 | 13 | [1, 3, 7, 15, 17, 21, 25, 57] |
| [144] | 0.13 | Not measured | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Alexei Baevski, ..., and Michael Auli | ArXiv | 2020 | 5869 | [1, 3, 7, 12, 14, 20, 26, 29, 32, 39, 104, 108] |
| [84] | 0.13 | 98% | KSoF: The Kassel State of Fluency Dataset – A Therapy Centered Dataset of Stuttering | S. Bayerl, ..., and K. Riedhammer | ArXiv | 2022 | 36 | [1, 3, 7, 12, 14, 20, 29, 32, 74] |
| [145] | 0.13 | Not measured | Machine Learning for Stuttering Identification: Review, Challenges & Future Directions | S. A. Sheikh, ..., and Slim Ouni | ArXiv | 2021 | 50 | [4, 14, 20, 26, 29, 30, 32, 60, 87, 104] |

## Adjacent Work

### Which papers cite the same foundational papers as relevant papers?

Use this table to discover related papers on adjacent topics, to gain a broader understanding of the field and help generate ideas for useful new research directions.

| Ref. | Adjacency score | Topic Match | Title | Authors | Journal | Year | Total Citations | References These Foundational Papers |
|---|---|---|---|---|---|---|---|---|
| [148] | 7.37 | Not measured | SSDM: Scalable Speech Dysfluency Modeling | No author found | Journal Not Provided | N/A | 0 | [2, 3, 6, 11, 17, 49, 50, 76, 90] |
| [149] | 6.41 | Not measured | A novel attention model across heterogeneous features for stuttering event detection | Abedal-Kareem Al-Banna, ..., and E. Edirisinghe | Expert Syst. Appl. | 2024 | 9 | [6, 19, 29, 33, 35, 43, 45, 49] |
| [150] | 5.37 | Not measured | SSDM: Scalable Speech Dysfluency Modeling | No author found | Journal Not Provided | N/A | 0 | [2, 3, 6, 17, 49, 76, 90] |
| [145] | 5.33 | Not measured | Machine Learning for Stuttering Identification: Review, Challenges & Future Directions | S. A. Sheikh, ..., and Slim Ouni | ArXiv | 2021 | 50 | [6, 19, 26, 43, 49, 64, 81, 83] |
| [151] | 4.13 | Not measured | FGCL: Fine-Grained Contrastive Learning For Mandarin Stuttering Event Detection | Han Jiang, ..., and Jihua Zhu | 2024 IEEE Spoken Language Technology Workshop (SLT) | 2024 | 0 | [19, 35, 43, 45, 49, 83] |
| [153] | 3.53 | Not measured | Seamless Dysfluent Speech Text Alignment for Disordered Speech Analysis | Zongli Ye, ..., and G. Anumanchipalli | ArXiv | 2025 | 0 | [11, 17, 39, 50] |
| [154] | 3.46 | Not measured | A CNN-based Stutter Detection Using MFCC Features with Binary Cross-Entropy Loss Function | Rohit Waddar, ..., and S. Budihal | 2024 IEEE International Conference on Contemporary Computing and Communications (InC4) | 2024 | 1 | [6, 19, 23, 41, 43, 45, 95] |
| [155] | 3.10 | Not measured | The Impact of Stuttering Event Representation on Detection Performance | Abedal-Kareem Al-Banna, ..., and Saif B Abuaisheh | 2024 2nd International Conference on Cyber Resilience (ICCR) | 2024 | 0 | [35, 43, 45, 49] |
| [159] | 2.46 | 18% | Towards Accurate Phonetic Error Detection Through Phoneme Similarity Modeling | Xuanru Zhou, ..., and G. Anumanchipalli | ArXiv | 2025 | 2 | [11, 17, 50] |
| [125] | 2.03 | 1% | “The World is Designed for Fluent People”: Benefits and Challenges of Videoconferencing Technologies for People Who Stutter | Shaomei Wu | Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems | 2023 | 16 | [6, 19, 43, 76] |
| [114] | 1.92 | 8% | Enhancing Stuttering Detection: A Syllable-Level Stutter Dataset | Vamshiraghusimha Narasinga, ..., and A. Vuppala | 2024 International Conference on Signal Processing and Communications (SPCOM) | 2024 | 1 | [3, 6, 19, 43] |
| [157] | 1.83 | 39% | EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Spoken Dialogue Systems | Jingwen Liu, ..., and G. Anumanchipalli | ArXiv | 2025 | 0 | [11, 17, 50] |
| [138] | 1.54 | 0% | Detection and Classification of Stuttering from Text | Fateme Moghimi and Mehran Yazdi | 2024 11th International Symposium on Telecommunications (IST) | 2024 | 1 | [6, 43, 61, 95] |
| [123] | 1.53 | 1% | Detecting Dysfluencies in Stuttering Therapy Using wav2vec 2.0 | S. Bayerl, ..., and K. Riedhammer | ArXiv | 2022 | 51 | [6, 19, 43] |
| [99] | 1.42 | 75% | Computational Intelligence-Based Stuttering Detection: A Systematic Review | Raghad Alnashwan, ..., and Waleed M. Al-nuwaiser | Diagnostics | 2023 | 5 | [4, 6, 19, 26, 30, 41, 43, 45, 71] |
| [152] | 0.97 | 76% | Automatic Detection of Articulatory-Based Disfluencies in Primary Progressive Aphasia | Jiachen Lian, ..., and G. Anumanchipalli | IEEE Journal of Selected Topics in Signal Processing | 2025 | 7 | [11, 17, 50, 90] |
| [121] | 0.76 | 2% | Efficient Recognition and Classification of Stuttered Word from Speech Signal using Deep Learning Technique | Kalpana Murugan, ..., and Sai Subhash Donthu | 2022 IEEE World Conference on Applied Intelligence and Computing (AIC) | 2022 | 2 | [6, 95] |
| [162] | 0.75 | 63% | K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function | Shuhe Li, ..., and G. Anumanchipalli | ArXiv | 2025 | 1 | [11, 17] |
| [104] | 0.73 | 44% | Inclusive ASR for Disfluent Speech: Cascaded Large-Scale Self-Supervised Learning with Targeted Fine-Tuning and Data Augmentation | Dena F. Mujtaba, ..., and Jia Bin | ArXiv | 2024 | 1 | [43, 83, 141, 145] |
| [160] | 0.73 | 76% | AV-EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Omni-modal LLMS with Audio-visual Cues | Krish Patel, ..., and G. Anumanchipalli | ArXiv | 2025 | 0 | [11, 17, 50] |