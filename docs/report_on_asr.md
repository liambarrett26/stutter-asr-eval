# Innovative cross-modal sequence learning for speech-text alignment and disentangled representation

## Overview

The search yielded strong advancements in cross-modal fine-grained sequence representation learning for minimally-supervised tasks, with key approaches like VQ-CTAP [1], CTAP [2], and VQMIVC [3] developing robust token-acoustic contrastive learning, semantic-paralinguistic disentanglement, and high-compression vector quantization, showcasing direct applicability to ASR, TTS, and voice conversion.

---

### Key Findings:

#### **1. Frame-Level Speech-Text Alignment Using Token-Acoustic Contrastive Learning**
- **VQ-CTAP [1] and CTAP [2]**:
  - Achieve fine-grained token-frame alignment by leveraging contrastive pretraining with InfoNCE variants.
  - Introduce vector quantization (VQ) to reduce redundancy, enabling scalable alignment across minimally-supervised tasks (e.g., ASR, TTS, VC).
  - Use large datasets like 210k phoneme-audio pairs for robust multimodal alignment with plug-and-play downstream adaptability.

#### **2. Semantic-Paralinguistic Disentanglement**
- **VQMIVC [3]**:
  - Employs VQ and mutual information minimization to disentangle speaker identity, prosody, and linguistic content in voice conversion tasks with state-of-the-art naturalness and intelligibility.
- **Adversarial Techniques [7,15]**:
  - Use adversarial loss or cycle consistency to ensure robust disentanglement of semantic and paralinguistic factors (e.g., prosody, pitch) in zero-shot or expressive generation settings.
- **Challenges**: Prosody disentanglement ([9]) and nuanced paralinguistic features remain less robust compared to speaker-based separation.

#### **3. High-Compression Techniques with Vector Quantization**
- **VQ and Adaptations**:
  - Models like VQ-CTAP [1], DVQVC [14], and AVQVC [19] use VQ to balance compression and content retention, creating discrete latent spaces critical for scalability in low-supervision tasks.
  - Adaptive codebooks and clustering ([1][3]) enhance preservation of linguistic and paralinguistic detail.

#### **4. Multi-Task Learning and Flexible Architectures**
- **ComSL [4] and CKDST [6]**:
  - Combine cross-modal contrastive methods efficiently for multi-task learning (e.g., ASR, speech translation), achieving advances with pre-trained modular systems in data-scarce conditions.
- Shared embeddings (e.g., speech-text common bases in [28]) unify alignment across modalities for various downstream uses.

#### **5. Comprehensive Training Strategies**
- Hybrid loss functions and dynamic augmentation techniques ([1][28]) ensure smooth convergence while balancing alignment, disentanglement, and compression goals.
- Robust results observed in low-resource conditions with minimally-supervised frameworks like CTAP/VQ-CTAP ([1][2]).

---

### Overall Implications:
Token-acoustic contrastive learning combined with disentanglement and VQ-based compression demonstrates high potential for scalable, generalizable cross-modal tasks (e.g., ASR, TTS, VC) using minimal supervision. VQ-CTAP [1], CTAP [2], and VQMIVC [3] stand out as key contributions for advancing this field.

## Categories

### Categories of Resources for the Specific Topic

---

#### **1. Token-Acoustic Contrastive Learning for Frame-Level Speech-Text Alignment**
- **Description**: Techniques using contrastive learning to bridge token-level text representations (e.g., phonemes) with frame-level acoustic features while resolving granularity mismatches.
- **References**: [1,2,16,27,6,20,28]
- **Details**:
  - [1]: VQ-CTAP uses contrastive pretraining with vector quantization for aligning text and speech at the frame level, enabling plug-and-play for TTS, VC, and ASR.
  - [2]: CTAP aligns phonemes and speech via two encoders with contrastive loss applied to 210k pairs; scalable for low-supervised ASR, TTS, VC.
  - [16]: CPSP introduces three encoders for phoneme-speech alignment in minimally-supervised fine-grained sequence tasks.
  - [27]: Tokenwise contrastive loss aligns speech encoder outputs with BERT embeddings for fine-grained intent recognition.
  - [6]: CKDST applies contrastive loss to promote mutual information transfer between speech and text for speech translation.
  - [20]: ConST employs cross-modal speech-text contrastive learning for end-to-end multilingual speech translation.
  - [28]: Combines shared codebooks and hard-negative mining to address frame-level correspondence and improve cross-modal alignment.

---

#### **2. Disentanglement of Semantic and Paralinguistic Features**
- **Description**: Methods disentangling content (semantics) from speaker and prosodic features for robust VC, TTS, or speech analysis.
- **References**: [3,5,7,9,10,11,15,17,24]
- **Details**:
  - [3]: VQMIVC reduces mutual dependence between semantic and paralinguistic representations via VQ and MI minimization; robust for one-shot VC.
  - [7]: Disentangles speaker and content embeddings using variational autoencoder (VAE) with on-the-fly training augmentation.
  - [5]: Conditional DSVAE refines content embeddings with phonetic structure for improved disentanglement in zero-shot VC.
  - [9]: SAVC disentangles prosody and content using soft speech units and adversarial augmentation, enhancing expressiveness for voice synthesis.
  - [10]: Adversarial VAE-based disentanglement using GAN and domain adaptation focuses on speaker removal for VC representation.
  - [11]: SpeechSplit applies triple bottleneck for segregating semantic, timbre, pitch, and rhythm, achieving broader disentanglement.
  - [15]: Adversarial Mask-And-Predict framework separates speaker, rhythm, pitch, and other prosody features in multi-factor VC.
  - [17]: CTVC applies contrastive learning with time-invariant retrieval for more robust disentangled speech representations.
  - [24]: Proposes hierarchical modeling and semi-supervised style extraction to disentangle prosody/style in TTS with improved generalization.

---

#### **3. Vector Quantization (VQ) for Compression and Alignment**
- **Description**: Methods leveraging VQ to compress speech or text representations into discrete embeddings, reducing redundancy while preserving semantics and task adaptability.
- **References**: [1,3,8,14,19,21,25,22]
- **Details**:
  - [1]: VQ-CTAP uses vector quantization to translate frame-level representations into compact, discrete codebooks aligned with token-level text.
  - [3]: VQMIVC uses VQ to encode content features while disentangling speaker identity in one-shot VC.
  - [8]: VQ-based one-shot voice conversion (VQVC) separates speaker/content with minimal supervision using discrete content embeddings.
  - [14]: DVQVC combines advanced content encoder with perceptual/adversarial loss to counter VQ-induced imperfections in zero-shot VC.
  - [19]: AVQVC incorporates VQ and contrastive training to effectively disentangle content and timbre for one-shot VC.
  - [21]: VQVC+ integrates U-Net with VQ quantization to improve model fidelity for linguistic and timbre features.
  - [25]: TVQVC uses transformer-based VQ-VAE for linguistic encoding combined with CTC loss for high-quality VC.
  - [22]: Combines K-means quantization and self-supervised features for one-shot VC; balances speaking variation and content fidelity.

---

#### **4. Multi-Task Learning and Modular Architectures**
- **Description**: Frameworks combining pretraining, modular encoders, or multi-task optimization for scalable cross-modal speech and text tasks (e.g., ASR, TTS, VC, translation).
- **References**: [1,6,4,20,13]
- **Details**:
  - [1]: VQ-CTAP employs modular pre-trained components for direct task adaptability in ASR, TTS, and VC, with no fine-tuning required.
  - [6]: CKDST unifies machine translation and speech modalities via multi-task contrastive learning, excelling in translation benchmarks.
  - [4]: ComSL utilizes composite architectures (speech-only and text-only pretrained models) for efficient multilingual translation.
  - [20]: ConST optimizes cross-modal representation for multi-task speech-to-text pipelines (speech retrieval, text alignment).
  - [13]: Flow-VAE VC integrates end-to-end latent disentanglement for zero-shot VC without dependency on specific vocoder pipelines.

---

#### **5. Loss Design and Advanced Training Objectives**
- **Description**: Innovative loss designs (e.g., hybrid or augmentation-enhanced) for convergence, semantic alignment, disentanglement, and generalization in low-supervision contexts.
- **References**: [1,5,7,15,28]
- **Details**:
  - [1]: VQ-CTAP introduces a stepping optimization strategy to balance multiple loss terms, achieving smoother convergence for TTS/VC.
  - [5]: Conditional DSVAE reshapes prior distributions in content embeddings for phoneme preservation, refining learning stability.
  - [7]: Noise-invariant data augmentation during training complements disentanglement in VAE for robust speech representation learning.
  - [15]: Adversarial MAP framework minimizes representation correlation while enhancing prosody disentanglement through augmented masking.
  - [28]: Hard-negative guided loss optimizes hybrid granularities (word/frame) for more fine-grained alignment in contrastive learning.

---

#### **6. Zero-Shot Voice Conversion with Disentanglement**
- **Description**: Focus on zero-shot VC models, balancing speaker-content separation under unseen conditions while ensuring naturalness or prosody preservation.
- **References**: [3,5,7,8,14,19,21]
- **Details**:
  - [3]: VQMIVC effectively disentangles and transfers speaker identity while preserving linguistic information in one-shot settings.
  - [5]: Conditional DSVAE achieves phonetic-rich semantic embeddings for zero-shot VC under unstable speaker dynamics.
  - [7]: A noise-robust disentanglement VAE model enhances speaker/content separation for zero-shot VC.
  - [8]: VQ-based one-shot VC uses unsupervised quantization for lightweight cross-speaker adaptation.
  - [14]: DVQVC tackles imperfections in VQ embeddings while improving intelligibility in unseen speaker VC tasks.
  - [19]: AVQVC strengthens quantized disentanglement through contrastive loss for cross-speaker tasks.
  - [21]: VQVC+ introduces U-Net and bottleneck quantization, achieving high-quality naturalness for novel voices.

---

### Summary
The references overwhelmingly highlight advancements in **token-acoustic contrastive learning**, **disentanglement**, and **vector quantization**, with specific focus on fine-grained alignment, semantic-paralinguistic factorization, and lightweight yet scalable architectures for multi-task applications like **ASR**, **TTS**, and **VC** in minimally supervised or zero-shot scenarios. Token-acoustic contrastive pretraining ([1][2]) and disentanglement frameworks ([3][7]) set benchmarks for robustness and efficiency.

## Timeline

### Timeline and Development of Ideas:

#### **Early Foundations (2020-2021):**
- **Vector Quantization and Disentanglement:**
  - **[8] (2020)**: Proposed one-shot voice conversion (VC) using VQ to disentangle content/speaker features, paving the way for heavy reliance on codebooks in VC.
  - **[3] (2021)**: Introduced VQMIVC, integrating VQ with mutual information (MI) to discard speaker leakage, achieving improved disentanglement for one-shot VC.
  - **Parallel Contributions**: [10][21] advanced adversarial and autoencoder-based disentanglement using VAE and instance normalization.

#### **Emergence of Cross-Modal and Frame-Level Alignment (2021-2022):**
- **Token-Acoustic Contrastive Learning Gains Traction:**
  - **[27] (2022)**: Token-wise contrastive learning aligned speech with text embeddings. Early designs (e.g., cross-modal attention) tackled token-frame granularity mismatch.
  - **[6][20] (2022)**: Broad contrastive speech-text models advanced multi-modal alignment, but models like ConST were coarse-grained, focusing more on global semantics.
- **Integration of Hierarchical and Loss Approaches:**
  - **[7][15] (2022)**: Robust disentanglement techniques using adversarial learning and cycle-consistent losses emerged, targeting both content-timbre and prosody disentanglement.

#### **Maturation of Methods (2023-2024):**
- **Sophistication in Token-Acoustic Alignment:**
  - **CTAP ([2], 2023)** and **VQ-CTAP ([1], 2024)** refined token-frame alignment by coupling contrastive loss with scalable VQ. These became benchmarks for minimally-supervised ASR, VC, and TTS.
  - **Hybrid Advances ([12][17])**: Soft speech units and time-invariant retrieval bridged frame-level alignment while encouraging disentanglement.
- **Focus on Prosody and High-Compression:**
  - **[14][19][24] (2023-2024)** innovated prosody retention via K-means quantization or hierarchical modeling for expressive speech synthesis.
  - **[22] (2024)** introduced VQ with self-supervised learning, balancing compression, prosody, and semantic nuances for voice conversion.

---

### Clusters of Research Groups and Contributions:

1. **Chunyu Qiang et al. (CTAP/VQ-CTAP Group)**:
   - Dominant in 2023-2024 with CTAP ([2]) and VQ-CTAP ([1]), advancing token-acoustic contrastive pretraining and VQ techniques for low-supervision tasks. They addressed redundancy in intermediate representations and optimized task-agnostic sequence learning.

2. **Disong Wang and H. Meng (VQMIVC Group)**:
   - Key contributions in 2021 ([3]) with VQMIVC, introducing MI-based disentanglement with VQ for one-shot VC. Strong focus on mutual information and disentanglement robustness has cited influence on [9][14].

3. **Jiachen Lian et al. (DSVAE Group)**:
   - Significant impact with disentanglement frameworks ([5][7]) using variational autoencoders and adversarial learning in 2022, extended to zero-shot VC with improved phonetic structure preservation.

4. **Da-Yi Wu and Hung-yi Lee (VQVC Group)**:
   - Pioneers of VQ in VC ([8][21]), proposing methods focused on disentangled representation learning and lightweight architectures. Their early works laid the foundation for vectorized speech-content compression now extended in [19][22].

5. **Yimin Deng et al. (Soft Units and Prosody)**:
   - Recent contributions ([9][17]), combining soft-unit speech embeddings with adversarial augmentation for expressive disentanglement, focusing on naturalness and prosodic richness in 2023-2024.

---

### Core Evolution Highlights:
- **2020-2022**: Foundations established for disentanglement (adversarial and MI-based) and early exploration of VQ for compression.
- **2023-2024**: Token-acoustic contrastive learning ([1][2]) combines with VQ for scalable, minimally-supervised cross-modal alignment; improved prosody modeling ([9][24]) and shared codebook strategies broaden applications to expressive voice conversion and speech synthesis.

## Foundational Work

### Which papers form the foundational references on this topic?

The below table shows the resources that are most often cited by the relevant papers on this topic. This is measured by the **reference rate**, which is the fraction of relevant papers that cite a resource. Use this table to determine the most important core papers to be familiar with if you want to deeply understand this topic. Some of these core papers may not be directly relevant to the topic, but provide important context.

| Ref. | Reference Rate | Topic Match | Title | Authors | Journal | Year | Total Citations | Cited By These Relevant Papers |
|---|---|---|---|---|---|---|---|---|
| [65] | 0.50 | 4% | Zero-Shot Voice Style Transfer with Only Autoencoder Loss | Kaizhi Qian, ..., and M. Hasegawa-Johnson | ArXiv | 2019 | 433 | [3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 21, 22, 23, 25, 29] |
| [47] | 0.43 | 20% | HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | Wei-Ning Hsu, ..., and Abdel-rahman Mohamed | IEEE/ACM Transactions on Audio, Speech, and Language Processing | 2021 | 2365 | [1, 2, 5, 9, 12, 16, 17, 22, 26, 27, 30, 31, 33] |
| [141] | 0.41 | Not measured | One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization | Ju-Chieh Chou, ..., and Hung-yi Lee | ArXiv | 2019 | 226 | [3, 5, 7, 8, 9, 10, 11, 12, 15, 21, 22, 23] |
| [59] | 0.39 | 15% | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | Alexei Baevski, ..., and Michael Auli | ArXiv | 2020 | 4839 | [1, 2, 6, 9, 12, 14, 16, 20, 26, 27, 30, 31, 32, 33] |
| [21] | 0.37 | 33% | VQVC+: One-Shot Voice Conversion by Vector Quantization and U-Net architecture | Da-Yi Wu, ..., and Hung-yi Lee | ArXiv | 2020 | 97 | [1, 3, 9, 12, 14, 17, 19, 22, 23, 25, 29] |
| [142] | 0.33 | Not measured | StarGAN-VC: non-parallel many-to-many Voice Conversion Using Star Generative Adversarial Networks | H. Kameoka, ..., and Nobukatsu Hojo | 2018 IEEE Spoken Language Technology Workshop (SLT) | 2018 | 359 | [3, 5, 7, 8, 10, 11, 13, 19, 21, 25] |
| [111] | 0.31 | 0% | Phonetic posteriorgrams for many-to-one voice conversion without parallel data training | Lifa Sun, ..., and H. Meng | 2016 IEEE International Conference on Multimedia and Expo (ICME) | 2016 | 303 | [1, 2, 3, 5, 7, 10, 13, 16, 21, 25] |
| [100] | 0.30 | 0% | Minimally-Supervised Speech Synthesis with Conditional Diffusion Model and Language Model: A Comparative Study of Semantic Coding | Chunyu Qiang, ..., and J. Dang | ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2023 | 5 | [1, 2, 16, 26] |
| [143] | 0.29 | Not measured | HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis | Jungil Kong, ..., and Jaekyoung Bae | ArXiv | 2020 | 1602 | [5, 9, 12, 13, 14, 17, 18, 22, 25, 30] |
| [79] | 0.29 | 0% | CLAP Learning Audio Concepts from Natural Language Supervision | Benjamin Elizalde, ..., and Huaming Wang | ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2023 | 333 | [1, 2, 16, 28] |
| [144] | 0.28 | Not measured | WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing | Sanyuan Chen, ..., and Furu Wei | IEEE Journal of Selected Topics in Signal Processing | 2021 | 1430 | [1, 5, 9, 17, 22, 30, 33, 37] |
| [45] | 0.27 | 21% | Neural Discrete Representation Learning | Aäron van den Oord, ..., and K. Kavukcuoglu | N/A | 2017 | 4076 | [1, 5, 8, 12, 13, 21, 22, 25, 28, 30] |
| [39] | 0.27 | 23% | vq-wav2vec: Self-Supervised Learning of Discrete Speech Representations | Alexei Baevski, ..., and Michael Auli | ArXiv | 2019 | 630 | [1, 2, 3, 9, 14, 16, 18, 19] |
| [87] | 0.24 | 0% | PMVC: Data Augmentation-Based Prosody Modeling for Expressive Voice Conversion | Yimin Deng, ..., and Jing Xiao | Proceedings of the 31st ACM International Conference on Multimedia | 2023 | 6 | [9, 17, 35] |
| [145] | 0.23 | Not measured | Voice conversion from non-parallel corpora using variational auto-encoder | Chin-Cheng Hsu, ..., and H. Wang | 2016 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA) | 2016 | 295 | [8, 10, 11, 12, 13, 15, 19, 21, 25] |
| [146] | 0.23 | Not measured | Freevc: Towards High-Quality Text-Free One-Shot Voice Conversion | Jingyi Li, ..., and Li Xiao | ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2022 | 78 | [1, 9, 17, 22] |
| [137] | 0.23 | 0% | Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision | E. Kharitonov, ..., and Neil Zeghidour | Transactions of the Association for Computational Linguistics | 2023 | 161 | [1, 2, 16, 26, 36] |
| [2] | 0.22 | 100% | Learning Speech Representation from Contrastive Token-Acoustic Pretraining | Chunyu Qiang, ..., and J. Dang | ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2023 | 3 | [1, 26, 164, 167, 176] |
| [8] | 0.21 | 99% | One-Shot Voice Conversion by Vector Quantization | Da-Yi Wu and Hung-yi Lee | ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2020 | 80 | [9, 12, 15, 19, 21, 22, 25, 29, 35] |
| [24] | 0.21 | 29% | Improving Prosody for Cross-Speaker Style Transfer by Semi-Supervised Style Extractor and Hierarchical Modeling in Speech Synthesis | Chunyu Qiang, ..., and Zhong-ming Wang | ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2023 | 9 | [1, 2, 16, 26] |

## Adjacent Work

### Which papers cite the same foundational papers as relevant papers?

Use this table to discover related papers on adjacent topics, to gain a broader understanding of the field and help generate ideas for useful new research directions.

| Ref. | Adjacency score | Topic Match | Title | Authors | Journal | Year | Total Citations | References These Foundational Papers |
|---|---|---|---|---|---|---|---|---|
| [87] | 1.91 | 0% | PMVC: Data Augmentation-Based Prosody Modeling for Expressive Voice Conversion | Yimin Deng, ..., and Jing Xiao | Proceedings of the 31st ACM International Conference on Multimedia | 2023 | 6 | [11, 12, 19, 39, 65, 67] |
| [154] | 1.85 | Not measured | Reimagining speech: a scoping review of deep learning-based methods for non-parallel voice conversion | A. R. Bargum, ..., and Cumhur Erkut | Frontiers in Signal Processing | 2024 | 3 | [3, 11, 18, 19, 21, 65] |
| [155] | 1.85 | Not measured | Reimagining Speech: A Scoping Review of Deep Learning-Powered Voice Conversion | A. R. Bargum, ..., and Cumhur Erkut | ArXiv | 2023 | 3 | [3, 11, 18, 19, 21, 65] |
| [22] | 1.60 | 31% | SKQVC: One-Shot Voice Conversion by K-Means Quantization with Self-Supervised Speech Representations | Youngjun Sim, ..., and Young-Joo Suh | ArXiv | 2024 | 0 | [3, 12, 14, 21, 47, 65] |
| [156] | 1.48 | Not measured | Takin-VC: Zero-shot Voice Conversion via Jointly Hybrid Content and Memory-Augmented Context-Aware Timbre Modeling | Yuguang Yang, ..., and Jianjun Zhao | ArXiv | 2024 | 2 | [3, 14, 19, 21] |
| [157] | 1.41 | Not measured | MAIN-VC: Lightweight Speech Representation Disentanglement for One-shot Voice Conversion | Pengcheng Li, ..., and Ning Cheng | 2024 International Joint Conference on Neural Networks (IJCNN) | 2024 | 1 | [3, 8, 11, 19, 21, 65] |
| [35] | 1.36 | 24% | CLN-VC: Text-Free Voice Conversion Based on Fine-Grained Style Control and Contrastive Learning with Negative Samples Augmentation | Yimin Deng, ..., and Jing Xiao | 2023 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Big Data & Cloud Computing, Sustainable Computing & Communications, Social Computing & Networking (ISPA/BDCloud/SocialCom/SustainCom) | 2023 | 2 | [3, 11, 12, 19, 65, 67] |
| [158] | 1.29 | Not measured | DQR-TTS: Semi-supervised Text-to-speech Synthesis with Dynamic Quantized Representation | Jiangzong Wang, ..., and Jing Xiao | 2023 IEEE Intl Conf on Parallel & Distributed Processing with Applications, Big Data & Cloud Computing, Sustainable Computing & Communications, Social Computing & Networking (ISPA/BDCloud/SocialCom/SustainCom) | 2023 | 0 | [8, 12, 19, 21] |
| [41] | 1.26 | 22% | Discrete Unit based Masking for Improving Disentanglement in Voice Conversion | Philip H. Lee, ..., and Berrak Sisman | ArXiv | 2024 | 0 | [3, 21, 47, 53, 65] |
| [159] | 1.20 | Not measured | Multi-Level Temporal-Channel Speaker Retrieval for Zero-Shot Voice Conversion | Zhichao Wang, ..., and Yuping Wang | IEEE/ACM Transactions on Audio, Speech, and Language Processing | 2023 | 3 | [3, 11, 19, 21] |
| [160] | 1.20 | Not measured | Vec-Tok-VC+: Residual-enhanced Robust Zero-shot Voice Conversion with Progressive Constraints in a Dual-mode Training Strategy | Linhan Ma, ..., and Lei Xie | ArXiv | 2024 | 2 | [3, 11, 19, 21] |
| [161] | 1.20 | Not measured | Random Cycle Loss and Its Application to Voice Conversion | Haoran Sun, ..., and T. Zheng | IEEE Transactions on Pattern Analysis and Machine Intelligence | 2023 | 5 | [3, 11, 19, 21] |
| [162] | 1.19 | 0% | Kling-Foley: Multimodal Diffusion Transformer for High-Quality Video-to-Audio Generation | Jun Wang, ..., and Kun Gai | ArXiv | 2025 | 1 | [1, 2] |
| [163] | 1.19 | 0% | InstructAudio: Unified speech and music generation with natural language instruction | Chunyu Qiang, ..., and Jianwu Dang | ArXiv | 2025 | 2 | [1, 2] |
| [165] | 1.14 | Not measured | Takin-VC: Expressive Zero-Shot Voice Conversion via Adaptive Hybrid Content Encoding and Enhanced Timbre Modeling | Yuguang Yang, ..., and Jianjun Zhao | ArXiv | 2024 | 2 | [14, 19, 21] |
| [166] | 1.09 | Not measured | Disentangling segmental and prosodic factors to non-native speech comprehensibility | Waris Quamer and Ricardo Gutierrez-Osuna | ArXiv | 2024 | 0 | [3, 8, 11, 21, 65] |
| [25] | 1.07 | 28% | TVQVC: Transformer Based Vector Quantized Variational Autoencoder with CTC Loss for Voice Conversion | Ziyi Chen and Pengyuan Zhang | N/A | 2021 | 3 | [21, 53, 65, 111] |
| [19] | 1.06 | 34% | Avqvc: One-Shot Voice Conversion By Vector Quantization With Applying Contrastive Learning | Huaizhen Tang, ..., and Jing Xiao | ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2022 | 52 | [21, 39, 53, 67] |
| [12] | 1.04 | 53% | VQ-CL: Learning Disentangled Speech Representations with Contrastive Learning and Vector Quantization | Huaizhen Tang, ..., and Jing Xiao | ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2023 | 8 | [11, 19, 21, 47, 53, 65, 67] |
| [168] | 1.03 | Not measured | EmoPro: A Prompt Selection Strategy for Emotional Expression in LM-based Speech Synthesis | Haoyu Wang, ..., and Chen Zhang | ArXiv | 2024 | 1 | [2, 24, 34] |