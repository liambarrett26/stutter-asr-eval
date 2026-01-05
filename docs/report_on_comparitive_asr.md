# Comparative benchmarks of modern ASR systems on stuttered and disfluent speech

## Overview

The search reveals a small but growing core of **true multi‑system benchmarks** of modern ASR on stuttered/disfluent speech—most notably Mujtaba et al.’s cross‑API study [1] and Batra et al.’s Boli dataset comparison [3], with additional, more constrained multi‑system results from Mandarin datasets and challenges [2,4,5,6,12]—alongside a larger body of **single‑system** but methodologically rich work that clarifies error patterns, fairness gaps, and adaptation levers [7,8,9,10,11,13].

---

### ### 1. Overall Landscape: How Close Are We to a True Benchmark Ecosystem?

#### **1.1. Direct multi‑system benchmarks on stuttered/disfluent speech**

**Most central to your goal:**

- **Lost in Transcription (Mujtaba et al., 2024)** [1]
  - Evaluates **six contemporary ASR systems** (mix of commercial APIs, including **Google Cloud Speech‑to‑Text V1**, and modern open‑source architectures in the wav2vec2/Whisper family) on:
    - **FluencyBank** with disfluencies included (FB‑Y) and a cleaned “intended speech” version (FB‑N).
    - **LibriSpeech** (LS‑N) and a **synthetically disfluent version** (LS‑Y).
  - Reports **WER and CER per system**, per condition, plus **semantic similarity metrics** (FBERT‑style), and performs **statistical analyses** linking particular disfluency types (e.g., prolongations, interjections) to WER increases (p < 0.05).
  - Focuses explicitly on **accuracy bias against disfluent speech**, not on adaptation; this is the clearest example of the kind of benchmark you asked about.

- **Boli dataset (Batra et al., ICASSP 2025)** [3]
  - Introduces a **multilingual stuttered‑speech corpus** (Indian languages + English) with per‑file labels for five stutter categories (blocks, prolongations, interjections, sound repetitions, word repetitions).
  - Provides a **direct comparison of two open‑source SOTA models** on an English subset:
    - **wav2vec 2.0 vs Whisper**, with WER reported under:
      - **Concatenated/pooled evaluation:** 29.22% vs 4.55%.
      - **Speaker‑wise averaging:** 55.81% vs 21.27% [3].
  - Confirms **large performance gaps between contemporary architectures** on stuttered speech, even when both are “SOTA” in general ASR.

**Other multi‑system or multi‑variant results that are somewhat narrower:**

- **Mandarin, community‑led dataset (Li & Wu, CHI EA 2024; Li et al., FAccT 2025)** [2,4]
  - Introduce and further analyze a **community‑created Mandarin stuttered‑speech dataset** (50+ hours, 72 speakers, conversations + commands), and **benchmark multiple “popular ASR models”** to expose bias; details are less API‑explicit than [1] but provide a **multi‑system baseline for Mandarin PWS**.
- **AS‑70 + StutteringSpeech Challenge (Gong et al. 2024; Xue et al. SLT 2024)** [5,6]
  - **AS‑70** [6] offers a large **Mandarin stuttered‑speech corpus** with speaker‑exclusive splits and severity bands; baselines involve modern pretrained models (e.g., Whisper, HuBERT) evaluated **zero‑shot vs fine‑tuned**.
  - The **2024 challenge** [5] reports performance of **multiple submitted systems** (often SOTA architectures like Zipformer) on the same AS‑70‑based test set—functionally a **multi‑system benchmark**, though systems are research submissions rather than named commercial APIs.
- **StutterZero / StutterFormer vs multiple Whisper baselines (Xu, 2025)** [12]
  - Compares two **end‑to‑end stutter‑to‑fluent models** against **Whisper‑Tiny/Small/Medium** on SEP‑28k/LibriStutter‑derived data and FluencyBank [12], yielding multi‑system WER/CER/semantic comparisons but with a primary emphasis on the new models.

**Bottom line:**
You have **only one clearly designed, heterogeneous cross‑API benchmark** in English [1] and **one clean open‑source‑only comparison** [3]; for Mandarin, benchmarks are split across **dataset papers** [2,4,6] and a **challenge** [5] rather than a single unified cross‑API study. Most other work is **single‑system** or **single‑architecture‑with‑variants**.

---

### ### 2. Quantitative Patterns: How Bad Is the WER Degradation, and For Whom?

#### **2.1. Magnitude of WER gaps: stuttered vs non‑stuttered speech**

Across studies, the **relative degradation** on stuttered/disfluent speech is substantial and consistent:

- **Cross‑system English benchmark (Lost in Transcription)** [1]
  - Every evaluated ASR system shows **substantially higher WER** on disfluent conditions (FB‑Y, LS‑Y) than on clean or edited conditions (FB‑N, LS‑N).
  - The bias is systematic: **WER distributions shift upward** for disfluent sets across all six systems, not just weaker ones [1].
  - CER exhibits similar patterns, but **semantic metrics show that even small WER increases can correspond to meaningful semantic degradation** when disfluencies are involved.

- **Voice‑assistant systems (Mitra et al. 2021; Lea et al. 2023)** [8,9]
  - **VA‑Fluent vs VA‑Dysfluent**: intended‑speech WER (isWER) jumps from **5.65% to 19.29%** [9].
    - By stuttering severity:
      - Grade 1: 8.39% isWER
      - Grade 2: 16.64%
      - Grade 3: 47.86% [9].
  - **Apple Speech framework in a CHI study**: mean WER for PWS is **19.8%** overall; mild ≈4.8%, moderate–severe ≈25.4% [8].
  - Both works show **order‑of‑magnitude disparities** between typical and highly stuttered speech on the same system.

- **Mandarin AS‑70 and challenge** [5,6]
  - Zero‑shot modern ASR (e.g., Whisper) performs **poorly** on Mandarin stuttered speech, with **large WERs**; fine‑tuning on AS‑70 yields **substantial relative reductions** [6].
  - Challenge results [5] show further WER improvements with specialized architectures and augmentations, but still clearly **higher error rates than on non‑stuttered Mandarin**.

- **Whisper vs wav2vec2.0 (Boli)** [3]
  - On English Boli speech, even a strong model (wav2vec2) sees **29–56% WER**, while Whisper reduces this to **≈5–21%**, depending on aggregation [3].
  - This demonstrates both **severe absolute degradation** vs fluent speech, and **strong dependence on architecture choice**.

#### **2.2. System ranking: who does best among current SOTA models?**

From the available cross‑model studies:

- **Whisper tends to dominate earlier open‑source SOTA**:
  - On Boli, Whisper dramatically outperforms wav2vec2.0 in WER [3].
  - In several single‑system studies, “raw” Whisper is treated as a high‑performing baseline [7,10,12,13].
- **Commercial APIs show similar qualitative biases but absolute performance varies**:
  - Lost in Transcription [1] includes at least Google Cloud V1 and finds that **all systems share the same direction of bias** (higher WER on disfluent speech), with **different absolute WER levels and different sensitivities** to disfluency types.
  - The paper does not establish a single universally best vendor but clearly shows that **no off‑the‑shelf commercial system is robust to stuttering**.
- **Adapted / specialized models can beat generic Whisper‑scale baselines on stuttered speech**:
  - StutterZero and StutterFormer achieve **24–28% relative WER reductions** and 31–34% BERTScore gains vs Whisper‑Medium [12].
  - Personalized Whisper models achieve **notable WER/CER reductions for individual PWS**, especially in spontaneous speech [13].
  - Challenge systems on AS‑70 outperform naive fine‑tunes and zero‑shot models [5,6].

#### **2.3. Disfluency‑ and severity‑specific vulnerabilities**

Several works dissect **which stutter types and severities are most problematic**:

- **Disfluency type correlations (Lost in Transcription)** [1]
  - Prolongations, interjections, and other marked disfluencies show **significant positive correlations** with WER across models, statistically tested (p < 0.05).
  - This suggests that **the distribution of disfluency types in a dataset directly affects system rankings and apparent robustness**.

- **Stutter‑type annotations (Boli)** [3]
  - Boli’s per‑word labels allow per‑type analysis; preliminary observations indicate **word‑repetition stutters are relatively well handled** by both wav2vec2 and Whisper [3], implying that **not all stutters are equally harmful**.

- **Sound repetitions and hallucinations (Sridhar & Wu, 2025)** [10]
  - On real podcasts with PWS, Whisper shows **particularly poor performance on sound repetitions**, a distinctly stuttering‑specific phenomenon.
  - Sound repetitions trigger **hallucinations in >20% of cases**, i.e., Whisper fabricates extra words or phrases, a failure mode not captured by WER alone [10].

- **Clinical severity and stuttering‑rate (Mitra et al., Gong et al.)** [6,9]
  - WER rises sharply with **clinical severity level** in VA‑Dysfluent [9].
  - AS‑70 introduces a **stuttering‑rate (SR) metric** (mild ≤7%, moderate 7–12%, severe >12%) and organizes splits to enable **severity‑stratified ASR evaluation** [6].
  - This strongly supports **severity‑aware benchmarking**; averaged WER hides extreme degradation for severe speakers.

---

### ### 3. Metrics Beyond WER and Fairness Considerations

#### **3.1. CER, semantic metrics, and hallucinations**

- **CER (Character Error Rate)** is reported in several key papers [1,6,12,13], especially:
  - Where **partial words** and **broken words** are common (stuttered speech) and where languages are **morphologically rich** (Mandarin).
  - CER often reveals improvements or regressions not visible at the word level, particularly for stutter‑related fragments.

- **Semantic metrics (FBERT/BERTScore)**:
  - **Lost in Transcription** uses semantic similarity (e.g., FBERT) to show that **even if WER is high, some systems preserve semantics better** [1].
  - **StutterZero/StutterFormer** use **BERTScore** and show **31–34% improvements** over Whisper‑Medium [12], indicating these models better preserve meaning even when surface errors remain.
  - These metrics are crucial when disfluencies are normalized away or altered; **WER alone cannot distinguish “wrong words but right meaning” cases**.

- **Hallucination rate**:
  - **Whisper on sound repetitions** shows >20% hallucinations [10], a qualitatively different failure mode from standard substitutions or deletions.
  - Hallucinations matter for **safety, trust, and fairness**; they can be more damaging than simple misrecognitions.

#### **3.2. Task‑level metrics and user‑perceived quality**

- **Voice‑assistant context (Lea et al., Mitra et al.)** [8,9]:
  - **Intent Error Rate**: measured by feeding ASR output into NLU; often remains acceptable even when WER is moderately high, but **degrades sharply for severe stuttering** [8,9].
  - **Truncation rates**: PWS suffer **much higher truncation** due to endpointing interacting with blocks and prolongations; interventions reduce truncation by ∼79% and WER from 25.4% to 9.9% for moderate–severe PWS without acoustic retraining [8].
  - These show that **decoding and endpoint tuning are major levers** for improving real‑world usability, especially for streaming ASR.

- **User experience and acceptability**:
  - CHI‑style work [8] combines WER with **survey‑based measures** of frustration, perceived effort, and willingness to use ASR, showing that **technical improvements translate into meaningful UX gains** for PWS.
  - Later fairness‑oriented work [2,4,10] explicitly frames performance gaps as **access barriers** and draws on disability‑led methods.

#### **3.3. Fairness and group‑comparative analyses**

- **PWS vs non‑PWS gaps**:
  - VA‑Fluent vs VA‑Dysfluent: 5.65% vs 19.29% isWER [9].
  - FluencyBank vs Switchboard (FluencyBank Timestamped) enables cross‑group robustness comparisons [7].
  - Lost in Transcription [1] analyzes **bias via WER distributions** and correlations with disfluency labels; community‑led Mandarin datasets foreground **equity and inclusion** in technical analysis [2,4].

- **Within‑group disparities (severity, disfluency patterns)**:
  - Clinical severity strongly modulates performance [8,9].
  - Disfluency‑type breakdowns show that **stuttering‑specific phenomena (sound repetitions, blocks, partial words)** are more damaging than generic hesitations [1,10,11].

**Implications for your work:**
If you want to make **fairness claims**, you will need:

- Severity‑aware analyses (e.g., SR bands [6], clinical grades [9]).
- Disfluency‑type stratification (blocks vs repetitions vs prolongations) where possible [1,3,7,10].
- Group comparisons (PWS vs controls) where a non‑stuttered counterpart dataset exists [7,9].

WER alone is insufficient; **CER, semantic measures, hallucination metrics, truncation, and intent‑level performance** all capture important dimensions of “how bad” the bias is for people who stutter.

---

### ### 4. Methodological Lessons and Gaps for Designing New Benchmarks

#### **4.1. Reference conventions and normalization can invert rankings**

The literature uses two main transcription strategies:

- **Intended‑speech references** (disfluencies removed/normalized):
  - VA‑Dysfluent / VA‑Fluent [9], FluencyBank intended‑speech task [7], portions of [8,11].
  - Penalize systems that **faithfully reproduce disfluencies**; favor those that “clean them up.”

- **Surface‑faithful references** (disfluencies included):
  - AS‑70 [6], Boli [3], FluencyBank FB‑Y [1], podcast work [10].
  - Penalize systems that **normalize disfluencies away**.

Many papers emphasize that **this choice materially affects WER and system ranking** [1,7,9,11]. For example:

- An ASR that aggressively deletes “uh/um” and some repetitions may:
  - Look **better** on intended‑speech WER,
  - Look **worse** when disfluencies are part of the reference.

For a **comparative benchmark**, you should:

- Decide explicitly whether your target is **surface transcription** or **intended speech**, and
- Apply **identical normalization to all systems**, including:
  - Punctuation stripping, casing, number normalization.
  - Hesitation token mapping (“um”, “uh” → one canonical token or ignored).
  - Policy for partial words, blocks, and non‑verbal events.

#### **4.2. Streaming vs offline: endpointing and stability matter for PWS**

Most multi‑system benchmarks [1,3,5,6,12] work in **offline mode**, focusing on final transcripts. However:

- **Streaming voice‑assistant studies** [8,9] show that **endpointing and VAD** interact badly with:
  - Prolongations,
  - Blocks,
  - Long pauses between repetitions.
- This leads to:
  - **Truncations** (utterance cut off early), and
  - Unstable partial hypotheses—issues not visible in offline WER.

For a **realistic PWS benchmark**, especially for assistants:

- Consider separate evaluations for:
  - **Final WER** vs
  - **Streaming quality metrics** (truncation rate, revisions, latency).
- Ensure that for commercial APIs, you **configure disfluency filtering and endpoint behavior consistently** where possible.

#### **4.3. System selection and comparability**

The literature suggests a practical template for system selection:

- **Commercial / production APIs:**
  - At least one or two of the major providers (e.g., Google, Microsoft, Amazon, Apple), as in [1,8].
  - When possible, use **“raw” transcripts or options that disable disfluency removal** to keep outputs comparable.
- **Open‑source / research SOTA:**
  - **Whisper** (various sizes) is now the de facto baseline [1,3,7,10,12,13].
  - A strong **encoder‑only model** (e.g., wav2vec2, HuBERT) for comparison [3,6].
  - If relevant, a **RNN‑T/transducer** representing streaming architectures [11].
- **Adapted models**:
  - Including **fine‑tuned Whisper** or stutter‑specialized models (e.g., StutterZero/StutterFormer) can contextualize **how far off‑the‑shelf systems are from achievable performance** [6,11,12,13].

Crucially, all should be run:

- On the **same audio**,
- With **matched or documented decoding settings**, and
- Scored with a **single, shared evaluation script** and normalization pipeline, as done in [1,3].

#### **4.4. Remaining gaps and opportunities**

Based on what is *not* yet well covered:

- **Cross‑API benchmarks are still rare.**
  - Beyond Lost in Transcription [1], we lack comprehensive head‑to‑head evaluations of **all major commercial APIs vs multiple open‑source SOTA** on the same real stuttered corpora.
- **Streaming behavior benchmarks are underdeveloped.**
  - We have evidence from single‑system studies [8,9] but no **multi‑system, controlled comparison of streaming performance** (truncation, latency, partial‑hypothesis stability) specifically for stuttered speech.
- **Language and dialect diversity is limited.**
  - English (FluencyBank, VA‑Dysfluent, podcasts, Boli English) and Mandarin (AS‑70, StammerTalk dataset) dominate [1,2,3,4,5,6,7,10].
  - Boli starts to cover Indian languages [3], but there is **little benchmarking on other languages** where stuttering interacts differently with phonotactics and orthography.
- **Task‑oriented multi‑system benchmarks are missing.**
  - Intent error, task success, and user metrics are mostly studied on **single systems** [8,9]; a cross‑system study comparing **WER vs task success trade‑offs** on PWS data is absent.
- **Fairness metrics are not standardized.**
  - Several papers measure PWS vs control gaps or severity‑band WER [1,2,4,6,7,8,9], but there is no common fairness metric (e.g., specific gap thresholds, variance, confidence intervals) that all adopt.

These gaps point directly to **high‑value directions** for your own work:

- Design a **multi‑system, multi‑metric benchmark** that:
  - Uses **real stuttered corpora** (FluencyBank, AS‑70, Boli, Mandarin community dataset) under clear reference conventions.
  - Evaluates **multiple commercial APIs and open‑source SOTA models** in both **offline and streaming settings**.
  - Reports **WER, CER, semantic metrics, hallucination rate, task success, and fairness gaps** (severity, PWS vs non‑PWS).
- Leverage methodological lessons from [1,3,5,6,7,8,9,11,12,13] to:
  - Ensure **fair normalization**,
  - Stratify by **disfluency type and severity**, and
  - Connect technical outcomes to **user and fairness implications**.

In summary, the field now has **strong ingredients**—datasets, initial cross‑system benchmarks, and rich single‑system analyses—but **no single, comprehensive, standardized benchmark suite** yet; your project can meaningfully fill that gap by integrating these strands into a unified evaluation of modern ASR on stuttered speech.

## Categories

### Comparative Scope of the Papers: What Is Actually Benchmarked?

#### **Benchmark Type and System Coverage**

The table below synthesizes, for each paper, whether it provides:
- **Multi‑system benchmarking** on stuttered/disfluent speech,
- Inclusion of **commercial/production APIs**,
- Inclusion of **modern open‑source SOTA models** (e.g., Whisper, wav2vec2, HuBERT, RNN‑T),
- Or focuses primarily on **single‑system tuning/adaptation**.

| Ref | Main focus vis‑à‑vis your criteria | Multi‑system benchmark on stuttered/disfluent speech? | Commercial / production ASR evaluated? | Open‑source / research SOTA evaluated? | Role of adaptation / tuning |
|-----|------------------------------------|--------------------------------------------------------|----------------------------------------|----------------------------------------|-----------------------------|
| **[1]** Lost in Transcription (2024) | Systematic, **multi‑ASR benchmarking** of disfluent vs non‑disfluent speech (real & synthetic) with WER/CER + semantic metrics | **Yes – six ASR systems** on FluencyBank and LibriSpeech variants | **Yes – includes Google Cloud Speech‑to‑Text V1** [1] | **Yes – includes state‑of‑the‑art architectures such as wav2vec 2.0 / Whisper‑like systems** [1] | No training of systems; focus on **zero‑shot performance and bias quantification** |
| **[2]** Community‑led Chinese stuttered speech dataset (CHI EA 2024) | **New Mandarin stuttered dataset** + initial benchmarking of “popular ASR models” for fairness | Likely **yes**, but the extended abstract summary does not enumerate all systems in detail [2] | Implied “popular ASR models”; unclear which vendors [2] | Likely modern pretrained ASR (building on [6]) | Benchmarking is mainly **baseline characterization**, not heavy adaptation |
| **[3]** Boli dataset (ICASSP 2025) | New multilingual stuttered dataset; **technical validation via model comparison** on English subset | **Yes – Wav2Vec2.0 vs Whisper** [3] | No commercial APIs; both are research/open‑source | **Yes – Wav2Vec2.0 and Whisper** [3] | Only **off‑the‑shelf models**, no fine‑tuning; used for dataset validation |
| **[4]** Our Collective Voices (FAccT 2025) | Sociotechnical analysis of Chinese stuttered dataset + **benchmarking and fine‑tuning** | Yes for baseline benchmarking; details build on [2,6] | Possibly includes commercial APIs (not explicit in summary) | Yes – fine‑tuning modern ASR on the dataset [4] | **Substantial fine‑tuning/adaptation** of models on stuttered data |
| **[5]** Mandarin Stuttering Challenge findings (SLT 2024) | Challenge report: **multiple submitted ASR systems** on AS‑70 stuttered ASR track [5] | **Yes – many systems from participants**, but not public commercial APIs per se | Not obvious; systems are challenge submissions, not named APIs [5] | Yes – many are state‑of‑the‑art architectures (e.g., Zipformer‑based) [5] | Focus is on **participant systems and their techniques**, not on off‑the‑shelf production APIs |
| **[6]** AS‑70 dataset (2024) | New Mandarin stuttered dataset with ASR & SED baselines | Baselines only; **single‑system comparisons and variants**, not broad multi‑system benchmarking [6] | Not emphasized | **Yes – fine‑tuning modern pretrained models like Whisper/HuBERT** [6] | Core is **adaptation/fine‑tuning** of pretrained ASR on AS‑70 |
| **[7]** FluencyBank Timestamped (JSLHR 2024) | New annotations + **benchmark for intended speech recognition** and disfluency detection | For ASR, **single modern ASR (Whisper)** on intended speech [7] | No | **Yes – Whisper** [7] | No multi‑system comparison; adaptation not main focus |
| **[8]** User perceptions & technical improvements (CHI 2023) | **Single commercial VA ASR** evaluated and tuned for PWS [8] | No – one production system + its tuned variants | **Yes – consumer‑grade Apple Speech framework** [8] | No | **Decoder & endpoint tuning** and design interventions are central |
| **[9]** Voice assistant system for dysfluent speech (Interspeech 2021) | **Single hybrid ASR** (VA system) evaluated on dysfluent vs fluent sets [9] | No – only one system + tuned variants | Yes – a production‑style hybrid VA ASR [9] | No | **Decoding‑level tuning**; no model re‑training |
| **[10]** Just Stutter: Whisper disparities (Interspeech 2025) | **Single SOTA model (Whisper)** analyzed by disfluency type using real podcast data [10] | No – Whisper only | No | **Yes – Whisper** [10] | No adaptation; focus on error analysis and hallucinations |
| **[11]** RNN‑T robustness to disfluencies (ICASSP 2021) | **Single RNN‑T architecture** evaluated across training variants on disfluency and stutter tests [11] | No – internal RNN‑T variants only | Not framed as a commercial API, though RNN‑T is production‑like | Yes – production‑style RNN‑T [11] | Core contribution is **data/labeling strategies for training** |
| **[12]** StutterZero/StutterFormer (IEEE Access 2025) | New stutter‑to‑fluent conversion ASR models vs several Whisper baselines [12] | **Yes – multiple Whisper variants vs two proposed models** | No | **Yes – Whisper‑Tiny/Small/Medium** [12] | Proposed systems are trained; Whisper is baseline off‑the‑shelf |
| **[13]** Fine‑Tuning ASR for Stuttered Speech (2025) | Personalized vs generalized **fine‑tuning of Whisper** [13] | No – Whisper Small variants only | No | **Yes – Whisper Small** [13] | **Fine‑tuning (LoRA)** is central; not a multi‑system comparison |

**Key comparative observation:**
Among the references, **[1]** stands out as the only paper that (a) systematically compares **multiple commercial and open‑source SOTA ASR systems** under **matched conditions**, (b) includes both **real stuttered speech (FluencyBank)** and **synthetic disfluencies**, and (c) reports WER/CER plus semantic metrics and statistical tests focused explicitly on **accuracy bias against disfluency**.
Papers **[2,3,4,5,6,12]** provide multi‑system or multi‑variant results in more limited scopes (language‑specific datasets, challenge submissions, or comparison with an in‑house method), while **[7,8,9,10,11,13]** are predominantly single‑system studies (with internal variants).

---

### Dataset and Disfluency Coverage Across Studies

#### **Corpora, Languages, and Disfluency Types**

This table emphasizes which **stuttering/disfluency datasets** are used, their **languages**, and whether they cover **clinical stuttering** vs generic spontaneous disfluencies.

| Ref | Datasets used for ASR evaluation | Language(s) | Stuttering / disfluency type coverage |
|-----|----------------------------------|-------------|----------------------------------------|
| **[1]** Lost in Transcription | **FluencyBank (FB‑Y)**: real speech from adults who stutter; **FluencyBank cleaned (FB‑N)**: “intended speech” version; **LibriSpeech original (LS‑N)**; **synthetic disfluent LibriSpeech (LS‑Y)** created via TTS and manipulations [1] | English | FB‑Y contains clinical stuttering and typical disfluencies; LS‑Y has **synthetic disfluencies** (prolongations, interjections, etc.) [1]. Explicit correlation analyses per disfluency type vs WER. |
| **[2]** Community Chinese stuttered dataset | Newly collected **Mandarin stuttered speech** from 72 PWS, both spontaneous conversation and voice‑command dictation [2] | Mandarin Chinese | Wide range of stuttering characteristics; not just ums/uhs but clinically stuttered speech [2]. |
| **[3]** Boli dataset | **Project Boli**: multilingual stuttered speech, with annotations for blocks, prolongations, interjections, sound repetitions, word repetitions. ASR validation on English subset [3] | Primarily Indian languages + English | Each audio file contains **a single stuttered word plus surrounding fluent words**, enabling fine‑grained disfluency‑type analysis [3]. |
| **[4]** Our Collective Voices | Same Mandarin dataset as [2], but extended analysis including fine‑tuning benchmarks [4] | Mandarin Chinese | Same as [2]; emphasis on diversity and stuttering experience. |
| **[5]** StutteringSpeech Challenge (AS‑70) | **AS‑70** challenge split: conversational and command reading speech from PWS [5,6] | Mandarin Chinese | Annotated stuttering events, used for both SED and ASR; contains repetitions, prolongations, etc. [6]. |
| **[6]** AS‑70 dataset | AS‑70, with **speaker‑exclusive splits** and **stuttering‑rate (SR) severity bands** (mild, moderate, severe) [6] | Mandarin Chinese | Verbatim transcripts with stuttering events; severity stratification via SR metric (mild ≤7%, moderate 7–12%, severe >12%) [6]. |
| **[7]** FluencyBank Timestamped | **FluencyBank (updated)** with timestamps and disfluency labels; comparisons to Switchboard (fluent/disfluent but not stuttering) [7] | English | Labels partial words, filled pauses, repetitions, revisions; includes speech from people who stutter vs typical Switchboard speakers [7]. |
| **[8]** CHI 2023 PWS study | New dataset of VA‑style utterances from 91 PWS (plus survey data), with severity annotations [8] | English | Natural stuttering (sound/syllable repetitions, prolongations, blocks, etc.); analyzed by severity group and task type. |
| **[9]** VA‑Dysfluent / VA‑Fluent | **VA‑Dysfluent**: 18 PWS; **VA‑Fluent**: matched fluent speakers [9] | English | Voice‑assistant tasks with clinically graded stutter severity; intended‑speech transcripts used (disfluencies removed/normalized) [9]. |
| **[10]** Podcasts with PWS | Real‑world **podcast recordings featuring people who stutter**, segmented for analysis [10] | Likely English | Mix of fluent and stuttered segments; special analysis of **sound repetitions** and their effect on hallucinations [10]. |
| **[11]** RNN‑T disfluency robustness | Internal **Disfluencies Test** (field data) and **Stutter Test** (quiet room, limited prompts) [11] | Not explicitly stated; context suggests English | Disfluencies Test: ordinary disfluencies; Stutter Test: speech from PWS; partial words and repetitions emphasized [11]. |
| **[12]** StutterZero/StutterFormer | Training on **SEP‑28k‑derived synthetic stuttered→fluent pairs** and **LibriStutter**, evaluation on **unseen FluencyBank speakers** [12] | English | Stutter types synthesized from SEP‑28k and LibriStutter; FluencyBank evaluation uses real PWS speech [12]. |
| **[13]** Whisper fine‑tuning | **FluencyBank** + new **HAI dataset** (job‑interview‑style spontaneous & read prompts from 18 PWS) [13] | English | Real stuttering, with repeated syllables and prolongations analyzed; personalized vs generalized models studied [13]. |

**Comparative observations:**

- **Mandarin vs English:** Mandarin stuttered ASR is covered primarily by **AS‑70 and the StammerTalk community dataset** ([2,4,5,6]). English stuttered ASR is covered by **FluencyBank** and related derivatives ([1,7,12,13]), **VA‑Dysfluent** ([9]), **Boli** ([3]), and **podcast data** ([10]).
- **Synthetic vs real stuttering:** **[1]** and **[12]** explicitly use **synthetic disfluencies** (LS‑Y, LibriStutter, synthetic SEP‑28k pairs) in addition to real PWS data, enabling controlled comparisons but raising questions of ecological validity versus purely real datasets like FluencyBank, AS‑70, Boli, VA‑Dysfluent, etc.

---

### Comparative Performance Findings: WER Degradation and Error Patterns

#### **System‑Level WER Comparisons Where Available**

Below is a consolidated view of the **most prominent quantitative WER findings**, restricted to what is explicitly described in the summaries.

| Ref | Systems compared | Stuttered/disfluent WER (or isWER) vs fluent | Notable relative degradations / findings |
|-----|------------------|----------------------------------------------|------------------------------------------|
| **[1]** Lost in Transcription | **Six contemporary ASR systems** (mix of commercial + open‑source, including Google Cloud V1; architectures referencing wav2vec2 & Whisper) [1] | WER & CER are reported per system for **FB‑Y vs FB‑N** and **LS‑Y vs LS‑N**; summary notes systematic WER increases in disfluent conditions [1]. | Shows **consistent bias**: all systems have **higher WER on disfluent (Y) vs non‑disfluent (N)** datasets. Correlations: specific disfluency types (prolongations, interjections, etc.) are significantly associated with increased WER (p < 0.05) [1]. Semantic metrics (FBERT) show semantic degradation beyond surface errors. |
| **[3]** Boli | **Wav2Vec2.0 vs Whisper** on English Boli subset [3] | For concatenated/pooled utterances: **Wav2Vec2 WER 29.22% vs Whisper 4.55%**; for speaker‑wise averaging: **Wav2Vec2 55.81% vs Whisper 21.27%** [3]. | Whisper dramatically outperforms Wav2Vec2. Authors note **word‑repetition stutters were well identified by both models**, but absolute WER gap is large [3]. |
| **[5]** StutteringSpeech Challenge findings | Multiple top systems from challenge teams [5] | Paper reports **reductions in recognition error rates for stuttered speech** for top systems vs baselines [5]. Exact WERs by system are not in the summary. | Shows that **specialized architectures (e.g., Zipformer)** and data augmentation can meaningfully reduce WER on stuttered Mandarin [5]. |
| **[6]** AS‑70 baseline | Pretrained models (e.g., Whisper, HuBERT) fine‑tuned vs zero‑shot [6] | Fine‑tuned models show **significant WER reductions** vs zero‑shot pretrained models on AS‑70 [6]. | Highlights **poor zero‑shot performance** of generic SOTA models on Mandarin stuttered speech and the benefits of targeted fine‑tuning [6]. |
| **[7]** FluencyBank Timestamped | **Whisper** for intended speech recognition [7] | Benchmarks Whisper on **intended speech (disfluencies removed)** for PWS vs Switchboard; WER differences indicate robustness gaps (exact numbers not in summary) [7]. | Intended‑speech framing: disfluencies are removed in references, so models that “normalize” disfluencies may be favored. |
| **[8]** CHI 2023 PWS study | Single **Apple Speech framework** VA ASR; multiple tuned configurations [8] | Baseline mean WER across PWS: **19.8%**; mild severity ~**4.8%**, moderate–severe baseline **25.4%** [8]. After best intervention combo, WER for moderate–severe group reduced to **9.9%** [8]. | Shows **large disparity by severity** and that **endpoint & decoding tuning** can yield >50% relative WER reductions for moderate–severe PWS without retraining [8]. |
| **[9]** VA‑Dysfluent | Single hybrid VA ASR baseline vs tuned decoding [9] | Baseline intended‑speech WER (isWER): **VA‑Fluent 5.65% vs VA‑Dysfluent 19.29%** [9]. By severity: Grade1 8.39%, Grade2 16.64%, Grade3 47.86% [9]. Adjusted decoding yields **24% relative isWER reduction** for PWS [9]. | One of the clearest demonstrations of **severity‑dependent degradation** and of **benefits from decoding adjustments** without acoustic retraining [9]. |
| **[10]** Just Stutter | **Whisper** (single model) [10] | Shows **significant disparities between fluent and stuttered segments**; particularly poor performance on **sound repetitions** which also cause >20% hallucination rate [10]. | Provides granular error patterns: **sound repetition stutters** uniquely problematic, both for WER and hallucinations [10]. |
| **[11]** RNN‑T robustness | Several RNN‑T variants with different training sets/transcript treatments [11] | Best model achieves **22.5% and 16.4% relative WER reductions** on disfluencies and stutter tests, respectively, vs baseline [11]. | Demonstrates that adding disfluent training data and appropriate **partial‑word labeling strategies** can significantly reduce WER on stuttered/disfluent sets without harming fluent test performance [11]. |
| **[12]** StutterZero/StutterFormer | **StutterZero** and **StutterFormer** vs **Whisper‑Tiny/Small/Medium** [12] | StutterZero yields **24% WER reduction** and **31% BERTScore improvement** vs Whisper‑Medium; StutterFormer yields **28% WER reduction** and **34% BERTScore improvement** [12]. Accuracy: StutterFormer 88–90%, StutterZero 84–88% on validation/FluencyBank [12]. | Claims statistically significant (Wilcoxon p<0.05) WER and semantic improvements over Whisper baselines on stuttered speech, via end‑to‑end stutter‑to‑fluent conversion [12]. |
| **[13]** Fine‑tuning Whisper | Whisper‑Small baseline vs generalized vs personalized LoRA‑fine‑tuned models [13] | Personalized models **substantially reduce WER**, especially on spontaneous speech; exact numbers not given in summary [13]. | Suggests **speaker‑specific fine‑tuning** is particularly beneficial for PWS, complementing generalized adaptation [13]. |

**Comparative observations specifically for multi‑system / cross‑system findings:**

- On **English stuttered speech**, **Whisper** is consistently shown as **stronger than previous open‑source models**:
  - Boli: Whisper WER 4.55–21.27% vs Wav2Vec2.0 29.22–55.81% [3].
  - StutterZero/StutterFormer: Whisper‑Medium is outperformed by specialized stutter‑conversion models but still provides a strong baseline [12].
- On **Mandarin stuttered speech**, **zero‑shot SOTA models perform poorly** and require adaptation:
  - AS‑70 fine‑tuning significantly reduces WER vs pretrained models [6].
  - StutteringSpeech Challenge shows that **specialized architectures and augmentations** can further reduce WER [5].
- **Disfluency‑type sensitivity** is a consistent theme:
  - **[1]** quantifies WER increases associated with specific disfluency types using correlation & statistical tests.
  - **[10]** highlights sound repetitions as the worst case for Whisper, including hallucinations.
  - **[3]** notes word‑repetition stutters are relatively well captured by both tested models, suggesting some stutter categories are less harmful than others.
  - **[11]** shows that **partial‑word treatment** (remove/replace/preserve) in training has measurable impact on disfluency/stutter WER.

---

### Metrics Beyond WER: CER, Semantic Accuracy, User Experience, and Fairness

#### **Metric Coverage by Paper**

| Ref | Metrics used (beyond WER) | Focus / contribution |
|-----|---------------------------|----------------------|
| **[1]** Lost in Transcription | **WER, CER, semantic‑similarity metrics (e.g., FBERT)**; per‑model WER distributions; correlation analyses [1] | Offers the most **comprehensive metric suite**: surface error, character‑level, and semantics. It enables distinguishing cases where WER is high but semantics preserved, and quantifies **bias via WER distributions and correlations with disfluency types**. |
| **[2]** Community Chinese dataset | WER and possibly other ASR metrics; fairness analysis on PWS [2] | Emphasis on **fairness and inclusion**: benchmarking “popular ASR models” to expose biases against Chinese PWS; details expanded in [4]. |
| **[3]** Boli | WER (two aggregation schemes) [3] | Focus on using WER to validate the dataset and highlight **model differences**; no user or semantic metrics. |
| **[4]** Our Collective Voices | WER and fine‑tuned performance; likely fairness analyses [4] | Integrates **technical benchmark results with sociotechnical and equity considerations** for Chinese PWS. |
| **[5]** StutteringSpeech Challenge | Recognition error rates (WER or similar), plus SED metrics [5] | Combines **ASR and stuttering‑event detection metrics**, allowing joint evaluation of recognition and event detection. |
| **[6]** AS‑70 | WER and SED metrics; severity‑stratified results [6] | Highlights **severity‑conditioned ASR performance**, enabling fairness‑flavored analyses across SR bands. |
| **[7]** FluencyBank Timestamped | WER for intended speech recognition; disfluency detection precision/recall metrics [7] | Provides a **shared resource** and baseline where ASR is evaluated on intended speech; disfluency detection metrics help separate recognition vs detection performance. |
| **[8]** CHI 2023 PWS study | **WER, truncation rate, thresholded WER (<10%,<15%), Intent Error Rate**, user‑reported usability measures [8] | Strong emphasis on **task success and UX**: shows that lower WER and fewer truncations track with better downstream intent and user experience. |
| **[9]** VA‑Dysfluent | **Intended‑speech WER (isWER)**, insertion/deletion/substitution error rates, domain‑ and intent‑recognition accuracy [9] | Detailed **error‑type breakdown** and **task‑oriented metrics** (domain/intent), showing that disfluencies especially increase **insertions** and may affect NLU. |
| **[10]** Just Stutter | WER, hallucination rate (>20% on some sound repetition cases) [10] | Focuses on **hallucinations as a qualitative/quantitative failure mode**, not just conventional WER. |
| **[11]** RNN‑T robustness | WER and insertion behavior [11] | Primarily WER, with supporting analysis of **insertion errors** for disfluency conditions. |
| **[12]** StutterZero/StutterFormer | **WER, CER, BERTScore (semantic similarity)**; significance tests (Wilcoxon p<0.05) [12] | Similar to [1], provides both surface and semantic metrics, plus **statistical significance**, making cross‑model comparisons more rigorous. |
| **[13]** Fine‑tuning Whisper | **WER, CER** [13] | Focuses on **absolute and relative WER/CER gains** from generalized vs personalized fine‑tuning. |

#### **Fairness, Severity, and Group‑Comparative Analyses**

- **Severity‑stratified performance:**
  - **[8]** and **[9]** explicitly report WER as a function of **clinical stutter severity**, revealing dramatic gaps (e.g., Grade 3 isWER ≈47.86% vs Grade 1 8.39% [9]; moderate–severe group 25.4% baseline WER vs ~4.8% for mild [8]).
  - **[6]** introduces a quantitative **Stuttering‑Rate metric (SR)** with thresholds for mild, moderate, severe and constructs splits balanced across severity, which can be used to analyze WER disparities [6].
- **PWS vs non‑PWS comparisons:**
  - **[9]**: VA‑Fluent vs VA‑Dysfluent (5.65% vs 19.29% isWER) [9].
  - **[7]**: FluencyBank vs Switchboard, enabling comparisons between PWS and typical speakers; used to assess robustness differences [7].
- **Fairness / inclusion framing:**
  - **[2]** and **[4]** frame their benchmarking explicitly in terms of **fairness, inclusion, and community voice** for Chinese PWS.
  - **[1]** emphasizes **accuracy bias** via disaggregated WER distributions and statistically tested correlations between disfluency types and error rates.
  - **[10]** is **disability‑led** and interprets Whisper’s disparities and hallucinations as an equity issue for people who stutter.

- **Task‑level and user‑perceived quality:**
  - **[8]**: Richest integration of ASR metrics with **user metrics**: truncation, intent error rate, user surveys. Shows that technical tuning can significantly reduce user‑experienced frustration and improve acceptability for PWS [8].
  - **[9]**: Links improvements in isWER to **domain and intent recognition gains** (e.g., +3.6% domain and +1.7% intent recognition after decoding tuning) [9], showing that even moderate WER changes can yield downstream task benefits.

---

### Methodological Contrasts Relevant for Comparative Benchmark Design

#### **Reference Conventions (Surface vs Intended Speech)**

- **Intended‑speech references (disfluencies removed/normalized):**
  - VA‑Dysfluent / VA‑Fluent [9], FluencyBank intended‑speech task [7], and some analyses in [8] evaluate ASR on **intended fluent transcripts**, penalizing models that faithfully reproduce actual disfluencies.
  - This is similar in spirit to FluencyBank **FB‑N** condition in [1] and the intended‑speech focus in [11].
- **Surface‑faithful references (disfluencies included):**
  - **AS‑70** [6], **Boli** [3], the FluencyBank **FB‑Y** condition in [1], and the podcast analyses in [10] keep disfluencies in transcripts; systems that “clean up” disfluencies can incur higher WER.
- **Implication:** When comparing across papers or designing new benchmarks, it is crucial to **match reference conventions**, since they can invert which systems appear better (a model that aggressively normalizes may look good on intended‑speech WER but bad on surface‑faithful WER).

#### **Streaming vs offline and decoding controls**

- **Streaming/VA context:**
  **[8]** and **[9]** directly address **endpointing, truncation, and streaming behavior**, showing that stuttering interacts strongly with VAD and partial results. They demonstrate substantial WER improvements and truncation reductions via **decoder tuning and endpointer adjustments** without retraining the acoustic model.
- **Offline / batch context:**
  Most others ([1,2,3,4,5,6,7,10,11,12,13]) operate in **offline recognition** mode, with focus on transcript accuracy rather than latency or partial hypothesis stability.

#### **Comparative evaluation structure**

- **True multi‑system matched benchmarks:**
  - **[1]** and **[3]** are the clearest examples with fully matched evaluation pipelines across multiple off‑the‑shelf ASR systems on the same disfluent/stuttered data.
- **Multi‑variant within one architecture:**
  - **[5]** (challenge systems), **[6]**, **[11]**, **[12]**, and **[13]** all compare **variants of SOTA models** (different training data, architectures, or fine‑tuning strategies) rather than heterogeneous APIs.
- **Single‑system + configuration tuning:**
  - **[7,8,9,10]** examine the behavior of a **single ASR system** (Whisper or a proprietary VA ASR) under different data conditions, tuning, or analysis breakdowns.

---

### Summary: Distinctive Contributions for an Expert Focusing on Comparative Benchmarking

- **[1]** provides the **most comprehensive cross‑system benchmark** on disfluent and stuttered speech, uniquely combining:
  - Multiple commercial and open‑source SOTA models.
  - Real vs synthetic disfluency datasets (FluencyBank vs LS‑Y).
  - WER/CER + semantic similarity and statistical bias analyses.

- **[3]** gives **clean, concrete WER numbers** comparing Wav2Vec2.0 vs Whisper on a carefully annotated stuttered dataset, illustrating current SOTA dominance of Whisper and showing how strongly model choice matters.

- **[2,4,5,6]** collectively establish **Mandarin stuttered benchmarks**, including:
  - Community‑led dataset creation ([2,4]).
  - A challenge setting with multiple advanced systems ([5]).
  - Detailed, severity‑aware baselines and fine‑tuning analyses ([6]).

- **[8,9]** foreground **user and severity perspectives** in production VA contexts, showing both:
  - The magnitude of performance gaps between PWS and non‑PWS, and across severity levels.
  - That substantial gains are possible via **decoding/endpoint tuning** and interface interventions, even without retraining.

- **[10]** exposes **failure modes (hallucinations)** of a modern SOTA model (Whisper) on specific stuttering patterns (sound repetitions), suggesting that WER alone underestimates the problem.

- **[11,12,13]** demonstrate diverse **adaptation strategies** (data selection & labeling [11], stutter‑to‑fluent conversion [12], generalized vs personalized fine‑tuning [13]) that substantially reduce WER on stuttered speech, reinforcing the conclusion drawn in [1,6,8,9] that **off‑the‑shelf SOTA ASR remains insufficiently robust** to stuttering unless explicitly adapted.

For future benchmarking work aligned with your goals, **[1,3,5,6], and [2]/[4]** are the most relevant anchors: they provide multi‑system or multi‑variant performance baselines and methodological patterns for fair, cross‑system evaluation on stuttered speech, while **[8,9,10,11,12,13]** offer rich design insights on metrics, annotation conventions, and adaptation levers that significantly influence comparative WER outcomes.

## Timeline

### Overview of the Research Trajectory

#### **High-level progression**

Across these works, you can see a clear trajectory:

- **2020–2021:** *Robustness to disfluencies* is treated mostly as an engineering problem within a single production ASR stack, using proprietary data and models. Focus is on **tuning and training choices** rather than cross-system benchmarking [9,11].
- **2023:** The field begins to explicitly integrate **human–computer interaction and user perspectives**, focusing on how people who stutter actually experience ASR and which error patterns matter most in practice [8].
- **2024:** Major **infrastructural milestones**—large stutter-specific corpora (AS‑70, FluencyBank Timestamped) and community-led Mandarin datasets—are released, and the first *challenge* on stuttering ASR appears [2,5,6,7]. Evaluation is still mostly per-system, but now firmly grounded in publicly describable resources.
- **2024–2025:** We start to see **true comparative benchmarking across multiple modern ASR systems**, including commercial APIs and open-source SOTA models such as Whisper and wav2vec 2.0 [1,3]. There is also a growing focus on **bias/fairness, disability-led research, and semantic/intent-level metrics** [1,2,4,8,10].
- **2025 onward:** Fine-grained **error analysis by disfluency type**, **personalized vs generalized adaptation**, and **end-to-end stutter-to-fluent conversion** suggest a shift from merely documenting deficits to designing **targeted remediation strategies** benchmarked against state-of-the-art ASR [10,12,13].

---

### Early Work: Making a Single ASR More Robust to Disfluencies (2020–2021)

#### **RNN‑T robustness and disfluency-aware training (2020–2021)**

- **Mendelev et al. (2020/ICASSP 2021)** [11] is an early technical milestone:
  - Focuses on **RNN‑Transducer robustness** to disfluencies, with a dedicated *Stutter Test* set alongside a more general *Disfluencies Test* and a fluent *Ordinary Test*.
  - Systematically varies **training data composition** (how much disfluent data to include) and **representation choices** for partial words (removal, special tag, or character-preserving).
  - Reports **relative WER reductions** of ~22.5% and 16.4% on disfluency and stutter sets, respectively, with no degradation on fluent speech [11].
- Conceptual significance:
  - Treats **stuttering and disfluencies as part of the robustness envelope** for mainstream RNN‑T systems, rather than as an entirely separate special case.
  - However, **only one underlying ASR system** is studied; the focus is intra-system design choices, not cross-system benchmarking.

#### **Voice assistant tuning for dysfluent speech (2021)**

- **Mitra et al. (Interspeech 2021)** [9]:
  - Evaluates a single **consumer hybrid voice-assistant ASR** on **VA-Fluent vs VA-Dysfluent** datasets from 18 adults who stutter.
  - Introduces **intended-speech WER (isWER)** as the primary metric, emphasizing *cleaned* fluent intent despite surface disfluencies:
    - isWER rises from **5.65% (fluent)** to **19.29% (dysfluent)**.
    - Within the stuttering group, severity strongly drives WER (e.g., grade 3 ≈ 48% isWER) [9].
  - Investigates **decoder tuning** and stutter detector-informed pipeline changes, achieving ~24% relative isWER improvement without changing the acoustic model [9].
- Methodological footprint:
  - Establishes **severity-stratified performance analysis** and explicit **intent-level evaluation** (domain and intent recognition rates).
  - Still **single-system** and proprietary, with no comparison against other commercial or open-source ASRs.

**Takeaway for this phase:**
The main concern is *how to keep one production system usable on disfluent/stuttered speech*. WER and isWER are central, but the perspective is system-centric and closed. There is not yet a culture of comparative benchmarking across heterogeneous ASR models.

---

### Human-Centered Turn: User Experience and Intent-Level Metrics (2023)

#### **CHI 2023: From user perceptions to technical tuning**

- **Lea et al. (CHI 2023)** [8] marks a critical pivot:
  - Combines **survey data** from 61 people who stutter with a **recording study** of 91 participants to quantify **real-world usability** of a consumer-grade ASR (Apple Speech framework).
  - Primary quantitative metrics:
    - **WER** stratified by stuttering severity (e.g., mild ≈ 4.8%, moderate–severe baseline ≈ 25.4%) [8].
    - **Truncation rates** and **thresholded WER** (e.g., proportion of utterances with WER < 10% or < 15%).
    - **Intent Error Rate**, by feeding ASR output into an NLU model and assessing downstream task accuracy [8].
  - Evaluates three interventions (endpointer tuning, decoder tweaks, and “dysfluency refinements”), showing large reductions in truncation and WER for more severely affected users [8].
- Conceptual shifts:
  - Strong focus on **people who stutter as end-users**, not just on model robustness.
  - Puts **task success and user-perceived quality** on equal footing with WER, foreshadowing later fairness and disability-centered work [2,4,10].
  - Nonetheless, still benchmarks **only one commercial system**; no multi-system comparison.

**Impact on later work:**
This paper is widely cited in subsequent stuttering-ASR research [1,2,3,4,5,6,7,10,12,13] and helps normalize:
- The use of **intended vs surface transcription distinctions**.
- The inclusion of **user-centered and task-based metrics** alongside WER.
- The framing of disparities as an **accessibility and fairness problem**, not just a robustness issue.

---

### Infrastructural Milestones: Public Stuttered-Speech Corpora & Challenges (2024)

#### **AS‑70: Large Mandarin stuttered-speech dataset for ASR and SED (2024)**

- **Gong et al. (AS‑70)** [6]:
  - Introduces the **first large, publicly available Mandarin stuttered-speech dataset** explicitly designed for ASR and stuttering event detection.
  - Contains both **conversational** and **command-reading** speech, with **verbatim manual transcriptions** and **stuttering-rate–based severity labels** [6].
  - Provides **baseline ASR experiments** with modern pretrained encoders (e.g., Whisper, HuBERT), showing:
    - Large performance gaps when models are evaluated **zero-shot** on stuttered speech.
    - Significant improvements when **fine-tuned** on AS‑70 [6].
- Importance for benchmarking:
  - Establishes a **standard resource** for evaluating and comparing ASR on Mandarin stuttered speech.
  - Supports **speaker-exclusive splits and severity-balanced partitions**, enabling robust subgroup analysis.

#### **FluencyBank Timestamped: enabling systematic stuttered-speech evaluation (2024)**

- **Romana et al. (2024)** [7]:
  - Retrofit the widely known **FluencyBank** corpus with **timestamps and detailed disfluency labels** for each token.
  - Benchmark **Whisper** for **“intended speech recognition”** (transcribing the underlying fluent message) on FluencyBank vs Switchboard [7].
- Significance:
  - FluencyBank is already a key clinical stuttering dataset; adding **time-aligned, type-labeled disfluencies** makes it a **central benchmark** for ASR robustness studies.
  - Provides shared infrastructure for **cross-ASR comparisons** and **disfluency-type–specific error analysis**, which later work (e.g., [1,10,13]) leverages.

#### **Mandarin stuttering challenge and benchmarking (SLT 2024)**

- **Xue et al. (SLT 2024 StutteringSpeech Challenge)** [5]:
  - Launches a **challenge with three tracks:** stuttering event detection, ASR on stuttered speech, and open research.
  - Uses a processed form of the open AS‑70 corpus as the shared dataset [5].
  - Summarizes the performance of **multiple submitted systems** (research teams), highlighting methods such as Zipformer architectures and augmentation strategies that reduce WER on stuttered speech [5].
- Role in the field:
  - Moves the community toward **shared, comparative evaluation** on a common stuttered-speech benchmark—albeit across contest submissions rather than standardized commercial APIs.

#### **Community-led Mandarin dataset and initial benchmarking (CHI EA 2024)**

- **Li & Wu (CHI EA 2024)** [2]:
  - Report creation of the **first community-led Mandarin stuttered speech dataset**, with both spontaneous conversation and voice commands from 72 people who stutter.
  - Conduct **benchmarking of popular ASR models** on this dataset to probe **biases against disfluent speech** [2].
- Conceptual importance:
  - Introduces a **grassroots, disability-led data collection model** where the stuttering community shapes dataset design and consent.
  - Begins to systematically examine **fairness and inclusion** in Mandarin ASR, beyond accuracy alone.

**Takeaway for 2024:**
This year is dominated by **dataset and infrastructure creation**—AS‑70, FluencyBank Timestamped, the Mandarin community dataset, and the StutteringSpeech Challenge. These works make comparative benchmarking *possible at scale* and anchor the field around a few shared corpora. They also emphasize **Mandarin and multilingual** perspectives, pushing beyond the earlier English-centric focus.

---

### True Cross-System Benchmarks & Bias Quantification (2024–2025)

#### **Lost in Transcription: multi-system ASR bias analysis (2024)**

- **Mujtaba et al. (2024)** [1] is arguably the first **systematic, multi-model benchmark** squarely aligned with your goal:
  - Evaluates **six contemporary ASR systems** (mix of commercial and open-source SOTA—explicitly includes Google Cloud Speech‑to‑Text V1 and models inspired by wav2vec 2.0/Whisper) [1].
  - Uses two real vs synthetic pairs:
    - **FluencyBank (FB‑Y)** vs a **cleaned/edited version (FB‑N)**.
    - **LibriSpeech (LS‑N)** vs a **synthetically disfluent LibriSpeech (LS‑Y)** created via TTS and transformations [1].
  - Primary metrics:
    - **WER** and **CER** across systems and conditions.
    - **Semantic metrics** (e.g., FBERT) capturing semantic accuracy beyond surface WER.
    - **Disfluency-type correlations** with WER (e.g., prolongations, interjections) with statistical significance (p < 0.05) [1].
  - Provides **per-model WER distributions**, matched-condition comparisons, and explicit **bias quantification**: how much each system’s error rate inflates from clean to disfluent/stuttered conditions.
- Why it is a milestone:
  - Delivers the **kind of comparative benchmark the field lacked earlier**: multiple heterogeneous ASRs evaluated under *matched conditions* on both real and synthetic disfluencies.
  - Integrates **semantics** and **statistical testing**, moving evaluation beyond raw WER to a more nuanced view of performance and bias.

#### **Project Boli: multilingual stutter dataset with Whisper vs wav2vec 2.0 (ICASSP 2025)**

- **Batra et al. (Project Boli)** [3]:
  - Introduces a **multilingual stuttered-speech dataset** (primarily Indian languages) with **word-level labels for five stutter types** and rich metadata.
  - For technical validation, conducts a **direct comparison**:
    - **wav2vec 2.0 vs Whisper** on English audio.
    - Reports WER under two evaluation schemes: **concatenated/pooled** and **speaker-wise averaged** [3].
      - Wav2Vec2: 29.22% / 55.81%.
      - Whisper: 4.55% / 21.27% [3].
  - Notes qualitative patterns—for example, **word-repetition stutters are well recognized by both models** [3].
- Importance:
  - Combines **explicit stutter-type annotation** with **cross-model benchmarking**, enabling analysis not just of overall WER but of **which stutter types are easier/harder for different architectures**.
  - Confirms that **modern large-scale models (Whisper)** can substantially outperform earlier pretrained encoders on stuttered speech, even without adaptation.

#### **Our Collective Voices: deepening the Mandarin community dataset (FAccT 2025)**

- **J. Li et al. (FAccT 2025)** [4]:
  - Extends the CHI EA 2024 work [2] with both **technical and social analysis** of the community-led Mandarin dataset.
  - Includes **benchmarking and fine-tuning** of ASR models on this dataset, and examines implications for **fluency bias, inclusion, and structural change** [4].
- Trend:
  - This work further solidifies **disability-led, fairness-oriented research** in the ASR-for-stuttering space, connecting benchmarks to **social values and advocacy**.

#### **Whisper-focused fairness and hallucination analysis (Interspeech 2025)**

- **Sridhar & Wu (Interspeech 2025)** [10]:
  - Evaluates **OpenAI Whisper** on speech samples from **podcasts featuring people who stutter**.
  - Provides **fine-grained analysis by disfluency type**, finding:
    - Larger performance disparities for **sound repetitions**, a stuttering-specific disfluency.
    - **>20% hallucination rate** triggered by sound repetitions—i.e., Whisper fabricates content [10].
- Significance:
  - Deepens understanding of **failure modes** of state-of-the-art open-source ASR on stuttered speech, especially **hallucinations** rather than mere mis-recognition.
  - Connects technical analysis with **disability-led research practices**, echoing the community-led ethos of [2,4].

**Takeaway for this phase:**
These works, particularly [1] and [3], finally deliver **explicit multi-system benchmarks** on stuttered speech with WER (and related metrics) as primary outcomes. The research community is shifting from “can we make our single system cope with stuttering?” to “**how do different SOTA systems compare, and where are the biases and hallucinations?**” Fairness and ethics become central themes, not side remarks.

---

### Adaptation, Personalization, and Stutter-to-Fluent Conversion (2025)

Although your focus is on benchmarking rather than adaptation, **recent method-focused papers are themselves evaluated against state-of-the-art baselines on stuttered speech**, and their experimental designs reflect how the field thinks about evaluation.

#### **Personalized vs generalized fine-tuning of Whisper (2025)**

- **Mujtaba & Mahapatra (2025)** [13]:
  - Fine-tune **Whisper-Small** using parameter-efficient **LoRA** adapters on:
    - **FluencyBank** and
    - A new **HAI dataset** of stuttered job-interview–style speech and read prompts [13].
  - Compare **generalized models** (trained across speakers) versus **personalized models** (speaker-specific adapters) using **WER and CER**.
  - Show that **personalized models significantly reduce WER**, particularly for spontaneous speech [13].
- Relevance to benchmarking:
  - Uses the same modern backbone (Whisper) that appears in multi-system benchmarks [1,3,7,10,12], allowing **cross-paper comparability** of performance ranges.
  - Continues the trend of **severity-, style-, and disfluency-aware evaluation**, though primarily comparing model variants, not different vendors.

#### **StutterZero and StutterFormer: end-to-end stutter-to-fluent conversion (2025)**

- **Xu (2025)** [12]:
  - Proposes **waveform-to-waveform models** (StutterZero, StutterFormer) that convert stuttered speech to fluent audio and simultaneously generate transcripts.
  - Benchmarks these models against **Whisper-Tiny/Small/Medium** on **SEP-28K/LibriStutter-derived training data** and **FluencyBank** test speakers [12].
  - Reports:
    - **WER and CER**, plus **BERTScore** for semantic similarity.
    - Relative WER reductions of **24% (StutterZero)** and **28% (StutterFormer)** vs Whisper-Medium; BERTScore improvements of 31% and 34%, respectively [12].
- Significance:
  - Although centered on a new modeling paradigm, **the evaluation is inherently comparative**, using multiple Whisper variants as baselines on real stuttered speech.
  - Strengthens the norm that **stutter-specific systems must be compared against modern SOTA ASR**, not just against older baselines.

**Takeaway for this phase:**
The conversation moves from “stuttering breaks ASR” toward “which adaptation strategies or novel architectures can restore parity, and how do they fare against the best existing systems?” Evaluation protocols increasingly include:

- **Multiple Whisper-sized baselines**,
- **WER + CER + semantic metrics**, and
- Testing on **real stuttered corpora** (FluencyBank, SEP‑28k derivatives), not just synthetic disfluencies.

---

### Key Trends in Methods, Metrics, and Framing

#### **1. From single proprietary systems to broad multi-system comparisons**

- Early work [9,11] focuses on **one proprietary or internal system** with tuning and training variation.
- Mid-period user-centered work [8] is still single-system, but adds **user-level and task-level metrics**.
- Later works [1,2,3,5,6,12] evaluate:
  - **Multiple off-the-shelf models** (e.g., Whisper vs wav2vec 2.0 [3]),
  - **Multiple vendor APIs** (e.g., inclusion of Google Cloud Speech-to-Text V1 and other systems [1]),
  - **Multiple model variants within the same architecture** (Whisper-Tiny/Small/Medium [12]).
- This reflects a **maturation of benchmarking culture**, aligning stuttered-speech evaluation with mainstream ASR practice.

#### **2. Increasing reliance on shared, well-annotated stuttering resources**

- Early stutter evaluation sets in [11] are internal, not clearly reusable.
- **FluencyBank** becomes progressively central:
  - Used implicitly in [9,11], then explicitly instrumented with detailed annotations and timestamps in [7].
  - Deployed in **comparative benchmarks** [1,12,13].
- **Mandarin corpora and challenges** (AS‑70 [6], StutteringSpeech Challenge [5], community dataset [2,4]) broaden the **language and dialect coverage** and formalize **shared benchmarks**.
- **Project Boli** [3] introduces multilingual coverage with fine-grained stutter-type labeling, enabling **cross-lingual and cross-type analyses**.

#### **3. Evolution of evaluation metrics**

- **WER** is consistently primary, but its usage becomes more nuanced:
  - **Intended-speech WER (isWER)** in [9] and **“intended speech recognition”** in [7] emphasize evaluating models on *underlying fluent content*.
  - Stratification by **severity** and **utterance type** (commands vs conversational) appears in [8,9,6].
- **CER** gains prominence in later work [1,6,12,13], reflecting:
  - Interest in **partial-word and character-level robustness**, particularly relevant for stuttered fragments and morphologically rich languages.
- **Semantic and task-based metrics**:
  - NLU-based **Intent Error Rate** and task success in [8].
  - **FBERT/BERTScore** for semantic similarity in [1,12].
- **Higher-level outcomes**:
  - **Truncation rates** and stability metrics in streaming voice-assistant contexts [8,9].
  - **Hallucination rates** (e.g., >20% under sound repetitions for Whisper [10]).
  - **Fairness and bias measures** (group-level gaps, community-led interpretations) in [1,2,4,10].

#### **4. Interpretation of disfluencies and annotation conventions**

- Early work experiments with **removing vs tagging partial words** [11], highlighting how transcript conventions impact WER.
- Later works become explicit about:
  - **Intended vs surface transcripts** [7,9].
  - Inclusion/exclusion of **filled pauses and interjections** [1,7].
  - Use of **detailed stutter-type labels** for analysis (blocks, prolongations, sound repetitions, word repetitions) [3,6,7,10].
- This reflects an awareness that **evaluation protocols can materially change which systems appear best**, especially when some models “normalize” disfluencies while others reproduce them.

#### **5. Shift in framing: from robustness to fairness and disability rights**

- **Robustness framing** dominates in [9,11]: stuttering is a stressor on ASR performance.
- **User-centered and accessibility framing** emerges in [8], where user perceptions and task success are central.
- **Fairness and inclusion** become explicit themes in:
  - Community-led dataset creation [2,4].
  - Bias quantification and semantic disparity analysis [1].
  - Disability-led research analyzing hallucination and error patterns [10].
- This shift suggests that future benchmarks will increasingly be evaluated not only on **absolute WER**, but also on **equity metrics** and **alignment with the lived experience of people who stutter**.

---

### Key Research Clusters and Their Contributions

#### **Lea / Findlater / Wu and collaborators: HCI and community-led stuttering datasets**

- **Lea et al. (CHI 2023)** [8]:
  - Set the standard for **user-centered ASR evaluation** for people who stutter, integrating WER, truncation, and intent metrics.
- **Li & Wu (CHI EA 2024)** [2] and **J. Li et al. (FAccT 2025)** [4]:
  - Lead the creation and deeper analysis of a **community-led Mandarin stuttered speech dataset**, emphasizing **fairness, lived experience, and participatory methods**.
- **Sridhar & Wu (Interspeech 2025)** [10]:
  - Provide a **Whisper-focused** analysis of **disfluency-type–specific disparities and hallucinations**, conducted by researchers who stutter.
- **Implication:**
  This cluster is pushing the field toward **disability-led, ethics-aware benchmarking** that connects technical metrics with **social outcomes and advocacy**. Future work from this group will likely continue to foreground **equity metrics and participatory evaluation**.

#### **Mujtaba / Mahapatra and collaborators: bias quantification and adaptation**

- **Mujtaba et al. (2024)** [1]:
  - Provide the first **large-scale multi-system bias analysis** across commercial and open-source ASR on both real and synthetic disfluencies, with WER/CER and semantic metrics plus statistical significance.
- **Mujtaba & Mahapatra (2025)** [13]:
  - Explore **personalized vs generalized fine-tuning** of Whisper on stuttered speech, systematically measuring gains in WER/CER on FluencyBank and HAI.
- **Implication:**
  This thread combines **rigorous benchmarking** with **adaptation strategies**, and it is positioned to drive future work on **personalization, fairness-aware training, and standardization of benchmarks**.

#### **Mandarin / AS‑70 / challenge community (Gong, Xue, Li, Wu, others)**

- **Gong et al. (AS‑70)** [6]:
  - Foundational dataset and baseline paper for **Mandarin stuttered speech**.
- **Xue et al. (StutteringSpeech Challenge)** [5]:
  - Catalyze a **competitive benchmarking environment** on AS‑70, encouraging diverse modeling approaches.
- **Li & Wu / J. Li et al.** [2,4]:
  - Build a complementary **community-led Mandarin dataset**, bringing in social and fairness dimensions.
- **Implication:**
  This cluster is likely to sustain a **Mandarin-focused benchmarking ecosystem**, with standardized datasets and recurring challenges, similar to how Switchboard anchored general ASR benchmarking.

#### **FluencyBank & clinical stuttering community (Romana, Provost, etc.)**

- **Romana et al. (FluencyBank Timestamped)** [7]:
  - Add **time-aligned disfluency labels** and provide initial benchmarks with Whisper.
- Many later works (e.g., [1,12,13]) rely on **FluencyBank** as a primary evaluation set.
- **Implication:**
  FluencyBank is becoming the **de facto English benchmark** for real stuttered speech in ASR studies, analogous to AS‑70 in Mandarin and Boli for Indian languages.

#### **Novel modeling paradigms (Xu, others)**

- **Xu (StutterZero/StutterFormer)** [12]:
  - Pioneers **end-to-end waveform-to-waveform stutter-to-fluent conversion** evaluated against multiple Whisper baselines, with WER, CER, and semantic metrics.
- **Implication:**
  This line suggests that **pre-ASR speech conversion** could become a distinct subfield, requiring its own **benchmarking standards** relative to off-the-shelf ASR.

---

### Synthesis: Where the Field Stands and What the Timeline Suggests

- The **early 2020s** focused on **proprietary single-system robustness** with WER improvements from architectural tweaks and tuning [9,11].
- By **2023**, user-centered HCI work had reframed ASR on stuttered speech as an **accessibility and fairness issue**, adding task and perception metrics to WER [8].
- **2024** brought the **infrastructure** needed for scalable benchmarking: large annotated stuttered-speech corpora, timestamped datasets, and challenge tracks [2,5,6,7].
- **2024–2025** marks the emergence of **true multi-system benchmarking**, with direct comparisons between commercial APIs and open-source SOTA models [1,3], fine-grained disfluency-type analyses [7,10], and a strong fairness/disability-led focus [2,4,10].
- Parallel to this, **adaptation and novel modeling** (personalized Whisper, stutter-to-fluent conversion) are systematically evaluated against these benchmarks [12,13].

For your specific goal—**comparing modern ASR systems on stuttered speech with WER and related metrics**—the most relevant milestones, in chronological order, are:

- **Disfluency-aware RNN‑T robustness** [11] and **isWER-oriented VA tuning** [9] (foundational, though single-system).
- **User- and intent-focused evaluation of a consumer ASR** [8].
- **Creation of shared stuttered-speech benchmarks**: AS‑70 [6], FluencyBank Timestamped [7], Mandarin community dataset [2], Boli [3].
- **Challenge-based benchmarking** on stuttered speech [5].
- **Multi-system, bias-centric benchmarking** across commercial and open-source ASR [1].
- **Cross-model comparison of Whisper vs wav2vec 2.0** on stuttered speech [3].
- **Whisper-focused, disfluency-type–specific failure and hallucination analysis** [10].
- **Comparative baselines for stutter-to-fluent models vs Whisper variants** [12].

The trajectory suggests that forthcoming work will likely:

- Further **standardize evaluation pipelines** (normalization, reference conventions, severity stratification),
- Incorporate more **group-comparative fairness metrics**, and
- Expand **cross-system benchmarks** to include **streaming behavior**, **partial-result stability**, and **user-centered outcomes**—all on top of the WER/CER-centric evaluations that now form the backbone of this field.

## Foundational Work

### Which papers form the foundational references on this topic?

The below table shows the resources that are most often cited by the relevant papers on this topic. This is measured by the **reference rate**, which is the fraction of relevant papers that cite a resource. Use this table to determine the most important core papers to be familiar with if you want to deeply understand this topic. Some of these core papers may not be directly relevant to the topic, but provide important context.

| Ref. | Reference Rate | Topic Match | Title | Authors | Journal | Year | Total Citations | Cited By These Relevant Papers |
|---|---|---|---|---|---|---|---|---|
| [8] | 1.00 | 42% | From User Perceptions to Technical Improvement: Enabling People Who Stutter to Better Use Speech Recognition | Colin S. Lea, ..., and Leah Findlater | Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems | 2023 | 43 | [1, 2, 3, 4, 5, 6, 7] |
| [55] | 1.00 | 0% | SEP-28k: A Dataset for Stuttering Event Detection from Podcasts with People Who Stutter | Colin S. Lea, ..., and Jeffrey P. Bigham | ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2021 | 124 | [1, 2, 3, 4, 5, 6, 7, 8, 9] |
| [17] | 1.00 | 5% | Enhancing ASR for Stuttered Speech with Limited Data Using Detect and Pass | Olabanji Shonibare, ..., and Venkatesh Ravichandran | ArXiv | 2022 | 33 | [1, 2, 3, 4, 5, 6, 7, 8] |
| [91] | 1.00 | 0% | Fluency Bank: A new resource for fluency research and practice. | N. Bernstein Ratner and B. MacWhinney | Journal of fluency disorders | 2018 | 114 | [1, 2, 3, 4, 5, 6, 7, 8, 9] |
| [9] | 0.99 | 26% | Analysis and Tuning of a Voice Assistant System for Dysfluent Speech | V. Mitra, ..., and Jefferey Bigham | N/A | 2021 | 34 | [1, 2, 3, 4, 5, 6, 7, 8] |
| [6] | 0.83 | 75% | AS-70: A Mandarin stuttered speech dataset for automatic speech recognition and stuttering event detection | Rong Gong, ..., and Ming Li | ArXiv | 2024 | 14 | [3, 4, 5] |
| [42] | 0.75 | 0% | Disordered Speech Data Collection: Lessons Learned at 1 Million Utterances from Project Euphonia | R. L. MacDonald, ..., and K. Tomanek | N/A | 2021 | 77 | [1, 2, 4, 5, 6, 8] |
| [88] | 0.67 | 0% | The University College London Archive of Stuttered Speech (UCLASS). | P. Howell, ..., and J. Bartrip | Journal of speech, language, and hearing research : JSLHR | 2009 | 106 | [2, 4, 5, 6, 7, 8] |
| [22] | 0.56 | 3% | Stutter-TTS: Controlled Synthesis and Improved Recognition of Stuttered Speech | Xin Zhang, ..., and Venkatesh Ravichandran | ArXiv | 2022 | 11 | [2, 4, 5, 6] |
| [60] | 0.51 | 0% | Robust Speech Recognition via Large-Scale Weak Supervision | Alec Radford, ..., and I. Sutskever | N/A | 2022 | 5631 | [1, 4, 6, 7] |
| [115] | 0.50 | 0% | Variability of Stuttering: Behavior and Impact. | Seth E. Tichenor and J Scott Yaruss | American journal of speech-language pathology | 2020 | 50 | [1, 2, 4, 8] |
| [46] | 0.48 | 0% | "I Want to Publicize My Stutter": Community-led Collection and Curation of Chinese Stuttered Speech Data | Qisheng Li and Shaomei Wu | Proceedings of the ACM on Human-Computer Interaction | 2024 | 4 | [4] |
| [113] | 0.48 | 0% | “The World is Designed for Fluent People”: Benefits and Challenges of Videoconferencing Technologies for People Who Stutter | Shaomei Wu | Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems | 2023 | 19 | [1, 2, 4] |
| [130] | 0.39 | 0% | Conformer: Convolution-augmented Transformer for Speech Recognition | Anmol Gulati, ..., and Ruoming Pang | ArXiv | 2020 | 3746 | [1, 5, 6] |
| [2] | 0.37 | 100% | Towards Fair and Inclusive Speech Recognition for Stuttering: Community-led Chinese Stuttered Speech Dataset Creation and Benchmarking | Qisheng Li and Shaomei Wu | Extended Abstracts of the CHI Conference on Human Factors in Computing Systems | 2024 | 8 | [4, 5, 10] |
| [11] | 0.37 | 11% | Improved Robustness to Disfluencies in Rnn-Transducer Based Speech Recognition | Valentin Mendelev, ..., and Manuel Giollo | ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2020 | 23 | [2, 4, 8, 9] |
| [15] | 0.36 | 6% | Personalized Automatic Speech Recognition Trained on Small Disordered Speech Datasets | Jimmy Tobin and K. Tomanek | ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) | 2021 | 42 | [1, 2, 8] |
| [21] | 0.35 | 3% | Exploring Smart Speaker User Experience for People Who Stammer | Anna Bleakley, ..., and L. Clark | Proceedings of the 24th International ACM SIGACCESS Conference on Computers and Accessibility | 2022 | 16 | [2, 4, 8] |
| [40] | 0.32 | 0% | Automatic recognition of children's read speech for stuttering application | Sadeen Alharbi, ..., and P. Green | N/A | 2017 | 20 | [5, 6, 8, 9] |
| [114] | 0.29 | 0% | Re-envisioning Remote Meetings: Co-designing Inclusive and Empowering Videoconferencing with People Who Stutter | Jingjin Li, ..., and Gilly Leshed | Proceedings of the 2024 ACM Designing Interactive Systems Conference | 2024 | 9 | [4, 10, 13] |

## Adjacent Work

### Which papers cite the same foundational papers as relevant papers?

Use this table to discover related papers on adjacent topics, to gain a broader understanding of the field and help generate ideas for useful new research directions.

| Ref. | Adjacency score | Topic Match | Title | Authors | Journal | Year | Total Citations | References These Foundational Papers |
|---|---|---|---|---|---|---|---|---|
| [10] | 2.16 | 15% | J-j-j-just Stutter: Benchmarking Whisper's Performance Disparities on Different Stuttering Patterns | Charan Sridhar and Shaomei Wu | Interspeech 2025 | 2025 | 0 | [2, 6, 8, 17, 55] |
| [46] | 2.15 | 0% | "I Want to Publicize My Stutter": Community-led Collection and Curation of Chinese Stuttered Speech Data | Qisheng Li and Shaomei Wu | Proceedings of the ACM on Human-Computer Interaction | 2024 | 4 | [2, 6, 8, 55] |
| [19] | 1.92 | 4% | Enhanced ASR FOR Stuttering Speech: Combining Adversarial and Signal-Based Data Augmentation | Shangkun Huang, ..., and Rong Zheng | 2024 IEEE Spoken Language Technology Workshop (SLT) | 2024 | 3 | [6, 8, 9, 17, 55] |
| [30] | 1.75 | 1% | Leveraging LLM for Stuttering Speech: A Unified Architecture Bridging Recognition and Event Detection | Shangkun Huang, ..., and Rong Zheng | ArXiv | 2025 | 1 | [6, 8, 9, 17, 55] |
| [14] | 1.59 | 6% | Inclusive ASR for Disfluent Speech: Cascaded Large-Scale Self-Supervised Learning with Targeted Fine-Tuning and Data Augmentation | Dena F. Mujtaba, ..., and Jia Bin | ArXiv | 2024 | 8 | [8, 9, 55] |
| [110] | 1.41 | 0% | Large Language Models for Dysfluency Detection in Stuttered Speech | Dominik Wagner, ..., and Tobias Bocklet | ArXiv | 2024 | 16 | [8, 9, 55, 63, 91] |
| [27] | 1.34 | 2% | Automatic Disfluency Detection From Untranscribed Speech | Amrit Romana, ..., and E. Provost | IEEE/ACM Transactions on Audio, Speech, and Language Processing | 2023 | 16 | [8, 9, 11, 55] |
| [141] | 1.32 | Not measured | Exploring Whisper Embeddings for Stutter Detection: A Layer-Wise Study | Ashita Batra, ..., and Pradip K. Das | 2025 33rd European Signal Processing Conference (EUSIPCO) | 2025 | 0 | [8, 9, 55, 91] |
| [13] | 1.28 | 8% | Fine-Tuning ASR for Stuttered Speech: Personalized vs. Generalized Approaches | Dena F. Mujtaba and N. Mahapatra | ArXiv | 2025 | 1 | [8, 17, 42, 55, 60] |
| [101] | 1.28 | 0% | A CNN-based Stutter Detection Using MFCC Features with Binary Cross-Entropy Loss Function | Rohit Waddar, ..., and S. Budihal | 2024 IEEE International Conference on Contemporary Computing and Communications (InC4) | 2024 | 3 | [8, 11, 17, 55] |
| [142] | 1.28 | Not measured | Speech AI for All: Promoting Accessibility, Fairness, Inclusivity, and Equity | Shaomei Wu, ..., and N. Bernstein Ratner | Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems | 2025 | 0 | [2, 8, 91] |
| [120] | 1.18 | 0% | Govern with, Not For: Understanding the Stuttering Community’s Preferences and Goals for Speech AI Data Governance in the US and China | Jingjin Li, ..., and Shaomei Wu | Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society | 2025 | 0 | [8, 42, 55, 60] |
| [113] | 1.13 | 0% | “The World is Designed for Fluent People”: Benefits and Challenges of Videoconferencing Technologies for People Who Stutter | Shaomei Wu | Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems | 2023 | 19 | [9, 17, 40, 55, 91] |
| [8] | 1.09 | 42% | From User Perceptions to Technical Improvement: Enabling People Who Stutter to Better Use Speech Recognition | Colin S. Lea, ..., and Leah Findlater | Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems | 2023 | 43 | [9, 11, 17, 40, 55] |
| [47] | 1.06 | 0% | Limitations in speech recognition for young adults with down syndrome | Franceli L. Cibrian, ..., and V. Motti | Universal Access in the Information Society | 2025 | 3 | [8, 9, 11] |
| [143] | 0.98 | Not measured | Anonymization of Stuttered Speech -- Removing Speaker Information while Preserving the Utterance | Jan Hintz, ..., and Ingo Siegert | 3rd Symposium on Security and Privacy in Speech Communication | 2023 | 2 | [8, 55, 91] |
| [94] | 0.98 | 0% | Comparative Analysis of Classifiers using Wav2Vec2.0 Layer Embeddings for Imbalanced Stuttering Datasets | Madhurima Sen, ..., and Pradip K. Das | 2024 First International Conference on Electronics, Communication and Signal Processing (ICECSP) | 2024 | 2 | [8, 55, 91] |
| [12] | 0.98 | 9% | StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction | Qianheng Xu | IEEE Access | 2025 | 0 | [6, 8, 60, 91] |
| [114] | 0.95 | 0% | Re-envisioning Remote Meetings: Co-designing Inclusive and Empowering Videoconferencing with People Who Stutter | Jingjin Li, ..., and Gilly Leshed | Proceedings of the 2024 ACM Designing Interactive Systems Conference | 2024 | 9 | [8, 9] |
| [23] | 0.92 | 3% | Comparing ASR Systems in the Context of Speech Disfluencies | Maria Teleki, ..., and James Caverlee | Interspeech 2024 | 2024 | 5 | [8, 9] |