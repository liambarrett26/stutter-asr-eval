# Research Roadmap: ASR Evaluation on Stuttered Speech

This document outlines the research plan for a high-impact journal article investigating the effect of stuttered speech on ASR systems and methods for improvement.

---

## Phase 1: Data Acquisition and Preparation

### 1.1 Obtain Stuttered Speech Corpora

- [x] **FluencyBank** - Primary English benchmark
  - TalkBank account registered; auth requires browser cookie (HttpOnly, expires ~24h)
  - Transcript zips downloaded to `/Volumes/FATSPEECH/fluencybank/raw/` (8 corpora):
    - **Voices-AWS** (66MB): 57 interview + 45 reading sessions, adults who stutter, CHAT format with timestamps + morphology + dependency parse
    - **Voices-CWS** (62MB): 26 interview + 22 reading sessions, children who stutter
    - **Voices-AWC** (6.4MB): 3 adults who clutter (interview + reading + stuttering samples)
    - **UMD-CMU** (1.8MB): 15 CWS + 15 controls, longitudinal conversation with mothers
    - **Hakim** (273K): 8 CWS + 8 controls, nonword repetition
    - **Examples** (4K), **VanZaalen** (19K), **Brejon** (205K, French)
  - CHAT files contain utterance-level timestamps (ms), speaker labels (*PAR/*INV), disfluency markers
  - Romana et al. (2024) improved timestamped version not available (no response to emails)
  - [x] Download media (MP4 video) files: 305/302 files, 130.1 GB, verified (spot-check 20 files, 1 truncation found and repaired). Download script at `src/data/download_fluencybank_media.py`
  - [x] Extract audio from MP4 videos: 270 WAV files, 46.9 hours, 5.0 GB (16kHz mono) → `/Volumes/FATSPEECH/fluencybank/processed/audio/`; 1 corrupt file re-downloaded and fixed
  - [x] Write CHAT transcript parser (`src/data/parse_chat.py`): extracts speaker utterances, parses \x15-delimited timestamps, strips CHAT coding to clean text, exports aligned CSV + flat text
  - [x] Process all FluencyBank transcripts: 346 CHAT files → 61,035 utterances (55,003 with timestamps) across 8 corpora; output in `/Volumes/FATSPEECH/fluencybank/processed/`
  - [x] Build FluencyBank inventory: 332 sessions, 113 speakers (`processed/inventory.csv` + `processed/speakers.csv`); Voices-AWS 13.2h/51 spk, Voices-CWS 3.4h/17 spk, UMD-CMU 37.6h/29 spk, Hakim 6.1h/13 spk
  - FluencyBank README at `/Volumes/FATSPEECH/fluencybank/README.md` and `docs/fluencybank.md`

- [x] **UCLASS** - UCL Archive of Stuttered Speech
  - Downloaded to `/Volumes/FATSPEECH/uclass/` (~6GB total); see `README.md` there for full structure and audit
  - **Release 1** (2004): 138 WAV monologues, 81 speakers (12F, 69M), ages 5–47
    - 49 sessions have transcripts: 31 ortho + 25 phonetic + 16 TextGrid + 14 SFS with embedded stutter annotations
    - SFS annotated files contain rich disfluency-coded phonetic data (18,019 annotation records)
  - **Release 2** (2008): 82 monologue + 107 reading + 128 conversation WAVs; 85 speakers
    - Very few transcripts: 4 monologue ortho, 2 monologue SFS aligned, 2 reading ortho
  - **Release FSF**: 56 SFS files, 14 speakers × 4 conditions, read passages (known reference text)
  - **46 speakers overlap** between R1 and R2 (longitudinal recordings)
  - SFS transcription conventions documented in `docs/sfs_transcription_conventions.md`
  - Full Howell & Huckvale (2004) paper converted to `docs/howell_huckvale_2004_sfs_transcription.md`
  - SFS annotation parser (`src/data/extract_sfs.py`) updated to handle anonymised UCLASS history strings
  - SFS annotation audit written to `/Volumes/FATSPEECH/uclass/audit_sfs_annotations.csv`
  - [x] Unzip all audio archives on FATSPEECH
  - [x] Audit SFS files for embedded annotations (found 14 R1 + 2 R2 = 16 annotated files)
  - [x] Identify speaker overlap between R1 and R2 (46 shared speakers)
  - [x] Parse R1 TextGrid files into standardised CSV format (33 files → `processed/transcripts/aligned/`)
  - [x] Extract annotations from SFS files into standardised CSV format (16 files → `processed/transcripts/aligned/`)
  - [x] Convert flat ortho/phon transcripts to standardised CSV (64 files → `processed/transcripts/flat/`)
  - [x] Build unified UCLASS inventory CSV (`processed/inventory.csv`: 304 sessions, 56 usable with audio+transcript)
  - [x] Build speaker metadata CSV (`processed/speakers.csv`: 120 speakers with age range, releases, task types)
  - Processing script: `src/data/process_uclass.py`
  - 16 sessions have both TextGrid and SFS annotations (cross-validation possible)
  - [x] Enrich metadata with info page fields (`processed/sessions_metadata.csv`: 304 sessions with handedness, onset age, therapy type, language, quality scores, recording location)
  - [x] Flag non-English speakers: 13 sessions across 9 languages (Arabic x5, French, Italian, Punjabi, Somali, Tamil, Turkish, Urdu, Yoruba) — flagged in `inventory.csv` as `is_l2=True`, kept in dataset for sub-analysis
  - [x] Extract FSF audio: 56 SFS files to WAV (`processed/fsf_audio/`), 2.1h total
  - [x] Fix SFS parser for M_0061 and M_1102 (mixed-endian detection: marker vs actual data endianness)
  - [x] Cross-validate TextGrid vs SFS annotations for 15 overlapping sessions: **TextGrid timestamps are correct** (match WAV duration); SFS annotation timestamps are scaled incorrectly (different sample rate origin). Same labels, wrong timescale. See `processed/transcripts/TIMESTAMP_NOTES.md`

- [x] **SEP-28k** - EXCLUDED: 3-second audio chunks with stutter event labels but no word-level transcriptions. Cannot calculate WER/CER. Stutter-type annotations already available in richer form from SLASS (284 files with type labels, 236 with phonetic stutter coding).

- [x] **SLASS** - Speech Lab Archive of Stuttered Speech (UCL internal)
  - Located on external drives: `/Volumes/FATSPEECH/speech_sfs/` (9,961 SFS files, 110GB) and `/Volumes/SPEECH/from_jason/` (curated subsets)
  - SFS binary format reverse-engineered; extraction script at `src/data/extract_sfs.py`
  - Curated subset in `from_jason/`: Persistent (8-10, 10-12, 12+), Recovered, and Fluent control groups
  - 66 sessions with orthographic + stutter transcripts already in `Speech data Jason/input/`
  - Rich annotation layers per SFS file: orthographic, stutter, syllables, type, clause, PW, rhyme
  - Sample rates vary (10kHz, 20kHz, 22.05kHz, 24kHz, 44.1kHz); both big- and little-endian files
  - [x] Batch extraction on curated `from_jason/` subsets: 148/148 files, 0 failures, 9.1h audio, 882 annotation layers, 62 speakers → `/Volumes/FATSPEECH/slass/processed/`
  - [x] Build SLASS metadata: `processed/inventory.csv` (148 sessions) + `processed/speakers.csv` (62 speakers with fluency group, age range)
  - SLASS README at `/Volumes/FATSPEECH/slass/README.md` and `data/slass.md`
  - [x] **Full archive extraction**: 9,780/9,961 files in 6.1 min → `/Volumes/FATSPEECH/slass/full_archive/` (84.1GB audio, 329.6h, 6,159 with annotations, 683 with orthographic, 236 with stutter, 181 parse errors)
  - [x] Validate extracted WAV audio: curated vs full archive match confirmed; 9,789 resource forks cleaned; 8,971 valid WAVs + 818 tiny/empty; spot-check sample all valid
  - [x] Cross-reference with Jason ortho transcripts: 56/66 sessions match SFS extractions; content aligns well (17-19/20 first words match; minor parsing diffs in Jason format). 10 Jason sessions not in SFS archive, 520 SFS ortho files not in Jason.
  - [x] Cross-reference with Jason usage matrices: 58/68 have matching SFS ortho, 62/68 have SFS type annotations. Matrices contain per-word stutter type labels (Fluent/Block/Prolongation/PWR/WWR/Unknown) + 81 linguistic feature columns — directly usable for co-dependency analysis.

- [x] **UNWR (adult SSI cohort)** - Tang & Wong-lineage adult dataset
  - Source: `data/additions/UNWR.zip` (1.3 GB). Extracted to `/Volumes/FATSPEECH/unwr/raw/` and processed to `/Volumes/FATSPEECH/unwr/processed/`.
  - 58 speakers across 4 groups: Control (25), Stutter (12), Attention-ADHD (15), Attention+Stutter (6). Age range 18–60, adult cohort filling a gap between child-skewed SLASS/UCLASS and fluent-adult LibriSpeech.
  - Per-speaker: 2× ~60s SSI question recordings (q1, q2) + syllable-segmented orthographic transcript + full phenotype (age, gender, languages, ASRS hyperactivity/inattention, PSal total, sonority/modulation/cue scores).
  - Stutter labels are session-level SSI % stuttered syllables (range 0.0–6.0%) — not per-word annotations. Stratifies the corpus for ASR severity analysis but is coarser than SLASS/Jason.
  - [x] Processing pipeline: `src/data/process_unwr.py` → `speakers.csv`, `inventory.csv`, `transcripts/<PID>.csv`, `scoring_{unwr,srt,pdt}.csv`
  - [x] Standardisation: `python src/data/standardise.py --corpus unwr` → 56 unified-schema records under `/Volumes/FATSPEECH/standardised/unwr/`. `stutter_type` field carries `ssi_pct=...;group=...` for stratified analysis.
  - [x] EDA: dataset summary table row + `unwr_groups.png` (speakers/group, SSI/group, age/group).
  - Documentation: `docs/unwr.md`.
  - Outstanding: UNWR does not yet contribute to per-word duration KDE — needs forced alignment to slot in.

- [~] **UNWR Reliability (children's nonword reading)** - ON HOLD
  - Source: `data/additions/Reliability_Analyses.zip` (2.8 MB, TextGrids + R/Python scripts; **no audio**).
  - 1,735 transcribed items across schools (Hackney, Hatfield, Priory, Stanford, StHelen) and transcribers (Clarissa, Kaho, Roaa). Each TextGrid has ortho / target / response tiers.
  - [x] Processing pipeline: `src/data/process_unwr_reliability.py` → `items.csv` at `/Volumes/FATSPEECH/unwr_reliability/processed/`
  - [x] Inter-rater agreement: `src/data/unwr_reliability_agreement.py` → `agreement_pairwise.csv` + `agreement_summary.csv`. Gold-standard human ceiling ≈ 5% PER (Clarissa vs Kaho_original); naive retranscriber band ≈ 25% PER.
  - [ ] **HELD pending audio.** The matching `.wav` files are presumably in `X:\Speech\UNWR-Transcription-Paper\` in the speech-lab Windows store. Liam to follow up with colleagues; once received, this data unlocks phoneme-level ASR evaluation against the gold-standard transcriptions.
  - Documentation: `docs/unwr_reliability.md`.

### 1.2 Fluent Speech Control Dataset

- [x] **LibriSpeech** - Fluent baseline
  - Downloaded to `/Volumes/FATSPEECH/librispeech/LibriSpeech/` (2.1 GB extracted)
  - **test-clean**: 40 speakers, 2,620 FLAC files, 532 MB (~5.4h)
  - **test-other**: 33 speakers, 2,939 FLAC files, 544 MB (~5.3h)
  - **dev-clean**: 40 speakers, 2,703 FLAC files, 538 MB (~5.4h) — for parameter tuning
  - **dev-other**: 33 speakers, 2,864 FLAC files, 517 MB (~5.1h) — for parameter tuning
  - Format: FLAC audio (16kHz) + `.trans.txt` reference transcripts (uppercase, clean)
  - Speaker metadata in `SPEAKERS.TXT` (ID, gender, subset, minutes)
  - Note: models likely trained on LibriSpeech training splits — test performance represents best-case fluent ceiling. This is desirable for quantifying the PWS vs non-PWS gap (conservative estimate).
  - Training splits (960h, ~59GB) deferred to Phase 6 fine-tuning if needed

### 1.3 Data Preparation Pipeline

#### Audio standardisation
- [ ] Resample all audio to 16kHz mono 16-bit PCM WAV (model input standard for Whisper, wav2vec2, HuBERT)
- [ ] Compute audio quality metadata per file: duration, estimated SNR, silence ratio, clipping detection
- [ ] Flag problematic files (SNR < 5dB, excessive clipping, < 1s or > 60min)
- [ ] Run VAD (Silero VAD) on all audio to segment speech vs silence — critical for reducing Whisper hallucinations on stuttered speech with long blocks/pauses
- [ ] Segment long recordings into ~30s chunks at VAD boundaries (for Whisper-style models)

#### Transcript standardisation
- [ ] Define unified transcript CSV schema across all corpora: `file_id, start_ms, end_ms, speaker, text_surface, text_intended, stutter_type, corpus, split`
- [ ] Create **dual-reference transcripts** for all data:
  - **Surface**: preserves disfluencies as spoken (repetitions, prolongations, blocks, filled pauses)
  - **Intended**: normalised fluent speech (what the speaker meant to say)
  - SLASS already has both (stutter layer = surface, orthographic layer = intended)
  - UCLASS: TextGrid stutter-coded = surface, orthographic = intended
  - FluencyBank CHAT: currently unannotated — need to determine which reference type the transcripts represent
- [ ] Choose and implement text normaliser: **Whisper `EnglishTextNormalizer`** (Open ASR Leaderboard standard) — apply identically to both reference and hypothesis
- [ ] Handle filled pauses explicitly: decide scope (remove for intended evaluation, preserve for surface)
- [ ] Strip corpus-specific coding (CHAT markers, SFS `:` `/` prefixes, `x` boundaries, `Q` pauses) to produce clean reference text

#### Metadata and splits
- [ ] Build unified speaker metadata across all corpora: speaker_id, corpus, age, gender, language (L1/L2), stuttering_severity, fluency_group
- [ ] Estimate stuttering severity from transcripts where not clinically rated (stuttering rate = disfluent syllables / total syllables): mild SR <= 7%, moderate 7-12%, severe > 12%
- [ ] Build speaker-exclusive test splits (no speaker appears in both train and test)
- [ ] Create severity-stratified subsets (mild/moderate/severe)
- [ ] Create corpus-stratified subsets for cross-dataset analysis

#### Evaluation infrastructure
- [ ] Implement JSONL manifest format for raw ASR outputs: `{file_id, audio_path, model_name, raw_hypothesis, word_timestamps, confidence, processing_time_s}`
- [ ] Save raw outputs BEFORE normalisation so re-evaluation with different normalisers doesn't require re-running inference
- [ ] Implement metric computation: WER, CER (jiwer), BERTScore (deberta-xlarge-mnli), hallucination rate (I/N), per-stutter-type error rates
- [ ] Implement word-level alignment (`jiwer.process_words()`) for stutter-type conditioned error analysis
- [ ] Implement statistical testing: paired bootstrap resampling for WER confidence intervals, Wilcoxon signed-rank test for system comparison, MAPSSWE (NIST SCTK)
- [ ] Implement both corpus-level (micro) and utterance-level (macro) WER aggregation — report both

---

## Phase 2: ASR Model Benchmarking

### 2.1 Open-Source Models (Local Inference)

| Model | Priority | Rationale |
|-------|----------|-----------|
| **Whisper large-v3** | Critical | Gold standard, best accuracy |
| **Whisper medium** | High | Size/accuracy tradeoff analysis |
| **Whisper small/base** | Medium | Resource-constrained scenarios |
| **Faster-Whisper** | High | Production-viable speed |
| **wav2vec 2.0 large** | Critical | Key comparison (literature shows 30-55% WER vs 5-21% for Whisper) |
| **HuBERT large** | High | Different self-supervised approach |
| **Distil-Whisper** | Medium | Efficiency vs accuracy tradeoff |

- [ ] Implement unified evaluation harness for all local models
- [ ] Run baseline transcriptions on all corpora
- [ ] Measure inference time and resource usage

### 2.2 Commercial APIs

| API | Priority | Rationale |
|-----|----------|-----------|
| **Google Cloud STT** | High | Market leader (despite poor benchmarks on disfluent speech) |
| **Microsoft Azure** | High | Enterprise comparison |
| **Amazon Transcribe** | Medium | AWS ecosystem |
| **AssemblyAI** | High | Strong real-world audio performance |
| **Deepgram Nova-3** | High | Top commercial accuracy (<5% WER) |

- [ ] Implement API wrappers with consistent interface
- [ ] Document API configuration (disfluency filtering options, endpoints)
- [ ] Run evaluations with matched settings across APIs

### 2.3 Evaluation Execution

- [ ] Run all models on all datasets
- [ ] Store raw transcriptions for analysis
- [ ] Generate per-model, per-dataset metric tables
- [ ] Calculate statistical significance (paired t-tests, Wilcoxon)

---

## Phase 3: Core Metrics Analysis

### 3.1 Standard Metrics

- [ ] **WER (Word Error Rate)**
  - Overall WER per model per dataset
  - Breakdown: substitutions, deletions, insertions
  - Comparison: stuttered vs fluent speech (LibriSpeech baseline)

- [ ] **CER (Character Error Rate)**
  - Important for partial words and morphological analysis
  - Captures character-level errors in stuttered fragments

### 3.2 Semantic Metrics

- [ ] **BERTScore** (semantic similarity)
  - Captures meaning preservation beyond surface errors
  - Use DeBERTa-xlarge-mnli as base model

- [ ] **Intent Error Rate** (if applicable to voice-command subset)
  - Feed ASR output to NLU model
  - Measure downstream task success

### 3.3 Stuttering-Specific Metrics

- [ ] **Hallucination Rate**
  - Words inserted that don't correspond to any speech
  - Literature shows >20% for sound repetitions in Whisper

- [ ] **Truncation Analysis** (if streaming data available)
  - Utterances cut off early due to endpointing
  - Particularly problematic for blocks and prolongations

---

## Phase 4: Stutter-Type and Severity Analysis

### 4.1 Co-dependency Analysis

This is a key contribution outlined in the research docs. Create contingency tables:

- [ ] **P(error_type | stutter_type)**
  - Given prolongation, what's the probability of deletion/substitution/insertion?
  - Repeat for: blocks, part-word repetitions, whole-word repetitions, interjections

- [ ] **P(stutter_type | error_type)**
  - Inverse analysis: given an error, what stutter type likely caused it?
  - Useful for stutter detection from ASR output

- [ ] Statistical tests for co-dependency significance (chi-square, mutual information)

### 4.2 Severity-Stratified Analysis

Based on stuttering rate (SR) or clinical grades:

| Severity | Definition (SR) |
|----------|-----------------|
| Mild | SR ≤ 7% |
| Moderate | 7% < SR ≤ 12% |
| Severe | SR > 12% |

- [ ] Calculate WER/CER per severity band per model
- [ ] Plot severity vs WER curves for each model
- [ ] Identify severity threshold where performance degrades sharply
- [ ] Statistical comparison of model robustness across severity levels

### 4.3 Disfluency-Type Analysis

Per stutter type:
- [ ] Sound repetitions (most problematic per literature)
- [ ] Word repetitions
- [ ] Prolongations
- [ ] Blocks
- [ ] Interjections/filled pauses

For each:
- [ ] WER on segments containing this disfluency type
- [ ] Error type distribution
- [ ] Hallucination rate
- [ ] Model ranking (which models handle which types best)

---

## Phase 5: Model Interrogation and Mechanistic Analysis

This phase elevates the work from "documenting the problem" to "explaining why it occurs" — essential for Nature Machine Intelligence.

### 5.1 Attention and Representation Analysis

- [ ] **Attention map analysis (Whisper)**
  - Extract and visualise encoder self-attention patterns during disfluent vs fluent segments
  - Do attention heads attend to the correct acoustic context during blocks/prolongations, or do they "skip" over them?
  - Compare attention patterns for the same utterance content spoken fluently vs with stuttering
  - Layer-by-layer analysis: which layers are most affected by disfluency?

- [ ] **Hidden state probing**
  - Train linear probes on intermediate encoder representations to test: does the model internally detect stuttering events even when it fails to transcribe correctly?
  - Probe for: fluency state (fluent/disfluent), stutter type, word identity
  - Compare probe accuracy across layers and model architectures (Whisper vs wav2vec2 vs HuBERT)

- [ ] **CTC alignment analysis (wav2vec2, HuBERT)**
  - Examine CTC spike patterns during blocks, prolongations, repetitions
  - How does the CTC blank token interact with stuttering silence (blocks)?
  - Does the model produce repeated CTC spikes for repetitions, or does it collapse them?

- [ ] **Decoder behaviour analysis (Whisper)**
  - Examine token-level probabilities during disfluent segments
  - Does the autoregressive decoder suppress repeated tokens (interpreting repetitions as model errors rather than speech)?
  - Analyse how the decoder's language model prior overrides acoustic evidence during disfluent speech
  - Compare greedy vs beam search token selection paths during disfluencies

### 5.2 Ablation Studies

- [ ] **Architectural ablation: encoder-decoder vs CTC**
  - Why does Whisper (encoder-decoder) dramatically outperform wav2vec2 (CTC) on stuttered speech (5-21% vs 30-55% WER)?
  - Hypothesis: the language model decoder "fills in" intended speech, while CTC preserves surface disfluencies
  - Test by examining each architecture's error type distribution (S/D/I) per stutter type

- [ ] **Layer-wise ablation**
  - Systematically mask or zero-out encoder layers and measure WER change on stuttered vs fluent speech
  - Identify which layers are critical for handling disfluency

- [ ] **Decoding parameter sweep**
  - Systematic grid search over temperature (0.0–1.0), beam size (1–10), no_speech_threshold, compression_ratio_threshold
  - Measure WER separately on fluent and stuttered subsets at each setting
  - Identify parameter configurations that specifically improve stuttered speech performance

- [ ] **VAD interaction study**
  - How does VAD boundary placement interact with blocks and prolongations?
  - Do blocks get classified as silence and cut? Do prolongations get split mid-sound?
  - Compare Silero VAD vs energy-based VAD vs no VAD on stuttered speech

- [ ] **Language detection ablation**
  - Document Whisper's language misidentification on stuttered English (observed: Welsh, Spanish classification)
  - Forced English vs auto-detected: WER difference on stuttered speech
  - What acoustic features of stuttered speech trigger wrong language detection?

### 5.3 Acoustic Feature Analysis

- [ ] **Failure-correlated acoustic features**
  - Extract acoustic features (F0, energy, spectral centroid, formant transitions, speech rate, pause duration) for each word
  - Correlate with ASR error probability: what acoustic signatures predict failure?
  - Compare feature distributions of correctly vs incorrectly transcribed disfluent words

- [ ] **Synthetic perturbation experiments**
  - Generate controlled synthetic disfluencies (prolongations of varying length, repetitions of varying count, blocks of varying duration) from fluent speech
  - Systematically vary one parameter at a time to isolate the breaking point for each model
  - Establish dose-response curves: at what severity does each model degrade?

- [ ] **Representation similarity analysis**
  - For speakers who produced both fluent and disfluent versions of similar content, compare internal model representations
  - Use CKA (Centered Kernel Alignment) or SVCCA to measure representation similarity across layers
  - Does the model represent the same word differently when it's stuttered vs fluent?

---

## Phase 6: Improvement Strategies

### 6.1 Decoder and Endpoint Tuning (No Retraining)

Literature shows significant gains without acoustic model changes:

- [ ] Decoding parameter optimisation (from ablation results in 5.2)
- [ ] Endpoint timeout adjustments for blocks/prolongations
- [ ] Custom VAD tuned for stuttered speech (higher silence tolerance)
- [ ] Prompt engineering for Whisper: does the initial prompt affect disfluency handling?
- [ ] Measure WER improvement from decoding-only changes

### 6.2 Fine-tuning Approaches

| Approach | Description |
|----------|-------------|
| **LoRA adapters** | Parameter-efficient, good for limited data |
| **Full fine-tuning** | All parameters, requires substantial data |
| **Generalised** | Train on pooled PWS data |
| **Personalised** | Speaker-specific adapters |

- [ ] Implement fine-tuning pipeline for Whisper (LoRA on encoder, decoder, or both)
- [ ] Train generalised model on SLASS stuttered speech data
- [ ] Train personalised models for subset of speakers
- [ ] Compare generalised vs personalised performance
- [ ] **Informed fine-tuning**: use mechanistic insights from Phase 5 to guide which layers/components to fine-tune

### 6.3 Data Augmentation

- [ ] Synthetic disfluency injection into fluent speech (prolongations, repetitions via signal processing)
- [ ] LibriStutter-style synthetic corpus generation
- [ ] Evaluate whether synthetic disfluencies provide the same training signal as real stuttered speech

### 6.4 Architectural Recommendations

Based on findings from Phase 5, propose concrete changes for future ASR systems:

- [ ] What loss function modifications could improve disfluency robustness? (e.g., disfluency-aware CTC, modified cross-entropy that doesn't penalise surface repetitions)
- [ ] What training data composition would help? (proportion of disfluent speech, diversity of stutter types)
- [ ] What architectural modifications could help? (e.g., disfluency-aware attention masking, explicit fluency state modelling, separate decoder paths for surface vs intended output)
- [ ] What pre-training objectives would build better representations of disfluent speech?

---

## Phase 7: Reference Convention and Methodological Analysis

Critical methodological contribution — reference choice affects rankings:

### 7.1 Surface vs Intended Speech

- [ ] Evaluate all models on both reference types
- [ ] Document how model rankings change between surface and intended evaluation
- [ ] Quantify the magnitude of ranking inversions
- [ ] Provide guidance on when to use each convention

### 7.2 Normalisation Sensitivity Analysis

- [ ] Test multiple normalisers: Whisper EnglishTextNormalizer, BasicTextNormalizer, jiwer default
- [ ] Test filled pause scope: included vs excluded from reference
- [ ] Test partial word handling: included vs excluded
- [ ] Report sensitivity of model rankings to each normalisation choice
- [ ] Recommend a standard normalisation protocol for stuttered speech ASR evaluation

---

## Phase 8: Fairness and Accessibility Analysis

### 8.1 Group Comparison

- [ ] **PWS vs non-PWS gap**
  - Calculate absolute and relative WER differences (stuttered corpora vs LibriSpeech)
  - Statistical significance testing
  - Compare gap across models: which have smallest disparity?

- [ ] **Severity disparity**
  - WER gap between mild and severe stuttering
  - Identify severity threshold where performance degrades sharply
  - Compute disparity ratio: WER_severe / WER_mild per model

- [ ] **Demographic analysis**
  - Gender effects (limited by male-heavy datasets)
  - Age effects (children vs adults)
  - L1 vs L2 speaker comparison (13 UCLASS L2 sessions)

### 8.2 Practical Accessibility Implications

- [ ] Voice assistant usability analysis: at what WER does a VA become unusable?
- [ ] Recommendations for PWS-friendly ASR configuration (specific model + parameter choices)
- [ ] Cost-benefit analysis: accuracy vs latency vs cost for real-world deployment

---

## Phase 9: Results and Publication

### 9.1 Key Contributions (for Nature Machine Intelligence / IEEE TASLP)

1. **Largest stuttered speech ASR benchmark**: 66h transcribed stuttered speech (SLASS) + FluencyBank + UCLASS, multi-system evaluation
2. **Mechanistic analysis**: attention patterns, hidden state probes, and decoder behaviour reveal WHY ASR fails on disfluent speech — not just that it does
3. **Stutter-type co-dependency analysis**: novel quantification of P(error_type | stutter_type) linking specific disfluencies to specific failure modes
4. **Architectural ablation**: systematic comparison of encoder-decoder vs CTC architectures on stuttered speech, with layer-level analysis
5. **Dose-response characterisation**: synthetic perturbation experiments establishing precise failure thresholds per disfluency type per model
6. **Severity-stratified evaluation** with clinical relevance (persistent/recovered/fluent longitudinal design)
7. **Reference convention guidelines**: evidence-based recommendations for evaluating ASR on disfluent speech
8. **Improvement pathways**: decoder tuning, informed fine-tuning, and architectural recommendations with empirical validation
9. **Fairness quantification**: PWS vs non-PWS performance gap with disparity ratios across commercial and open-source systems

### 9.2 Visualisations

- [ ] Attention map heatmaps: fluent vs disfluent speech, by model layer
- [ ] Probe accuracy curves: fluency detection across layers (Whisper vs wav2vec2 vs HuBERT)
- [ ] Dose-response curves: synthetic disfluency severity vs WER per model
- [ ] Model comparison bar charts (WER by model, grouped by dataset)
- [ ] Severity vs WER line plots per model with confidence bands
- [ ] Confusion matrices for stutter-type x error-type (per model)
- [ ] Hallucination rate heatmaps by stutter type
- [ ] Surface vs intended WER scatter plots showing ranking inversions
- [ ] Representation similarity dendrograms: fluent vs stuttered same-content speech
- [ ] Normalisation sensitivity tornado plots

### 9.3 Writing and Submission

- [ ] Draft introduction and related work
- [ ] Methods section with reproducibility details
- [ ] Results with statistical analysis
- [ ] Discussion: mechanistic insights and implications for ASR architecture design
- [ ] Discussion: implications for accessibility and fairness
- [ ] Prepare supplementary materials and code release
- [ ] Target venue: Nature Machine Intelligence (primary), IEEE/ACM TASLP (fallback), Nature Communications (alternative)

---

## Model Selection Rationale

Based on literature review and current benchmarks:

### Open-Source (Critical Path)

| Model | Why Include |
|-------|-------------|
| Whisper large-v3 | SOTA accuracy, gold standard baseline |
| Faster-Whisper | Production-viable, same accuracy |
| wav2vec 2.0 | Key comparison - literature shows dramatic gap vs Whisper |
| HuBERT | Different SSL approach, mentioned in project docs |

### Commercial APIs (Comparison)

| API | Why Include |
|-----|-------------|
| Google Cloud STT | Market leader, important despite poor performance |
| AssemblyAI | Best on challenging audio, trained on real-world data |
| Deepgram Nova-3 | Top commercial accuracy |
| Azure STT | Enterprise baseline |

### Models Considered but Lower Priority

| Model | Reason |
|-------|--------|
| NVIDIA Canary | Top accuracy but limited accessibility |
| Amazon Transcribe | Good but slower, requires S3 setup |
| Kaldi | Legacy toolkit, less relevant for modern comparison |
| Vosk | Lightweight but lower accuracy |

---

## References

Key papers informing this roadmap:

1. Mujtaba et al. (2024) - "Lost in Transcription" - Multi-system benchmark
2. Batra et al. (2025) - Boli dataset, wav2vec2 vs Whisper comparison
3. Lea et al. (2023) - CHI paper on PWS user experience
4. Mitra et al. (2021) - VA system analysis with severity stratification
5. Gong et al. (2024) - AS-70 Mandarin dataset methodology
6. Sridhar & Wu (2025) - Whisper hallucination analysis on stuttered speech
