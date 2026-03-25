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
  - [ ] Download media (MP4 video) files — in progress (~302 files, ~12.9GB estimated); download script at `src/data/download_fluencybank_media.py`
  - [ ] Extract audio from MP4 videos with ffmpeg (→ 16kHz mono WAV)
  - [x] Write CHAT transcript parser (`src/data/parse_chat.py`): extracts speaker utterances, parses \x15-delimited timestamps, strips CHAT coding to clean text, exports aligned CSV + flat text
  - [x] Process all FluencyBank transcripts: 346 CHAT files → 61,035 utterances (55,003 with timestamps) across 8 corpora; output in `/Volumes/FATSPEECH/fluencybank/processed/`
  - [ ] Build FluencyBank inventory CSV (speakers, ages, gender, tasks, transcript/media availability)
  - FluencyBank README at `/Volumes/FATSPEECH/fluencybank/README.md` and `data/fluencybank.md`

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

- [ ] **SEP-28k** - For stutter event labels (not full transcriptions)
  - Publicly available, use for stutter-type annotations
  - ~28k clips with event-type labels (blocks, prolongations, repetitions)

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
  - [ ] Validate extracted WAV audio against known-good exports (spot-check playback)
  - [ ] Cross-reference extracted orthographic annotations with existing `Speech data Jason/input/` transcripts (66 sessions)
  - [ ] Cross-reference extracted annotations with `Speech data Jason/output/` usage matrices (68 CSVs with per-word linguistic features)
  - [ ] Decide on target sample rate for resampling (16kHz standard for ASR)

### 1.2 Fluent Speech Control Dataset

- [ ] **LibriSpeech** - Fluent baseline
  - Use test-clean and test-other splits for comparison
  - Enables quantification of PWS vs non-PWS performance gap

### 1.3 Data Preparation Pipeline

- [ ] Implement audio preprocessing (resampling to 16kHz, normalization)
- [ ] Create unified transcript format with stutter annotations
- [ ] Develop metadata schema (speaker ID, severity, stutter types)
- [ ] Build train/validation/test splits (speaker-exclusive)
- [ ] Create severity-stratified subsets (mild/moderate/severe)

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

## Phase 5: Fine-tuning Experiments

### 5.1 Data Augmentation Strategies

- [ ] Synthetic disfluency injection (prolongations, repetitions via TTS)
- [ ] Time-stretching augmentation
- [ ] Noise augmentation
- [ ] LibriStutter-style synthetic corpus generation

### 5.2 Fine-tuning Approaches

| Approach | Description |
|----------|-------------|
| **Full fine-tuning** | All parameters, requires substantial data |
| **LoRA adapters** | Parameter-efficient, good for limited data |
| **Generalized** | Train on pooled PWS data |
| **Personalized** | Speaker-specific adapters |

- [ ] Implement fine-tuning pipeline for Whisper
- [ ] Train generalized model on stuttered speech
- [ ] Train personalized models for subset of speakers
- [ ] Compare generalized vs personalized performance

### 5.3 Decoder/Endpoint Tuning

Literature shows significant gains without acoustic model changes:

- [ ] Experiment with decoding parameters (beam size, temperature)
- [ ] Test endpoint timeout adjustments for blocks/prolongations
- [ ] Measure WER improvement from decoding-only changes

---

## Phase 6: Reference Convention Analysis

Critical methodological contribution - reference choice affects rankings:

### 6.1 Surface vs Intended Speech

- [ ] Create both transcript versions for all data
  - **Surface**: preserves disfluencies as spoken
  - **Intended**: normalized fluent speech

- [ ] Evaluate all models on both reference types
- [ ] Document how rankings change between conventions
- [ ] Provide guidance on when to use each

### 6.2 Normalization Analysis

- [ ] Test different normalization strategies:
  - Punctuation handling
  - Filled pause mapping (um/uh → single token or removed)
  - Partial word handling
  - Block annotation handling

- [ ] Report sensitivity of results to normalization choices

---

## Phase 7: Fairness and Accessibility Analysis

### 7.1 Group Comparison

- [ ] **PWS vs non-PWS gap**
  - Calculate absolute and relative WER differences
  - Statistical significance testing

- [ ] **Severity disparity**
  - WER gap between mild and severe stuttering
  - Identify models with smallest disparity

### 7.2 Practical Accessibility Implications

- [ ] Voice assistant usability analysis
- [ ] Transcription service quality assessment
- [ ] Recommendations for PWS-friendly ASR configuration

---

## Phase 8: Results and Publication

### 8.1 Key Contributions

1. **First comprehensive multi-system benchmark** on stuttered speech including commercial APIs and open-source SOTA
2. **Stutter-type co-dependency analysis** - novel quantification of error patterns
3. **Severity-stratified evaluation** with clinical relevance
4. **Reference convention guidelines** for stuttered speech ASR evaluation
5. **Fine-tuning strategies** with generalized vs personalized comparison
6. **Practical recommendations** for improving ASR accessibility

### 8.2 Visualizations

- [ ] Model comparison bar charts (WER by model, grouped by dataset)
- [ ] Severity vs WER line plots per model
- [ ] Confusion matrices for stutter-type × error-type
- [ ] Hallucination rate heatmaps
- [ ] Surface vs intended WER scatter plots

### 8.3 Writing and Submission

- [ ] Draft introduction and related work
- [ ] Methods section with reproducibility details
- [ ] Results with statistical analysis
- [ ] Discussion of implications for accessibility
- [ ] Prepare supplementary materials and code release
- [ ] Target venue: Nature Comms, Nature Machine Intell., JSLHR, IEEE TASLP, or Interspeech/ICASSP

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
