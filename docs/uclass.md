# UCLASS — UCL Archive of Stuttered Speech

Downloaded from: https://www.uclass.psychol.ucl.ac.uk
Funded by: The Wellcome Trust
Institution: UCL Division of Psychology and Language Sciences

**Citation requirement:** All use must acknowledge the data source and credit Wellcome Trust support.

---

## Data Store Structure

```
uclass/
├── raw/
│   ├── release1/                         # 2004 — Spontaneous Dysfluent Monologues
│   │   ├── audio/wav/                    # 138 WAV files (unzipped from AllAudioWav.zip)
│   │   ├── sfs_audio/files/             # 138 SFS audio files (unzipped from AllAudioSfs.zip)
│   │   ├── sfs_annotated/files/         # 16 SFS files with embedded annotations (AllAnnotatedSfs.zip)
│   │   ├── transcripts/
│   │   │   ├── orthographic/files/      # 31 .orth flat text files
│   │   │   ├── phonetic/files/          # 25 .phon flat text files (JSRU format)
│   │   │   └── aligned/
│   │   │       ├── textgrid/files/      # 25 Praat TextGrid files (.grid)
│   │   │       ├── chat/files/          # 25 CHILDES CHAT files (.cha)
│   │   │       └── sfs_annotation/files/# 25 SFS annotation export files
│   │   └── metadata/                    # information.htm (full speaker listing)
│   │
│   ├── release2/                         # 2008 — Monologue + Reading + Conversation
│   │   ├── monologue/
│   │   │   ├── audio/wav/               # 82 WAV files
│   │   │   ├── sfs_aligned/             # 2 SFS with aligned phonetic annotations
│   │   │   └── transcripts/
│   │   │       ├── orthographic/        # 4 flat text files
│   │   │       ├── phonetic/            # 2 flat text files
│   │   │       └── aligned/
│   │   │           ├── textgrid/        # 6 Praat TextGrid files
│   │   │           ├── chat/            # 6 CHILDES CHAT files
│   │   │           └── sfs_annotation/  # 6 SFS annotation export files
│   │   ├── reading/
│   │   │   ├── audio/wav/               # 107 WAV files
│   │   │   └── transcripts/
│   │   │       ├── orthographic/        # 2 flat text files
│   │   │       └── aligned/
│   │   │           ├── textgrid/        # 2 Praat TextGrid files
│   │   │           ├── chat/            # 2 CHILDES CHAT files
│   │   │           └── sfs_annotation/  # 2 SFS annotation export files
│   │   ├── conversation/
│   │   │   └── audio/wav/               # 128 WAV files
│   │   └── metadata/                    # info pages (monologue, reading, conversation)
│   │
│   └── fsf/                              # Frequency-Shifted Feedback experiment
│       ├── audio/sfs/                    # 56 SFS files (14 speakers × 4 conditions)
│       └── metadata/                     # hdbw.pdf, HowellAndHuckvale.pdf
│
├── processed/                            # Standardised outputs for ASR pipeline
│   ├── transcripts/
│   │   ├── aligned/                     # 49 CSVs (time,duration,label) from TextGrids + SFS
│   │   ├── flat/                        # 64 CSVs (type,text) from flat ortho/phon
│   │   └── TIMESTAMP_NOTES.md           # TextGrid timestamps are correct; SFS are not
│   ├── fsf_audio/                       # 56 WAV files extracted from FSF SFS files (2.1h)
│   ├── inventory.csv                    # 304 sessions with audio/transcript/language flags
│   ├── speakers.csv                     # 120 speakers with age range, releases, tasks
│   └── sessions_metadata.csv            # Full metadata from info pages (304 sessions)
│
├── audit_sfs_annotations.csv            # Full SFS annotation audit (2026-03-25)
└── README.md
```

**Status:** All zips unzipped. Audit complete (2026-03-25).

---

## Audit Results (2026-03-25)

### Totals
- **304 unique recording sessions** across all releases
- **120 unique speakers** (81 in R1, 85 in R2, 46 in both)
- **49 R1 sessions have at least one transcript** (of 138 total)
- **14 SFS files contain embedded stutter annotations** (18,019 annotation records total)

### R1 Sessions with Transcripts (49 total)

Sessions with **all four** formats (ortho + phon + textgrid + sfs_annotation):
- M_0078_16y5m_1

Sessions with **time-aligned annotations** (TextGrid + SFS embedded, stutter-coded):
- M_0030_16y4m_1, M_0061_16y9m-1, M_0107_07y7m_1, M_0121_11y1m_1, M_0121_15y1m_1, M_0553_10y0m_1, M_0553_11y0m_1, M_1064_47y0m_1, M_1099_25y0m_1, M_1100_28y0m_1, M_1101_35y0m_1, M_1103_20y0m_1, M_1104_40y0m_1, M_1105_21y0m_1, M_1106_25y0m_1

Sessions with **flat ortho + phonetic** (no time alignment):
- F_0101_10y4m_1, F_0101_13y1m_1, M_0028_15y11m_1, M_0030_12y1m_1, M_0030_17y9m_1, M_0052_16y4m_1, M_0061_14y8m_1, M_0078_12y4m_1, M_0078_14y4m_1, M_0095_08y10m_1, M_0098_07y8m_1, M_0098_10y6m_1, M_0100_11y2m_1, M_0100_12y3m_1, M_0100_13y10m_1, M_0104_10y3m_1, M_0138_12y2m_1, M_0210_11y3m_1, M_0213_10y10m_1, M_0394_09y2m_1, M_0399_12y4m_1

Sessions with **ortho only** (no phonetic):
- M_0095_07y7m_1, M_0098_09y8m_1, M_0138_13y3m_1, M_0219_11y2m_1, M_0234_09y9m_1, M_0394_08y10m_1, M_0394_09y5m_1, M_0556_07y8m_1, M_0556_08y0m_1

Sessions with **phonetic only** (no ortho):
- M_1097_26y0m_1, M_1102_24y0m_1, M_1107_38y0m_1

### Parse Issues
- M_0061_16y9m-1.sfs: invalid sample period (both annotated and audio SFS)
- M_1102_24y0m_1.sfs: invalid sample period (audio SFS only)

---

## Release Summaries

### Release 1 (2004) — Spontaneous Monologues
- **138 WAV recordings** from 81 unique speakers (12F, 69M)
- Ages: 5y4m – 47y0m (mostly children, some adults)
- British English, all people who stutter (PWS)
- Recording locations: Clinic, Home, UCL
- Quality metrics per recording: tape noise, env noise, speaker quality, interruptions (1–4 scale)
- Speaker metadata: handedness, family history, age of onset, therapy type (Holistic/Family)
- **31 flat orthographic + 25 flat phonetic + 16 time-aligned TextGrid + 14 SFS with embedded annotations**
- SFS annotations use stutter-coded JSRU phonetic notation (see `docs/sfs_transcription_conventions.md`)
- Annotation tiers in aligned files: word, syllable, phonetic word (varies by speaker)

### Release 2 (2008) — Monologue, Reading, Conversation
- **82 monologue** + **107 reading** + **128 conversation** WAV recordings
- 85 unique speakers, ages 7y10m – 20y1m
- Additional metadata: word count, therapy gap (months since therapy), hearing/language flags
- **Multilingual speakers**: English (majority), Arabic, Somali, Tamil, Yoruba, Punjabi, French, Turkish, Urdu
- Reading material: SSI-3 passages (125 words), ATR passages (370 words), OMW passages (420 words)
- **Very few transcripts**: 4 monologue ortho, 2 monologue phonetic, 2 reading ortho, 6 monologue aligned, 2 reading aligned

### Release FSF — Frequency-Shifted Feedback
- **14 speakers × 4 conditions = 56 SFS files**
- Read passages ("Alice" and "Kate") alternating easy/difficult linguistic sections
- Conditions: normal listening (n) vs frequency-shifted feedback (f)
- Filename convention: `[speakerID]_r[run]_[text]_[feedback_sequence]` (e.g., `0075_r1_2_fnfn`)
- **No transcriptions available** (but reference text is known — the Alice/Kate passages)
- SFS format only — use `src/data/extract_sfs.py` to extract audio

---

## What is Usable for the ASR Evaluation Project

### Immediately usable (audio + reference transcript available)

**Tier 1 — Time-aligned stutter annotations (best quality for ASR + co-dependency analysis):**
- 15 R1 sessions with TextGrid + SFS embedded annotations (stutter-coded phonetic)
- 2 R2 monologue sessions with SFS aligned phonetic (M_1022, M_1023)
- All adult male speakers; 1.6h audio; 18,019 annotation records
- Contains disfluency markers (blocks, repetitions, prolongations) essential for co-dependency analysis

**Tier 2 — Flat orthographic transcripts (usable for WER calculation):**
- 31 R1 sessions with orthographic text
- 4 R2 monologue sessions with orthographic text (M_0017, M_0065, M_1017×2)
- Mix of children and adults; no time alignment but sufficient for whole-utterance WER

**Tier 3 — Known reference text (reading passages):**
- 107 R2 reading WAV files with known passage text (SSI-3/ATR/OMW)
- 56 FSF SFS files with known Alice/Kate passages (needs SFS extraction)
- Reference is the passage text itself; however disfluencies mean ASR output will diverge from reference in informative ways

**Total immediately usable: ~49 R1 + ~6 R2 monologue + 107 R2 reading + 56 FSF = ~218 sessions**

### Usable with effort

| Source | Recordings | What's needed |
|--------|------------|---------------|
| R1 audio without transcripts | 89 | Manual transcription or ASR-then-correct |
| R2 monologue without transcripts | ~76 | Manual transcription or ASR-then-correct |
| R2 conversation audio | 128 | Manual transcription (hardest — multi-speaker) |

### Key limitations

- **Heavy gender imbalance**: overwhelmingly male speakers across all releases
- **Mostly children**: adult speakers are rare and clustered in R1 (IDs 1064–1107)
- **Inconsistent annotation formats**: R1 adult speakers use stutter-coded phonetic annotation (`:` `/` `Q` `x`); R1 younger speakers use plain orthographic; R2 has very sparse transcription
- **Multilingual speakers in R2** (Arabic, Somali, Tamil, Yoruba, Punjabi, French, Turkish, Urdu) — must be excluded from British English evaluation or analysed separately
- **No severity labels**: stuttering severity must be estimated from transcripts (stuttering rate) or from SSI-3 passage scores where available
- **46 speakers appear in both R1 and R2** — same individuals recorded at different ages/timepoints; important for longitudinal analysis but must handle carefully to avoid data leakage in train/test splits
- **2 SFS parse failures**: M_0061_16y9m-1 and M_1102_24y0m_1 have non-standard sample period encoding
