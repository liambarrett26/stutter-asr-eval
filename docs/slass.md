# SLASS — Speech Lab Archive of Stuttered Speech

Source: UCL Speech Lab (Division of Psychology and Language Sciences)
Collected by: Peter Howell and colleagues
**Not publicly available** — UCL internal research data

---

## Overview

SLASS is a longitudinal corpus of speech recordings from children who stutter, children who recovered from stuttering, and fluent controls, collected over ~20 years at UCL. Recordings span ages ~5 to 19+, with many speakers recorded at multiple timepoints across clinical age bands (8–10, 10–12, 12+ years).

The data exists in two forms:
1. **Full archive**: 9,961 SFS files, 110GB (`/Volumes/FATSPEECH/speech_sfs/`)
2. **Curated subset**: 148 files organised into clinical groups (`/Volumes/SPEECH/from_jason/`)

Each SFS file contains 16-bit PCM audio + multiple annotation layers. The extraction script (`src/data/extract_sfs.py`) handles all known format variants including mixed-endian files.

---

## Key Numbers

| | Files | Hours | Notes |
|---|---|---|---|
| **Total archive** | 9,961 | 329.6h | All recordings |
| Extracted successfully | 9,780 | — | 181 parse errors (1.8%) |
| With any annotation | 6,159 | 213.5h | Some annotation layer present |
| **With orthographic transcripts** | **683** | **66.2h** | **ASR-ready: word-level reference text** |
| With stutter annotations | 236 | 22.2h | Phonetic disfluency coding |
| With type annotations | 284 | 17.5h | Stutter type per word |
| With syllable annotations | 674 | 51.2h | Sub-word segmentation |
| Audio only (no annotations) | 3,621 | 116.1h | Speech signal only |
| **Curated subset** | **148** | **9.1h** | Clinically grouped (persistent/recovered/fluent) |

**For ASR evaluation: 66.2 hours of transcribed stuttered speech** — the largest known dataset of its kind.

---

## Data Store Structure

```
/Volumes/FATSPEECH/slass/
├── processed/                           # Curated subset (148 files, clinically grouped)
│   ├── audio/
│   │   ├── persistent_8-10/             # 22 WAV files
│   │   ├── persistent_10-12/            # 22 WAV files
│   │   ├── persistent_12+/              # 22 WAV files
│   │   ├── recovered_8-10/              # 23 WAV files
│   │   ├── recovered_10-12/             # 23 WAV files
│   │   ├── recovered_12+/               # 23 WAV files
│   │   ├── fluent_-8/                   # 3 WAV files
│   │   ├── fluent_8-10/                 # 4 WAV files
│   │   ├── fluent_10-12/                # 2 WAV files
│   │   └── fluent_12+/                  # 4 WAV files
│   ├── annotations/                     # Per-group CSVs (time, duration, label)
│   │   └── {group}/                     # ~6-9 CSVs per file
│   ├── inventory.csv                    # 148 sessions with metadata
│   └── speakers.csv                     # 62 speakers
│
├── full_archive/                        # Complete extraction (9,780 files)
│   ├── audio/                           # 9,780 WAV files, 84.1 GB
│   ├── annotations/                     # CSVs for 6,159 annotated files
│   └── inventory.csv                    # Full inventory with layer flags
│
└── README.md

Source data (not copied):
  /Volumes/FATSPEECH/speech_sfs/          # Original 9,961 SFS files
  /Volumes/SPEECH/from_jason/             # Curated subset source
  /Volumes/SPEECH/from_jason/Speech data Jason/input/   # 66 ortho + stutter transcript pairs
  /Volumes/SPEECH/from_jason/Speech data Jason/output/  # 68 per-word linguistic feature CSVs
```

---

## Extracted Annotation Layers

### Layers extracted (text-based annotations)

These are parsed from SFS annotation records and exported as CSV (time, duration, label):

| Layer | Files | Hours | Description |
|-------|-------|-------|-------------|
| **orthographic** | 683 | 66.2h | Word-level transcription. `:` = content word, `/` = function word, `x` = boundary, `Q` = pause |
| **syllable** | 674 | 51.2h | Syllable-level segmentation |
| **type** | 284 | 17.5h | Stutter type classification per word (blocks, prolongations, repetitions) |
| **stutter** | 236 | 22.2h | JSRU phonetic transcription preserving disfluencies as produced |
| **PW** | ~300 | — | Phonological word boundaries |
| **clause** | ~110 | — | Clause/utterance boundaries |
| **rhyme** | ~100 | — | Rhyme/rime phonological structure |
| **im** | ~27 | — | Interjection markers |
| **dysfluencies** | ~200 | — | Disfluency event annotations (various naming conventions) |
| **markers/marked points** | ~880 | — | Experimental time markers |

### Layers NOT extracted (signal analysis data)

These are numerical/acoustic analysis outputs stored as SFS data items, not text annotations. They are excluded because:
- They contain thousands of frames per second of numerical data (not parseable as text annotations)
- They represent derived acoustic measurements, not transcriptions
- Extracting them as annotation records would produce meaningless output

| Layer | Files | Description | Why excluded |
|-------|-------|-------------|--------------|
| **voc19** | 1,327 | Voice analysis (19-parameter voicing analysis) | Numerical frame data, ~100 frames/sec |
| **fmanal** (formant analysis) | 813 | Formant tracks (F1–F4 frequency and bandwidth) | Numerical, ~100 frames/sec per formant |
| **Various signal processing** | ~200 | MFCC, LPC, spectral analysis, dicode | Numerical feature vectors |

These layers can be re-extracted from the original SFS files using SFS tools or the MATLAB API if needed for acoustic analysis. The audio WAV files are always fully extracted regardless of annotation content.

---

## Curated Subset Detail (148 files, 9.1 hours)

### By clinical group

| Group | Age Band | Files | Duration | Notes |
|-------|----------|-------|----------|-------|
| **Persistent** | 8–10 yrs | 22 | 69 min | Children who continue to stutter |
| **Persistent** | 10–12 yrs | 22 | 62 min | |
| **Persistent** | 12+ yrs | 22 | 75 min | |
| **Recovered** | 8–10 yrs | 23 | 116 min | Children who recovered from stuttering |
| **Recovered** | 10–12 yrs | 23 | 57 min | |
| **Recovered** | 12+ yrs | 23 | 63 min | |
| **Fluent** | <8 yrs | 3 | 65 min | Fluent controls |
| **Fluent** | 8–10 yrs | 4 | 22 min | |
| **Fluent** | 10–12 yrs | 2 | 4 min | |
| **Fluent** | 12+ yrs | 4 | 11 min | |
| **Total** | | **148** | **9.1 hours** | **62 unique speakers** |

All curated files have orthographic annotations. Most persistent and recovered files also have stutter, type, and syllable layers.

### Audio properties
- Sample rates: 10kHz, 12kHz, 20kHz, 22.05kHz, 24kHz, 44.1kHz (varies by recording era)
- Format: 16-bit mono PCM (exported as WAV)
- Both big-endian and little-endian SFS files (parser auto-detects, including mixed-endian edge cases)

### Filename convention
`{gender}_{speakerID}_{unknown}_{age}.sfs`
- e.g. `f_0050_7_11y6m.sfs` = female, speaker 0050, age 11 years 6 months
- `m_0553_7_11y0m.sfs` = male, speaker 0553, age 11 years 0 months

---

## Existing Transcripts (Speech data Jason)

In `/Volumes/SPEECH/from_jason/Speech data Jason/`:

### Input transcripts (66 sessions)
Paired `*_ortho.txt` + `*_stutter.txt` files with timestamped word-level transcription:
- Orthographic: standard English with `:` (content) and `/` (function word) prefixes
- Stutter: JSRU phonetic transcription with disfluency coding (repetitions, blocks, prolongations)
- 4 female speakers x 3 age points + 18 male speakers x 3 age points

### Output usage matrices (68 CSVs)
Per-word linguistic feature matrices with columns including:
- ID, Gender, Age_In_Months, Timestamp, Phonetic_Content
- Fluency, Stutter_Type, Stutter_Type_2
- Word, Syllable, Word_Type, Word_Position, Utterance_Length
- Segmental/Biphone Phonotactic_Probability
- Neighbourhood_Density, Neighbourhood_Frequency
- Onset_Complexity, Syntactic_Complexity
- Phonological_Difficulty, Consonant_Onset
- PSal scores (Sonority, Modulation, Cue)
- Following-word context features (all of the above for the next word)

These matrices are invaluable for the co-dependency analysis (P(error_type | stutter_type)).

---

## What is Usable for ASR Evaluation

### Tier 1 — Ready for ASR evaluation (audio + orthographic transcript)

| Source | Files | Hours | Notes |
|--------|-------|-------|-------|
| Full archive with orthographic | 683 | 66.2h | Word-level reference text from SFS annotations |
| Curated subset (all have ortho) | 148 | 9.1h | Clinically grouped; overlaps with above |
| Jason transcripts | 66 | ~5h est | Independent ortho+stutter pairs; cross-validation |

**Total unique ASR-ready: ~66 hours** (683 files from full archive, which includes the curated subset)

### Tier 2 — Usable for co-dependency / error analysis

| Source | Files | Hours | Notes |
|--------|-------|-------|-------|
| With stutter + type annotations | ~280 | ~20h | Disfluency-coded phonetic + stutter type classification |
| Jason usage matrices | 68 | — | Per-word linguistic features for co-dependency tables |

### Tier 3 — Audio only (could be transcribed)

| Source | Files | Hours | Notes |
|--------|-------|-------|-------|
| Audio without annotations | 3,621 | 116.1h | Manual transcription or ASR-then-correct |
| Files with non-text annotations only | ~2,500 | ~97h | Have acoustic analysis but no transcription |

---

## Annotation Conventions

See `docs/sfs_transcription_conventions.md` for full documentation including JSRU phonetic alphabet.

### Quick reference

| Marker | Meaning |
|--------|---------|
| `:word` | Content word |
| `/word` | Function word |
| `x` | Word boundary (silence) |
| `Q` | Silent pause / block |
| `"` | Syllable stress marker |
| `(UM)`, `(ER)` | Filled pause |
| `{xN}` | N repetitions of preceding sound |
| `[:]` | Intended form annotation |
| `{U blocks}` | Articulatory block |
