# FluencyBank — Shared Database for the Study of Fluency

Source: https://talkbank.org/fluency/
Part of: TalkBank (Carnegie Mellon University)
Funded by: NIH NIDCD grant R01-DC015494
PI: Nan Bernstein Ratner (University of Maryland)

**Citation requirement:** All published work must cite:
> Bernstein Ratner, N. & MacWhinney, B. (2018). Fluency Bank: A new resource for fluency research and practice. *Journal of Fluency Disorders*, 56, 69–80.

**Access:** Requires TalkBank account. Media server requires authenticated browser cookie (HttpOnly, domain=talkbank.org). Transcript zips available via git.talkbank.org with same auth.

---

## Data Store Structure

```
fluencybank/
├── raw/
│   ├── Voices-AWS/                    # Adults Who Stutter (primary corpus)
│   │   ├── transcripts/
│   │   │   └── Voices-AWS.zip         # 102 CHAT files + OASES PDFs + ratings
│   │   └── media/
│   │       ├── interview/             # 57 MP4 videos (spontaneous speech)
│   │       └── reading/               # 45 MP4 videos (SSI-4 Friuli passage)
│   │
│   ├── Voices-CWS/                    # Children Who Stutter
│   │   ├── transcripts/
│   │   │   └── Voices-CWS.zip         # 48 CHAT files + OASES forms
│   │   └── media/
│   │       ├── interview/             # 26 MP4 videos
│   │       └── reading/               # 22 MP4 videos (SSI-4 grade-level passages)
│   │
│   ├── Voices-AWC/                    # Adults Who Clutter
│   │   ├── transcripts/
│   │   │   └── Voices-AWC.zip         # 7 CHAT files
│   │   └── media/
│   │       ├── interview/             # 3 MP4 videos
│   │       ├── reading/               # 3 MP4 videos
│   │       └── stuttering/            # 1 MP4 video
│   │
│   ├── UMD-CMU/                       # Longitudinal CWS + Controls
│   │   ├── transcripts/
│   │   │   └── UMD-CMU.zip            # 143 CHAT files
│   │   └── media/
│   │       ├── CWS/                   # 65 MP4 videos (clinician/parent/frog tasks × years)
│   │       └── Control/               # 48 MP4 videos (same tasks, fluent children)
│   │
│   ├── Hakim/                         # Nonword Repetition Study
│   │   ├── transcripts/
│   │   │   └── Hakim.zip              # 32 CHAT files
│   │   └── media/
│   │       ├── CWS/
│   │       │   ├── LS/                # 8 files (language sample)
│   │       │   └── Nonword/           # 8 files (nonword repetition)
│   │       └── TD/
│   │           ├── LS/                # 8 files
│   │           └── Nonword/           # 8 files
│   │
│   ├── Examples/                      # Single speaker demo
│   │   └── transcripts/
│   │       └── Examples.zip           # 1 CHAT file
│   │
│   ├── VanZaalen/                     # Single adult clutterer
│   │   └── transcripts/
│   │       └── VanZaalen.zip          # 5 CHAT files
│   │
│   └── Brejon/                        # French children (final stutter)
│       └── transcripts/
│           └── Brejon.zip             # 8 CHAT files
│
└── README.md
```

---

## Corpus Summaries

### Voices-AWS — Adults Who Stutter (PRIMARY)

- **38 self-identified adults** who stutter (American English)
- **Interview**: 6 structured questions about stuttering's impact on daily life, work, social interactions
- **Reading**: SSI-4 Friuli passage (from June 2017 onward)
- **Transcripts**: CHAT format with utterance-level timestamps (ms), unannotated for disfluency type
- **Additional data**: OASES (Overall Assessment of the Speaker's Experience of Stuttering) forms (unscored), ratings spreadsheet
- **Filename convention**: `{age}{gender}{variant}.mp4` (e.g., `24fb.mp4` = 24-year-old female, variant b)
- **Most participants have two samples** (interview + reading)
- **DOI**: 10.21415/T5VC91

### Voices-CWS — Children Who Stutter

- **22 children/teenagers** who stutter, ages 9–17
- **Interview**: Questions about support meetings, family, peers, speech, therapy
- **Reading**: Grade-appropriate SSI-4 passages
- **OASES-C/OASES-T** forms (from 2021 onward, unscored)
- **Filename convention**: `{age}{gender}{variant}.mp4`
- **DOI**: 10.21415/T5Q692

### Voices-AWC — Adults Who Clutter

- **4 adults** who clutter (different disorder from stuttering)
- **3 speakers** with interview, reading, and 1 with stuttering sample
- Smaller corpus; useful for differential diagnosis comparison
- **DOI**: 10.21415/T5VC91

### UMD-CMU — Longitudinal CWS + Controls

- **15 children** within 6 months of stuttering onset + **15 age/sex/SES-matched** fluent controls
- **3-year longitudinal** design with annual assessments (y1, y2, y3)
- **Tasks**: Clinician interaction, parent interaction, frog story narration
- **143 CHAT transcripts** — conversational speech between children and adults
- Video recordings of each session
- Additional cohorts of late-talking and Spanish-English bilingual children
- Demographic data available on request from PI (nratner@umd.edu)

### Hakim — Nonword Repetition

- **8 CWS** (4;1–8;4, mean 5;10) + **8 typically-developing** controls (matched age ± 4 months, gender, maternal education)
- **Tasks**: Language sample (LS) + nonword repetition
- Audio recordings
- **Note**: Nonword repetition is not natural speech — limited ASR evaluation utility, but language samples are usable
- **DOI**: 10.21415/T5N682
- **Citation**: Hakim & Ratner (2004)

### Not Downloaded (low utility for ASR evaluation)

| Corpus | Reason for exclusion |
|--------|---------------------|
| **IISRP / IISRP-new** | Password-restricted; ages 2–6 (very young children) |
| **Ratner** | Password-restricted; ages 2–4 |
| **Sawyer** | Password-restricted; ages 6–8, n=17 |
| **Tellis** | Password-restricted; ages 6–12, n=8 |
| **Wagovich** | No media available |
| **EllisWeismer** | Late talkers, not stuttering |
| **POLER** | Children with epilepsy, not stuttering |
| **Rescorla** | Late talkers, not stuttering |
| **Purdue** | No media available |
| **Ulm** | No media; German |

---

## CHAT Transcript Format

All transcripts use the CHILDES CHAT format. Key features for ASR evaluation:

### Utterance structure
```
*PAR:	I'd stutter on like every syllable of every word . 62597_69962
```
- `*PAR:` = participant (person who stutters); `*INV:` = investigator
- Timestamps at end: `62597_69962` = start_ms to end_ms
- `.` marks utterance boundary

### Annotation tiers
```
%mor:	pron|I~aux|would verb|stutter adp|on det|every noun|syllable ...
%gra:	1|3|NSUBJ 2|3|AUX 3|0|ROOT 4|3|ADVMOD ...
```
- `%mor:` = morphological analysis (POS tags)
- `%gra:` = dependency grammar relations

### Disfluency conventions (standard CHAT)
- `&-um`, `&-uh` = filled pauses
- `[/]` = repetition (retracing without correction)
- `[//]` = revision (retracing with correction)
- `[///]` = reformulation
- `(.)` = short pause; `(..)` = medium pause; `(...)` = long pause

### Media reference
```
@Media:	20f, video
```
Links transcript to corresponding video file.

### For ASR evaluation
- Extract `*PAR:` lines only (speaker who stutters, not interviewer)
- Parse timestamps for utterance-level segmentation
- Strip CHAT coding markers to get clean reference text
- Pair with audio extracted from MP4 video files (`ffmpeg -i input.mp4 -vn -acodec pcm_s16le -ar 16000 output.wav`)

---

## What is Usable for ASR Evaluation

### Tier 1 — Spontaneous speech with transcripts + timestamps

| Corpus | Speakers | Sessions | Transcript | Notes |
|--------|----------|----------|------------|-------|
| Voices-AWS interview | 38 adults | 57 | CHAT, timestamped, unannotated | Primary evaluation set; comparable to Mujtaba et al. (2024) |
| Voices-CWS interview | 22 children | 26 | CHAT, timestamped, unannotated | Child/teen comparison |
| UMD-CMU CWS | 15 children | ~65 | CHAT, conversational | Longitudinal; multi-speaker (child + adult) |
| UMD-CMU Control | 15 children | ~48 | CHAT, conversational | Fluent baseline for CWS comparison |

### Tier 2 — Read speech with known reference text

| Corpus | Speakers | Sessions | Transcript | Notes |
|--------|----------|----------|------------|-------|
| Voices-AWS reading | ~38 adults | 45 | SSI-4 Friuli passage (known text) | Direct WER calculation possible |
| Voices-CWS reading | ~22 children | 22 | SSI-4 grade-level passages | Known reference text |

### Tier 3 — Supplementary

| Corpus | Speakers | Sessions | Notes |
|--------|----------|----------|-------|
| Hakim LS | 8 CWS + 8 TD | 16 | Language samples, young children |
| Voices-AWC | 4 adults | 7 | Cluttering (not stuttering) — differential analysis |

---

## Download Status (2026-03-25)

- [x] All transcript zips downloaded (8 corpora, 346 CHAT files total)
- [ ] Media downloads in progress (~12.9 GB estimated total)
  - Voices-AWS: IN PROGRESS (57 interview + 45 reading MP4s)
  - Voices-CWS: IN PROGRESS (26 interview + 22 reading MP4s)
  - Voices-AWC: IN PROGRESS (7 MP4s)
  - UMD-CMU: IN PROGRESS (65 CWS + 48 Control MP4s)
  - Hakim: COMPLETE (32 audio files)
- [ ] Extract audio from MP4 videos (ffmpeg → 16kHz mono WAV)
- [ ] Parse CHAT transcripts into standardised CSV format
- [ ] Build FluencyBank inventory and speaker metadata CSV

## Technical Notes

- Media server (`media.talkbank.org`) returns only first 11 bytes by default (partial content). Must send explicit `Range: bytes=0-99999999999` header to download full files.
- Server is per-connection throttled at ~0.45 MB/s; parallel streams increase total throughput.
- Cookie expires after ~24 hours; refresh by logging in via browser.
- Download script: `scripts/download_fluencybank_media.py`
