# Transcript Standardisation

This document describes how transcripts from each corpus are converted into the unified evaluation schema. The standardisation code is in `src/data/standardise.py`.

---

## Unified Schema

Every transcript record, regardless of source corpus, is standardised into a single CSV format:

| Column | Type | Description |
|--------|------|-------------|
| `file_id` | string | Unique identifier for the recording session |
| `start_s` | float | Start time in seconds (empty if unavailable) |
| `end_s` | float | End time in seconds (empty if unavailable) |
| `speaker` | string | Speaker code (e.g. PAR, CHI, or empty for single-speaker) |
| `text_intended` | string | What the speaker meant to say (disfluencies removed, clean English) |
| `text_surface` | string | What the speaker actually produced (disfluencies preserved where possible) |
| `stutter_type` | string | Stutter type label if annotated (e.g. block, repetition, filled_pause, fluent, or empty) |
| `corpus` | string | Source corpus (slass, uclass, fluencybank, librispeech) |

### Dual-reference design

ASR evaluation on stuttered speech requires two reference transcripts:

- **Intended**: evaluates whether the ASR recovered the communicative meaning. A model that "cleans up" disfluencies scores well here.
- **Surface**: evaluates whether the ASR faithfully captured what was spoken. A model that deletes repetitions and blocks scores poorly here.

These two references can produce different WER values and even invert model rankings. Both are reported in the evaluation.

---

## Corpus-Specific Conversion Rules

### SLASS

**Source layers**: orthographic, stutter (JSRU phonetic), type, syllable

**Intended text** is derived from the **orthographic layer**:

| Input | Rule | Output |
|-------|------|--------|
| `:new` | Strip `:` prefix (content word) | `new` |
| `/and` | Strip `/` prefix (function word) | `and` |
| `:school.` | Strip prefix + trailing period | `school` |
| `x` | Remove (word boundary marker) | *(removed)* |
| `Q` | Remove (silent pause) | *(removed)* |
| `:"ev-ree` | Strip prefix + `"` stress markers | `ev-ree` |

**Surface text** and **stutter type** are derived from the **stutter layer** (JSRU phonetic with disfluency coding):

The stutter layer uses JSRU phonetic notation with embedded disfluency markers. We do NOT use the JSRU phonetic spelling as the surface text (e.g., `nyuu` for "new") because that conflates pronunciation variation with stuttering. Instead:

- `text_surface` = the same English word from the orthographic layer
- `stutter_type` = classified from the disfluency markers in the stutter label

The stutter layer is used for **disfluency classification only**, not for surface spelling.

**Disfluency classification rules** (from stutter layer labels):

| Marker in label | Classification | Example |
|----------------|---------------|---------|
| `Q` (standalone) | `block` | `Q` → silent block |
| `Q` within word + letters | `block` | `:JQQQus[:"JUST]` → block in "just" |
| `{U blocks}` | `block` | articulatory block |
| `{xN}` | `repetition` | `{x14}` → 14 repetitions |
| Repeated segments with spaces | `repetition` | `AA AA AA` |
| 4+ identical consecutive chars | `prolongation` | `MMMMM`, `IIIII` |
| `(UM)`, `(ER)`, `(ERM)` | `filled_pause` | filled pause |
| `(HA)` | `hesitation` | hesitation |
| `[:]` bracket (no other markers) | `other_disfluency` | disfluent but type unclear |
| No markers | `fluent` | clean production |

Multiple markers can co-occur, producing combined types: `block+repetition`, `block+repetition+prolongation`, etc.

**Special surface text cases**:
- Filled pauses: `text_surface` = `um`, `er` (these ARE different words from the intended speech)
- Standalone blocks: `text_surface` = `[block]` (silence, no word produced)
- All other words: `text_surface` = same as `text_intended` (the English word)

**Word alignment**: Orthographic and stutter records are matched by timestamp (nearest within 50ms tolerance).

**Type layer** (`*_type.csv`) is used as a fallback if the stutter layer is unavailable. It uses a different notation with cluster position markers (`[N]`, `{N]`) but the same classification logic applies.

### UCLASS

**Source layers**: word_orth (TextGrid), word (stutter-coded TextGrid), flat ortho

**Intended text** from orthographic TextGrid (`*_word_orth.csv`):

| Input | Rule | Output |
|-------|------|--------|
| `well` | Lowercase | `well` |
| `MY` | Lowercase (CAPS = stressed/stuttered) | `my` |
| `suppose` | Lowercase | `suppose` |

**Surface text** from stutter-coded TextGrid (`*_word.csv`):

Same rules as SLASS stutter layer (JSRU phonetic + disfluency coding).

**Flat orthographic** transcripts (no timestamps):

| Input | Rule | Output |
|-------|------|--------|
| Full paragraph with punctuation | Lowercase, strip punctuation (keep apostrophes), fix encoding | Clean text |
| CAPS words | Lowercased (stress/stutter info lost) | Normal case |
| `er`, `erm`, `um` | Preserved (part of the transcript) | Kept |

When only flat orthographic is available, `text_intended` and `text_surface` are identical (no separate surface annotation exists).

### FluencyBank

**Source**: CHAT transcripts parsed by `src/data/parse_chat.py`

| Input (from CHAT parser) | Rule | Output |
|--------------------------|------|--------|
| Clean text from `parse_chat.clean_chat_text()` | Lowercase, strip remaining punctuation | Clean text |
| `&-um`, `&-uh` | Already removed by CHAT parser | *(not present)* |
| `[/]`, `[//]` retracing | Already removed by CHAT parser | *(not present)* |

FluencyBank Voices-AWS/CWS transcripts are **unannotated for disfluency type**. The clean text represents the words as transcribed, with filled pauses and CHAT markers already stripped. Both `text_intended` and `text_surface` are set to the same clean text.

**Timestamps**: Converted from milliseconds (CHAT format) to seconds.

### LibriSpeech

**Source**: `.trans.txt` files (UTTERANCE_ID followed by UPPERCASE text)

| Input | Rule | Output |
|-------|------|--------|
| `HE HOPED THERE WOULD BE STEW` | Lowercase | `he hoped there would be stew` |

LibriSpeech is fluent read speech. Both `text_intended` and `text_surface` are identical. `stutter_type` is set to `fluent`.

**No within-file timestamps**: Each FLAC file is one utterance.

---

## Normalisation Pipeline

After corpus-specific stripping, ASR evaluation applies additional normalisation using the **Whisper `EnglishTextNormalizer`** (the Open ASR Leaderboard standard). This is applied identically to both reference and hypothesis text at evaluation time. It handles:

- Case folding
- Punctuation removal
- Number normalisation ("ten" -> "10")
- Contraction expansion ("I'm" -> "I am")
- British to American spelling conversion

The corpus-specific stripping in `standardise.py` is a **pre-normalisation** step that removes annotation-specific formatting before the general-purpose normaliser is applied.

### Pipeline order

```
Raw annotation (corpus-specific format)
    ↓
Corpus-specific stripping (standardise.py)
    → text_intended (clean English, disfluencies removed)
    → text_surface (disfluencies preserved where available)
    ↓
Whisper EnglishTextNormalizer (at evaluation time)
    → normalised_reference
    → normalised_hypothesis
    ↓
jiwer WER/CER computation
```

---

## Filled Pause Handling

Filled pauses (um, uh, er, erm) require explicit handling because they sit at the boundary between disfluency and normal speech:

| Evaluation mode | Filled pause in reference | Rationale |
|----------------|--------------------------|-----------|
| **Intended** | Removed | The speaker did not intend to say "um" |
| **Surface** | Preserved | The speaker did produce "um" |

Note: The Whisper `EnglishTextNormalizer` removes "um", "uh", "mm", "mhm", "hmm" by default. For surface evaluation, these must be retained in the reference **before** the normaliser runs, or the normaliser must be modified to preserve them.

---

## Data Availability Summary

| Corpus | Intended text | Surface text | Stutter type | Timestamps |
|--------|--------------|-------------|-------------|------------|
| SLASS (ortho + stutter) | Yes (English words) | Same as intended (English) | Yes (classified from stutter layer: block, repetition, prolongation, filled_pause, etc.) | Yes |
| SLASS (ortho only) | Yes (English words) | Same as intended | No | Yes |
| UCLASS (word_orth TextGrid) | Yes | Same as intended | No | Yes |
| UCLASS (flat ortho only) | Yes | Same as intended | No | No |
| FluencyBank | Yes | Same as intended | No | Yes (ms) |
| LibriSpeech | Yes (fluent) | Same as intended | `fluent` | No (per-file) |

**Key design decision**: `text_surface` and `text_intended` contain the same English word for all stuttered words. The disfluency information is captured in `stutter_type`, not in alternative spelling. This enables straightforward WER computation (ASR output compared against English words) while the `stutter_type` column enables conditional error analysis (P(error | stutter_type)).
