# SFS Stuttering Transcription Conventions

Source: Howell, P. & Huckvale, M. (2004). "Facilities to assist people to research into stammered speech." *Stammering Research*, 1.
Original document: `sr.howell+huckvale.doc`

This document describes the annotation and transcription conventions used in the UCL Speech Filing System (SFS) for stuttered speech data, as used in UCLASS and SLASS corpora.

---

## 1. Overview

SFS stores audio recordings with multiple time-aligned annotation layers. For stuttered speech, the key layers are:

- **Orthographic**: Standard spelling of what was said
- **Phonetic**: JSRU phonetic alphabet transcription of how it was said (including disfluencies)
- **Stutter/type annotations**: Disfluency event labels

Annotations are stored as time-stamped records. Each record has a start time, duration, and text label.

---

## 2. Annotation Label Conventions

### Word type prefixes (used in both orthographic and phonetic layers)

| Prefix | Meaning | Example |
|--------|---------|---------|
| `:` | Content word (nouns, verbs, adjectives, adverbs) | `:school`, `:anyway` |
| `/` | Function word (articles, prepositions, conjunctions, pronouns) | `/and`, `/the`, `/i` |
| `x` | Word/syllable boundary marker (silence between words) | `x` |
| `Q` | Silent pause (longer silence, not a word boundary) | `Q` |

### Disfluency markers (phonetic/stutter layers)

| Marker | Meaning | Example |
|--------|---------|---------|
| `"` | Syllable stress marker / syllable boundary in multi-syllable words | `:"star-tid"` |
| `(UM)` | Filled pause "um" | `(UM)` |
| `(ER)` | Filled pause "er/uh" | `(ER)` |
| `(HA Q ER)` | Hesitation sequence (e.g., "ha" + pause + "er") | `(HA Q ER)` |
| `{xN}` | Repetition of preceding sound, N times | `{x9}` = 9 repetitions |
| `[:]` | Intended form annotation (what the speaker meant to say) | `[:"STAR-tid]` |
| `{U blocks}` | Block annotation (articulatory block) | `{U blocks}` |
| CAPS in ortho | Stressed/stuttered word in orthographic transcription | `MY`, `JUST`, `WENT` |

### Example: Orthographic vs Phonetic/Stutter annotation

**Orthographic layer** (what was intended):
```
2.08580 :new
2.44190 :high
2.69620 :school.
4.45140 /and
4.59130 :every
4.90780 :morning
```

**Stutter layer** (what was actually produced):
```
2.08580 :nyuu
2.44190 :hie
2.69620 :skuul.
4.45140 /aand
4.59130 :"ev-ree
4.90780 :"mawn-ingx
```

### Example: Severe stuttering with blocks and repetitions

Orthographic: `:JUST`
Stutter: `:JQQQJQQQQJQJQJQQQUQQQus[:"JUST]`

This reads as: repeated attempts at "J" with silent blocks (Q), then "U" with blocks, finally producing "us" — the intended word was "JUST".

---

## 3. The JSRU Phonetic Alphabet

The JSRU (Joint Speech Research Unit) alphabet is an ASCII-compatible phonetic transcription system used in SFS. It maps standard keyboard characters to phonetic symbols.

### Consonants

| JSRU | IPA | Example (initial) | Example (non-initial) |
|------|-----|-------------------|-----------------------|
| `p` | p | pen | lip |
| `b` | b | bad | nib |
| `t` | t | tea | light |
| `d` | d | do | lad |
| `k` | k | cat | crack |
| `g` | g | got | fog |
| `ch` | tʃ | chin | watch |
| `j` | dʒ | June | village |
| `f` | f | food | off |
| `v` | v | voice | give |
| `th` | θ | thin | tenth |
| `dh` | ð | then | with |
| `s` | s | same | success |
| `z` | z | zoo | was |
| `sh` | ʃ | show | wash |
| `zh` | ʒ | genre | beige |
| `h` | h | happy | behave |
| `m` | m | man | swim |
| `n` | n | know | gone |
| `ng` | ŋ | — | sing |
| `l` | l | leg | girl |
| `r` | ɹ | red | arrow |
| `y` | j | year | value |
| `w` | w | wet | quick |
| `x` | x | — | loch (Scots) |
| `gx` | ʔ | got (Cockney) | glottal stop |

### Vowels

| JSRU | IPA | Example |
|------|-----|---------|
| `i` | ɪ | sit |
| `o` | ɒ | got |
| `oo` | ʊ | put |
| `aa` | æ | hat |
| `e` | ɛ | ten |
| `U` | ʌ | cup |
| `a` | ə | ago (schwa) |
| `ee` | iː | see |
| `aw` | ɔː | saw |
| `uu` | uː | too |
| `ar` | ɑː | arm |
| `er` | ɜː | fur |
| `ai` | eɪ | page |
| `ie` | aɪ | eye |
| `oi` | ɔɪ | boy |
| `oa` | əʊ | home |
| `ou` | aʊ | now |
| `ia` | ɪə | beer |
| `ei` | ɛə | bare |
| `ur` | ʊə | tour |

---

## 4. SFS Annotation Workflow

The standard workflow for creating transcriptions in SFS is:

1. **Chunking**: Audio is segmented into sentence-length regions using the `npoint` endpoint detector (energy-based). Pauses are labelled `/`, speech regions labelled `chunkNN`.

2. **Orthographic transcription**: Chunk labels are replaced with the spoken text using `anedit`. Conventions:
   - Lower case except proper nouns
   - No punctuation
   - Numbers and abbreviations spelled out
   - Non-speech sounds marked as `[cough]`, `[breath]`, etc.
   - SFS annotations limited to 250 characters per label

3. **Phonetic transcription**: Orthography is automatically converted to JSRU phonetic form using `antrans` (built-in pronunciation dictionary). An exceptions file handles unknown words. Output is then manually corrected.

4. **Automatic alignment**: The `analign` program performs forced alignment of the phonetic transcription to the audio signal using HMM-based acoustic models. This produces word-level and optionally phone-level time alignments.

5. **Manual correction**: Aligned transcriptions are manually checked and corrected in the `eswin` display program, adjusting boundaries and fixing transcription errors.

---

## 5. Data Formats for Export

SFS can export annotations in multiple formats:

| Format | Extension | Description |
|--------|-----------|-------------|
| SFS Annotation | `.sfs` (embedded) | Native format, stored within the SFS file alongside audio |
| Praat TextGrid | `.grid` / `.TextGrid` | Praat-compatible, `"ooTextFile"` header, point or interval tiers |
| CHILDES CHAT | `.cha` | CHILDES-compatible, `@Begin`/`@End` headers, `*SPK:` lines |
| Flat text | `.orth` / `.phon` | Plain text, one annotation per line: `timestamp label` |
| CSV | `.csv` | Comma-separated: `line,tmin,tier,text,tmax` |

### TextGrid format variants found in UCLASS/SLASS

- **`.word.orth.grid`** — Word-level orthographic (plain English words with timestamps)
- **`.word.grid`** — Word-level with stutter annotations (JSRU phonetic + disfluency markers)
- **`.syll.orth.grid`** — Syllable-level orthographic
- **`.syll.grid`** — Syllable-level phonetic
- **`.pw.grid`** — Phonological word level

---

## 6. Interpreting Stutter Annotations for ASR Evaluation

When using these transcriptions as ASR reference text:

### For "intended speech" (surface disfluencies removed):
- Use the **orthographic layer** directly
- Strip `:` and `/` prefixes
- Remove `x` boundary markers and `Q` pauses
- Remove `(UM)`, `(ER)` filled pauses (or map to canonical tokens)
- CAPS words represent the intended word

### For "surface speech" (disfluencies preserved):
- Use the **stutter/phonetic layer**
- The `[:]` bracketed forms show intended vs actual production
- Repetitions (`{xN}`), blocks (`Q`), and filled pauses are part of the surface form
- This is what the speaker actually produced and what the ASR system heard

### Mapping stutter types to standard categories:
| Annotation pattern | Stutter type |
|-------------------|--------------|
| `{xN}` repetitions in phonetic form | Sound/syllable repetition |
| Whole word repeated (same word appears consecutively) | Whole-word repetition |
| `Q` within a word attempt | Block |
| Extended phoneme in transcription | Prolongation |
| `(UM)`, `(ER)`, `(HA)` | Filled pause / interjection |
