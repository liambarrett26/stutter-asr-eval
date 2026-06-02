# UNWR Reliability (children's nonword reading) — AUDIO RECEIVED

The Reliability dataset from Kevin Tang (gold-standard children's
non-word reading retranscriptions, originally for Clarissa's
reliability paper). TextGrids parsed and inter-rater agreement computed
earlier; **the corresponding audio arrived 2026-06-02** and has been
linked to the TextGrid items. Phoneme-level ASR evaluation against the
gold standard is now unblocked.

## Status

- TextGrids parsed: 1,763 files → 1,735 valid rows (transcriber tree).
- Transcribers: Clarissa (gold-standard), Kaho, Roaa, plus the
  "original" school-cohort buckets.
- Inter-rater PER computed (see below).
- **Audio received and verified.** The transcription-audio bundle adds
  2,758 WAVs (2,721 per-item + 37 whole-session) with co-located
  reference TextGrids and a master demographics sheet.
- Audio↔TextGrid alignment verified: of 2,370 per-item WAVs paired with
  a parseable TextGrid extent, **2,369 (100.0%) match within 50 ms**
  (mean |wav − xmax| = 0.000 s). The audio and transcriptions line up.

## Layout on disk

```
/Volumes/FATSPEECH/unwr_reliability/
├── raw/
│   ├── Reliability_Analyses/      transcriber retranscription tree
│   │   ├── Step1_Fidelity/        R + Python analysis scripts and outputs
│   │   └── data/Retranscription/  per-transcriber dirs, 1,763 TextGrids
│   └── transcription_audio/data/  AUDIO BUNDLE (2026-06-02), 1.9 GB
│       ├── <cohort>/<child>/syllN/<item>.wav + <item>.TextGrid
│       └── meta/                  master demographics (xlsx + tsv),
│                                  feature data, seg.data, filename lists
└── processed/
    ├── items.csv                  one row per (transcriber, school,
                                   child, item) with ortho / target /
                                   response and interval times
    ├── audio_items.csv            one row per WAV: cohort/child/syll/item,
                                   wav path + duration + sample rate,
                                   co-located TextGrid + xmax, duration
                                   match flag, ortho/target/response
    ├── agreement_pairwise.csv     1,255 rows of per-item PER comparisons
                                   between transcriber pairs
    └── agreement_summary.csv      per-pair + per-rater rolled-up metrics
```

## Inter-rater agreement baseline

Mean phoneme-error-rate (PER) computed via token-level Levenshtein on
the response tier:

| Rater pair | n | PER | Exact-match |
| ---------- | -: | --: | ----------: |
| Clarissa vs Clarissa_original | 68 | 0.004 | 98.5% |
| Clarissa vs Kaho_original | 129 | 0.046 | 93.8% |
| Clarissa_original vs Roaa_28Nov | 68 | 0.094 | 61.8% |
| Clarissa vs Roaa_28Nov | 210 | 0.173 | 39.0% |
| Kaho vs Roaa_28Nov | 244 | 0.245 | 18.0% |
| Clarissa vs Kaho | 210 | 0.246 | 21.4% |
| Kaho_original vs Roaa_28Nov | 129 | 0.255 | 24.0% |
| Kaho vs Kaho_original | 129 | 0.304 | 15.5% |

Accuracy against published targets:

| Rater | n | PER vs target |
| ----- | -: | ------------: |
| Clarissa_original | 36 | 0.113 |
| Roaa_28Nov | 200 | 0.255 |
| Clarissa | 167 | 0.273 |
| Kaho_original | 118 | 0.323 |
| Kaho | 167 | 0.339 |

**Human ceiling band for any future phoneme-level ASR system:**
roughly **5%** (gold-standard humans agreeing with each other) to
**25%** (naive humans agreeing without prior context). A phoneme-level
ASR inside that band is matching human performance on this task.

## Phoneme-level ASR evaluation (now unblocked)

The audio has arrived and is verified aligned to the TextGrids, so the
plan below can proceed:

1. ~~Verify TextGrid times line up with the audio~~ — DONE: 100% of
   paired per-item WAVs match TextGrid xmax within 50 ms
   (`processed/audio_items.csv`).
2. Use Clarissa's retranscription as the per-item phoneme reference;
   evaluate Whisper / wav2vec2 / HuBERT / phoneme-level commercial APIs
   against it.
3. Stratify by syllable length (syll2 / syll3 / syll4) and by item
   familiarity (prac vs test) for a fairness breakdown.
4. Report ASR PER alongside the human inter-rater band (≈5% gold-standard
   to ≈25% naive) so reviewers can see where the system sits relative to
   the human ceiling.

## Pipeline

- `src/data/process_unwr_reliability.py` — walks the raw tree, parses
  TextGrids (minimal stdlib parser), writes `items.csv`.
- `src/data/unwr_reliability_agreement.py` — joins by (school, child,
  item) across transcribers, computes pairwise edit distance and PER
  against the target, writes `agreement_pairwise.csv` and
  `agreement_summary.csv`.
- `src/data/link_unwr_reliability_audio.py` — walks the transcription-
  audio bundle, pairs each WAV with its co-located TextGrid, probes WAV
  duration vs TextGrid xmax, writes `audio_items.csv`.
