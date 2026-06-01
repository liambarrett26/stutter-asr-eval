# UNWR Reliability (children's nonword reading) — ON HOLD pending audio

The Reliability dataset from Kevin Tang (gold-standard children's
non-word reading retranscriptions, originally for Clarissa's
reliability paper) has been parsed and the inter-rater agreement
baseline computed. **Further analysis is on hold pending the
corresponding audio recordings.**

The Reliability_Analyses zip contains only the TextGrid transcriptions
and analysis scripts — no `.wav` files. The audio is presumably back
in the speech-lab Windows store under
`X:\Speech\UNWR-Transcription-Paper\` (visible in the archive listing
during the egress design). It is a natural addition to the egress
manifest when we are next at the lab.

## Status

- TextGrids parsed: 1,763 files → 1,735 valid rows
- Transcribers identified: Clarissa (gold-standard), Kaho, Roaa (28Nov),
  plus the corresponding "original" buckets from the school-cohort
  directories that the inner zips dropped alongside the
  retranscriptions.
- Inter-rater PER computed.
- **No audio available yet** — therefore no ASR evaluation possible
  against this gold standard.

## Layout on disk

```
/Volumes/FATSPEECH/unwr_reliability/
├── raw/Reliability_Analyses/      original archive contents
│   ├── Step1_Fidelity/            R + Python analysis scripts and outputs
│   └── data/Retranscription/      school-cohort and per-transcriber dirs
│                                  with 1,763 TextGrid files
└── processed/
    ├── items.csv                  one row per (transcriber, school,
                                   child, item) with ortho / target /
                                   response and interval times
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

## When the audio arrives

Once the matching `.wav` files are recovered (egress trip):

1. Verify each TextGrid's interval times line up with the audio
   (the TextGrid xmax should match the WAV duration).
2. Use Clarissa's retranscription as the per-item phoneme reference;
   evaluate Whisper / wav2vec2 / HuBERT / phoneme-level commercial APIs
   against it.
3. Stratify by syllable length (syll2 / syll3 / syll4) and by item
   familiarity (prac vs test) for a fairness breakdown.
4. Report ASR PER alongside the human inter-rater band so reviewers can
   see where the system sits relative to human ceiling.

## Pipeline

- `src/data/process_unwr_reliability.py` — walks the raw tree, parses
  TextGrids (minimal stdlib parser), writes `items.csv`.
- `src/data/unwr_reliability_agreement.py` — joins by (school, child,
  item) across transcribers, computes pairwise edit distance and PER
  against the target, writes `agreement_pairwise.csv` and
  `agreement_summary.csv`.
