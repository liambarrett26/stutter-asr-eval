# UNWR (adult SSI cohort)

UCL's adult cohort for the Tang & Wong-lineage Stuttering Severity
Instrument work. Wired into the project pipeline as a parallel corpus
to SLASS, UCLASS, FluencyBank and LibriSpeech.

## Contents

| Statistic | Value |
| --------- | ----- |
| Speakers | 58 |
| Audio | 1.87 h total (each speaker has q1 + q2 SSI clips of ~60 s) |
| Transcripts | 56 syllable-segmented `.docx` files |
| Master metadata | `Data Directory.xlsx` — 8 sheets covering demographics, SSI, ASRS, PSal, UNWR/SRT/PDT item-level scoring |

Cohort breakdown:

| Group | n | Notes |
| ----- | -: | ----- |
| Control (C) | 25 | Essentially fluent (SSI 0.0–1.74%) |
| Stutter (S) | 12 | Clinical stutter diagnosis (SSI 0.0–6.00%) |
| Attention-ADHD (A) | 15 | ADHD diagnosis, no stutter |
| Attention+Stutter (AS) | 6 | Both diagnoses |

Age range 18–60. Adult cohort — fills a gap between the child-skewed
SLASS / UCLASS holdings and the fluent-adult LibriSpeech baseline.

## Layout on disk

```
/Volumes/FATSPEECH/unwr/
├── raw/UNWR/                      original archive contents
│   ├── Control (C)/<PID> (<ID>)/<PID>_SSI/<PID>_{q1,q2}.wav
│   ├── Control (C)/<PID> (<ID>)/<PID>_SSI/<PID>_transcript.docx
│   ├── Stutter (S)/...
│   ├── Attention-ADHD (A)/...
│   ├── Attention + Stutter (AS)/...
│   └── Data Directory.xlsx
└── processed/
    ├── speakers.csv               one row per PID with full phenotype
    ├── inventory.csv              audio paths, durations, sample rates
    ├── scoring_unwr.csv           per-item nonword reading scores
    ├── scoring_srt.csv            per-item speech-repetition scores
    ├── scoring_pdt.csv            per-item phonetic-discrimination scores
    └── transcripts/<PID>.csv      clean + syllable-segmented transcript

/Volumes/FATSPEECH/standardised/unwr/
└── <PID>.csv                      unified-schema rows; stutter_type =
                                   "ssi_pct=<value>;group=<label>"
```

## Pipeline

- `src/data/process_unwr.py` — one-shot processor: reads the raw archive,
  parses the xlsx, the `.docx` transcripts and the WAV headers, writes
  the `processed/` outputs above. Stdlib + openpyxl only.
- `src/data/standardise.py --corpus unwr` — emits one record per speaker
  into `standardised/unwr/`. `text_intended` is period-stripped clean
  prose; `text_surface` preserves the syllable boundaries; `stutter_type`
  carries the session-level SSI score and group label so downstream
  analyses can stratify.

## What's in the EDA

- Dataset summary row (`figures/dataset_summary.png`).
- Group breakdown panel (`figures/unwr_groups.png`): speakers per group,
  SSI severity by group, age distribution by group.

UNWR does not yet contribute to the per-word duration KDE
(`figures/kde_duration_cross_dataset.png`) because we have transcripts
but no time alignment. Forced alignment would unlock that.

## Known quirks

- Transcript `.docx` files use period-delimited syllable segmentation
  (`Fa.vo.rite. hob.bies.`). The processor strips periods for the
  clean text column and preserves the segmented form in
  `text_syllabified`.
- A handful of `.docx` files end with inline summary text
  (e.g. "343 syllables 0 stuttered"). Not stripped — they make it into
  `text_intended`. Trivial to filter at WER time if needed.
- A couple of speakers carry `group_directory = "Attention + Stutter (AS)"`
  while `all_data_group = "Stutter"` — the recruitment group differs
  from the All_data clinical group. Both are stored; pick deliberately.
- Three sheets in the xlsx (PDT_scoring, UNWR_scoring, SRT_scoring) hold
  per-item-per-speaker scores; they are written verbatim to the
  corresponding `scoring_*.csv` files for downstream use.
