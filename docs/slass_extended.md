# SLASS extended data & supplementary metadata

In June 2026 two further drops were egressed from the speech-lab Windows
store and organised by `src/data/organise_slass_extended.py`. They are
partitioned according to an explicit policy (agreed with Pete / Liam,
2026-06-02): **the additional audio is held separate** from the base
dataset so the refined analyses run on an unchanged base, while
**transcripts and rosters may enrich the base**.

## Where things live

| Location | What | Held separate? |
| -------- | ---- | -------------- |
| `/Volumes/FATSPEECH/slass_extended/audio/` | 1,376 participant-format audio files (8.6 GB): mp3 823, wma 362, m4a 181, flac 7, aiff/aifc/ra 3 | **Yes** |
| `/Volumes/FATSPEECH/slass_extended/out_of_scope/` | 10,346 `.au` stimulus files (10.9 GB) — tone/word stimuli from PhD sub-projects, TIMIT, Simulator. Quarantined, not deleted. | **Yes** |
| `/Volumes/FATSPEECH/slass/supplementary/transcripts/` | 709 orthographic / stutter-bracketed / SALT / CHAT transcripts | No (may enrich base) |
| `/Volumes/FATSPEECH/slass/supplementary/master_rosters/` | 529 participant master lists & demographics spreadsheets | No (may enrich base) |
| `/Volumes/FATSPEECH/slass/supplementary/speaker_metadata/` | 466 per-speaker admin docs | No (may enrich base) |

Each tree has an `index/` CSV and a preserved provenance log
(`_audio_copy_log.csv` / `_egress_log.csv`).

## Base dataset is untouched

The base SLASS dataset that the EDA and Pete's analyses run on —
`/Volumes/FATSPEECH/slass/{full_archive,processed}`, the Jason matrices,
and `/Volumes/FATSPEECH/standardised/slass/` — was not modified by this
housekeeping. The `extra_audio/` and `slass_additions/` landing dirs
were emptied and removed.

## Audio index — id resolution status

`slass_extended/index/extended_audio_index.csv`:

| id_status | n | meaning |
| --------- | -: | ------- |
| known_base_speaker | 468 | filename speaker id matches a base SLASS speaker (110 distinct) |
| unresolved | 898 | no parseable id in filename; includes 75 word-stimulus mp3s <50 KB (mostly `Diane_Leung`) that should be dropped at integration |
| new_speaker | 10 | speaker id absent from the base archive |

## Why `.au` is out-of-scope

The Windows audio scan (`Find-AudioFiles.ps1`) had no sub-project
skip-list, so `.au` swept in 10,346 stimulus files from `eryk_PhD_data`,
`Liam_PhD_data`, `speech2_data`, `Simulator` and `TIMIT`. These are tone
and word stimuli, not SLASS participant recordings. They are quarantined
under `out_of_scope/` rather than deleted, pending a review decision.

## Integration backlog (future)

1. Resolve the 898 `unresolved` audio files via parent-dir conventions
   and the master rosters; exclude the word-stimulus mp3s.
2. Settle provenance for borderline trees (`UCLASS Recode` is probably
   UCLASS).
3. Convert held audio to 16 kHz mono WAV and add to the base inventory +
   standardisation.
4. Roster join to resolve the 8,839 opaque `c###/d###` SLASS sessions to
   demographics.
5. Parse SALT / stutter-bracketed transcripts to extend the ASR-ready
   reference set.

## Reproduce

```
python src/data/organise_slass_extended.py --dry-run   # preview
python src/data/organise_slass_extended.py             # move + index
```

Idempotent: re-running cleans up any leftover sources and rebuilds the
indexes from the destination trees.
