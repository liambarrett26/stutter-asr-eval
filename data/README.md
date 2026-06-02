# data/ — documentation only

This directory holds **documentation and small description files** about
the project's data. The actual data — audio, transcripts, derived
inventories, listings, zips — lives on the mounted drive
(`/Volumes/FATSPEECH/`), never in the repo. `data/**` is gitignored.

## What's here

- `additions/sizes_structure.md` — describes the speech-res store
  listing dumps and where they now live on the drive.
- `additions/reliability.md` — provenance notes for the UNWR reliability
  data (the K. Tang correspondence).
- `raw/`, `processed/`, `results/` — empty scaffolding mirroring the
  intended on-drive layout (see CLAUDE.md).

## Where the data actually lives (`/Volumes/FATSPEECH/`)

| Path | Contents |
| ---- | -------- |
| `slass/` | base SLASS: `full_archive/`, `processed/`, `supplementary/` |
| `slass_extended/` | held-separate additional SLASS audio + out-of-scope stimuli |
| `uclass/`, `fluencybank/`, `librispeech/` | other corpora |
| `unwr/` | UNWR adult SSI cohort |
| `unwr_reliability/` | children's nonword reading: TextGrids + audio |
| `standardised/` | unified-schema CSVs per corpus |
| `store_metadata/` | store listings, inventory, egress manifests/logs |
| `source_archives/` | original egress zips (provenance) |
| `scripts/` | operational egress scripts (meta_egress, audio_egress) |

The version-controlled processing/analysis code is under `src/data/` and
`results/eda/`; per-corpus documentation is under `docs/`.
