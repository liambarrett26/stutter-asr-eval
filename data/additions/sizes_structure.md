# Info on sizes and structure files

These files came from the original speech res data archive from which the wav and sfs files for slass originate.

On the oriiginal extraction, only sfs and wav files were extracted with no preservation of the file's providence and structure.

For extended analyses, the metainformation about these files are worth egressing and linking up to the current slass archive. This includes the age, sex, native language and other meta information about the speaker. Additionally, some separate transcription and annotation files might be available. This will be across various files, formats and little to no standardisation. However, in the main should be deducible from the information available and knowledge on the project. Any truly ambigous cases can be raised for 2x checking with Pete.

## Where the actual files now live

The large listing and derived files are NOT kept in this repo (this `data/`
directory holds documentation only). They live on the mounted drive:

- `/Volumes/FATSPEECH/store_metadata/sizes.txt`, `structure.txt`
  (and their UTF-8 conversions) — the original PowerShell directory dumps
  of the speech-res store.
- `/Volumes/FATSPEECH/store_metadata/archive_inventory.csv` — parsed
  inventory (994k rows) produced by `src/data/archive_inventory.py`.
- `/Volumes/FATSPEECH/store_metadata/archive_match_manifest.csv`,
  `archive_match_summary.csv`, `archive_egress_log.csv` — egress
  matching + run records.
- `/Volumes/FATSPEECH/source_archives/` — the original egress zips
  (`UNWR.zip`, `Reliability_Analyses.zip`,
  `data_UNWR_Transcritpion_Audio.zip`), retained as provenance.
- Operational egress scripts: `/Volumes/FATSPEECH/scripts/meta_egress/`
  (supplementary files) and `/Volumes/FATSPEECH/scripts/audio_egress/`
  (audio formats). The canonical Python pipeline is version-controlled in
  `src/data/` (`archive_inventory.py`, `archive_match.py`,
  `archive_egress.py`).

See `docs/slass_extended.md` for how the egressed data was organised.



