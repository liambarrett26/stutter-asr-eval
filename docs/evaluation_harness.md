# ASR evaluation harness (Phase 2, step 1)

Three stages, manifest-driven, so inference runs once and re-scoring with
a different normaliser/reference is free.

```
build_manifest.py  →  manifest.jsonl  →  run_asr.py  →  hyp/<model>.jsonl  →  score.py  →  scores/<model>.json
   (this Mac)                              (GPU box)                            (anywhere)
```

## 1. Manifest — `src/evaluation/build_manifest.py`

One JSON unit per audio file: `unit_id, dataset, condition (stuttered|
fluent), audio_path, speaker_id, split, reference_intended,
reference_surface, n_ref_words, duration_s, needs_chunking,
has_stutter_labels, stutter_type_counts`.

Current v1 (`/Volumes/FATSPEECH/manifests/benchmark_v1.jsonl`):

| condition | units | speakers | ref words | audio |
| --------- | ----: | -------: | --------: | ----: |
| stuttered (slass_full, clean≥0.9) | 420 sessions | 122 | 146,136 | 29.8 h |
| fluent (LibriSpeech test+dev) | 11,126 utts | 146 | 210,269 | 21.3 h |

Corpora are added via resolver functions. **FluencyBank is the next
resolver to wire** — its standardised `file_id` (`Hakim_CWS_LS_03`)
doesn't map 1:1 to audio names, so it needs an inventory join
(`fluencybank/processed/inventory.csv`: corpus/session_id/task →
`audio/<corpus>/<task>/<id>.wav`).

```
python -m src.evaluation.build_manifest --out /Volumes/FATSPEECH/manifests/benchmark_v1.jsonl
```

## 2. Inference — `src/evaluation/run_asr.py` (GPU box)

Reads the manifest, transcribes each unit, writes raw hypotheses
**before normalisation**. Resume-safe (skips done unit_ids). Long
recordings (`needs_chunking`, 418 SLASS sessions >30 s) are split into
windows for CTC models; pass `--no-chunk` for Whisper's native handling.
Models resample to 16 kHz on load (Whisper internally; wav2vec2/HuBERT
via `librosa.load(sr=16000)`), so no pre-resampling pass is needed.

```
# headline pair on the 2×5090 box
python -m src.evaluation.run_asr --manifest .../benchmark_v1.jsonl \
    --model whisper-large-v3 --device cuda --no-chunk \
    --out /Volumes/FATSPEECH/results/hyp/whisper-large-v3.jsonl

python -m src.evaluation.run_asr --manifest .../benchmark_v1.jsonl \
    --model wav2vec2-large --device cuda --max-chunk-s 20 \
    --out /Volumes/FATSPEECH/results/hyp/wav2vec2-large.jsonl
```

Model names: `whisper-large-v3|medium|small`, `faster-whisper-large-v3`,
`wav2vec2-large`, `hubert-large`.

## 3. Scoring — `src/evaluation/score.py`

Normalises ref+hyp identically (Whisper `EnglishTextNormalizer`, falling
back to a basic normaliser if Whisper isn't importable), computes WER/CER
against **both** `reference_intended` and `reference_surface` (surface vs
intended can invert rankings on stuttered speech), and aggregates:
overall, by condition (the H1 gap), stuttered-with-labels, speaker-level,
plus a paired-bootstrap 95% CI on the stuttered−fluent gap. WER/CER use a
self-contained Levenshtein (no jiwer dependency). `--selftest` validates
the logic.

```
python -m src.evaluation.score --manifest .../benchmark_v1.jsonl \
    --hyp .../hyp/whisper-large-v3.jsonl \
    --out /Volumes/FATSPEECH/results/scores/whisper-large-v3.json
```

## Status / caveats

- build_manifest + score validated locally (scorer self-test + a
  synthetic fluent-vs-stuttered end-to-end recovering the expected gap).
- run_asr is **GPU-box-ready but not run locally** (needs torch/whisper);
  smoke-test it with `--limit 10` first.
- Unit granularity differs by corpus (SLASS = session, LibriSpeech =
  utterance); `duration_s` is recorded so length can be controlled for.
  Speaker-level aggregation + cluster bootstrap handle within-speaker
  correlation.
- Only 27/420 stuttered sessions carry per-word stutter labels here; the
  type-conditioned analysis (H2) still anchors on the Jason matrices.
