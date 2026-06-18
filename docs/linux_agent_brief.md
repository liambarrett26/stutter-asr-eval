# Brief for a Claude Code instance on the Linux GPU box

Orientation for an agent picking up this repo on the lab Linux machine
(2× NVIDIA 5090) to run models. The persistent memory from the Mac
sessions does NOT travel with you, so the project state you need is
captured here. Read `CLAUDE.md` first (project overview), then this.

## What the project is

Evaluate and fine-tune SOTA ASR on stuttered speech, then explain the
gaps mechanistically. Four nested hypotheses:

- **H1** ASR error gap: stuttered ≫ fluent WER, and it varies by model
  (Whisper ≪ wav2vec2 expected). The headline result.
- **H2** the gap is conditioned on disfluency type (block/prolongation/
  PWR/WWR), examined at the word level — and the *why* via xAI/probing
  of model activations at specific stutter events.
- **H3** severity stratification (mild/moderate/severe).
- **H4** phoneme-level ASR vs the human inter-rater ceiling (UNWR
  reliability set).

H2/H3/H4 are nested under H1. The bigger arc is mechanistic
interpretability of *why* the nets fail — so word-level, audio-aligned
stutter labels matter, not just corpus WER.

## Where everything lives

Code is in the repo (`src/`, `results/eda/`); **all data lives in the
governed UCL RDSS store** (migrated June 2026). Manifests store the Mac mount
path `/Volumes/ritd-ag-project-rd02dw-lbarr63`; on this box the same tree
mounts elsewhere — find the mount and either set `ASR_DATA_ROOT=<mount>` (the
builders derive every path from it) or pass `run_asr.py --audio-root-map`. The
repo `data/` holds docs only. Each store has a `README.md`; the store root has
a `CHANGELOG.md` (RDSS has no version control — record data changes there).

Key data locations (paths shown with the Mac RDSS prefix):
```
/Volumes/ritd-ag-project-rd02dw-lbarr63/
├── manifests/
│   ├── benchmark_v1.jsonl        H1 manifest, one unit per audio file
│   └── h2_word_events.jsonl      H2 manifest, one row per word event
├── standardised/<corpus>/        unified-schema transcripts per corpus
│   └── severity_index.csv        per-session stuttering-rate severity
├── slass/{full_archive,processed,supplementary}/   SLASS audio+annotations
├── slass_extended/               HELD-SEPARATE extra audio (not base)
├── fluencybank/processed/        FB audio + disfluency_index/text
├── uclass/, librispeech/, unwr/, unwr_reliability/
├── results/{hyp,scores}/         put model outputs here
└── source_archives/, store_metadata/   provenance
```

## Data state (what's ready)

- **H1 manifest** (`benchmark_v1.jsonl`, 11,777 units):
  stuttered 602 units / 230 speakers / 61.4 h (SLASS-full 420 +
  FluencyBank 195 + UCLASS 36); fluent 11,174 / 155 / 33.1 h
  (LibriSpeech + UMD-CMU controls); 1 cluttered. Both intended +
  surface references; `needs_chunking` flag for >30 s; speaker_id for
  speaker-exclusive / cluster analysis.
- **H2 word-event manifest** (`h2_word_events.jsonl`): 11,663 words from
  64 Jason-matrix sessions (all with audio), 1,838 typed stuttered
  (Prol 481 / Block 308 / PWR 247 / WWR 158 / Combined 644). Each row:
  audio_path + start_s/end_s + word + ordered+simplified stutter_type +
  syllable + word_type + linguistic covariates. This is the engine for
  H2 and the activation-probing.
- **Severity** (`severity_index.csv`, 823 rows): Jason gold has 46
  severe sessions; FluencyBank adds 7. Use the Jason SR (the
  slass_full auto-labels under-detect → skew mild).
- **H4**: `unwr_reliability/` has audio + gold phoneme transcriptions
  (verified aligned) and the human inter-rater ceiling (≈5–25 % PER).

## The harness (how to run)

`src/evaluation/build_manifest.py` (build), `run_asr.py` (inference,
GPU), `score.py` (WER/CER vs both refs, bootstrap gap). Full runbook:
**`docs/running_models_linux.md`**. Self-test the scorer:
`python -m src.evaluation.score --selftest`.

Model wrappers (`src/models/`): `whisper-large-v3|medium|small`,
`faster-whisper-large-v3`, `wav2vec2-large`, `hubert-large`, plus
commercial API stubs (`src/models/api/`, need keys). All resample to
16 kHz on load.

## Gotchas / conventions

- **Path handling on Linux:** the manifest stores the Mac RDSS path
  (`/Volumes/ritd-ag-project-rd02dw-lbarr63/...`). Either set
  `ASR_DATA_ROOT=<local mount>` and regenerate, or pass `run_asr.py
  --audio-root-map /Volumes/ritd-ag-project-rd02dw-lbarr63=<local mount>`, or
  symlink the mount to that path. (`*.fatspeech.jsonl.bak` keeps the old
  FATSPEECH-path manifests if a FATSPEECH-rooted copy is still in use.)
- **Data never goes in the repo** (`data/**` is gitignored except `.md`
  docs). Write model outputs to the drive's `results/`.
- **Two SLASS sets**: `slass_full` (comprehensive, in the manifest) vs
  curated `slass` (Pete's stable analysis base). They overlap at the
  recording level under different naming — dedup by speaker before
  pooling.
- **slass_extended/** audio is deliberately held separate from the base;
  don't pull it into analyses without the planned roster-join + dedup.
- **No auto-memory here** — record durable state in the repo (docs/,
  todo.md), not in a memory dir.
- Conda env `asr-eval-env` (see `environment.yml`); Python 3.10+.

## What's done vs next

Done: all data prep, EDA (Pete's round-3 review addressed), H1/H2
manifests, severity, harness (validated on synthetic + scorer
self-test), FluencyBank + UCLASS wired in, FB CHAT disfluency extracted.

Next (your job): run the **H1 headline pair** (whisper-large-v3 +
wav2vec2-large) per the runbook → score → first gap result; then the
model sweep; then H2 (word-event manifest + activation probing) and H4
(phoneme PER vs human ceiling). Open follow-ups: wire FB reconstructed
surface into the manifest; FluencyBank UMD-CMU has ~75 more sessions
recoverable with a per-session name map.

See `todo.md` for the full roadmap and `docs/` for per-corpus details.
