# Running the ASR models (Linux GPU box)

Operational runbook for the inference + scoring stage. Inference runs on
the lab Linux machine (2× NVIDIA 5090); the Mac is data-prep only. The
harness is `src/evaluation/{build_manifest,run_asr,score}.py` — see
`docs/evaluation_harness.md` for the design.

## 0. Prerequisites

```bash
git pull
# core (torch, torchaudio, transformers, librosa, jiwer) + local model extras
pip install -e ".[whisper,faster-whisper]"
#   wav2vec2 + HuBERT need no extra (they run via transformers)
nvidia-smi          # confirm both 5090s visible
```

The project data must be mounted. As of June 2026 the governed home is the
**UCL RDSS** share, and the manifests store RDSS paths
(`/Volumes/ritd-ag-project-rd02dw-lbarr63/...` — the Mac mount point). On the
GPU box the same tree mounts elsewhere; note that path. Say it mounts at
`/mnt/data` (whether that's the RDSS share over CIFS or a local working copy).

## 1. The path-remap gotcha (important)

The manifests store absolute Mac paths for the RDSS root
(`/Volumes/ritd-ag-project-rd02dw-lbarr63/...`). Two ways to point the harness
at the local mount — pick one:

```bash
# (a) remap at run time — no manifest edit
MAP="/Volumes/ritd-ag-project-rd02dw-lbarr63=/mnt/data"   # adjust RHS to the real mount
MANI=/mnt/data/manifests/benchmark_v1.jsonl
OUT=/mnt/data/results
```

```bash
# (b) regenerate the manifest natively — builders read ASR_DATA_ROOT
ASR_DATA_ROOT=/mnt/data python -m src.evaluation.build_manifest \
  --out /mnt/data/manifests/benchmark_v1.jsonl
# (h2: also export JASON_ROOT to wherever the Jason matrices are mounted)
```

With (b) the manifest paths are native to the local mount, so no
`--audio-root-map` is needed at run time. If the GPU box still uses an old
FATSPEECH-rooted copy, the original Mac-path manifests are preserved alongside
as `*.fatspeech.jsonl.bak` (remap from `/Volumes/FATSPEECH` instead).

## 2. Smoke test (always do this first)

```bash
python -m src.evaluation.run_asr --manifest $MANI \
  --model whisper-large-v3 --device cuda --no-chunk --limit 10 \
  --audio-root-map "$MAP" --out $OUT/hyp/_smoke.jsonl
```
Check: model loads, 10 hypotheses written, no "missing"/path errors in
the log. Then delete `_smoke.jsonl`.

## 3. Headline H1 pair — full runs

Two 5090s → run the two models concurrently, one per GPU:

```bash
# GPU 0 — Whisper large-v3 (handles long audio natively, so --no-chunk)
CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.run_asr --manifest $MANI \
  --model whisper-large-v3 --device cuda --no-chunk --audio-root-map "$MAP" \
  --out $OUT/hyp/whisper-large-v3.jsonl &

# GPU 1 — wav2vec2-large (CTC; chunk the 418 long sessions at 20 s)
CUDA_VISIBLE_DEVICES=1 python -m src.evaluation.run_asr --manifest $MANI \
  --model wav2vec2-large --device cuda --max-chunk-s 20 --audio-root-map "$MAP" \
  --out $OUT/hyp/wav2vec2-large.jsonl &
wait
```
Both are resume-safe: re-running skips unit_ids already in the output.

## 4. Score → first H1 result

```bash
for m in whisper-large-v3 wav2vec2-large; do
  python -m src.evaluation.score --manifest $MANI \
    --hyp $OUT/hyp/$m.jsonl --out $OUT/scores/$m.json
done
```
Prints stuttered vs fluent WER (micro + macro) against BOTH references
plus the bootstrap 95% CI on the gap. Literature expectation: Whisper
~5–21% vs wav2vec2 ~30–55% on the stuttered side.

## 5. After the headline pair

- Model sweep: `whisper-medium`, `whisper-small`, `faster-whisper-large-v3`,
  `hubert-large` (same commands, new `--out`).
- Commercial APIs: wrappers exist under `src/models/api/` (need keys);
  send only short segments, enable no-retention where offered.
- H2 (type-conditioned + xAI): the word-event manifest is ready at
  `$OUT/../manifests/h2_word_events.jsonl` — see the agent brief.

## Expectations / caveats

- **Runtime:** ~95 h of audio total. Whisper-large-v3 runs
  faster-than-realtime on a 5090; budget a few hours per model, less for
  wav2vec2. Parallelise across both GPUs.
- **Surface ≈ intended** for FluencyBank and LibriSpeech in the current
  manifest; SLASS is where surface vs intended genuinely differ.
  (FB's real surface refs are reconstructed in
  `fluencybank/processed/disfluency_text/` but not yet wired into the
  manifest.)
- **wav2vec2 chunking** at 20 s concatenates per-window hypotheses —
  fine for corpus-level WER, slightly pessimistic at chunk boundaries.
  Note it in any writeup.
- **Two SLASS sets:** the manifest uses `slass_full` (comprehensive);
  the curated `standardised/slass` is Pete's stable analysis base. Don't
  pool them without dedup-by-speaker (they overlap at recording level).

## Outputs to keep

```
$OUT/hyp/<model>.jsonl       raw hypotheses (pre-normalisation; reusable)
$OUT/scores/<model>.json     WER/CER/gap per reference convention
```
Both live on the data drive, not the repo.
