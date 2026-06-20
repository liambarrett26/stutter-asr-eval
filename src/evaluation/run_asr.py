#!/usr/bin/env python3
"""Run an ASR model over an evaluation manifest, saving raw hypotheses.

Reads the JSONL manifest from build_manifest.py, transcribes each unit's
audio with the chosen model, and writes one raw-hypothesis JSON per unit
to an output JSONL. Hypotheses are saved BEFORE any normalisation so that
re-scoring with a different normaliser (score.py) never requires re-running
inference.

Designed to run on the GPU box (the model wrappers pull in torch/whisper).
Resume-safe: unit_ids already present in the output are skipped.

Long recordings (manifest `needs_chunking`) are split into fixed windows,
transcribed per-window, and concatenated — necessary for CTC models
(wav2vec2/HuBERT) that cannot take minutes-long audio in one pass. Whisper
chunks internally, so pass --no-chunk to use its native handling.

Models (via src/models wrappers):
    whisper-large-v3, whisper-medium, whisper-small, faster-whisper-large-v3,
    wav2vec2-large, hubert-large

Usage (on the GPU box):
    python -m src.evaluation.run_asr \\
        --manifest /Volumes/FATSPEECH/manifests/benchmark_v1.jsonl \\
        --model whisper-large-v3 --device cuda \\
        --out /Volumes/FATSPEECH/results/hyp/whisper-large-v3.jsonl

    python -m src.evaluation.run_asr --manifest ... \\
        --model wav2vec2-large --device cuda --max-chunk-s 20 \\
        --out /Volumes/FATSPEECH/results/hyp/wav2vec2-large.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def build_model(name: str, device: str | None):
    """Instantiate an ASRModel wrapper by short name."""
    # Imports are local so the manifest/scoring tools don't need torch.
    if name.startswith("faster-whisper"):
        from src.models.whisper_model import FasterWhisperModel
        size = name.replace("faster-whisper-", "") or "large-v3"
        return FasterWhisperModel(size=size, device=device)
    if name.startswith("whisper"):
        from src.models.whisper_model import WhisperModel
        size = name.replace("whisper-", "") or "large-v3"
        return WhisperModel(size=size, device=device)
    if name.startswith("wav2vec2"):
        from src.models.wav2vec2_model import Wav2Vec2Model
        return Wav2Vec2Model(device=device)
    if name.startswith("hubert"):
        from src.models.hubert_model import HuBERTModel
        return HuBERTModel(device=device)
    raise ValueError(f"unknown model: {name}")


def transcribe_chunked(model, audio_path: Path, max_chunk_s: float,
                       tmp_dir: Path) -> str:
    """Split long audio into windows, transcribe each, join hypotheses."""
    import soundfile as sf
    data, sr = sf.read(str(audio_path))
    if data.ndim > 1:          # downmix to mono
        data = data.mean(axis=1)
    win = int(max_chunk_s * sr)
    parts = []
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i, start in enumerate(range(0, len(data), win)):
        seg = data[start:start + win]
        if len(seg) < int(0.1 * sr):
            continue
        seg_path = tmp_dir / f"{audio_path.stem}_chunk{i}.wav"
        sf.write(str(seg_path), seg, sr)
        try:
            parts.append(model.transcribe(seg_path).text.strip())
        finally:
            seg_path.unlink(missing_ok=True)
    return " ".join(p for p in parts if p)


def already_done(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            try:
                done.add(json.loads(line)["unit_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None, help="cuda | cuda:0 | cpu")
    ap.add_argument("--language", default="en")
    ap.add_argument("--max-chunk-s", type=float, default=20.0,
                    help="window length for chunking long audio")
    ap.add_argument("--no-chunk", action="store_true",
                    help="never chunk (let the model handle long audio, "
                         "e.g. Whisper's native 30s windowing)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0,
                    help="this shard's id in [0, shard-count) — process only "
                         "units where (position %% shard-count) == shard-index")
    ap.add_argument("--shard-count", type=int, default=1,
                    help="number of shards the manifest is split across (e.g. "
                         "one per GPU). Shards are disjoint and round-robin by "
                         "position, so each carries a balanced mix of long/short.")
    ap.add_argument("--audio-root-map", default="",
                    help="remap absolute audio paths across machines, "
                         "OLD=NEW (e.g. '/Volumes/FATSPEECH=/mnt/fatspeech'). "
                         "The manifest stores Mac mount paths; use this when "
                         "the drive mounts elsewhere on the GPU box.")
    args = ap.parse_args()

    old_root = new_root = None
    if args.audio_root_map:
        if "=" not in args.audio_root_map:
            ap.error("--audio-root-map must be OLD=NEW")
        old_root, new_root = args.audio_root_map.split("=", 1)

    units = [json.loads(l) for l in args.manifest.open(encoding="utf-8")]
    if old_root:
        for u in units:
            if u.get("audio_path", "").startswith(old_root):
                u["audio_path"] = new_root + u["audio_path"][len(old_root):]
    if args.shard_count > 1:
        if not (0 <= args.shard_index < args.shard_count):
            ap.error("--shard-index must be in [0, --shard-count)")
        units = [u for i, u in enumerate(units)
                 if i % args.shard_count == args.shard_index]
        print(f"shard {args.shard_index}/{args.shard_count}: {len(units)} units")
    if args.limit:
        units = units[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(args.out)
    if done:
        print(f"resuming: {len(done)} units already done")

    print(f"loading model {args.model} on {args.device or 'auto'} ...")
    model = build_model(args.model, args.device)
    model.load()

    tmp_dir = args.out.parent / "_chunks"
    n_done = n_err = 0
    with args.out.open("a", encoding="utf-8") as fh:
        for i, u in enumerate(units, 1):
            if u["unit_id"] in done:
                continue
            audio = Path(u["audio_path"])
            t0 = time.time()
            try:
                if u.get("needs_chunking") and not args.no_chunk:
                    hyp = transcribe_chunked(model, audio,
                                             args.max_chunk_s, tmp_dir)
                else:
                    hyp = model.transcribe(audio,
                                           language=args.language).text
                rec = {
                    "unit_id": u["unit_id"],
                    "dataset": u["dataset"],
                    "condition": u["condition"],
                    "model": args.model,
                    "hypothesis": hyp,
                    "processing_time_s": round(time.time() - t0, 3),
                }
                n_done += 1
            except Exception as e:
                rec = {"unit_id": u["unit_id"], "model": args.model,
                       "hypothesis": "", "error": str(e)[:200]}
                n_err += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(units)}  done={n_done} err={n_err}")

    print(f"finished: {n_done} transcribed, {n_err} errors -> {args.out}")


if __name__ == "__main__":
    main()
