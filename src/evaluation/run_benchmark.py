#!/usr/bin/env python3
"""Orchestrate the open-source ASR benchmark across models and GPUs.

This is the Python control layer over ``run_asr.py`` (which runs ONE model over
ONE manifest). It:

  * optionally builds a small **stratified ratification subset** of the manifest
    (a spread across dataset x condition, including long/chunked units) so the
    outputs can be eyeballed before committing to the full-corpus run;
  * assigns models to GPUs and launches them concurrently (one process per GPU,
    ``CUDA_VISIBLE_DEVICES`` set per process), waving through if there are more
    models than GPUs;
  * is resume-safe (delegates to ``run_asr.py``, which skips done unit_ids);
  * writes a side-by-side reference-vs-hypothesis **ratification report** for
    qualitative sign-off.

Inference is intentionally decoupled from scoring: raw hypotheses are saved
pre-normalisation, so ``score.py`` (and any later metric) runs off these outputs
without re-running models.

Usage (on the GPU box, inside the venv):

    # 1) Ratify: two largest models on a stratified subset, then inspect
    python -m src.evaluation.run_benchmark \\
        --manifest /media/liamb/FATSPEECH/manifests/benchmark_v1.jsonl \\
        --out-dir  /media/liamb/FATSPEECH/results/ratify \\
        --audio-root-map "/Volumes/ritd-ag-project-rd02dw-lbarr63=/media/liamb/FATSPEECH" \\
        --models whisper-large-v3 wav2vec2-large --gpus 0,1 \\
        --ratify --n-per-cell 4

    # 2) Full corpus (same command without --ratify, new out-dir)
    python -m src.evaluation.run_benchmark --manifest ... \\
        --out-dir /media/liamb/FATSPEECH/results \\
        --audio-root-map "..." --models whisper-large-v3 wav2vec2-large --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Per-model flags for run_asr. Whisper does its own 30 s windowing (so never
# chunk it); CTC models (wav2vec2/HuBERT) cannot take minutes-long audio and
# are chunked at a fixed window.
MODEL_FLAGS: dict[str, list[str]] = {
    "whisper": ["--no-chunk"],
    "faster-whisper": ["--no-chunk"],
    "wav2vec2": ["--max-chunk-s", "20"],
    "hubert": ["--max-chunk-s", "20"],
}


def flags_for(model: str) -> list[str]:
    for prefix, fl in MODEL_FLAGS.items():
        if model.startswith(prefix):
            return fl
    return []


def load_manifest(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def build_subset(units: list[dict], n_per_cell: int) -> list[dict]:
    """Deterministic stratified subset: a spread across (dataset, condition).

    Within each cell we force-include a long (needs_chunking) unit if present
    (to exercise the chunking path) and then take evenly spaced units for
    variety. Deterministic — no RNG — so the subset is reproducible.
    """
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for u in units:
        cells[(u.get("dataset"), u.get("condition"))].append(u)

    chosen: list[dict] = []
    for key in sorted(cells, key=lambda k: tuple(str(x) for x in k)):
        group = sorted(cells[key], key=lambda u: u["unit_id"])
        picks: list[dict] = []
        seen: set[str] = set()

        def add(u: dict) -> None:
            if u["unit_id"] not in seen and len(picks) < n_per_cell:
                seen.add(u["unit_id"])
                picks.append(u)

        longs = [u for u in group if u.get("needs_chunking")]
        if longs:
            add(longs[len(longs) // 2])
        step = max(1, len(group) // n_per_cell)
        for i in range(0, len(group), step):
            add(group[i])
        chosen.extend(picks)
    return chosen


def assign_gpus(models: list[str], gpus: list[str]) -> list[tuple[str, str]]:
    """Round-robin models onto GPUs, preserving order."""
    return [(m, gpus[i % len(gpus)]) for i, m in enumerate(models)]


def run_wave(jobs: list[tuple[str, str, list[str], Path]]) -> dict[str, int]:
    """Launch a set of model processes concurrently, wait for all."""
    procs = []
    for model, gpu, cmd, logpath in jobs:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        logpath.parent.mkdir(parents=True, exist_ok=True)
        lf = logpath.open("w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        procs.append((model, gpu, p, lf))
        print(f"  launched {model} on GPU {gpu} (pid {p.pid}) -> {logpath}")
    rc = {}
    for model, gpu, p, lf in procs:
        p.wait()
        lf.close()
        rc[model] = p.returncode
        print(f"  {model} (GPU {gpu}) exited rc={p.returncode}")
    return rc


def summarise_out(path: Path) -> tuple[int, int, float]:
    """(records, errors, mean processing_time_s) for a run_asr output file."""
    n = errs = 0
    tsum = 0.0
    if path.exists():
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            if r.get("error"):
                errs += 1
            tsum += float(r.get("processing_time_s") or 0.0)
    return n, errs, (tsum / n if n else 0.0)


def ratify_report(units: list[dict], hyp_dir: Path, models: list[str],
                  report_path: Path, n_print: int = 12) -> None:
    """Write/print side-by-side reference vs each model's hypothesis."""
    hyps: dict[str, dict[str, dict]] = {}
    for m in models:
        rows = {}
        f = hyp_dir / f"{m}.jsonl"
        if f.exists():
            for line in f.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    rows[r["unit_id"]] = r
        hyps[m] = rows

    lines: list[str] = []
    for u in units:
        uid = u["unit_id"]
        head = (f"[{u.get('dataset')}/{u.get('condition')}"
                f"{' /chunked' if u.get('needs_chunking') else ''}] {uid}")
        lines.append(head)
        ref = (u.get("reference_intended") or "").strip()
        lines.append(f"    REF(intended): {ref[:160]}")
        for m in models:
            r = hyps[m].get(uid, {})
            if r.get("error"):
                lines.append(f"    {m}: <ERROR> {str(r['error'])[:120]}")
            else:
                lines.append(f"    {m}: {(r.get('hypothesis') or '').strip()[:160]}")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n===== RATIFICATION REPORT (first "
          f"{n_print} of {len(units)} units; full file: {report_path}) =====")
    shown = 0
    for block in "\n".join(lines).split("\n\n"):
        if block.strip():
            print(block)
            print()
            shown += 1
        if shown >= n_print:
            break
    print("===== per-model summary =====")
    for m in models:
        n, errs, mt = summarise_out(hyp_dir / f"{m}.jsonl")
        print(f"  {m}: {n} units, {errs} errors, mean {mt:.2f}s/unit")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--audio-root-map", default="")
    ap.add_argument("--models", nargs="+",
                    default=["whisper-large-v3", "wav2vec2-large"])
    ap.add_argument("--gpus", default="0,1",
                    help="comma-separated GPU ids, e.g. 0,1")
    ap.add_argument("--ratify", action="store_true",
                    help="build + run a stratified subset for sign-off")
    ap.add_argument("--n-per-cell", type=int, default=4,
                    help="units per (dataset,condition) cell in the subset")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", action="store_true",
                    help="shard each model across ALL GPUs (split the manifest "
                         "round-robin, one shard per GPU, run concurrently, then "
                         "merge), processing models sequentially so the slow "
                         "model uses the whole machine. Without this, models run "
                         "one-per-GPU in parallel.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hyp_dir = args.out_dir / "hyp"
    log_dir = args.out_dir / "logs"

    units = load_manifest(args.manifest)
    manifest = args.manifest
    if args.ratify:
        subset = build_subset(units, args.n_per_cell)
        manifest = args.out_dir / "subset_manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as fh:
            for u in subset:
                fh.write(json.dumps(u) + "\n")
        cells = defaultdict(int)
        for u in subset:
            cells[(u.get("dataset"), u.get("condition"))] += 1
        print(f"ratification subset: {len(subset)} units -> {manifest}")
        for k in sorted(cells, key=lambda k: tuple(str(x) for x in k)):
            print(f"    {k[0]}/{k[1]}: {cells[k]}")
        units = subset

    def build_cmd(model: str, out: Path, shard_index: int = 0,
                  shard_count: int = 1) -> list[str]:
        cmd = [sys.executable, "-m", "src.evaluation.run_asr",
               "--manifest", str(manifest), "--model", model,
               "--device", args.device, "--out", str(out)]
        if shard_count > 1:
            cmd += ["--shard-index", str(shard_index),
                    "--shard-count", str(shard_count)]
        if args.audio_root_map:
            cmd += ["--audio-root-map", args.audio_root_map]
        cmd += flags_for(model)
        return cmd

    t0 = time.time()
    if args.shard:
        # Each model is split across ALL GPUs (one shard per GPU), run
        # concurrently, then shard outputs are concatenated into <model>.jsonl.
        # Models run sequentially, so the slow model (Whisper) uses the whole
        # machine. Each shard file is independently resume-safe.
        print(f"\nsharding each model across GPUs {gpus} (sequential per model)")
        for model in args.models:
            jobs, shard_outs = [], []
            for gi, gpu in enumerate(gpus):
                out = hyp_dir / f"{model}.shard{gi}.jsonl"
                shard_outs.append(out)
                jobs.append((f"{model}.s{gi}", gpu,
                             build_cmd(model, out, gi, len(gpus)),
                             log_dir / f"{model}.shard{gi}.log"))
            if args.dry_run:
                for name, gpu, cmd, _ in jobs:
                    print(f"[dry-run] GPU {gpu}: {' '.join(cmd)}")
                continue
            print(f"\n=== {model}: {len(gpus)} shards across GPUs {gpus} ===")
            run_wave(jobs)
            merged = hyp_dir / f"{model}.jsonl"
            with merged.open("w", encoding="utf-8") as out:
                for so in shard_outs:
                    if so.exists():
                        out.write(so.read_text(encoding="utf-8"))
            print(f"  merged {len(shard_outs)} shards -> {merged}")
    else:
        plan = assign_gpus(args.models, gpus)
        print("\nmodel -> GPU plan: "
              + ", ".join(f"{m}:GPU{g}" for m, g in plan))
        jobs = [(model, gpu, build_cmd(model, hyp_dir / f"{model}.jsonl"),
                 log_dir / f"{model}.log") for model, gpu in plan]
        if args.dry_run:
            for model, gpu, cmd, _ in jobs:
                print(f"[dry-run] GPU {gpu}: {' '.join(cmd)}")
            return
        by_gpu: dict[str, list] = defaultdict(list)
        for j in jobs:
            by_gpu[j[1]].append(j)
        n_waves = max(len(v) for v in by_gpu.values()) if by_gpu else 0
        for w in range(n_waves):
            wave = [v[w] for v in by_gpu.values() if w < len(v)]
            print(f"\n=== wave {w + 1}/{n_waves}: {[j[0] for j in wave]} ===")
            run_wave(wave)

    if args.dry_run:
        return
    print(f"\nall models finished in {time.time() - t0:.0f}s")
    ratify_report(units, hyp_dir, args.models,
                  args.out_dir / "ratify_report.txt")


if __name__ == "__main__":
    main()
