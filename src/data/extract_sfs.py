#!/usr/bin/env python3
"""
Extract audio (.wav) and annotation (.txt/.csv) data from UCL Speech Filing
System (.sfs) files.

The SFS binary format (Mark Huckvale, UCL) stores audio waveforms and multiple
annotation layers in a single file.  This script reverse-engineers the on-disk
layout to extract:

  1. Audio  → 16-bit PCM WAV (resampled to a target rate if desired)
  2. Annotations → per-layer CSV files with columns: time, duration, label

Usage
-----
    # Extract a single file
    python scripts/extract_sfs.py /path/to/file.sfs -o output_dir

    # Batch-extract an entire directory tree
    python scripts/extract_sfs.py /Volumes/SPEECH/from_jason -o data/raw --recursive

    # Extract only orthographic annotations + audio
    python scripts/extract_sfs.py /path/to/file.sfs -o out --layers orthographic
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path


# ── SFS binary layout constants ──────────────────────────────────────────────
_MAGIC = b"UC2\x00"
_HEADER_SIZE = 0x400          # main header (0x000–0x1FF) + first item header (0x200–0x3FF)
_HISTORY_OFFSET = 0x200       # processing-history string inside header block
_ITEM_HEADER_LEN = 0x200      # each item: 0x200 history/header, then data
_ITEM_HDR_META = 0x180        # offset within item block to metadata fields


@dataclass
class AnnotationRecord:
    time: float       # seconds
    duration: float   # seconds
    label: str


@dataclass
class AnnotationLayer:
    name: str                             # e.g. "orthographic", "stutter", "type"
    history: str                          # raw SFS history string
    records: list[AnnotationRecord] = field(default_factory=list)


@dataclass
class SFSFile:
    path: Path
    endian: str                           # "big" or "little"
    sample_rate: int
    n_samples: int
    duration: float                       # seconds
    audio_offset: int                     # byte offset where PCM starts
    audio_bytes: int                      # length of PCM data in bytes
    layers: list[AnnotationLayer] = field(default_factory=list)


# ── Low-level parsing ────────────────────────────────────────────────────────

def _detect_endianness(data: bytes) -> str:
    """Detect endianness from the format marker at offset 0x1C."""
    marker = data[0x1C:0x20]
    if marker == b"sfs\x00":
        return "big"
    elif marker == b"SFS\x00":
        return "little"
    # Fallback heuristic: try both and pick the one giving a sane sample rate
    for endian, fmt in [("little", "<"), ("big", ">")]:
        try:
            period = struct.unpack(f"{fmt}d", data[0x3A0:0x3A8])[0]
            if 1e-6 < period < 1:
                return endian
        except struct.error:
            continue
    raise ValueError("Cannot determine SFS file endianness")


def _unpack_u32(data: bytes, offset: int, fmt: str) -> int:
    return struct.unpack(f"{fmt}I", data[offset : offset + 4])[0]


def _unpack_f64(data: bytes, offset: int, fmt: str) -> float:
    return struct.unpack(f"{fmt}d", data[offset : offset + 8])[0]


def _parse_annotation_records(
    data: bytes, offset: int, n_records: int, sample_rate: int
) -> list[AnnotationRecord]:
    """Parse variable-length annotation records.

    Each record:
        byte 0       : data_length (bytes following this byte)
        bytes 1–4    : timestamp in samples (uint32 LE)
        bytes 5–8    : duration in samples  (uint32 LE)
        bytes 9–len  : label text (null-terminated)

    Note: annotation records are always little-endian regardless of the file's
    audio endianness.
    """
    records: list[AnnotationRecord] = []
    pos = offset
    for _ in range(n_records):
        if pos >= len(data):
            break
        rec_len = data[pos]
        if rec_len < 8:
            break
        rec = data[pos + 1 : pos + 1 + rec_len]
        ts = struct.unpack("<I", rec[0:4])[0]
        dur = struct.unpack("<I", rec[4:8])[0]
        label = rec[8:].decode("latin-1").rstrip("\x00")
        records.append(
            AnnotationRecord(
                time=ts / sample_rate,
                duration=dur / sample_rate,
                label=label,
            )
        )
        pos += 1 + rec_len
    return records


def parse_sfs(filepath: str | Path) -> SFSFile:
    """Parse an SFS file and return its metadata + annotation layers."""
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        data = f.read()

    if data[:4] != _MAGIC:
        raise ValueError(f"Not an SFS file (bad magic): {filepath}")

    endian = _detect_endianness(data)
    fmt = ">" if endian == "big" else "<"

    # Audio item header sits at 0x380–0x3FF
    byte_count = _unpack_u32(data, 0x39C, fmt)
    period = _unpack_f64(data, 0x3A0, fmt)
    if period <= 0 or period > 1:
        raise ValueError(f"Invalid sample period {period} in {filepath}")
    sample_rate = round(1.0 / period)
    n_samples = byte_count // 2  # 16-bit audio

    sfs = SFSFile(
        path=filepath,
        endian=endian,
        sample_rate=sample_rate,
        n_samples=n_samples,
        duration=n_samples / sample_rate,
        audio_offset=_HEADER_SIZE,
        audio_bytes=byte_count,
    )

    # Walk successive annotation items after the audio data
    pos = _HEADER_SIZE + byte_count
    while pos + _ITEM_HEADER_LEN <= len(data):
        # Item history string (256 bytes, null-terminated)
        history = data[pos : pos + 256].decode("latin-1").split("\x00")[0]
        if not history:
            break

        # Extract annotation type from history like "Eswin/AN(type=orthographic)"
        layer_name = history
        if "type=" in history:
            layer_name = history.split("type=")[1].rstrip(")")

        # Item metadata at pos + 0x180
        hdr = pos + _ITEM_HDR_META
        # Annotation records count and byte size are always LE
        n_records = struct.unpack("<I", data[hdr + 0x18 : hdr + 0x1C])[0]
        n_bytes = struct.unpack("<I", data[hdr + 0x1C : hdr + 0x20])[0]

        # Skip non-annotation items (e.g. formant tracks)
        data_start = pos + _ITEM_HEADER_LEN
        if "AN(" in history or "AN " in history:
            records = _parse_annotation_records(
                data, data_start, n_records, sample_rate
            )
            sfs.layers.append(
                AnnotationLayer(name=layer_name, history=history, records=records)
            )

        pos = data_start + n_bytes

    return sfs


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_audio(sfs: SFSFile, output_path: Path) -> Path:
    """Extract the audio waveform to a 16-bit PCM WAV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(sfs.path, "rb") as f:
        f.seek(sfs.audio_offset)
        pcm_data = f.read(sfs.audio_bytes)

    # Handle big-endian audio: swap bytes to little-endian for WAV
    if sfs.endian == "big":
        samples = struct.unpack(f">{sfs.n_samples}h", pcm_data)
        pcm_data = struct.pack(f"<{sfs.n_samples}h", *samples)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sfs.sample_rate)
        wf.writeframes(pcm_data)

    return output_path


def extract_annotations(
    sfs: SFSFile,
    output_dir: Path,
    stem: str,
    layer_filter: set[str] | None = None,
) -> list[Path]:
    """Export annotation layers as CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for layer in sfs.layers:
        # Normalize the layer name for filtering and filenames
        norm_name = layer.name.lower().strip()

        if layer_filter and norm_name not in layer_filter:
            continue

        if not layer.records:
            continue

        # Sanitize filename
        safe_name = norm_name.replace(" ", "_").replace("/", "_")
        out_path = output_dir / f"{stem}_{safe_name}.csv"

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "duration", "label"])
            for rec in layer.records:
                writer.writerow([f"{rec.time:.6f}", f"{rec.duration:.6f}", rec.label])

        paths.append(out_path)

    return paths


# ── CLI ──────────────────────────────────────────────────────────────────────

def find_sfs_files(path: Path, recursive: bool = False) -> list[Path]:
    """Find all .sfs files at a path (file or directory)."""
    if path.is_file():
        return [path] if path.suffix.lower() == ".sfs" else []

    pattern = "**/*.sfs" if recursive else "*.sfs"
    return sorted(path.glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract audio and annotations from UCL SFS files"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="SFS file or directory containing SFS files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively search for SFS files in subdirectories",
    )
    parser.add_argument(
        "--layers",
        nargs="*",
        default=None,
        help="Annotation layers to extract (e.g. orthographic stutter). "
             "Default: all layers.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio extraction (annotations only)",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip annotation extraction (audio only)",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="List available annotation layers without extracting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and report contents without writing output",
    )

    args = parser.parse_args()

    sfs_files = find_sfs_files(args.input, args.recursive)
    if not sfs_files:
        print(f"No SFS files found at {args.input}", file=sys.stderr)
        sys.exit(1)

    layer_filter = {l.lower() for l in args.layers} if args.layers else None

    n_success = 0
    n_failed = 0
    total_duration = 0.0

    for sfs_path in sfs_files:
        try:
            sfs = parse_sfs(sfs_path)
        except Exception as e:
            print(f"FAIL  {sfs_path.name}: {e}", file=sys.stderr)
            n_failed += 1
            continue

        total_duration += sfs.duration
        stem = sfs_path.stem

        if args.list_layers or args.dry_run:
            print(
                f"{sfs_path.name}: {sfs.endian}-endian, "
                f"{sfs.sample_rate}Hz, {sfs.duration:.1f}s, "
                f"{len(sfs.layers)} layers"
            )
            for layer in sfs.layers:
                n_words = sum(
                    1
                    for r in layer.records
                    if r.label not in ("x", "X", "Q", "")
                )
                print(
                    f"  - {layer.name} ({len(layer.records)} records, "
                    f"{n_words} non-boundary)"
                )
            if args.list_layers:
                continue

        if args.dry_run:
            n_success += 1
            continue

        # Determine output subdirectory from relative path
        try:
            rel = sfs_path.relative_to(args.input)
            out_subdir = args.output / rel.parent
        except ValueError:
            out_subdir = args.output

        if not args.no_audio:
            wav_path = out_subdir / "audio" / f"{stem}.wav"
            extract_audio(sfs, wav_path)

        if not args.no_annotations:
            ann_dir = out_subdir / "annotations"
            extract_annotations(sfs, ann_dir, stem, layer_filter)

        n_success += 1
        print(f"  OK  {sfs_path.name} → {out_subdir}")

    print(
        f"\nDone: {n_success} succeeded, {n_failed} failed, "
        f"{total_duration/3600:.1f}h total audio"
    )


if __name__ == "__main__":
    main()
