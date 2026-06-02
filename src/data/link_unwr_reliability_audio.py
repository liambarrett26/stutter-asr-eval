#!/usr/bin/env python3
"""Link the UNWR reliability audio bundle to its TextGrid items.

The transcription-audio bundle
(/Volumes/FATSPEECH/unwr_reliability/raw/transcription_audio/data/) holds,
per child, one WAV per nonword item co-located with a reference TextGrid
in the same syllable folder. This script:

  1. Walks the bundle, pairing each per-item WAV with its co-located
     TextGrid (same stem in the same directory).
  2. Probes WAV duration from the RIFF header and reads the TextGrid xmax,
     reporting how well they agree (a sanity check that the audio and the
     transcription line up).
  3. Writes an audio index CSV keyed by (cohort, child, syllable, item).

Output:
  /Volumes/FATSPEECH/unwr_reliability/processed/audio_items.csv

Reuses the TextGrid parser and the WAV header probe from the sibling
processing modules. Standard library only.

Usage:
    python src/data/link_unwr_reliability_audio.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

# Import sibling helpers without requiring the package to import cleanly.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from process_unwr_reliability import parse_textgrid  # noqa: E402
from process_unwr import probe_wav  # noqa: E402


BUNDLE = Path("/Volumes/FATSPEECH/unwr_reliability/raw/"
              "transcription_audio/data")
OUT = Path("/Volumes/FATSPEECH/unwr_reliability/processed/audio_items.csv")

ITEM_RE = re.compile(r"^(\d+)(prac|test)(\d+)(.+)$", re.I)


def child_fields(child_dir: str) -> tuple[str, str]:
    parts = child_dir.split("_", 2)
    cid = parts[0]
    name = "_".join(parts[1:]) if len(parts) >= 2 else child_dir
    return cid, name


def main() -> None:
    if not BUNDLE.exists():
        print(f"bundle not found: {BUNDLE}", file=sys.stderr)
        sys.exit(1)

    rows = []
    n_wav = 0
    n_paired = 0
    n_dur_ok = 0
    dur_deltas = []

    for wav in sorted(BUNDLE.rglob("*.wav")):
        if wav.name.startswith("."):
            continue
        n_wav += 1
        rel = wav.relative_to(BUNDLE)
        parts = rel.parts
        # cohort/child/syllN/item.wav  (per-item) OR cohort/child/x_SB.wav
        m = ITEM_RE.match(wav.stem)
        cohort = parts[0] if len(parts) >= 1 else ""
        child_dir = parts[1] if len(parts) >= 2 else ""
        cid, cname = child_fields(child_dir)

        tg = wav.with_suffix(".TextGrid")
        if not tg.exists():
            # Try case variant
            alt = wav.with_suffix(".textgrid")
            tg = alt if alt.exists() else None

        wav_dur, wav_sr = probe_wav(wav)
        tg_xmax = None
        target = response = ortho = ""
        if tg is not None:
            n_paired += 1
            try:
                tiers = parse_textgrid(tg)
                # Largest xmax across tiers/intervals = TextGrid extent.
                xs = [iv["xmax"] for ivs in tiers.values() for iv in ivs
                      if iv.get("xmax") is not None]
                tg_xmax = max(xs) if xs else None
                for tname, ivs in tiers.items():
                    joined = " ".join(i["text"].strip() for i in ivs
                                      if i.get("text", "").strip())
                    if tname in ("ortho", "orthography", "orth"):
                        ortho = joined
                    elif tname == "target":
                        target = joined
                    elif tname in ("response", "resp"):
                        response = joined
            except Exception:
                pass

        if wav_dur is not None and tg_xmax is not None:
            delta = abs(wav_dur - tg_xmax)
            dur_deltas.append(delta)
            if delta <= 0.05:
                n_dur_ok += 1

        rows.append({
            "cohort": cohort,
            "child_id": cid,
            "child_name": cname,
            "syllable_dir": parts[2] if len(parts) >= 3 else "",
            "item_kind": m.group(2).lower() if m else "",
            "item_idx": m.group(3) if m else "",
            "item_ortho": m.group(4).lower() if m else "",
            "is_per_item": "yes" if m else "no",
            "wav_path": str(wav),
            "wav_duration_s": f"{wav_dur:.3f}" if wav_dur else "",
            "wav_sample_rate": wav_sr or "",
            "textgrid_path": str(tg) if tg else "",
            "tg_xmax_s": f"{tg_xmax:.3f}" if tg_xmax else "",
            "duration_match": ("yes" if (wav_dur and tg_xmax
                                         and abs(wav_dur - tg_xmax) <= 0.05)
                               else "no"),
            "ortho": ortho,
            "target": target,
            "response": response,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["cohort", "child_id", "child_name", "syllable_dir",
            "item_kind", "item_idx", "item_ortho", "is_per_item",
            "wav_path", "wav_duration_s", "wav_sample_rate",
            "textgrid_path", "tg_xmax_s", "duration_match",
            "ortho", "target", "response"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    per_item = sum(1 for r in rows if r["is_per_item"] == "yes")
    print(f"WAV files            : {n_wav}")
    print(f"  per-item           : {per_item}")
    print(f"  whole-session      : {n_wav - per_item}")
    print(f"Paired with TextGrid : {n_paired}")
    if dur_deltas:
        mean_d = sum(dur_deltas) / len(dur_deltas)
        print(f"Duration check (paired with parseable xmax): "
              f"{len(dur_deltas)} compared")
        print(f"  within 50 ms     : {n_dur_ok} "
              f"({n_dur_ok / len(dur_deltas) * 100:.1f}%)")
        print(f"  mean |wav - xmax|: {mean_d:.3f} s")
    print(f"Wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
