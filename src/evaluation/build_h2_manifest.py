#!/usr/bin/env python3
"""Build the H2 word-event manifest from the Jason usage matrices.

Unit of analysis is a WORD (not a session): one row per word that has a
resolvable onset, with its disfluency type, audio segment times, and the
linguistic covariates needed for type-conditioned error analysis and for
indexing model activations at specific stutter events (the xAI track).

Both fluent and stuttered words are emitted so that P(ASR error | type)
can be compared against the fluent baseline within the same recordings.

Audio is resolved per session, preferring the curated `slass/processed`
extraction (which shares the Jason f_/m_ naming) and falling back to the
full archive.

Timing:
  * WWR rows store `start | end` -> used directly.
  * other rows store a single onset -> end is the next word's onset
    (inter-onset interval), capped at 30 s.

Output (JSONL):
  {unit_id, session, audio_path, word_index, start_s, end_s,
   word, fluency, stutter_type_raw, stutter_type, first_type,
   syllable, word_type, phonetic_content, frequency,
   neighbourhood_density, sonority, word_position}

Usage:
    python -m src.evaluation.build_h2_manifest \\
        --out /Volumes/FATSPEECH/manifests/h2_word_events.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

# Audio root: the governed RDSS share since June 2026 (override per machine
# with ASR_DATA_ROOT). JASON is the source label matrices and lives on a
# separate lab drive, not under the data root — override with JASON_ROOT.
DATA_ROOT = Path(os.environ.get(
    "ASR_DATA_ROOT", "/Volumes/ritd-ag-project-rd02dw-lbarr63"))
JASON = Path(os.environ.get(
    "JASON_ROOT", "/Volumes/SPEECH/from_jason/Speech data Jason/output"))
PROC_AUDIO = DATA_ROOT / "slass" / "processed" / "audio"
FULL_AUDIO = DATA_ROOT / "slass" / "full_archive" / "audio"

PURE = ("Block", "Prolongation", "PWR", "WWR")


def simplify(st: str) -> str:
    st = st.strip()
    if st == "Fluent" or not st:
        return "Fluent"
    if st == "Unknown":
        return "Unknown"
    pure = set(PURE)
    for k in pure:
        if k in st and not any(o in st for o in pure - {k}):
            return k
    return "Combined"


def first_type(st: str) -> str:
    st = st.strip()
    if not st or st in ("Fluent", "Unknown"):
        return ""
    head = st.split("+")[0].strip()
    return head if head in PURE else ""


def build_audio_index() -> dict[str, Path]:
    """stem -> audio path, preferring curated processed extraction."""
    idx: dict[str, Path] = {}
    for root in (FULL_AUDIO, PROC_AUDIO):  # processed overwrites -> preferred
        if not root.exists():
            continue
        for p in root.rglob("*.wav"):
            if not p.name.startswith("."):
                idx[p.stem] = p
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    audio_idx = build_audio_index()
    rows_out = []
    n_sessions = n_no_audio = 0
    type_counts = Counter()

    for f in sorted(JASON.glob("*_Usage_Matrix.csv")):
        session = f.stem.replace("_Usage_Matrix", "")
        audio = audio_idx.get(session)
        if audio is None:
            n_no_audio += 1
            continue
        n_sessions += 1

        rows = list(csv.DictReader(open(f, encoding="latin-1"), delimiter="\t"))
        # Pre-parse onsets for inter-onset end times.
        parsed = []
        for r in rows:
            ts = (r.get("Timestamp", "") or "").strip()
            start = end = None
            if "|" in ts:
                a, _, b = ts.partition("|")
                try:
                    start, end = float(a), float(b)
                except ValueError:
                    pass
            elif ts:
                try:
                    start = float(ts)
                except ValueError:
                    pass
            parsed.append((r, start, end))

        for i, (r, start, end) in enumerate(parsed):
            word = (r.get("Word", "") or "").strip()
            if not word or start is None:
                continue
            if end is None:
                # inter-onset interval to next parseable onset
                for j in range(i + 1, len(parsed)):
                    if parsed[j][1] is not None:
                        cand = parsed[j][1] - start
                        if 0 < cand < 30:
                            end = parsed[j][1]
                        break
            st_raw = (r.get("Stutter_Type", "") or "").strip()
            simp = simplify(st_raw)
            type_counts[simp] += 1
            rows_out.append({
                "unit_id": f"{session}#{i}",
                "session": session,
                "audio_path": str(audio),
                "word_index": i,
                "start_s": round(start, 3),
                "end_s": round(end, 3) if end is not None else None,
                "word": word,
                "fluency": (r.get("Fluency", "") or "").strip(),
                "stutter_type_raw": st_raw,
                "stutter_type": simp,
                "first_type": first_type(st_raw),
                "syllable": r.get("Syllable", ""),
                "word_type": r.get("Word_Type", ""),
                "phonetic_content": r.get("Phonetic_Content", ""),
                "frequency": r.get("Frequency", ""),
                "neighbourhood_density": r.get("Neighbourhood_Density", ""),
                "sonority": r.get("Sonority", ""),
                "word_position": r.get("Word_Position", ""),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r) + "\n")

    print(f"sessions with audio: {n_sessions} (skipped {n_no_audio} no-audio)")
    print(f"word events: {len(rows_out):,}")
    print("by type:", {k: type_counts[k] for k in
                       ("Fluent", "Prolongation", "Block", "PWR", "WWR",
                        "Combined", "Unknown") if k in type_counts})
    stut = sum(v for k, v in type_counts.items()
               if k not in ("Fluent", "Unknown"))
    print(f"stuttered (typed) events: {stut:,}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
