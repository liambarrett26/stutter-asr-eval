#!/usr/bin/env python3
"""
Process all UCLASS data into standardised formats for the ASR evaluation pipeline.

Outputs to /Volumes/FATSPEECH/uclass/processed/:
  - transcripts/aligned/   CSV files: time,duration,label (from TextGrids + SFS annotations)
  - transcripts/flat/      CSV files: text (full utterance, from flat ortho/phon files)
  - speakers.csv            Unified speaker metadata
  - inventory.csv           Complete session inventory with data availability flags

Usage:
    python -m src.data.process_uclass
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .extract_sfs import parse_sfs, extract_annotations

UCLASS_ROOT = Path("/Volumes/FATSPEECH/uclass")
RAW = UCLASS_ROOT / "raw"
PROCESSED = UCLASS_ROOT / "processed"


# ── TextGrid parsing ─────────────────────────────────────────────────────────

def parse_textgrid(filepath: Path) -> list[dict]:
    """Parse a Praat TextTier (point-based) TextGrid file.

    Returns list of dicts with keys: time, duration, label.
    Duration is computed as time gap to next point (or 0 for last).
    """
    text = filepath.read_text(encoding="latin-1")
    lines = text.strip().split("\n")

    # Find the data lines: timestamp "label"
    records = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match: number "text" (on same line) or number\n"text" (on two lines)
        m = re.match(r'^([\d.]+)\s+"(.*)"$', line)
        if m:
            time = float(m.group(1))
            label = m.group(2)
            records.append({"time": time, "label": label})
        i += 1

    # Compute durations from gaps between consecutive points
    for j in range(len(records) - 1):
        records[j]["duration"] = records[j + 1]["time"] - records[j]["time"]
    if records:
        records[-1]["duration"] = 0.0

    return records


def session_id_from_textgrid(filepath: Path) -> tuple[str, str]:
    """Extract session ID and tier type from TextGrid filename.

    e.g. M_1099_25y0m_1.word.grid -> ('M_1099_25y0m_1', 'word')
         M_0030_16y4m_1.syll.grid -> ('M_0030_16y4m_1', 'syll')
         M_1064_47y0m_1.pw.grid   -> ('M_1064_47y0m_1', 'pw')
         M_0017_19y2m_1.word.orth.grid -> ('M_0017_19y2m_1', 'word_orth')
    """
    name = filepath.name
    # Remove .grid extension
    name = re.sub(r"\.grid$", "", name)
    # Split on first tier separator
    for sep in [".word.orth", ".syll.orth", ".syll.phon", ".word", ".syll", ".pw"]:
        if sep in name:
            session = name.split(sep)[0]
            tier = sep.lstrip(".").replace(".", "_")
            return session, tier
    return name, "unknown"


# ── Flat transcript parsing ──────────────────────────────────────────────────

def parse_flat_transcript(filepath: Path) -> str:
    """Read a flat orthographic or phonetic transcript file."""
    return filepath.read_text(encoding="latin-1").strip()


# ── Speaker metadata parsing ─────────────────────────────────────────────────

def parse_info_page(filepath: Path) -> list[dict]:
    """Parse an HTML info page to extract speaker metadata.

    Returns list of dicts with available fields.
    """
    if not filepath.exists():
        return []

    text = filepath.read_text(encoding="latin-1", errors="replace")

    # Extract table rows - look for speaker ID patterns
    speakers = []
    # Match lines with speaker IDs like F_0050_10y9m_1 or M_0017_08y9m_1
    for line in text.split("\n"):
        # Find speaker IDs in the line
        ids = re.findall(r"[FM]_\d{4}_\d+y\d+m_\d+", line)
        if ids:
            for sid in ids:
                parts = sid.split("_")
                gender = parts[0]
                speaker_num = parts[1]
                age_str = parts[2]
                rec_num = parts[3] if len(parts) > 3 else "1"

                # Parse age
                age_match = re.match(r"(\d+)y(\d+)m", age_str)
                if age_match:
                    age_months = int(age_match.group(1)) * 12 + int(age_match.group(2))
                else:
                    age_months = None

                speakers.append({
                    "session_id": sid,
                    "speaker_id": f"{gender}_{speaker_num}",
                    "gender": gender,
                    "age_str": age_str,
                    "age_months": age_months,
                })
    return speakers


def parse_session_id(session_id: str) -> dict:
    """Extract structured info from a session ID like M_0017_19y2m_1."""
    parts = session_id.replace("-", "_").split("_")
    gender = parts[0] if len(parts) >= 1 else "?"
    speaker_num = parts[1] if len(parts) >= 2 else "?"
    age_str = parts[2] if len(parts) >= 3 else "?"
    rec_num = parts[3] if len(parts) >= 4 else "1"

    age_months = None
    age_match = re.match(r"(\d+)y(\d+)m", age_str)
    if age_match:
        age_months = int(age_match.group(1)) * 12 + int(age_match.group(2))

    return {
        "session_id": session_id,
        "speaker_id": f"{gender}_{speaker_num}",
        "gender": gender,
        "age_str": age_str,
        "age_months": age_months,
        "recording_num": rec_num,
    }


# ── Main processing ──────────────────────────────────────────────────────────

def process_textgrids():
    """Parse all TextGrid files into standardised CSVs."""
    out_dir = PROCESSED / "transcripts" / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    tg_dirs = [
        RAW / "release1" / "transcripts" / "aligned" / "textgrid" / "files",
        RAW / "release2" / "monologue" / "transcripts" / "aligned" / "textgrid",
        RAW / "release2" / "reading" / "transcripts" / "aligned" / "textgrid",
    ]

    count = 0
    for tg_dir in tg_dirs:
        if not tg_dir.exists():
            continue
        for fp in sorted(tg_dir.glob("*.grid")):
            session, tier = session_id_from_textgrid(fp)
            records = parse_textgrid(fp)
            if not records:
                print(f"  SKIP {fp.name}: no records")
                continue

            out_path = out_dir / f"{session}_{tier}.csv"
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["time", "duration", "label"])
                for rec in records:
                    w.writerow([f"{rec['time']:.6f}", f"{rec['duration']:.6f}", rec["label"]])
            count += 1

    print(f"TextGrids: {count} CSV files written to {out_dir}")
    return count


def process_sfs_annotations():
    """Extract annotations from SFS files into standardised CSVs."""
    out_dir = PROCESSED / "transcripts" / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    sfs_dirs = [
        RAW / "release1" / "sfs_annotated" / "files",
        RAW / "release2" / "monologue" / "sfs_aligned",
    ]

    count = 0
    for sfs_dir in sfs_dirs:
        if not sfs_dir.exists():
            continue
        for fp in sorted(sfs_dir.glob("*.sfs")):
            try:
                sfs = parse_sfs(fp)
            except Exception as e:
                print(f"  ERROR {fp.name}: {e}")
                continue

            for layer in sfs.layers:
                if not layer.records:
                    continue
                # Name the output after session + layer
                safe_name = layer.name.lower().replace(" ", "_").replace("/", "_")
                # Truncate long anonymised names
                if len(safe_name) > 30:
                    safe_name = "sfs_annotation"
                out_path = out_dir / f"{fp.stem}_{safe_name}.csv"
                with open(out_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["time", "duration", "label"])
                    for rec in layer.records:
                        w.writerow([f"{rec.time:.6f}", f"{rec.duration:.6f}", rec.label])
                count += 1

    print(f"SFS annotations: {count} CSV files written to {out_dir}")
    return count


def process_flat_transcripts():
    """Convert flat ortho/phon files into standardised CSVs."""
    out_dir = PROCESSED / "transcripts" / "flat"
    out_dir.mkdir(parents=True, exist_ok=True)

    flat_dirs = [
        (RAW / "release1" / "transcripts" / "orthographic" / "files", "ortho"),
        (RAW / "release1" / "transcripts" / "phonetic" / "files", "phon"),
        (RAW / "release2" / "monologue" / "transcripts" / "orthographic", "ortho"),
        (RAW / "release2" / "monologue" / "transcripts" / "phonetic", "phon"),
        (RAW / "release2" / "reading" / "transcripts" / "orthographic", "ortho"),
    ]

    count = 0
    for flat_dir, ttype in flat_dirs:
        if not flat_dir.exists():
            continue
        for fp in sorted(flat_dir.iterdir()):
            if fp.name.startswith(".") or fp.suffix == ".zip":
                continue
            text = parse_flat_transcript(fp)
            if not text:
                continue

            session = fp.stem
            out_path = out_dir / f"{session}_{ttype}.csv"
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["type", "text"])
                w.writerow([ttype, text])
            count += 1

    print(f"Flat transcripts: {count} CSV files written to {out_dir}")
    return count


def build_inventory():
    """Build a complete inventory of all sessions and their data availability."""
    inventory = {}  # session_id -> dict of flags

    # R1 audio
    r1_wav = RAW / "release1" / "audio" / "wav"
    if r1_wav.exists():
        for fp in r1_wav.glob("*.wav"):
            sid = fp.stem
            if sid not in inventory:
                inventory[sid] = {"release": "R1", "task": "monologue"}
            inventory[sid]["has_audio"] = True

    # R2 audio
    for task, subdir in [("monologue", "monologue"), ("reading", "reading"), ("conversation", "conversation")]:
        wav_dir = RAW / "release2" / subdir / "audio" / "wav"
        if wav_dir.exists():
            for fp in wav_dir.glob("*.wav"):
                sid = fp.stem
                if sid not in inventory:
                    inventory[sid] = {"release": "R2", "task": task}
                elif inventory[sid]["release"] == "R1":
                    inventory[sid]["release"] = "R1+R2"
                    inventory[sid]["task"] = inventory[sid].get("task", "") + f"+{task}"
                else:
                    inventory[sid]["task"] = inventory[sid].get("task", "") + f"+{task}"
                inventory[sid]["has_audio"] = True

    # Check processed transcript availability
    aligned_dir = PROCESSED / "transcripts" / "aligned"
    flat_dir = PROCESSED / "transcripts" / "flat"

    if aligned_dir.exists():
        for fp in aligned_dir.glob("*.csv"):
            # Extract session ID from filename like M_0030_16y4m_1_syll.csv
            name = fp.stem
            # Try to match session ID pattern
            m = re.match(r"([FM]_\d{4}_[\dy]+m[_-]\d+)", name)
            if m:
                sid = m.group(1)
                if sid in inventory:
                    inventory[sid]["has_aligned_transcript"] = True

    if flat_dir.exists():
        for fp in flat_dir.glob("*.csv"):
            name = fp.stem
            m = re.match(r"([FM]_\d{4}_[\dy]+m[_-]\d+)", name)
            if m:
                sid = m.group(1)
                if sid in inventory:
                    inventory[sid]["has_flat_transcript"] = True

    # Write inventory
    out_path = PROCESSED / "inventory.csv"
    rows = []
    for sid in sorted(inventory.keys()):
        info = parse_session_id(sid)
        entry = inventory[sid]
        rows.append({
            "session_id": sid,
            "speaker_id": info["speaker_id"],
            "gender": info["gender"],
            "age_str": info["age_str"],
            "age_months": info["age_months"],
            "release": entry.get("release", ""),
            "task": entry.get("task", ""),
            "has_audio": entry.get("has_audio", False),
            "has_aligned_transcript": entry.get("has_aligned_transcript", False),
            "has_flat_transcript": entry.get("has_flat_transcript", False),
        })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # Summary stats
    total = len(rows)
    with_audio = sum(1 for r in rows if r["has_audio"])
    with_aligned = sum(1 for r in rows if r["has_aligned_transcript"])
    with_flat = sum(1 for r in rows if r["has_flat_transcript"])
    with_any = sum(1 for r in rows if r["has_aligned_transcript"] or r["has_flat_transcript"])
    usable = sum(1 for r in rows if r["has_audio"] and (r["has_aligned_transcript"] or r["has_flat_transcript"]))

    unique_speakers = len(set(r["speaker_id"] for r in rows))

    print(f"\nInventory: {total} sessions, {unique_speakers} speakers")
    print(f"  With audio: {with_audio}")
    print(f"  With aligned transcript: {with_aligned}")
    print(f"  With flat transcript: {with_flat}")
    print(f"  With any transcript: {with_any}")
    print(f"  USABLE (audio + transcript): {usable}")
    print(f"Written to {out_path}")

    # Also write a speakers.csv
    speakers_path = PROCESSED / "speakers.csv"
    speaker_rows = {}
    for r in rows:
        spk = r["speaker_id"]
        if spk not in speaker_rows:
            speaker_rows[spk] = {
                "speaker_id": spk,
                "gender": r["gender"],
                "n_sessions": 0,
                "age_range": [],
                "releases": set(),
                "tasks": set(),
                "n_with_transcript": 0,
            }
        speaker_rows[spk]["n_sessions"] += 1
        if r["age_months"]:
            speaker_rows[spk]["age_range"].append(r["age_months"])
        speaker_rows[spk]["releases"].add(r["release"])
        for t in r["task"].split("+"):
            if t:
                speaker_rows[spk]["tasks"].add(t)
        if r["has_aligned_transcript"] or r["has_flat_transcript"]:
            speaker_rows[spk]["n_with_transcript"] += 1

    spk_rows = []
    for spk in sorted(speaker_rows.keys()):
        s = speaker_rows[spk]
        ages = sorted(s["age_range"])
        spk_rows.append({
            "speaker_id": spk,
            "gender": s["gender"],
            "n_sessions": s["n_sessions"],
            "min_age_months": ages[0] if ages else "",
            "max_age_months": ages[-1] if ages else "",
            "releases": "+".join(sorted(s["releases"])),
            "tasks": "+".join(sorted(s["tasks"])),
            "n_with_transcript": s["n_with_transcript"],
        })

    with open(speakers_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=spk_rows[0].keys())
        w.writeheader()
        w.writerows(spk_rows)

    print(f"Speakers: {len(spk_rows)} written to {speakers_path}")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("=== Processing TextGrid files ===")
    process_textgrids()

    print("\n=== Processing SFS annotations ===")
    process_sfs_annotations()

    print("\n=== Processing flat transcripts ===")
    process_flat_transcripts()

    print("\n=== Building inventory and speaker metadata ===")
    build_inventory()

    print("\nDone!")


if __name__ == "__main__":
    main()
