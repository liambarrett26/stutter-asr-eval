#!/usr/bin/env python3
"""Process the UNWR adult-cohort corpus into the project's standard layout.

Inputs (under /Volumes/FATSPEECH/unwr/raw/UNWR/):
    Data Directory.xlsx          Master speaker + scoring metadata (8 sheets)
    <Group>/<PID> (<ID>)/<PID>_SSI/<PID>_q{1,2}.wav   SSI speech recordings
    <Group>/<PID> (<ID>)/<PID>_SSI/<PID>_transcript.docx   Syllable-segmented
                                                          transcript

Outputs (under /Volumes/FATSPEECH/unwr/processed/):
    speakers.csv                 One row per speaker with demographics,
                                 diagnoses, SSI scores, ASRS scores
    scoring_unwr.csv             Per-item UNWR nonword-reading scores
    scoring_srt.csv              Per-item SRT syllable-repetition scores
    scoring_pdt.csv              Per-item PDT phonetic-discrimination scores
    transcripts/<PID>.csv        Per-speaker transcript:
                                 text (period-stripped clean) +
                                 text_syllabified (syllable-period form)
    inventory.csv                Per-session inventory keyed by PID:
                                 audio paths (q1, q2), transcript path,
                                 audio duration, sample rate, group

This module has no project-internal dependencies; it only uses the
Python standard library plus openpyxl for the XLSX parse.

Usage:
    python src/data/process_unwr.py
"""

from __future__ import annotations

import csv
import re
import struct
import sys
import zipfile
from pathlib import Path

try:
    from openpyxl import load_workbook
except ImportError:
    print("openpyxl is required: pip install openpyxl", file=sys.stderr)
    sys.exit(1)


RAW_ROOT = Path("/Volumes/FATSPEECH/unwr/raw/UNWR")
OUT_ROOT = Path("/Volumes/FATSPEECH/unwr/processed")

GROUP_DIRS = {
    "Control (C)": "Control",
    "Stutter (S)": "Stutter",
    "Attention-ADHD (A)": "Attention",
    "Attention + Stutter (AS)": "Attention+Stutter",
}


# ── Master metadata ──────────────────────────────────────────────────────────

def parse_data_directory(xlsx_path: Path) -> dict[str, list[dict]]:
    """Return a dict of sheet_name -> list of row dicts (header keyed)."""
    wb = load_workbook(xlsx_path, data_only=True)
    out: dict[str, list[dict]] = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = [str(c).strip() if c is not None else "" for c in rows[0]]
        records = []
        for row in rows[1:]:
            if all(c is None or c == "" for c in row):
                continue
            rec = {header[i]: row[i] for i in range(min(len(header), len(row)))}
            records.append(rec)
        out[ws.title.strip()] = records
    return out


def build_speakers_csv(sheets: dict[str, list[dict]], pid_by_id: dict[int, str],
                       group_by_pid: dict[str, str], out_path: Path) -> int:
    """One row per ID, keyed by PID where available, joining All_data + SSI."""
    all_data = {r["ID"]: r for r in sheets.get("All_data", []) if r.get("ID")}
    ssi = {r["ID"]: r for r in sheets.get("SSI", []) if r.get("ID")}
    pdt = {r["ID"]: r for r in sheets.get("PDT", []) if r.get("ID")}
    srt = {r["ID"]: r for r in sheets.get("SRT", []) if r.get("ID")}
    dir_rows = sheets.get("Recording Directory", [])

    ids: set = set()
    for d in (all_data, ssi, pdt, srt):
        ids.update(d.keys())
    for r in dir_rows:
        if r.get("ID"):
            ids.add(r["ID"])

    cols = [
        "id", "pid", "group_directory", "all_data_group",
        "age", "gender", "first_language", "additional_language",
        "stutter", "stutter_diagnosis", "attention", "attention_diagnosis",
        "bilingualism",
        "ssi_pct", "ssi_syllable_count", "ssi_stuttered_syllables",
        "srt_total", "srt_2", "srt_3", "srt_4",
        "pdt_total",
        "asrs_total", "asrs_hyperactivity", "asrs_inattention",
        "psal_total", "sonority", "modulation", "cue",
    ]
    n = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for id_ in sorted(ids, key=lambda x: (x is None, x)):
            ad = all_data.get(id_, {})
            s = ssi.get(id_, {})
            p = pdt.get(id_, {})
            sr = srt.get(id_, {})
            pid = pid_by_id.get(id_, "")
            w.writerow({
                "id": id_,
                "pid": pid,
                "group_directory": group_by_pid.get(pid, ""),
                "all_data_group": ad.get("Group", ""),
                "age": ad.get("age", ""),
                "gender": ad.get("gender", ""),
                "first_language": ad.get("first.language", ""),
                "additional_language": ad.get("additional.language", ""),
                "stutter": ad.get("stutter", ""),
                "stutter_diagnosis": ad.get("stutter.diagnosis", ""),
                "attention": ad.get("attention", ""),
                "attention_diagnosis": ad.get("attention.diagnosis", ""),
                "bilingualism": ad.get("bilingualism", ""),
                "ssi_pct": s.get("% SSI", ad.get("X..SSI", "")),
                "ssi_syllable_count": s.get("Syllable", ad.get("Syllable", "")),
                "ssi_stuttered_syllables":
                    s.get("Stuttered", ad.get("Stuttered", "")),
                "srt_total": sr.get("srt", ad.get("srt", "")),
                "srt_2": sr.get("srt_2", ad.get("srt_2", "")),
                "srt_3": sr.get("srt_3", ad.get("srt_3", "")),
                "srt_4": sr.get("srt_4", ad.get("srt_4", "")),
                "pdt_total": p.get("pdt_total", ad.get("pdt_total", "")),
                "asrs_total": ad.get("ASRS_total", ""),
                "asrs_hyperactivity": ad.get("hyperactivity", ""),
                "asrs_inattention": ad.get("inattention", ""),
                "psal_total": ad.get("son_total_norm", ""),
                "sonority": ad.get("sonority_norm", ""),
                "modulation": ad.get("modulation_score_norm", ""),
                "cue": ad.get("cue_norm", ""),
            })
            n += 1
    return n


def build_scoring_csvs(sheets: dict[str, list[dict]], out_dir: Path) -> dict[str, int]:
    """Write the per-item UNWR / SRT / PDT scoring sheets verbatim."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for sheet, name in (("UNWR_scoring", "scoring_unwr.csv"),
                        ("SRT_scoring", "scoring_srt.csv"),
                        ("PDT_scoring", "scoring_pdt.csv")):
        rows = sheets.get(sheet, [])
        if not rows:
            counts[name] = 0
            continue
        cols = list(rows[0].keys())
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (r.get(c) if r.get(c) is not None else "")
                            for c in cols})
        counts[name] = len(rows)
    return counts


# ── Discovery: speaker / file layout ─────────────────────────────────────────

PID_DIR_RE = re.compile(r"^([A-Z]+\d+)\s*\(?(\d+)?\)?$", re.I)


def discover_speakers() -> list[dict]:
    """Walk the RAW_ROOT and return per-speaker file info."""
    speakers = []
    for group_dir_name in GROUP_DIRS:
        group_dir = RAW_ROOT / group_dir_name
        if not group_dir.exists():
            continue
        for spk_dir in sorted(group_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            m = PID_DIR_RE.match(spk_dir.name.strip())
            if not m:
                continue
            pid = m.group(1).upper()
            id_str = m.group(2)
            id_int = int(id_str) if id_str and id_str.isdigit() else None
            ssi_dir = spk_dir / f"{pid}_SSI"
            if not ssi_dir.exists():
                # Some directories use a lowercased subfolder name.
                candidates = list(spk_dir.glob(f"{pid}_*"))
                ssi_dir = candidates[0] if candidates else None
            audio_q1 = ssi_dir / f"{pid}_q1.wav" if ssi_dir else None
            audio_q2 = ssi_dir / f"{pid}_q2.wav" if ssi_dir else None
            transcript = (ssi_dir / f"{pid}_transcript.docx"
                          if ssi_dir else None)
            speakers.append({
                "pid": pid,
                "id": id_int,
                "group": GROUP_DIRS[group_dir_name],
                "group_directory": group_dir_name,
                "ssi_dir": ssi_dir,
                "audio_q1": audio_q1 if (audio_q1 and audio_q1.exists())
                            else None,
                "audio_q2": audio_q2 if (audio_q2 and audio_q2.exists())
                            else None,
                "transcript": transcript if (transcript
                                             and transcript.exists())
                              else None,
            })
    return speakers


# ── WAV header probe (no third-party deps) ───────────────────────────────────

def probe_wav(path: Path) -> tuple[float | None, int | None]:
    """Return (duration_s, sample_rate) by reading the WAV/RIFF header."""
    try:
        with path.open("rb") as fh:
            riff = fh.read(12)
            if riff[:4] != b"RIFF" or riff[8:12] != b"WAVE":
                return (None, None)
            sample_rate = None
            byte_rate = None
            bits_per_sample = None
            num_channels = None
            data_size = None
            while True:
                hdr = fh.read(8)
                if len(hdr) < 8:
                    break
                chunk_id, chunk_size = struct.unpack("<4sI", hdr)
                if chunk_id == b"fmt ":
                    fmt = fh.read(chunk_size)
                    (audio_format, num_channels, sample_rate,
                     byte_rate, _block_align,
                     bits_per_sample) = struct.unpack("<HHIIHH", fmt[:16])
                elif chunk_id == b"data":
                    data_size = chunk_size
                    break
                else:
                    fh.seek(chunk_size, 1)
            if sample_rate and data_size and num_channels and bits_per_sample:
                bytes_per_sample = bits_per_sample // 8
                duration = data_size / (sample_rate * num_channels
                                        * bytes_per_sample)
                return (duration, sample_rate)
    except Exception:
        pass
    return (None, None)


# ── Transcript parser ────────────────────────────────────────────────────────

def parse_transcript(docx_path: Path) -> tuple[str, str]:
    """Return (text_syllabified, text_clean) parsed from a .docx transcript.

    text_syllabified preserves the period-delimited syllable segmentation
    exactly as in the original (e.g. 'Fa.vo.rite. hob.bies.').

    text_clean is normalised lowercase whitespace-joined words with the
    intra-word periods removed (e.g. 'favorite hobbies').
    """
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    # Strip XML tags
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"\s+", " ", text).strip()

    text_syllabified = text
    # Drop trailing periods on each token to recover whole words.
    # 'Fa.vo.rite.' -> 'Favorite'; 'O.K.' -> 'OK'; 'hob.bies.' -> 'hobbies'
    cleaned = re.sub(r"\.", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return (text_syllabified, cleaned)


def write_transcript_csv(pid: str, syll: str, clean: str,
                         out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pid}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pid", "text", "text_syllabified"])
        w.writerow([pid, clean, syll])


# ── Inventory ────────────────────────────────────────────────────────────────

def build_inventory(speakers: list[dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pid", "id", "group", "group_directory",
            "audio_q1", "audio_q2", "transcript",
            "q1_duration_s", "q1_sample_rate",
            "q2_duration_s", "q2_sample_rate"]
    n = 0
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in speakers:
            d1, sr1 = probe_wav(s["audio_q1"]) if s["audio_q1"] else (None, None)
            d2, sr2 = probe_wav(s["audio_q2"]) if s["audio_q2"] else (None, None)
            w.writerow({
                "pid": s["pid"],
                "id": s["id"] or "",
                "group": s["group"],
                "group_directory": s["group_directory"],
                "audio_q1": str(s["audio_q1"]) if s["audio_q1"] else "",
                "audio_q2": str(s["audio_q2"]) if s["audio_q2"] else "",
                "transcript": str(s["transcript"]) if s["transcript"] else "",
                "q1_duration_s": f"{d1:.2f}" if d1 else "",
                "q1_sample_rate": sr1 or "",
                "q2_duration_s": f"{d2:.2f}" if d2 else "",
                "q2_sample_rate": sr2 or "",
            })
            n += 1
    return n


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Raw root: {RAW_ROOT}")
    print(f"Output root: {OUT_ROOT}")

    # 1. Discover speakers from the filesystem
    speakers = discover_speakers()
    print(f"\nDiscovered {len(speakers)} speakers")
    pid_to_group = {s["pid"]: s["group_directory"] for s in speakers}

    # 2. Parse Data Directory.xlsx
    xlsx = RAW_ROOT / "Data Directory.xlsx"
    sheets = parse_data_directory(xlsx)
    print(f"Loaded sheets: {list(sheets.keys())}")
    pid_by_id = {}
    for r in sheets.get("Recording Directory", []):
        if r.get("ID") and r.get("PID"):
            pid_by_id[r["ID"]] = str(r["PID"]).strip().upper()

    # 3. Write speakers.csv + scoring CSVs
    n_spk = build_speakers_csv(sheets, pid_by_id, pid_to_group,
                               OUT_ROOT / "speakers.csv")
    print(f"Wrote speakers.csv: {n_spk} rows")
    sc_counts = build_scoring_csvs(sheets, OUT_ROOT)
    for k, v in sc_counts.items():
        print(f"Wrote {k}: {v} rows")

    # 4. Parse each transcript
    n_trans = 0
    for s in speakers:
        if not s["transcript"]:
            continue
        try:
            syll, clean = parse_transcript(s["transcript"])
        except Exception as e:
            print(f"  parse failure {s['pid']}: {e}")
            continue
        write_transcript_csv(s["pid"], syll, clean,
                             OUT_ROOT / "transcripts")
        n_trans += 1
    print(f"Wrote {n_trans} transcript CSVs")

    # 5. Inventory with WAV durations and sample rates
    n_inv = build_inventory(speakers, OUT_ROOT / "inventory.csv")
    print(f"Wrote inventory.csv: {n_inv} rows")


if __name__ == "__main__":
    main()
