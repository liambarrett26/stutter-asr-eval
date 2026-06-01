#!/usr/bin/env python3
"""Process the UNWR children's nonword-reading reliability dataset.

Inputs (under /Volumes/FATSPEECH/unwr_reliability/raw/Reliability_Analyses/):
    data/Retranscription/Retranscribed_Clarissa/<school>/<child>/syll{2,3,4}/<item>.TextGrid
    data/Retranscription/Retranscription_Kaho/<school>/<child>/syll{2,3,4}/<item>.TextGrid
    data/Retranscription/Retranscription_Roaa_28Nov2019/<school>/<child>/syll{2,3,4}/<item>.TextGrid
    data/Retranscription/Retranscription_Roaa_10Oct2019/... (intermediate, smaller)
    data/Retranscription/addition/... (Hatfield re-add files for Kaho)

Each TextGrid has up to three IntervalTiers: 'ortho' (English nonword),
'target' (phoneme target), 'response' (transcribed phoneme response).

Output (/Volumes/FATSPEECH/unwr_reliability/processed/):
    items.csv     One row per (transcriber, school, child, item) with the
                  ortho / target / response labels and the interval times.

The Clarissa retranscriptions are treated as the gold standard.

Usage:
    python src/data/process_unwr_reliability.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


RAW_ROOT = Path("/Volumes/FATSPEECH/unwr_reliability/raw/Reliability_Analyses/"
                "data/Retranscription")
OUT_ROOT = Path("/Volumes/FATSPEECH/unwr_reliability/processed")

# Map each top-level Retranscription_* / Retranscribed_* directory to a
# transcriber identifier. Sub-folders inside addition/ get classified by
# inspecting their inner path.
TRANSCRIBER_DIR = {
    "Retranscribed_Clarissa": "Clarissa",
    "Retranscription_Kaho": "Kaho",
    "Retranscription_Roaa_28Nov2019": "Roaa_28Nov",
    "Retranscription_Roaa_10Oct2019": "Roaa_10Oct",
}


# ── Minimal Praat TextGrid parser ────────────────────────────────────────────

def parse_textgrid(path: Path) -> dict[str, list[dict]]:
    """Return {tier_name: [{xmin, xmax, text}, ...]} from a Praat TextGrid.

    Handles the long-form (verbose) Praat TextGrid syntax used in this
    dataset. Robust to comments and inconsistent whitespace.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    tiers: dict[str, list[dict]] = {}
    current_tier_name: str | None = None
    current_intervals: list[dict] = []
    interval: dict | None = None

    name_re = re.compile(r'name\s*=\s*"([^"]*)"')
    class_re = re.compile(r'class\s*=\s*"([^"]*)"')
    xmin_re = re.compile(r'xmin\s*=\s*([\-0-9.eE]+)')
    xmax_re = re.compile(r'xmax\s*=\s*([\-0-9.eE]+)')
    text_re = re.compile(r'text\s*=\s*"((?:[^"\\]|\\.)*)"')

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        cm = class_re.search(stripped)
        nm = name_re.search(stripped)
        if cm and "IntervalTier" in cm.group(1):
            # Starting a new tier - flush prior tier
            if current_tier_name is not None:
                tiers[current_tier_name] = current_intervals
            current_tier_name = None
            current_intervals = []
            interval = None
            continue
        if nm and current_tier_name is None and not stripped.startswith("text"):
            current_tier_name = nm.group(1).strip().lower()
            continue
        if stripped.startswith("intervals ["):
            if interval is not None:
                current_intervals.append(interval)
            interval = {"xmin": None, "xmax": None, "text": ""}
            continue
        if interval is not None:
            m = xmin_re.search(stripped)
            if m:
                try:
                    interval["xmin"] = float(m.group(1))
                except ValueError:
                    pass
                continue
            m = xmax_re.search(stripped)
            if m:
                try:
                    interval["xmax"] = float(m.group(1))
                except ValueError:
                    pass
                continue
            m = text_re.search(stripped)
            if m:
                interval["text"] = m.group(1).replace('\\"', '"')
                continue
    if interval is not None:
        current_intervals.append(interval)
    if current_tier_name is not None:
        tiers[current_tier_name] = current_intervals
    return tiers


# ── Walk + extract ───────────────────────────────────────────────────────────

# Filename pattern: leading digits are the syllable group, then 'prac' or
# 'test' plus an item index, then the nonword orthography.
ITEM_RE = re.compile(r"^(\d+)(prac|test)(\d+)(.+)$", re.I)


def classify_transcriber(rel_path: Path) -> str:
    """Determine the transcriber identifier from a path relative to RAW_ROOT.

    When the leading component is a transcriber bucket (Retranscribed_*,
    Retranscription_*), use the map. Otherwise — for the top-level
    school directories that the nested zips dropped into RAW_ROOT —
    the transcriber identity is embedded in the school-cohort dir name
    itself (e.g. 'Hackney_2017_Clarissa_*' or 'Hatfield_Kaho_*'). These
    are the *original* transcriptions, retained alongside the
    retranscriptions for reliability comparison.
    """
    parts = rel_path.parts
    if parts and parts[0] in TRANSCRIBER_DIR:
        return TRANSCRIBER_DIR[parts[0]]
    if parts and parts[0] == "addition":
        s = "/".join(parts[1:]).lower()
        if "kaho" in s:
            return "Kaho_addition"
        if "clarissa" in s:
            return "Clarissa_addition"
        return "addition_other"
    if parts and parts[0] == "November_RoaaFolders":
        return "Roaa_Nov"
    if parts and parts[0] == "RoaaEmptyFolders":
        return "Roaa_empty"
    if parts and parts[0] == "Retranscription_Empty_Folders":
        return "empty_placeholder"
    # Original-cohort school dirs encode the transcriber in their name.
    if parts:
        head = parts[0].lower()
        if "clarissa" in head:
            return "Clarissa_original"
        if "kaho" in head:
            return "Kaho_original"
        if head.startswith("sthelen"):
            return "Kaho_original"  # StHelen cohort transcribed by Kaho
    return "unknown"


def extract_school_year_cohort(school_dir: str) -> tuple[str, str, str]:
    """Return (school, year, cohort_label) parsed from e.g.
    'Hackney_2017_Clarissa_Complete_20180731'."""
    m = re.match(r"^([A-Za-z]+)_(\d{4})_([A-Za-z]+)", school_dir)
    if m:
        return (m.group(1), m.group(2), school_dir)
    m = re.match(r"^([A-Za-z]+)_([A-Za-z]+)_Complete", school_dir)
    if m:
        return (m.group(1), "", school_dir)
    m = re.match(r"^([A-Za-z]+)_Complete", school_dir)
    if m:
        return (m.group(1), "", school_dir)
    return (school_dir, "", school_dir)


def main() -> None:
    if not RAW_ROOT.exists():
        print(f"raw root not found: {RAW_ROOT}", file=sys.stderr)
        sys.exit(1)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    n_tg = 0
    n_skip = 0
    for tg_path in sorted(RAW_ROOT.rglob("*.TextGrid")):
        if tg_path.name.startswith("."):
            continue
        n_tg += 1
        rel = tg_path.relative_to(RAW_ROOT)
        transcriber = classify_transcriber(rel)
        # Path layout under transcriber dir is
        #   <school_cohort>/<child_dir>/<syll_dir>/<item>.TextGrid
        parts = list(rel.parts)
        # Drop the leading transcriber-bucket folder
        if parts and parts[0] in TRANSCRIBER_DIR:
            parts = parts[1:]
        elif parts and parts[0] == "addition":
            # addition/<bucket>/<...>; drop two leading components
            parts = parts[2:]
        if len(parts) < 4:
            n_skip += 1
            continue
        school_dir, child_dir, syll_dir, item_file = parts[-4:]
        school, year, cohort = extract_school_year_cohort(school_dir)

        # child_dir formats observed:
        #   '2_Hackney_Amelia', '152_Hatfeild_Betsy', '36_Hackney_Albert'
        child_parts = child_dir.split("_", 2)
        child_id = child_parts[0]
        child_name = ("_".join(child_parts[1:])
                      if len(child_parts) >= 2 else child_dir)

        m = ITEM_RE.match(tg_path.stem)
        if m:
            syll_n = int(m.group(1))
            item_kind = m.group(2).lower()
            item_idx = int(m.group(3))
            ortho_in_name = m.group(4).lower()
        else:
            syll_n = None
            item_kind = ""
            item_idx = None
            ortho_in_name = tg_path.stem.lower()

        try:
            tiers = parse_textgrid(tg_path)
        except Exception as e:
            n_skip += 1
            continue

        ortho = ""
        target = ""
        response = ""
        t_start = ""
        t_end = ""
        for tname, intervals in tiers.items():
            non_empty = [i for i in intervals if i.get("text")]
            joined = " ".join(i["text"].strip() for i in non_empty
                              if i["text"].strip())
            if tname in ("ortho", "orthography", "orth"):
                ortho = joined
                if non_empty:
                    t_start = (f"{non_empty[0]['xmin']:.3f}"
                               if non_empty[0]['xmin'] is not None else "")
                    t_end = (f"{non_empty[-1]['xmax']:.3f}"
                             if non_empty[-1]['xmax'] is not None else "")
            elif tname == "target":
                target = joined
            elif tname in ("response", "resp"):
                response = joined

        rows.append({
            "transcriber": transcriber,
            "school": school,
            "year": year,
            "cohort": cohort,
            "child_id": child_id,
            "child_name": child_name,
            "syllable_length": syll_n if syll_n is not None else "",
            "item_kind": item_kind,
            "item_idx": item_idx if item_idx is not None else "",
            "item_ortho": ortho_in_name,
            "ortho": ortho,
            "target": target,
            "response": response,
            "t_start_s": t_start,
            "t_end_s": t_end,
            "textgrid_path": str(tg_path),
        })

    cols = ["transcriber", "school", "year", "cohort",
            "child_id", "child_name",
            "syllable_length", "item_kind", "item_idx", "item_ortho",
            "ortho", "target", "response",
            "t_start_s", "t_end_s", "textgrid_path"]

    out_csv = OUT_ROOT / "items.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Parsed {n_tg:,} TextGrids ({n_skip} skipped)")
    print(f"Wrote {len(rows):,} rows -> {out_csv}")


if __name__ == "__main__":
    main()
