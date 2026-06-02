#!/usr/bin/env python3
"""Parse the speech-lab archive Windows directory listing into a clean CSV inventory.

Input: /Volumes/FATSPEECH/store_metadata/sizes.txt (UTF-16 LE, produced by PowerShell
       Get-ChildItem | Format-Table FullName, SizeMB on the lab's Research2 volume).

Output: /Volumes/FATSPEECH/store_metadata/archive_inventory.csv with columns
       full_path, size_mb, top_level, parent_dir, basename, stem, ext

The script auto-converts UTF-16 to UTF-8 in memory; it does not require the
pre-converted .utf8.txt file. Paths are kept in their original Windows form
(backslash separators, X:\\Speech\\... prefix) so that downstream scripts can
either map them to a mounted path on a Mac or use them directly on Windows.

Usage:
    python -m src.data.archive_inventory \\
        --sizes /Volumes/FATSPEECH/store_metadata/sizes.txt \\
        --out /Volumes/FATSPEECH/store_metadata/archive_inventory.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path


SIZE_RE = re.compile(r"^(.*?)\s+([0-9]+(?:\.[0-9]+)?)\s*$")


def read_sizes(path: Path) -> list[tuple[str, float]]:
    """Yield (full_path, size_mb) from the PowerShell sizes dump.

    Handles UTF-16 LE input transparently. Skips header rows and any line
    whose final token isn't a number.
    """
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8", errors="replace")

    rows = []
    for line in io.StringIO(text):
        line = line.rstrip("\r\n")
        if not line:
            continue
        if line.startswith("FullName") or line.startswith("--------"):
            continue
        m = SIZE_RE.match(line)
        if not m:
            continue
        full = m.group(1).rstrip()
        try:
            size_mb = float(m.group(2))
        except ValueError:
            continue
        # PowerShell reports SizeMB == 0 for directories; we keep those
        # as inventory rows (helps in matching), but mark them later
        # by absence of an extension.
        rows.append((full, size_mb))
    return rows


def split_win_path(p: str) -> tuple[str, str, str, str, str]:
    """Return (top_level, parent_dir, basename, stem, ext) from a Windows path.

    top_level = first path component after X:\\Speech\\ (e.g. 'Alan').
    parent_dir = the directory containing the file (full path, no trailing sep).
    basename = file name with extension.
    stem = file name without final extension.
    ext = final extension lower-cased, without the leading dot ('' if none).
    """
    parts = p.split("\\")
    basename = parts[-1] if parts else ""
    parent_dir = "\\".join(parts[:-1])
    # Top-level under X:\Speech\
    top_level = ""
    if len(parts) >= 3 and parts[0].endswith(":") and parts[1].lower() == "speech":
        top_level = parts[2]
    # Extension
    if "." in basename:
        stem, _, ext = basename.rpartition(".")
        ext = ext.lower()
    else:
        stem, ext = basename, ""
    return top_level, parent_dir, basename, stem, ext


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=Path, required=True,
                    help="Path to sizes.txt (UTF-16 from PowerShell)")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV path")
    args = ap.parse_args()

    rows = read_sizes(args.sizes)
    print(f"Parsed {len(rows):,} rows from {args.sizes}")

    # First pass: collect every parent_dir seen, plus all ancestors of
    # each parent. Any path equal to one of those is a directory, even if
    # it has a dotted name that looks like an extension and even if its
    # immediate children are themselves directories.
    known_dirs: set[str] = set()
    parsed: list[tuple[str, float, str, str, str, str, str]] = []
    for full, size_mb in rows:
        top, parent, base, stem, ext = split_win_path(full)
        anc = parent
        while anc and anc not in known_dirs:
            known_dirs.add(anc)
            anc = anc.rsplit("\\", 1)[0] if "\\" in anc else ""
        parsed.append((full, size_mb, top, parent, base, stem, ext))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["full_path", "size_mb", "top_level",
                    "parent_dir", "basename", "stem", "ext"])
        n_dirs = n_files = 0
        for full, size_mb, top, parent, base, stem, ext in parsed:
            if full in known_dirs:
                n_dirs += 1
                continue
            n_files += 1
            w.writerow([full, f"{size_mb:.4f}", top, parent, base, stem, ext])

    print(f"Wrote {n_files:,} file rows ({n_dirs:,} directory rows skipped) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
