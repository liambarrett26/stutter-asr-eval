#!/usr/bin/env python3
"""Copy archive files from the speech-lab Windows store using a match manifest.

Reads data/additions/archive_match_manifest.csv (from archive_match.py) and
copies each unique source file from the lab Windows store to a destination
tree on local disk. The archive's directory structure under X:\\Speech\\ is
mirrored under --dest so provenance is preserved.

Usage (on the Windows machine with the X: drive mounted):

    python archive_egress.py \\
        --manifest archive_match_manifest.csv \\
        --source-root "X:\\Speech" \\
        --dest "D:\\slass_additions" \\
        --log archive_egress_log.csv

Cross-platform notes
--------------------
The script also runs on macOS/Linux if the archive is mounted there
(e.g. via SMB). Pass --source-root "/Volumes/Research2/Speech" to map the
Windows X:\\Speech prefix to the mounted Unix path. Paths in the manifest
are stored verbatim with backslash separators; only the prefix is
substituted.

Behaviour
---------
- Files are deduped: each unique src_path is copied once even if it
  appears under multiple slass_sessions in the manifest.
- Already-present destination files are skipped unless --overwrite is
  passed. This makes the script safe to re-run after partial transfers.
- A log CSV records the outcome (copied/skipped/missing/error) and the
  destination path for each unique source file.
- Use --dry-run to print actions without performing copies. Recommended
  for a first pass to verify path mapping is correct.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import traceback
from pathlib import Path

csv.field_size_limit(sys.maxsize)

WINDOWS_PREFIX_DEFAULT = "X:\\Speech"


def normalise_src(src_path: str, source_root: str) -> Path:
    """Map a Windows-style archive path to a Path under source_root.

    The manifest stores paths like ``X:\\Speech\\Alan\\foo.txt``. We strip
    the configured Windows prefix (default ``X:\\Speech``) and join the
    remainder onto --source-root, with native path separators.
    """
    prefix = WINDOWS_PREFIX_DEFAULT
    if src_path.lower().startswith(prefix.lower()):
        rel = src_path[len(prefix):].lstrip("\\/")
    else:
        # Already-relative or differently-rooted; treat as relative.
        rel = src_path
    # Convert any backslashes to the platform separator.
    rel_parts = rel.replace("\\", "/").split("/")
    return Path(source_root, *rel_parts)


def dest_for(src_path: str, dest_root: Path) -> Path:
    """Build the destination Path, mirroring the archive's tree under dest."""
    prefix = WINDOWS_PREFIX_DEFAULT
    if src_path.lower().startswith(prefix.lower()):
        rel = src_path[len(prefix):].lstrip("\\/")
    else:
        rel = src_path
    rel_parts = rel.replace("\\", "/").split("/")
    return Path(dest_root, *rel_parts)


def dedupe(manifest_rows: list[dict]) -> list[dict]:
    """Return one row per unique src_path, preserving first occurrence
    and remembering which slass_sessions referenced it."""
    seen: dict[str, dict] = {}
    sessions: dict[str, list[str]] = {}
    categories: dict[str, set[str]] = {}
    for r in manifest_rows:
        p = r["src_path"]
        if p not in seen:
            seen[p] = r
        sessions.setdefault(p, []).append(r["slass_session"])
        categories.setdefault(p, set()).add(r["category"])
    out = []
    for p, r in seen.items():
        r = dict(r)
        r["n_sessions"] = len(sessions[p])
        r["categories"] = "|".join(sorted(categories[p]))
        out.append(r)
    return out


def copy_one(src: Path, dest: Path, overwrite: bool,
             dry_run: bool) -> str:
    """Return one of: copied, skipped, missing, error."""
    if not src.exists():
        return "missing"
    if dest.exists() and not overwrite:
        return "skipped"
    if dry_run:
        return "copied"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, dest)
        return "copied"
    except Exception:
        traceback.print_exc(limit=1)
        return "error"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="archive_match_manifest.csv from archive_match.py")
    ap.add_argument("--source-root", required=True,
                    help="Where X:\\Speech is reachable on this machine. "
                         "On Windows, pass 'X:\\Speech'. On a Mac with "
                         "the Research2 volume mounted, pass e.g. "
                         "'/Volumes/Research2/Speech'.")
    ap.add_argument("--dest", type=Path, required=True,
                    help="Destination root directory for the egress")
    ap.add_argument("--log", type=Path, required=True,
                    help="Output CSV log of per-file outcomes")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite destination files that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without copying anything")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most this many unique files "
                         "(0 = no limit). Useful for staged egress.")
    args = ap.parse_args()

    rows = []
    with args.manifest.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    unique = dedupe(rows)
    print(f"Manifest has {len(rows):,} rows; "
          f"{len(unique):,} unique source files")

    if args.limit and args.limit < len(unique):
        unique = unique[: args.limit]
        print(f"Limiting to first {args.limit:,} unique files")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_cols = ["src_path", "dest_path", "size_mb", "categories",
                "n_sessions", "outcome"]

    outcome_counts: dict[str, int] = {}
    with args.log.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=log_cols)
        w.writeheader()
        for i, r in enumerate(unique, start=1):
            src = normalise_src(r["src_path"], args.source_root)
            dest = dest_for(r["src_path"], args.dest)
            outcome = copy_one(src, dest, args.overwrite, args.dry_run)
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            w.writerow({
                "src_path": r["src_path"],
                "dest_path": str(dest),
                "size_mb": r["src_size_mb"],
                "categories": r["categories"],
                "n_sessions": r["n_sessions"],
                "outcome": outcome,
            })
            if i % 200 == 0:
                print(f"  {i:,}/{len(unique):,}  "
                      + " ".join(f"{k}={v}"
                                 for k, v in outcome_counts.items()))

    print(f"\nDone. Outcomes: "
          + " ".join(f"{k}={v}" for k, v in outcome_counts.items()))
    print(f"Log written to {args.log}")


if __name__ == "__main__":
    main()
