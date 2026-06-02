#!/usr/bin/env python3
"""Match the speech-lab archive inventory against our extracted SLASS sessions.

Reads:
  - /Volumes/FATSPEECH/store_metadata/archive_inventory.csv  (from archive_inventory.py)
  - /Volumes/FATSPEECH/slass/full_archive/inventory.csv (default)

Writes:
  - /Volumes/FATSPEECH/store_metadata/archive_match_manifest.csv
        One row per (slass_session, archive_file) candidate. Used as input
        to archive_egress.py.
  - /Volumes/FATSPEECH/store_metadata/archive_match_summary.csv
        One row per slass_session with counts by category.

Matching strategy
-----------------
For each SLASS session basename we parse a canonical key of the form
``(speaker_id, age_y, age_m)`` from substrings like ``f_0050_7_11y6m`` or
``0014_10y10m_conv``. We then look up archive files whose basename contains
the same triple, regardless of which top-level directory they sit in.

Categories of egress candidate (a file may carry several):
  transcript   - text transcripts (.txt, .doc, .docx, .rpt, .rtf, .htm,
                 .html, .salt, .pdf) keyed by full session match
  annotation   - Praat/SFS-style annotation companions
                 (.textgrid, .label, .anw, .rlt) keyed by full session match
  metadata     - small admin docs (.doc, .docx, .xls, .xlsx, .sav, .pdf,
                 .csv) that mention the speaker_id only — typically
                 demographics, severity ratings, recording logs

User policy: "Egress anything plausible, filter later." So we emit a row for
every candidate we can plausibly tie to a SLASS session. Ambiguity is
quantified by the ``match_basis`` column rather than filtered out.

Usage:
    python src/data/archive_match.py \\
        --archive /Volumes/FATSPEECH/store_metadata/archive_inventory.csv \\
        --slass /Volumes/FATSPEECH/slass/full_archive/inventory.csv \\
        --out /Volumes/FATSPEECH/store_metadata/archive_match_manifest.csv \\
        --summary /Volumes/FATSPEECH/store_metadata/archive_match_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

# Match e.g. 0050_7_11y6m, 0014_10y10m, 0017_8y9m, M_22_15y4m
KEY_RE = re.compile(r"(?<!\d)(\d{2,4})(?:_(\d{1,2}))?_(\d{1,2})y(\d{1,2})m", re.I)

# Speaker_id with mandatory gender prefix: f_0050, M0017, F0014, m_22 ...
# This is the strict filter we use for metadata-style egress so that
# random 3-4 digit numbers (millisecond constants, sample counts) don't
# match.
PREFIXED_ID_RE = re.compile(r"(?<![a-z0-9])[fmFM]_?(\d{2,4})(?!\d)")

TRANSCRIPT_EXTS = {"txt", "doc", "docx", "rpt", "rtf", "htm", "html",
                   "salt", "pdf", "cha"}
ANNOTATION_EXTS = {"textgrid", "label", "anw", "rlt", "dot"}
METADATA_EXTS = {"doc", "docx", "xls", "xlsx", "sav", "pdf", "csv"}

# Parent-dir keywords that boost confidence for transcript/annotation matches
TRANSCRIPT_HINT_RE = re.compile(
    r"transcrip|orthograph|salt|stutter\s*sq|sq\s*bracket|annot|"
    r"clutter|recoding|recording",
    re.I,
)

# Directories we want to ignore wholesale - subprojects, code, etc.
SKIP_TOP_LEVEL = {
    "eryk_phd_data", "liam_phd_data", "ipc", "hmm", "simulator",
    "ssi manuals", "instruction manuals", "papers-store",
    "uci templates", "vtdemo", "movie files", "multimedia",
    "ucl templates", "wsjcam0", "timit",
}

SKIP_PATH_RE = re.compile(
    r"\\Software\\|\\C_Code\\|\\CIncludes\\|HTKDemo|"
    r"WWWT_Scanna\\|WWWTFastSlow|"
    r"My EndNote Library|"
    r"\\Movie files\\|\\Multimedia\\",
    re.I,
)

# Filename / parent_dir hints that suggest a project-level roster, master
# list, or speaker-mapping file. Used in addition to per-session matches
# so that lookup CSVs/XLSs containing many speaker IDs in their contents
# (but not in their filenames) get egressed too.
MASTER_NAME_HINT_RE = re.compile(
    r"(speaker|subject|demograph|roster|master|cohort|inventor|matrix|"
    r"metadata|participant|severity|sample[_\s]?list|info[_\s]?sheet|"
    r"lookup|database|f_files|m_files|cd[_\s]?list|all[_\s]?part)",
    re.I,
)
MASTER_EXTS = {"xls", "xlsx", "csv", "sav", "doc", "docx", "pdf"}
MASTER_SESSION_SENTINEL = "__GLOBAL__"


def session_key(stem: str) -> tuple[str, int, int] | None:
    """Return (speaker_id, age_y, age_m) parsed from a basename stem.

    Returns None when no canonical triple is recognisable. The optional
    middle digit group (study number, e.g. the ``7`` in ``f_0050_7_11y6m``)
    is ignored when forming the key so it matches across variants that
    omit it.
    """
    m = KEY_RE.search(stem)
    if not m:
        return None
    speaker_id, _study, y, mo = m.groups()
    return speaker_id.zfill(4), int(y), int(mo)


def speaker_id_from(text: str) -> str | None:
    """Return a 4-digit zero-padded speaker_id parsed from text.

    Only matches numbers carrying a gender prefix (f_/m_/F_/M_) so that
    we do not confuse random 3-4 digit numbers (millisecond constants,
    sample counts) for speaker IDs.
    """
    m = PREFIXED_ID_RE.search(text)
    return m.group(1).zfill(4) if m else None


def classify(ext: str, parent_dir: str, basename_match: bool) -> list[str]:
    """Return the list of categories the candidate belongs to."""
    cats: list[str] = []
    has_transcript_hint = bool(TRANSCRIPT_HINT_RE.search(parent_dir))
    if ext in TRANSCRIPT_EXTS and (basename_match or has_transcript_hint):
        cats.append("transcript")
    if ext in ANNOTATION_EXTS and basename_match:
        cats.append("annotation")
    if ext in METADATA_EXTS and not basename_match:
        cats.append("metadata")
    return cats


def load_archive(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            full = row["full_path"]
            if not full or full.endswith("..."):
                continue
            top = row["top_level"].lower()
            if top in SKIP_TOP_LEVEL:
                continue
            if SKIP_PATH_RE.search(full):
                continue
            ext = row["ext"].lower()
            if not ext:
                continue
            try:
                row["size_mb_f"] = float(row["size_mb"])
            except ValueError:
                row["size_mb_f"] = 0.0
            rows.append(row)
    return rows


def index_archive(rows: list[dict]) -> tuple[
    dict[tuple[str, int, int], list[dict]],
    dict[str, list[dict]],
]:
    """Build two indexes for fast lookup.

    by_session: canonical key -> archive rows whose basename matches.
    by_speaker: speaker_id -> rows whose basename or parent_dir mentions
                that id but no full canonical key matched. Used for
                metadata-style egress.
    """
    by_session: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        stem = row["stem"]
        parent = row["parent_dir"]
        key = session_key(stem)
        if key is not None:
            by_session[key].append(row)
        sid = speaker_id_from(stem) or speaker_id_from(parent)
        if sid is not None:
            by_speaker[sid].append(row)
    return by_session, by_speaker


def slass_speaker_id_from(stem: str) -> str | None:
    """Speaker_id extractor for SLASS session basenames.

    More permissive than ``speaker_id_from`` because SLASS extraction
    filenames often have no gender prefix (e.g. ``0014_10y10m_conv``,
    ``0050c116``, ``0017c105``). We accept a 3- or 4-digit number at the
    start of the stem, optionally preceded by ``C_F_``/``C_M_``/``F_``/
    ``M_``/``f_``/``m_`` prefixes.
    """
    m = re.match(r"^(?:[CcFfMm]_)*(\d{3,4})(?!\d)", stem)
    return m.group(1).zfill(4) if m else None


def load_slass_sessions(path: Path) -> list[dict]:
    """Load our existing SLASS extraction inventory and add canonical keys.

    Speaker_id for a SLASS session is derived from the canonical session
    key when present (the leading digit group). Otherwise we fall back to
    a leading-digit extractor so basenames like ``0050c116`` still get
    tied to speaker ``0050`` for metadata + roster matching.
    """
    out = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stem = row.get("filename") or row.get("session_id") or ""
            stem = stem.rsplit(".", 1)[0] if stem.endswith(".sfs") else stem
            key = session_key(stem)
            if key is not None:
                sid = key[0]
            else:
                sid = slass_speaker_id_from(stem)
            out.append({
                "filename": row.get("filename") or row.get("session_id"),
                "stem": stem,
                "session_key": key,
                "speaker_id": sid,
                "row": row,
            })
    return out


def build_master_metadata_rows(archive: list[dict]) -> list[dict]:
    """Emit one manifest row per project-level roster / master-list file.

    These files are not tied to a single SLASS session — they typically
    map many speaker IDs to demographics, severity ratings, recording
    inventories, etc. We tag them with the GLOBAL sentinel so they get
    pulled exactly once during egress.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for row in archive:
        ext = row["ext"].lower()
        if ext not in MASTER_EXTS:
            continue
        if not (MASTER_NAME_HINT_RE.search(row["stem"])
                or MASTER_NAME_HINT_RE.search(row["parent_dir"])):
            continue
        if row["full_path"] in seen:
            continue
        seen.add(row["full_path"])
        rows.append({
            "slass_session": MASTER_SESSION_SENTINEL,
            "slass_speaker_id": "",
            "slass_key": "",
            "src_path": row["full_path"],
            "src_size_mb": row["size_mb_f"],
            "src_ext": row["ext"],
            "src_top_level": row["top_level"],
            "src_parent_dir": row["parent_dir"],
            "src_basename": row["basename"],
            "category": "master_metadata",
            "match_basis": "name_hint",
        })
    return rows


def build_manifest(
    slass_sessions: list[dict],
    by_session: dict[tuple[str, int, int], list[dict]],
    by_speaker: dict[str, list[dict]],
) -> list[dict]:
    """One row per (slass_session, archive_file, category) candidate.

    Category precedence per file:
      transcript or annotation (basename matches session)  >
      transcript_other_session (transcript file for same speaker_id but
                                different age — useful context, not a
                                direct match)  >
      metadata (admin / demographics by speaker_id)

    Each file is emitted under at most one category per session.
    """
    manifest: list[dict] = []
    seen: set[tuple[str, str]] = set()  # (slass_session, src_path)

    for s in slass_sessions:
        sname = s["filename"]
        key = s["session_key"]
        sid = s["speaker_id"]

        # Pass 1: session-key matches (strongest evidence). One file may be
        # classified as transcript and/or annotation here.
        session_matched_paths: set[str] = set()
        if key is not None:
            for row in by_session.get(key, []):
                cats = classify(row["ext"], row["parent_dir"],
                                basename_match=True)
                for cat in cats:
                    k = (sname, row["full_path"])
                    if k in seen:
                        continue
                    seen.add(k)
                    session_matched_paths.add(row["full_path"])
                    manifest.append({
                        "slass_session": sname,
                        "slass_speaker_id": sid or "",
                        "slass_key": ("{}_{}y{}m".format(*key)
                                      if key else ""),
                        "src_path": row["full_path"],
                        "src_size_mb": row["size_mb_f"],
                        "src_ext": row["ext"],
                        "src_top_level": row["top_level"],
                        "src_parent_dir": row["parent_dir"],
                        "src_basename": row["basename"],
                        "category": cat,
                        "match_basis": "session+basename",
                    })

        # Pass 2: speaker_id-only matches. Transcripts at wrong-age get a
        # distinct category so the user can filter them out cheaply.
        if sid is not None:
            for row in by_speaker.get(sid, []):
                if row["full_path"] in session_matched_paths:
                    continue  # already linked via stronger basis
                k = (sname, row["full_path"])
                if k in seen:
                    continue
                ext = row["ext"]
                in_transcript_dir = bool(
                    TRANSCRIPT_HINT_RE.search(row["parent_dir"]))
                if ext in TRANSCRIPT_EXTS and in_transcript_dir:
                    cat = "transcript_other_session"
                elif ext in METADATA_EXTS:
                    cat = "metadata"
                else:
                    continue  # not interesting
                seen.add(k)
                manifest.append({
                    "slass_session": sname,
                    "slass_speaker_id": sid or "",
                    "slass_key": ("{}_{}y{}m".format(*key)
                                  if key else ""),
                    "src_path": row["full_path"],
                    "src_size_mb": row["size_mb_f"],
                    "src_ext": row["ext"],
                    "src_top_level": row["top_level"],
                    "src_parent_dir": row["parent_dir"],
                    "src_basename": row["basename"],
                    "category": cat,
                    "match_basis": "speaker_id",
                })
    return manifest


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["slass_session", "slass_speaker_id", "slass_key",
            "src_path", "src_size_mb", "src_ext", "src_top_level",
            "src_parent_dir", "src_basename", "category", "match_basis"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(path: Path, slass_sessions: list[dict],
                  manifest: list[dict]) -> None:
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    sizes: dict[str, float] = defaultdict(float)
    for r in manifest:
        counts[r["slass_session"]][r["category"]] += 1
        sizes[r["slass_session"]] += r["src_size_mb"]

    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["slass_session", "slass_speaker_id", "slass_key",
            "n_transcript", "n_annotation",
            "n_transcript_other_session", "n_metadata",
            "n_master_metadata", "total_mb"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in slass_sessions:
            sname = s["filename"]
            c = counts.get(sname, {})
            w.writerow({
                "slass_session": sname,
                "slass_speaker_id": s["speaker_id"] or "",
                "slass_key": ("{}_{}y{}m".format(*s["session_key"])
                              if s["session_key"] else ""),
                "n_transcript": c.get("transcript", 0),
                "n_annotation": c.get("annotation", 0),
                "n_transcript_other_session":
                    c.get("transcript_other_session", 0),
                "n_metadata": c.get("metadata", 0),
                "n_master_metadata": 0,
                "total_mb": f"{sizes.get(sname, 0.0):.2f}",
            })
        # Global row for project-level masters: not tied to any session,
        # listed at the bottom of the summary so it's easy to find.
        global_c = counts.get(MASTER_SESSION_SENTINEL, {})
        if global_c:
            w.writerow({
                "slass_session": MASTER_SESSION_SENTINEL,
                "slass_speaker_id": "",
                "slass_key": "",
                "n_transcript": 0,
                "n_annotation": 0,
                "n_transcript_other_session": 0,
                "n_metadata": 0,
                "n_master_metadata": global_c.get("master_metadata", 0),
                "total_mb": (f"{sizes.get(MASTER_SESSION_SENTINEL, 0.0):.2f}"),
            })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, required=True,
                    help="archive_inventory.csv from archive_inventory.py")
    ap.add_argument("--slass", type=Path, required=True,
                    help="existing SLASS extraction inventory CSV")
    ap.add_argument("--out", type=Path, required=True,
                    help="output manifest CSV")
    ap.add_argument("--summary", type=Path, required=True,
                    help="output per-session summary CSV")
    args = ap.parse_args()

    archive = load_archive(args.archive)
    print(f"Loaded {len(archive):,} candidate files from {args.archive}")
    by_session, by_speaker = index_archive(archive)
    print(f"  {len(by_session):,} canonical session keys, "
          f"{len(by_speaker):,} speaker_ids indexed")

    slass = load_slass_sessions(args.slass)
    print(f"Loaded {len(slass):,} SLASS sessions from {args.slass}")
    with_key = sum(1 for s in slass if s["session_key"])
    with_sid = sum(1 for s in slass if s["speaker_id"])
    print(f"  {with_key} have canonical session key, "
          f"{with_sid} have at least speaker_id")

    manifest = build_manifest(slass, by_session, by_speaker)
    master = build_master_metadata_rows(archive)
    manifest.extend(master)
    print(f"Built {len(manifest):,} candidate egress rows "
          f"({len(master):,} project-level master metadata files)")

    write_manifest(args.out, manifest)
    write_summary(args.summary, slass, manifest)
    print(f"Wrote {args.out} and {args.summary}")


if __name__ == "__main__":
    main()
