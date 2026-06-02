#!/usr/bin/env python3
"""Organise the newly-egressed SLASS supplementary data into a clean layout.

Two drops arrived from the speech-lab store:

  /Volumes/FATSPEECH/slass_additions/   metadata: transcripts, per-speaker
                                        docs, master rosters (526 MB)
  /Volumes/FATSPEECH/slass/extra_audio/ extra audio formats beyond the
                                        wav/sfs already held (20 GB)

Policy (agreed with Pete / Liam, 2026-06-02):
  * Additional AUDIO is held SEPARATE from the base SLASS dataset so the
    refined analyses run on the same base. It moves to a clearly-labelled
    top-level tree:
        /Volumes/FATSPEECH/slass_extended/audio/        (participant audio)
        /Volumes/FATSPEECH/slass_extended/out_of_scope/ (.au stimuli etc.)
  * Transcripts and master rosters MAY enrich the base, so they move
    base-adjacent (not held separate):
        /Volumes/FATSPEECH/slass/supplementary/transcripts/
        /Volumes/FATSPEECH/slass/supplementary/speaker_metadata/
        /Volumes/FATSPEECH/slass/supplementary/master_rosters/

This script moves files (same-volume renames, instant), strips macOS
resource forks, and writes two index CSVs mapping each file to a SLASS
speaker id where parseable. It is non-destructive beyond the moves and
can be dry-run first.

Usage:
    python src/data/organise_slass_extended.py --dry-run
    python src/data/organise_slass_extended.py
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import Counter
from pathlib import Path

FAT = Path("/Volumes/FATSPEECH")
ADDITIONS = FAT / "slass_additions"
EXTRA_AUDIO = FAT / "slass" / "extra_audio"

EXTENDED = FAT / "slass_extended"
EXT_AUDIO = EXTENDED / "audio"
EXT_OOS = EXTENDED / "out_of_scope"
EXT_INDEX = EXTENDED / "index"

SUPP = FAT / "slass" / "supplementary"
SUPP_TRANS = SUPP / "transcripts"
SUPP_SPK = SUPP / "speaker_metadata"
SUPP_ROST = SUPP / "master_rosters"
SUPP_INDEX = SUPP / "index"

BASE_INV = FAT / "slass" / "full_archive" / "inventory.csv"

# Extensions treated as out-of-scope stimulus audio (not participant
# recordings). Everything else in extra_audio is participant audio.
OUT_OF_SCOPE_EXTS = {"au"}

ID_RE = re.compile(r"(?<![0-9])([fmFM])_?(\d{3,4})(?![0-9])")
ANYID_RE = re.compile(r"(?<![0-9])(\d{3,4})(?![0-9])")


def load_base_ids() -> set[str]:
    ids: set[str] = set()
    if not BASE_INV.exists():
        return ids
    with BASE_INV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m = re.match(r"^[CcFfMm_]*(\d{3,4})", r["filename"].rsplit(".", 1)[0])
            if m:
                ids.add(m.group(1).zfill(4))
    return ids


def speaker_id_from(name: str, parent: str) -> tuple[str, str]:
    """Return (speaker_id, source) where source is 'name'/'parent'/''."""
    m = ID_RE.search(name)
    if m:
        return m.group(2).zfill(4), "name"
    m = ID_RE.search(parent)
    if m:
        return m.group(2).zfill(4), "parent"
    return "", ""


def clean_resource_forks(root: Path) -> int:
    n = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        if p.name.startswith("._") or p.name == ".DS_Store":
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
    return n


def rel_under(path: Path, root: Path) -> Path:
    return path.relative_to(root)


def move_file(src: Path, dest: Path, dry_run: bool) -> bool:
    """Move src -> dest. Returns True if a move happened this call.

    Robust to two complications on the exFAT volume:
      * macOS uchg (immutable) flags that block unlink — cleared first.
      * a previously interrupted run that copied to dest but failed to
        remove src (leaving a duplicate). On re-run, if dest already
        exists and matches src by size, we remove the leftover src.
    """
    if dest.exists():
        if src.exists() and src.resolve() != dest.resolve() and not dry_run:
            try:
                if src.stat().st_size == dest.stat().st_size:
                    _clear_uchg(src)
                    src.unlink()
            except OSError:
                pass
        return False
    if dry_run:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    _clear_uchg(src)
    shutil.move(str(src), str(dest))
    return True


def _clear_uchg(path: Path) -> None:
    """Best-effort clear of the macOS user-immutable flag before a move."""
    try:
        import os
        st = path.stat()
        flags = getattr(st, "st_flags", 0)
        UF_IMMUTABLE = 0x00000002
        if flags & UF_IMMUTABLE:
            os.chflags(path, flags & ~UF_IMMUTABLE)
    except (OSError, AttributeError):
        pass


def load_metadata_categories() -> dict[str, str]:
    """Map relative-path -> category from the additions egress log.

    The egress log records src_path like 'X:\\Speech\\Alan\\foo.txt' and a
    pipe-joined categories field. We key on the windows-relative path
    (forward-slashed, lowercased) so we can route the local file.
    """
    log = ADDITIONS / "_egress_log.csv"
    out: dict[str, str] = {}
    if not log.exists():
        return out
    with log.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            src = r.get("src_path", "")
            cats = (r.get("categories") or "").split("|")
            # Precedence: transcript > master_metadata > metadata
            if "transcript" in cats or "transcript_other_session" in cats:
                cat = "transcript"
            elif "master_metadata" in cats:
                cat = "master_rosters"
            elif "metadata" in cats:
                cat = "speaker_metadata"
            else:
                cat = "speaker_metadata"
            # windows rel path under X:\Speech
            low = src.lower()
            marker = "x:\\speech\\"
            rel = src[len(marker):] if low.startswith(marker) else src
            rel = rel.replace("\\", "/").lower()
            out[rel] = cat
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run
    tag = "[dry-run] " if dry else ""

    base_ids = load_base_ids()
    print(f"{tag}Base SLASS speaker ids: {len(base_ids)}")

    # ── Housekeeping: resource forks ──
    if not dry:
        n1 = clean_resource_forks(ADDITIONS)
        n2 = clean_resource_forks(EXTRA_AUDIO)
        print(f"Stripped {n1 + n2} resource-fork / .DS_Store files")

    # ── Audio routing ──
    moved_audio = moved_oos = 0
    if EXTRA_AUDIO.exists():
        for p in sorted(EXTRA_AUDIO.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name.startswith("_"):
                continue  # logs, hidden
            ext = p.suffix.lstrip(".").lower()
            rel = rel_under(p, EXTRA_AUDIO)
            oos = ext in OUT_OF_SCOPE_EXTS
            dest_root = EXT_OOS if oos else EXT_AUDIO
            if move_file(p, dest_root / rel, dry):
                if oos:
                    moved_oos += 1
                else:
                    moved_audio += 1

    # ── Metadata routing ──
    cat_map = load_metadata_categories()
    moved_meta = Counter()
    if ADDITIONS.exists():
        for p in sorted(ADDITIONS.rglob("*")):
            if not p.is_file():
                continue
            if p.name.startswith(".") or p.name == "_egress_log.csv":
                continue
            rel = rel_under(p, ADDITIONS)
            rel_key = str(rel).replace("\\", "/").lower()
            cat = cat_map.get(rel_key, "speaker_metadata")
            dest_root = {
                "transcript": SUPP_TRANS,
                "master_rosters": SUPP_ROST,
                "speaker_metadata": SUPP_SPK,
            }[cat]
            if move_file(p, dest_root / rel, dry):
                moved_meta[cat] += 1

    if dry:
        print(f"\n{tag}(dry-run: indexes built from destinations after a "
              f"real run)")
        return

    # Preserve the egress / copy logs as provenance in the new trees.
    for log_src, log_dest in (
        (ADDITIONS / "_egress_log.csv", SUPP / "_egress_log.csv"),
        (EXTRA_AUDIO / "_audio_copy_log.csv",
         EXTENDED / "_audio_copy_log.csv"),
    ):
        if log_src.exists() and not log_dest.exists():
            log_dest.parent.mkdir(parents=True, exist_ok=True)
            _clear_uchg(log_src)
            shutil.move(str(log_src), str(log_dest))

    # ── Build indexes from the DESTINATION trees (robust to re-runs) ──
    audio_index = []
    for p in sorted(EXT_AUDIO.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        rel = rel_under(p, EXT_AUDIO)
        sid, src = speaker_id_from(p.name, str(rel.parent))
        if sid and sid in base_ids:
            status = "known_base_speaker"
        elif sid:
            status = "new_speaker"
        else:
            status = "unresolved"
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        audio_index.append({
            "rel_path": str(rel),
            "top_level": rel.parts[0] if rel.parts else "",
            "ext": p.suffix.lstrip(".").lower(),
            "size_bytes": size,
            "speaker_id": sid,
            "id_source": src,
            "id_status": status,
            "held_separate": "yes",
        })

    meta_index = []
    for cat, root in (("transcript", SUPP_TRANS),
                      ("master_rosters", SUPP_ROST),
                      ("speaker_metadata", SUPP_SPK)):
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            rel = rel_under(p, root)
            sid, src = speaker_id_from(p.name, str(rel.parent))
            meta_index.append({
                "rel_path": str(rel),
                "category": cat,
                "top_level": rel.parts[0] if rel.parts else "",
                "ext": p.suffix.lstrip(".").lower(),
                "speaker_id": sid,
                "id_source": src,
                "in_base": "yes" if (sid and sid in base_ids) else "",
            })

    EXT_INDEX.mkdir(parents=True, exist_ok=True)
    SUPP_INDEX.mkdir(parents=True, exist_ok=True)
    with (EXT_INDEX / "extended_audio_index.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "rel_path", "top_level", "ext", "size_bytes",
            "speaker_id", "id_source", "id_status", "held_separate"])
        w.writeheader()
        w.writerows(audio_index)
    with (SUPP_INDEX / "supplementary_metadata_index.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "rel_path", "category", "top_level", "ext",
            "speaker_id", "id_source", "in_base"])
        w.writeheader()
        w.writerows(meta_index)

    # ── Summary ──
    print(f"\n{tag}Audio:")
    print(f"  participant audio -> slass_extended/audio/   : {moved_audio}")
    print(f"  out-of-scope .au  -> slass_extended/out_of_scope/ : {moved_oos}")
    if audio_index:
        st = Counter(r["id_status"] for r in audio_index)
        print(f"  id status: {dict(st)}")
    print(f"\n{tag}Metadata -> slass/supplementary/:")
    for k, v in moved_meta.most_common():
        print(f"  {k}: {v}")
    print(f"\n{tag}Index rows: audio={len(audio_index)}, "
          f"metadata={len(meta_index)}")


if __name__ == "__main__":
    main()
