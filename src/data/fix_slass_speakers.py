#!/usr/bin/env python3
"""Repair the curated SLASS speakers.csv (gender garbage + missing ages).

The original speakers.csv mis-parsed the name-based fluent-control
sessions: the `gender` column holds the name/id token (e.g. "1013",
"489 SFS", "ANNAMCMANUS") instead of M/F, and ages came out blank even
though the session filenames contain an age token (e.g.
`rachelg_161_9y1m_m`).

This rebuilds speakers.csv by aggregating the per-session inventory and
RE-DERIVING gender and age from each session_id:
  * gender  from a leading f_/m_/F_/M_ prefix; else "U" (unknown), with
            the original name token kept in a new `name_token` column.
  * age     from a `(\\d+)y(\\d+)m` token anywhere in the session_id.

The original is preserved as speakers.csv.bak. A `shared_numeric_id`
column flags speakers that may be fragments of the same person (same
embedded participant number under different name spellings) for manual
review — not auto-merged.

Usage:
    python -m src.data.fix_slass_speakers
"""

from __future__ import annotations

import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

PROC = Path("/Volumes/FATSPEECH/slass/processed")
INV = PROC / "inventory.csv"
SPK = PROC / "speakers.csv"

AGE_RE = re.compile(r"(\d{1,2})y(\d{1,2})m", re.I)
NUM_RE = re.compile(r"(\d{3,4})")


def derive_gender(session_id: str, speaker_id: str) -> tuple[str, str]:
    """Return (gender, name_token). gender in {M,F,U}."""
    for src in (session_id, speaker_id):
        s = src.strip().lower()
        if s.startswith("f_") or s.startswith("f "):
            return "F", ""
        if s.startswith("m_") or s.startswith("m "):
            return "M", ""
    # No f_/m_ prefix -> name-based control; gender unknown, keep the token.
    token = speaker_id.split("_")[0] if "_" in speaker_id else speaker_id
    return "U", token


def derive_age_months(session_id: str) -> int | None:
    m = AGE_RE.search(session_id)
    return int(m.group(1)) * 12 + int(m.group(2)) if m else None


def main() -> None:
    if not INV.exists():
        print(f"inventory not found: {INV}")
        return
    rows = [r for r in csv.DictReader(open(INV))
            if not r.get("session_id", "").startswith("._")]

    by_speaker: dict[str, dict] = {}
    sessions_by_speaker: dict[str, list] = defaultdict(list)
    for r in rows:
        sid = r.get("speaker_id", "").strip()
        if not sid:
            continue
        sessions_by_speaker[sid].append(r)

    out = []
    for sid, sess in sorted(sessions_by_speaker.items()):
        genders, names, ages, groups, bands = set(), set(), [], set(), set()
        for r in sess:
            g, name = derive_gender(r.get("session_id", ""), sid)
            genders.add(g)
            if name:
                names.add(name)
            a = derive_age_months(r.get("session_id", ""))
            if a:
                ages.append(a)
            if r.get("fluency_group"):
                groups.add(r["fluency_group"])
            if r.get("age_band"):
                bands.add(r["age_band"])
        # Resolve gender: prefer a definite M/F over U.
        gender = ("M" if "M" in genders else
                  "F" if "F" in genders else "U")
        num = NUM_RE.search(sid)
        out.append({
            "speaker_id": sid,
            "gender": gender,
            "name_token": ";".join(sorted(names)),
            "fluency_group": ";".join(sorted(groups)),
            "n_sessions": len(sess),
            "min_age_months": min(ages) if ages else "",
            "max_age_months": max(ages) if ages else "",
            "age_bands": "+".join(sorted(bands)),
            "shared_numeric_id": num.group(1) if num else "",
        })

    # Flag shared numeric ids (possible fragment duplicates).
    num_counts = defaultdict(int)
    for o in out:
        if o["shared_numeric_id"]:
            num_counts[o["shared_numeric_id"]] += 1
    for o in out:
        nid = o["shared_numeric_id"]
        o["shared_numeric_id"] = nid if (nid and num_counts[nid] > 1) else ""

    # Back up and write.
    if SPK.exists():
        shutil.copy2(SPK, SPK.with_suffix(".csv.bak"))
    cols = ["speaker_id", "gender", "name_token", "fluency_group",
            "n_sessions", "min_age_months", "max_age_months", "age_bands",
            "shared_numeric_id"]
    with SPK.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    from collections import Counter
    gc = Counter(o["gender"] for o in out)
    n_age = sum(1 for o in out if o["min_age_months"] != "")
    n_dup = sum(1 for o in out if o["shared_numeric_id"])
    print(f"Rebuilt {SPK} ({len(out)} speakers; backup at {SPK}.bak)")
    print(f"  gender: {dict(gc)}  (was M=40 F=11 + ~11 garbage)")
    print(f"  with age now: {n_age}/{len(out)} (was 51/62)")
    print(f"  flagged shared_numeric_id (possible fragment dups): {n_dup}")


if __name__ == "__main__":
    main()
