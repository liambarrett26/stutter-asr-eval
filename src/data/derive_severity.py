#!/usr/bin/env python3
"""Derive per-session stuttering-rate severity across corpora.

Stuttering rate (SR) = disfluent units / total units, expressed as a
percentage, binned with the project's clinical thresholds:
    mild      SR <= 7%
    moderate  7% < SR <= 12%
    severe    SR > 12%

Sources (best available per corpus):
  * SLASS (Jason matrices)   word-level Fluency=Stuttered / total words
                             (gold; also the H2 base)
  * SLASS standardised       disfluent stutter_type / total rows
  * UCLASS standardised       disfluent stutter_type / total rows
  * FluencyBank               no per-word stutter labels -> SR not derivable
                             here (flagged; would need CHAT disfluency parse)
  * UNWR                       SSI % already measured (carried through)

Output:
  /Volumes/FATSPEECH/standardised/severity_index.csv
  columns: corpus, session, speaker_id, source, n_units, n_disfluent,
           stuttering_rate_pct, severity

Usage:
    python -m src.data.derive_severity
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

STD = Path("/Volumes/FATSPEECH/standardised")
JASON = Path("/Volumes/SPEECH/from_jason/Speech data Jason/output")
OUT = STD / "severity_index.csv"

DISFLUENT = lambda s: bool(s) and s.strip().lower() not in ("", "fluent", "unknown")
SPK_RE = re.compile(r"(\d{3,4})")


def band(sr: float) -> str:
    return "mild" if sr <= 7 else ("moderate" if sr <= 12 else "severe")


def speaker_of(session: str) -> str:
    m = SPK_RE.search(session)
    return m.group(1).zfill(4) if m else ""


def from_jason(rows_out: list) -> None:
    for f in sorted(JASON.glob("*_Usage_Matrix.csv")):
        session = f.stem.replace("_Usage_Matrix", "")
        rows = list(csv.DictReader(open(f, encoding="latin-1"), delimiter="\t"))
        tot = sum(1 for r in rows if (r.get("Word", "") or "").strip())
        stut = sum(1 for r in rows if r.get("Fluency") == "Stuttered")
        if tot < 20:           # too short to rate reliably
            continue
        sr = stut / tot * 100
        rows_out.append(["slass", session, speaker_of(session), "jason_words",
                         tot, stut, round(sr, 2), band(sr)])


def from_standardised(corpus: str, rows_out: list) -> None:
    d = STD / corpus
    if not d.exists():
        return
    for f in d.rglob("*.csv"):
        if f.name.startswith(".") or f.name.startswith("_"):
            continue
        text = f.read_bytes().replace(b"\x00", b"").decode("latin-1")
        rows = list(csv.DictReader(text.splitlines()))
        tot = sum(1 for r in rows if (r.get("text_intended", "")
                  or r.get("text_surface", "") or "").strip())
        if tot < 20:
            continue
        stut = sum(1 for r in rows if DISFLUENT(r.get("stutter_type")))
        sr = stut / tot * 100
        rows_out.append([corpus, f.stem, speaker_of(f.stem),
                         f"{corpus}_standardised", tot, stut,
                         round(sr, 2), band(sr)])


def from_fluencybank(rows_out: list) -> None:
    """FluencyBank severity from the CHAT disfluency index (fb_disfluency.py)."""
    p = Path("/Volumes/FATSPEECH/fluencybank/processed/disfluency_index.csv")
    if not p.exists():
        return
    for r in csv.DictReader(open(p)):
        try:
            rate = float(r.get("disfluency_rate_pct", "") or "")
        except ValueError:
            continue
        n = r.get("n_surface_words", "")
        events = (int(r.get("repetition", 0) or 0) + int(r.get("retrace", 0) or 0)
                  + int(r.get("filled_pause", 0) or 0)
                  + int(r.get("fragment", 0) or 0))
        rows_out.append(["fluencybank", f"{r['corpus']}/{r['session']}",
                         "", "fb_chat_disfluency", n, events,
                         round(rate, 2), r.get("severity", band(rate))])


def from_unwr(rows_out: list) -> None:
    p = Path("/Volumes/FATSPEECH/unwr/processed/speakers.csv")
    if not p.exists():
        return
    for r in csv.DictReader(open(p)):
        pct = (r.get("ssi_pct", "") or "").strip()
        if not pct:
            continue
        try:
            sr = float(pct)
        except ValueError:
            continue
        rows_out.append(["unwr", r.get("pid", ""), r.get("pid", ""),
                         "unwr_ssi", r.get("ssi_syllable_count", ""),
                         r.get("ssi_stuttered_syllables", ""),
                         round(sr, 2), band(sr)])


def main() -> None:
    rows_out = []
    from_jason(rows_out)
    from_standardised("slass_full", rows_out)
    from_standardised("uclass", rows_out)
    from_fluencybank(rows_out)
    from_unwr(rows_out)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["corpus", "session", "speaker_id", "source",
                    "n_units", "n_disfluent", "stuttering_rate_pct",
                    "severity"])
        w.writerows(rows_out)

    # Summary
    from collections import Counter
    by_src = Counter(r[3] for r in rows_out)
    by_band = Counter(r[7] for r in rows_out)
    print(f"Wrote {len(rows_out)} session/speaker severity rows -> {OUT}")
    print("by source:", dict(by_src))
    print("by severity:", dict(by_band))
    # SLASS jason severity spread (the H2/H3 gold)
    jason = [r for r in rows_out if r[3] == "jason_words"]
    sev = Counter(r[7] for r in jason)
    print(f"SLASS (jason gold) severity: {dict(sev)} over {len(jason)} sessions")
    fb = [r for r in rows_out if r[3] == "fb_chat_disfluency"]
    fbsev = Counter(r[7] for r in fb)
    print(f"FluencyBank (CHAT disfluency rate) severity: {dict(fbsev)} "
          f"over {len(fb)} sessions")


if __name__ == "__main__":
    main()
