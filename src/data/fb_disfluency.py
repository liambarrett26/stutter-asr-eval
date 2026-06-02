#!/usr/bin/env python3
"""Extract disfluency annotations from FluencyBank CHAT transcripts.

The aligned CSVs produced earlier stripped CHAT coding to clean text, so
FluencyBank carried no surface/intended distinction and no disfluency
labels. The raw .cha (inside raw/<corpus>/transcripts/<corpus>.zip) do
carry CHAT disfluency coding; this recovers it.

Per *PAR utterance we reconstruct:
  * surface  — what was produced (repetitions and fragments kept)
  * intended — the fluent target (retraced/repeated material removed,
               filled pauses and fragments removed)
and count disfluency events by CHAT type.

CHAT codes handled:
  word [/]            repetition  — the token/scope before [/] is the
                                    disfluent first production
  <scope> [/]         scoped repetition
  word [//]           retracing (revision)
  &-um / &-uh         filled pause
  &+frag / &~frag     phonological / nonword fragment (part-word)
  (.) (..) (...)      pause
  xxx / yyy / www     unintelligible / untranscribed (dropped from both)
  [: target]          replacement target (kept as intended)
  0word               omitted word (dropped)

Outputs:
  fluencybank/processed/disfluency_index.csv   per-session counts + rate
  fluencybank/processed/disfluency_text/<stem>.csv  per-utt surface+intended

Severity (disfluency rate = disfluent events / surface words) is binned
mild<=7 / moderate 7-12 / severe>12 to match the project convention.

Usage:
    python -m src.data.fb_disfluency
"""

from __future__ import annotations

import csv
import re
import zipfile
from collections import Counter
from pathlib import Path

RAW = Path("/Volumes/FATSPEECH/fluencybank/raw")
PROC = Path("/Volumes/FATSPEECH/fluencybank/processed")
CHA_DIR = PROC / "cha"
OUT_INDEX = PROC / "disfluency_index.csv"
OUT_TEXT = PROC / "disfluency_text"

TS_RE = re.compile(r"\d+_\d+\s*$")           # trailing media timestamp
REPEAT = "[/]"
RETRACE = "[//]"


def extract_cha() -> int:
    """Unzip each corpus transcript zip into processed/cha/<corpus>/."""
    CHA_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for zp in RAW.rglob("*/transcripts/*.zip"):
        corpus = zp.parts[-3]
        dest = CHA_DIR / corpus
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zp) as z:
                z.extractall(dest)
        except zipfile.BadZipFile:
            continue
        n += 1
    # strip resource forks
    for p in CHA_DIR.rglob("._*"):
        p.unlink(missing_ok=True)
    return n


def clean_token(tok: str) -> str:
    """Strip CHAT word-internal annotation, return the lexical form or ''."""
    tok = tok.strip()
    if not tok:
        return ""
    if tok.startswith("0"):                  # omitted word
        return ""
    if tok in ("xxx", "yyy", "www"):         # unintelligible
        return ""
    tok = tok.lstrip("&")                    # &-um, &+fr, &~ -> um/fr
    tok = tok.lstrip("+-~")
    tok = re.sub(r"[^A-Za-z'’]", "", tok)
    return tok.lower()


def parse_utterance(text: str) -> tuple[list[str], list[str], Counter]:
    """Return (surface_tokens, intended_tokens, event_counts)."""
    text = TS_RE.sub("", text).strip()
    # Normalise scope groups: keep the words, mark group boundaries so a
    # following [/] can drop the whole group from the intended form.
    text = text.replace("<", " <").replace(">", "> ")
    raw = text.split()

    surface: list[str] = []
    intended: list[str] = []
    events: Counter = Counter()

    # Build a token stream where each item is (kind, payload):
    #   kind 'word'  -> a lexical token
    #   kind 'group' -> a list of lexical tokens (was <...>)
    #   kind 'rep'/'ret' -> repetition/retrace marker (acts on prev item)
    #   kind 'fill'/'frag'/'pause'/'drop'
    stream: list[tuple[str, object]] = []
    buf: list[str] | None = None
    for t in raw:
        if t.startswith("<"):
            buf = [clean_token(t[1:])] if clean_token(t[1:]) else []
            continue
        if t.endswith(">"):
            if buf is not None:
                w = clean_token(t[:-1])
                if w:
                    buf.append(w)
                stream.append(("group", buf))
                buf = None
            continue
        if t == REPEAT:
            stream.append(("rep", None)); continue
        if t == RETRACE or t == "[///]":
            stream.append(("ret", None)); continue
        if t.startswith("[:"):
            stream.append(("target", clean_token(t.strip("[]:")))); continue
        if t.startswith("[") or t.endswith("]"):
            continue                                   # other scoped codes
        if t.startswith("(.") and t.endswith(")"):
            stream.append(("pause", None)); continue
        if t.startswith("&-"):
            w = clean_token(t)
            stream.append(("fill", w)); continue
        if t.startswith("&+") or t.startswith("&~"):
            w = clean_token(t)
            stream.append(("frag", w)); continue
        w = clean_token(t)
        if buf is not None:
            if w:
                buf.append(w)
        elif w:
            stream.append(("word", w))

    # Walk the stream applying repetition/retrace to the previous item.
    i = 0
    items: list[tuple[str, object]] = stream
    while i < len(items):
        kind, payload = items[i]
        nxt = items[i + 1][0] if i + 1 < len(items) else None
        if kind == "word":
            if nxt in ("rep", "ret"):
                surface.append(payload)              # produced
                events["repetition" if nxt == "rep" else "retrace"] += 1
                i += 2                               # drop marker; intended skips this copy
                continue
            surface.append(payload); intended.append(payload)
        elif kind == "group":
            words = [w for w in payload if w]
            if nxt in ("rep", "ret"):
                surface.extend(words)
                events["repetition" if nxt == "rep" else "retrace"] += 1
                i += 2
                continue
            surface.extend(words); intended.extend(words)
        elif kind == "fill":
            if payload:
                surface.append(payload)
            events["filled_pause"] += 1
        elif kind == "frag":
            if payload:
                surface.append(payload)
            events["fragment"] += 1
        elif kind == "pause":
            events["pause"] += 1
        elif kind == "target":
            if payload:
                intended.append(payload)
        i += 1

    return surface, intended, events


def band(rate: float) -> str:
    return "mild" if rate <= 7 else ("moderate" if rate <= 12 else "severe")


def main() -> None:
    n_zip = extract_cha()
    cha = [p for p in CHA_DIR.rglob("*.cha") if not p.name.startswith(".")]
    print(f"extracted {n_zip} zips, {len(cha)} .cha files")

    OUT_TEXT.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for f in sorted(cha):
        rel = f.relative_to(CHA_DIR)
        corpus = rel.parts[0]
        # Include the task subdir(s) in the stem so interview/106 and
        # reading/106 don't collide.
        stem = "_".join(rel.with_suffix("").parts[1:])
        utt_rows = []
        ev_total = Counter()
        n_surf_words = 0
        for line in f.read_text(errors="replace").splitlines():
            if not line.startswith("*PAR"):
                continue
            text = line.split(":", 1)[1] if ":" in line else line
            surf, intend, ev = parse_utterance(text)
            if not surf and not intend:
                continue
            ev_total.update(ev)
            n_surf_words += len(surf)
            utt_rows.append({
                "surface": " ".join(surf),
                "intended": " ".join(intend),
                "n_surface": len(surf),
                "n_intended": len(intend),
                "repetition": ev.get("repetition", 0),
                "retrace": ev.get("retrace", 0),
                "filled_pause": ev.get("filled_pause", 0),
                "fragment": ev.get("fragment", 0),
            })
        if not utt_rows:
            continue
        # per-session text file
        with (OUT_TEXT / f"{corpus}__{stem}.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(utt_rows[0]))
            w.writeheader(); w.writerows(utt_rows)

        n_events = (ev_total.get("repetition", 0) + ev_total.get("retrace", 0)
                    + ev_total.get("filled_pause", 0)
                    + ev_total.get("fragment", 0))
        rate = (n_events / n_surf_words * 100) if n_surf_words else 0.0
        index_rows.append({
            "corpus": corpus, "session": stem,
            "n_utts": len(utt_rows), "n_surface_words": n_surf_words,
            "repetition": ev_total.get("repetition", 0),
            "retrace": ev_total.get("retrace", 0),
            "filled_pause": ev_total.get("filled_pause", 0),
            "fragment": ev_total.get("fragment", 0),
            "disfluency_rate_pct": round(rate, 2),
            "severity": band(rate),
        })

    with OUT_INDEX.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        w.writeheader(); w.writerows(index_rows)

    tot = Counter()
    for r in index_rows:
        for k in ("repetition", "retrace", "filled_pause", "fragment"):
            tot[k] += r[k]
    sev = Counter(r["severity"] for r in index_rows)
    print(f"sessions with disfluency parse: {len(index_rows)}")
    print(f"events: {dict(tot)}")
    print(f"severity bands: {dict(sev)}")
    print(f"-> {OUT_INDEX}")
    print(f"-> {OUT_TEXT}/ (per-session surface+intended)")


if __name__ == "__main__":
    main()
