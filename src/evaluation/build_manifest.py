#!/usr/bin/env python3
"""Build the ASR evaluation manifest (one JSON unit per audio file).

A "unit" is one audio file plus its reference text(s) and metadata. For
SLASS the unit is a session; for LibriSpeech it is an utterance. The
manifest is the single input to `run_asr.py` (inference) and `score.py`
(scoring), so all corpus-specific path/reference logic lives here.

Corpora are added via resolver functions; this version wires:
  * slass_full   stuttered, child-skewed (the expanded full-archive base)
  * librispeech  fluent control

FluencyBank is the next resolver to wire (its standardised file_id does
not map 1:1 to audio names; needs an inventory join — see TODO).

Output (JSONL, one unit per line):
  {
    "unit_id", "dataset", "condition" (stuttered|fluent),
    "audio_path", "speaker_id", "split",
    "reference_intended", "reference_surface",
    "n_ref_words", "duration_s",
    "has_stutter_labels", "stutter_type_counts"
  }

Usage:
    python -m src.evaluation.build_manifest \\
        --out /Volumes/FATSPEECH/manifests/benchmark_v1.jsonl
    python -m src.evaluation.build_manifest --out ... --limit 20   # smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import soundfile as sf
except ImportError:
    sf = None

STD = Path("/Volumes/FATSPEECH/standardised")
SLASS_FULL_AUDIO = Path("/Volumes/FATSPEECH/slass/full_archive/audio")
LIBRI_ROOT = Path("/Volumes/FATSPEECH/librispeech/LibriSpeech")

DISFLUENT = lambda s: bool(s) and s.strip().lower() not in ("", "fluent", "unknown")


def _read_csv(path: Path) -> list[dict]:
    text = path.read_bytes().replace(b"\x00", b"").decode("latin-1")
    return list(csv.DictReader(text.splitlines()))


def _duration(path: Path) -> float | None:
    if sf is None or not path.exists():
        return None
    try:
        info = sf.info(str(path))
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


# ── SLASS full-archive (stuttered) ───────────────────────────────────────────

def resolve_slass_full(min_clean: float = 0.9) -> list[dict]:
    """One unit per clean full-archive session with audio present."""
    src = STD / "slass_full"
    man = src / "_session_manifest.csv"
    if not man.exists():
        print(f"  [slass_full] manifest missing: {man}", file=sys.stderr)
        return []
    quality = {r["session"]: r for r in _read_csv(man)}

    units = []
    for csv_path in sorted(src.glob("*.csv")):
        if csv_path.name.startswith("_") or csv_path.name.startswith("."):
            continue
        stem = csv_path.stem
        q = quality.get(stem)
        if not q:
            continue
        try:
            clean = float(q["clean_token_rate"]) if q["clean_token_rate"] else 0.0
        except ValueError:
            clean = 0.0
        if clean < min_clean:
            continue
        audio = SLASS_FULL_AUDIO / f"{stem}.wav"
        if not audio.exists():
            continue
        dur = _duration(audio)
        if dur is not None and dur < 1.0:
            continue  # corrupt / empty recording

        rows = _read_csv(csv_path)
        intended = " ".join(r["text_intended"] for r in rows
                            if r.get("text_intended", "").strip())
        surface = " ".join(r["text_surface"] for r in rows
                          if r.get("text_surface", "").strip())
        if not intended.strip():
            continue
        type_counts = Counter(r["stutter_type"].strip().lower()
                              for r in rows if DISFLUENT(r.get("stutter_type")))
        units.append({
            "unit_id": f"slass_full/{stem}",
            "dataset": "slass_full",
            "condition": "stuttered",
            "audio_path": str(audio),
            "speaker_id": q.get("speaker_id", ""),
            "split": "test",
            "reference_intended": intended,
            "reference_surface": surface or intended,
            "n_ref_words": len(intended.split()),
            "duration_s": dur,
            "needs_chunking": bool(dur and dur > 30.0),
            "has_stutter_labels": bool(type_counts),
            "stutter_type_counts": dict(type_counts),
        })
    return units


# ── LibriSpeech (fluent control) ─────────────────────────────────────────────

def resolve_librispeech() -> list[dict]:
    """One unit per LibriSpeech utterance, audio path derived from id."""
    units = []
    for split_csv in sorted((STD / "librispeech").glob("*.csv")):
        if split_csv.name.startswith("."):
            continue
        split = split_csv.stem  # e.g. test-clean
        for r in _read_csv(split_csv):
            uid = r.get("file_id", "").strip()
            text = r.get("text_intended", "").strip()
            if not uid or not text:
                continue
            # id like 1089-134686-0000 -> <split>/1089/134686/<id>.flac
            try:
                spk, chap, _ = uid.split("-")
            except ValueError:
                continue
            audio = LIBRI_ROOT / split / spk / chap / f"{uid}.flac"
            if not audio.exists():
                continue
            units.append({
                "unit_id": f"librispeech/{split}/{uid}",
                "dataset": "librispeech",
                "condition": "fluent",
                "audio_path": str(audio),
                "speaker_id": f"libri_{spk}",
                "split": "test" if split.startswith("test") else "dev",
                "reference_intended": text,
                "reference_surface": text,   # fluent: surface == intended
                "n_ref_words": len(text.split()),
                "duration_s": _duration(audio),
                "needs_chunking": False,   # LibriSpeech utts are short
                "has_stutter_labels": False,
                "stutter_type_counts": {},
            })
    return units


# ── FluencyBank (stuttered adults+children; some controls/clutterers) ─────────

FB_STD = STD / "fluencybank"
FB_AUDIO = Path("/Volumes/FATSPEECH/fluencybank/processed/audio")


def _fb_condition(corpus: str, stem: str) -> str:
    """FB is not uniformly stuttered: tag by sub-corpus/group."""
    if corpus == "Voices-AWC":
        return "cluttered"          # adults who clutter — keep distinct
    if corpus == "UMD-CMU" and "control" in stem.lower():
        return "fluent"             # matched controls
    return "stuttered"              # AWS/CWS, UMD-CMU CWS


def _fb_speaker(corpus: str, stem: str) -> str:
    """Link a speaker's interview+reading sessions: corpus + trailing id,
    dropping the task token."""
    rest = stem[len(corpus) + 1:] if stem.startswith(corpus + "_") else stem
    toks = rest.split("_")
    sess = toks[-1]
    return f"{corpus}_{sess}"


def resolve_fluencybank() -> list[dict]:
    if not FB_STD.exists():
        return []
    # Index audio by stem; resolve each transcript by longest trailing match.
    audio_by_stem: dict[str, Path] = {}
    for w in FB_AUDIO.rglob("*.wav"):
        if not w.name.startswith("."):
            audio_by_stem.setdefault(w.stem, w)

    claimed: set[str] = set()
    units = []
    for csv_path in sorted(FB_STD.rglob("*.csv")):
        if csv_path.name.startswith("."):
            continue
        corpus = csv_path.parent.name
        stem = csv_path.stem
        rest = stem[len(corpus) + 1:] if stem.startswith(corpus + "_") else stem
        toks = rest.split("_")
        audio = None
        for i in range(len(toks)):
            key = "_".join(toks[i:])
            if key in audio_by_stem and str(audio_by_stem[key]) not in claimed:
                audio = audio_by_stem[key]
                break
        if audio is None:
            continue
        claimed.add(str(audio))

        rows = _read_csv(csv_path)
        text = " ".join(r.get("text_intended", "").strip() for r in rows
                        if r.get("text_intended", "").strip())
        if not text.strip():
            continue
        dur = _duration(audio)
        units.append({
            "unit_id": f"fluencybank/{stem}",
            "dataset": "fluencybank",
            "condition": _fb_condition(corpus, stem),
            "audio_path": str(audio),
            "speaker_id": _fb_speaker(corpus, stem),
            "split": "test",
            "reference_intended": text,
            "reference_surface": text,   # FB disfluency not yet extracted
            "n_ref_words": len(text.split()),
            "duration_s": dur,
            "needs_chunking": bool(dur and dur > 30.0),
            "has_stutter_labels": False,
            "stutter_type_counts": {},
        })
    return units


# ── UCLASS (stuttered; child + adult) ────────────────────────────────────────

UCLASS_STD = STD / "uclass"
UCLASS_ROOT = Path("/Volumes/FATSPEECH/uclass")
_SPK_DIGITS = __import__("re").compile(r"(\d{3,4})")


def resolve_uclass() -> list[dict]:
    if not UCLASS_STD.exists():
        return []
    audio_by_stem: dict[str, Path] = {}
    for w in UCLASS_ROOT.rglob("*.wav"):
        if not w.name.startswith("."):
            audio_by_stem.setdefault(w.stem, w)

    units = []
    for csv_path in sorted(UCLASS_STD.rglob("*.csv")):
        if csv_path.name.startswith("."):
            continue
        rows = _read_csv(csv_path)
        text = " ".join(r.get("text_intended", "").strip() for r in rows
                        if r.get("text_intended", "").strip())
        if not text.strip():
            continue  # skips the phonetic stutter-only sessions (no English)
        stem = csv_path.stem
        audio = audio_by_stem.get(stem) or audio_by_stem.get(
            stem.rsplit("_", 1)[0])
        if audio is None:
            continue
        dur = _duration(audio)
        m = _SPK_DIGITS.search(stem)
        type_counts = Counter(r["stutter_type"].strip().lower()
                              for r in rows if DISFLUENT(r.get("stutter_type")))
        units.append({
            "unit_id": f"uclass/{stem}",
            "dataset": "uclass",
            "condition": "stuttered",
            "audio_path": str(audio),
            "speaker_id": f"uclass_{m.group(1)}" if m else f"uclass_{stem}",
            "split": "test",
            "reference_intended": text,
            "reference_surface": text,
            "n_ref_words": len(text.split()),
            "duration_s": dur,
            "needs_chunking": bool(dur and dur > 30.0),
            "has_stutter_labels": bool(type_counts),
            "stutter_type_counts": dict(type_counts),
        })
    return units


RESOLVERS = {
    "slass_full": resolve_slass_full,
    "librispeech": resolve_librispeech,
    "fluencybank": resolve_fluencybank,
    "uclass": resolve_uclass,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=list(RESOLVERS),
                    choices=list(RESOLVERS))
    ap.add_argument("--min-clean", type=float, default=0.9,
                    help="min clean_token_rate for slass_full sessions")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap units per dataset (smoke test)")
    args = ap.parse_args()

    all_units = []
    for ds in args.datasets:
        if ds == "slass_full":
            units = resolve_slass_full(min_clean=args.min_clean)
        else:
            units = RESOLVERS[ds]()
        if args.limit:
            units = units[: args.limit]
        print(f"  {ds}: {len(units):,} units")
        all_units.extend(units)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for u in all_units:
            fh.write(json.dumps(u) + "\n")

    # Summary
    by_cond = Counter(u["condition"] for u in all_units)
    spk = defaultdict(set)
    words = Counter()
    dur = defaultdict(float)
    for u in all_units:
        spk[u["condition"]].add(u["speaker_id"])
        words[u["condition"]] += u["n_ref_words"]
        if u["duration_s"]:
            dur[u["condition"]] += u["duration_s"]
    print(f"\nWrote {len(all_units):,} units -> {args.out}")
    for c in by_cond:
        print(f"  {c}: {by_cond[c]:,} units, {len(spk[c])} speakers, "
              f"{words[c]:,} ref words, {dur[c]/3600:.1f} h audio")


if __name__ == "__main__":
    main()
