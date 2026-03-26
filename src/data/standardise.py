#!/usr/bin/env python3
"""
Standardise transcripts across all corpora into a unified format for ASR evaluation.

This module handles the conversion from corpus-specific annotation formats into
two clean reference forms:

    1. **Intended transcript**: what the speaker meant to say (disfluencies removed)
    2. **Surface transcript**: what the speaker actually produced (disfluencies preserved)

The output is a single unified CSV per session with the schema:
    file_id, start_s, end_s, speaker, text_intended, text_surface, stutter_type, corpus

See docs/transcript_standardisation.md for full documentation of the conversion
rules applied to each corpus.

Usage:
    python -m src.data.standardise                    # Standardise all corpora
    python -m src.data.standardise --corpus slass     # Single corpus
    python -m src.data.standardise --corpus librispeech
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Literal


# ── Corpus-specific stripping functions ──────────────────────────────────────

def strip_sfs_orthographic(label: str) -> str:
    """Convert an SFS orthographic annotation label to clean intended text.

    SFS orthographic labels use:
        :word  = content word
        /word  = function word
        x      = word boundary marker (silence between words)
        Q      = silent pause / block
        .      = trailing period on some words

    Examples:
        ':new'       -> 'new'
        '/and'       -> 'and'
        ':school.'   -> 'school'
        'x'          -> ''  (boundary, removed)
        'Q'          -> ''  (pause, removed)
    """
    if label in ("x", "X", "Q", ""):
        return ""

    text = label

    # Remove : and / prefixes
    if text.startswith(":") or text.startswith("/"):
        text = text[1:]

    # Remove " syllable stress markers
    text = text.replace('"', "")

    # Remove trailing periods
    text = text.rstrip(".")

    # Remove trailing 'x' that marks word boundary within the label
    if text.endswith("x") and len(text) > 1:
        # Only strip if it looks like a boundary marker, not part of a word
        # e.g. 'mawnx' -> 'mawn' but 'box' should stay
        pass  # Conservative: don't strip, too ambiguous

    return text.strip()


def classify_sfs_stutter(label: str) -> str:
    """Classify the disfluency type from an SFS stutter layer label.

    The stutter layer uses JSRU phonetic notation with embedded disfluency
    markers. We classify into standard disfluency categories based on the
    markers present, NOT the phonetic spelling.

    Disfluency markers:
        Q               = silent block
        {U blocks}      = articulatory block
        {xN}            = N repetitions of preceding sound
        (UM), (ER)      = filled pause
        (HA)            = hesitation
        [:/word]        = intended form (presence indicates disfluent production)
        repeated chars  = prolongation (e.g., MMMMM, IIIII)

    Returns one of: block, repetition, prolongation, filled_pause,
    hesitation, or combinations joined with '+' (e.g., 'block+repetition').
    Returns 'fluent' if no disfluency markers are present.
    Returns '' for boundary markers.
    """
    if label in ("x", "X"):
        return ""
    if label == "Q":
        return "block"

    # Filled pauses
    filled = re.match(r"^\((\w+)\)\.?$", label)
    if filled:
        fp = filled.group(1).upper()
        if fp in ("UM", "ER", "ERM", "AH", "UH", "MM"):
            return "filled_pause"
        if fp in ("HA",):
            return "hesitation"
        return "interjection"

    types = []

    # Blocks: Q within word or {U blocks}
    if "Q" in label and re.search(r"[a-zA-Z]", label):
        types.append("block")
    if "{U block" in label:
        if "block" not in types:
            types.append("block")

    # Repetitions: {xN} markers or space-separated repeated segments
    if re.search(r"\{x\d+\}", label):
        types.append("repetition")
    # Also detect repeated characters with spaces: "AA AA AA"
    if re.search(r"(\b\w+)\s+\1", label):
        if "repetition" not in types:
            types.append("repetition")

    # Prolongations: 4+ consecutive identical characters
    if re.search(r"(\w)\1{3,}", label):
        types.append("prolongation")

    # Intended form bracket = disfluent production (if no other markers caught it)
    if not types and "[" in label:
        types.append("other_disfluency")

    if types:
        return "+".join(types)

    return "fluent"


def strip_sfs_type(label: str) -> str:
    """Extract stutter type from an SFS type annotation label.

    Type layer labels describe the phonetic form with embedded type codes:
        Q                   = block
        /c[2]fa             = sound with cluster info
        {xN}                = repetition count
        (UM), (ER), (HA)    = filled pause / hesitation

    Returns a simplified stutter type string.
    """
    if label == "Q":
        return "block"

    filled = re.match(r"^\((\w+)\)\.?$", label)
    if filled:
        fp = filled.group(1).upper()
        if fp in ("UM", "ER", "ERM", "AH", "UH"):
            return "filled_pause"
        if fp in ("HA",):
            return "hesitation"
        return f"interjection:{fp.lower()}"

    # Check for repetition markers
    if re.search(r"\{x\d+\}", label):
        return "repetition"

    # Check for block markers within words
    if "Q" in label and re.search(r"[a-zA-Z]", label):
        return "block_within_word"

    # Content/function word markers (no stutter)
    if label.startswith(":") or label.startswith("/"):
        return "fluent"

    return "unknown"


def strip_uclass_textgrid_stutter(label: str) -> str:
    """Convert a UCLASS stutter-coded TextGrid label to clean surface text.

    Same convention as SFS stutter layer (JSRU phonetic + disfluency coding).
    """
    return strip_sfs_stutter(label)


def strip_uclass_textgrid_orth(label: str) -> str:
    """Convert a UCLASS orthographic TextGrid label to clean intended text.

    UCLASS orthographic TextGrids use plain English words.
    CAPS indicates stressed or stuttered words.

    Examples:
        'well'      -> 'well'
        'MY'        -> 'my'
        'suppose'   -> 'suppose'
    """
    if label in ("x", "X", "Q", ""):
        return ""
    return label.lower().strip()


def strip_uclass_flat_ortho(text: str) -> str:
    """Clean a UCLASS flat orthographic transcript.

    These are full paragraph transcripts with:
        - Standard English punctuation
        - CAPS for stuttered/stressed words
        - Filled pauses written out: 'er', 'erm', 'um'
        - Smart quotes and encoding artefacts

    Returns clean lowercase text suitable as intended reference.
    """
    result = text

    # Fix encoding artefacts
    result = result.replace("\x92", "'").replace("\x93", '"').replace("\x94", '"')
    result = result.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')

    # Lowercase
    result = result.lower()

    # Remove punctuation (keep apostrophes in contractions)
    result = re.sub(r"[^\w\s']", " ", result)

    # Normalise whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


def strip_chat_text(text: str) -> str:
    """Clean text already processed by parse_chat.py's clean_chat_text().

    The CHAT parser already strips most markers. This handles remaining
    normalisation for the unified schema.
    """
    result = text.lower().strip()
    # Remove any remaining punctuation except apostrophes
    result = re.sub(r"[^\w\s']", " ", result)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def strip_librispeech(text: str) -> str:
    """Clean a LibriSpeech transcript line.

    LibriSpeech transcripts are UPPERCASE with no punctuation.

    Example:
        'HE HOPED THERE WOULD BE STEW FOR DINNER' -> 'he hoped there would be stew for dinner'
    """
    return text.lower().strip()


# ── Unified record construction ──────────────────────────────────────────────

def make_unified_record(
    file_id: str,
    start_s: float | None,
    end_s: float | None,
    speaker: str,
    text_intended: str,
    text_surface: str,
    stutter_type: str,
    corpus: str,
) -> dict:
    """Create a single record in the unified schema."""
    return {
        "file_id": file_id,
        "start_s": f"{start_s:.3f}" if start_s is not None else "",
        "end_s": f"{end_s:.3f}" if end_s is not None else "",
        "speaker": speaker,
        "text_intended": text_intended,
        "text_surface": text_surface,
        "stutter_type": stutter_type,
        "corpus": corpus,
    }


UNIFIED_FIELDNAMES = [
    "file_id", "start_s", "end_s", "speaker",
    "text_intended", "text_surface", "stutter_type", "corpus",
]


# ── Corpus-level standardisation functions ───────────────────────────────────

def standardise_slass_session(
    ortho_csv: Path,
    stutter_csv: Path | None = None,
    type_csv: Path | None = None,
    file_id: str = "",
) -> list[dict]:
    """Standardise a SLASS session from orthographic + optional stutter/type CSVs.

    The orthographic layer provides intended text. The stutter layer provides
    surface text. The type layer provides stutter type labels.

    Words are matched by timestamp alignment (nearest match within 50ms).
    """
    # Load orthographic (intended)
    def _read_csv_safe(path: Path) -> list[dict]:
        """Read CSV handling NUL bytes and encoding issues."""
        text = path.read_bytes().replace(b"\x00", b"").decode("latin-1")
        reader = csv.DictReader(text.splitlines())
        return list(reader)

    ortho_records = _read_csv_safe(ortho_csv)

    # Load stutter (surface) if available
    stutter_records = []
    if stutter_csv and stutter_csv.exists():
        stutter_records = _read_csv_safe(stutter_csv)

    # Load type if available
    type_records = []
    if type_csv and type_csv.exists():
        type_records = _read_csv_safe(type_csv)

    # Build stutter and type lookups by timestamp (nearest within 50ms)
    def find_nearest(records: list[dict], time_s: float, tolerance: float = 0.05) -> dict | None:
        best = None
        best_dist = tolerance
        for r in records:
            dist = abs(float(r["time"]) - time_s)
            if dist < best_dist:
                best = r
                best_dist = dist
        return best

    # Detect WWR: consecutive identical words in the orthographic layer
    # Build a set of timestamps where WWR occurs
    wwr_timestamps = set()
    ortho_words = [(r["time"], strip_sfs_orthographic(r["label"]).upper())
                   for r in ortho_records if r["label"] not in ("x", "X", "Q", "")]
    i = 0
    while i < len(ortho_words) - 1:
        if (ortho_words[i][1] == ortho_words[i + 1][1]
                and ortho_words[i][1]):
            # Mark all instances in the run as WWR
            run_start = i
            while (i + 1 < len(ortho_words)
                   and ortho_words[i + 1][1] == ortho_words[run_start][1]):
                i += 1
            # Mark all but the last (which is the "successful" production)
            for j in range(run_start, i):
                wwr_timestamps.add(ortho_words[j][0])
            i += 1
        else:
            i += 1

    unified = []
    for rec in ortho_records:
        label = rec["label"]
        intended = strip_sfs_orthographic(label)
        if not intended:
            continue

        time_s = float(rec["time"])
        dur_s = float(rec["duration"])

        # Classify disfluency from stutter layer
        # The surface text is the ENGLISH word (from ortho), not JSRU phonetic.
        # The stutter layer tells us the TYPE of disfluency, not an alternative spelling.
        surface = intended  # Same English word — surface vs intended differ only in stutter_type
        stype = ""

        # Check for WWR first (consecutive identical words in ortho layer)
        if rec["time"] in wwr_timestamps:
            stype = "wwr"
        elif stutter_records:
            match = find_nearest(stutter_records, time_s)
            if match:
                stype = classify_sfs_stutter(match["label"])
                # Refine: if classifier said "repetition", it's PWR (not WWR,
                # since we already caught WWR above from consecutive ortho words)
                stype = stype.replace("repetition", "pwr")
                # For filled pauses, the surface text IS different (um, er)
                filled = re.match(r"^\((\w+)\)\.?$", match["label"])
                if filled:
                    surface = filled.group(1).lower()
                # For blocks (standalone Q), surface is silence
                elif match["label"] == "Q":
                    surface = "[block]"

        # Override with type layer if available and stutter layer didn't match
        if not stype and type_records:
            match = find_nearest(type_records, time_s)
            if match:
                stype = strip_sfs_type(match["label"])

        unified.append(make_unified_record(
            file_id=file_id,
            start_s=time_s,
            end_s=time_s + dur_s,
            speaker="",
            text_intended=intended,
            text_surface=surface,
            stutter_type=stype,
            corpus="slass",
        ))

    return unified


def standardise_uclass_session(
    orth_csv: Path | None = None,
    stutter_csv: Path | None = None,
    flat_ortho: Path | None = None,
    file_id: str = "",
) -> list[dict]:
    """Standardise a UCLASS session.

    Priority:
    1. Time-aligned orthographic TextGrid (intended) + stutter-coded TextGrid (surface)
    2. Flat orthographic transcript (intended only, no timestamps)
    """
    unified = []

    if orth_csv and orth_csv.exists():
        with open(orth_csv) as f:
            orth_records = list(csv.DictReader(f))

        stutter_records = []
        if stutter_csv and stutter_csv.exists():
            with open(stutter_csv) as f:
                stutter_records = list(csv.DictReader(f))

        for rec in orth_records:
            intended = strip_uclass_textgrid_orth(rec["label"])
            if not intended:
                continue

            time_s = float(rec["time"])
            dur_s = float(rec["duration"])
            surface = intended

            if stutter_records:
                # Find nearest stutter record
                best = None
                best_dist = 0.05
                for sr in stutter_records:
                    dist = abs(float(sr["time"]) - time_s)
                    if dist < best_dist:
                        best = sr
                        best_dist = dist
                if best:
                    surface = strip_uclass_textgrid_stutter(best["label"])
                    if not surface:
                        surface = intended

            unified.append(make_unified_record(
                file_id=file_id,
                start_s=time_s,
                end_s=time_s + dur_s,
                speaker="",
                text_intended=intended,
                text_surface=surface,
                stutter_type="",
                corpus="uclass",
            ))

    elif flat_ortho and flat_ortho.exists():
        with open(flat_ortho) as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = strip_uclass_flat_ortho(row["text"])
                if text:
                    unified.append(make_unified_record(
                        file_id=file_id,
                        start_s=None,
                        end_s=None,
                        speaker="",
                        text_intended=text,
                        text_surface=text,  # No surface info from flat ortho
                        stutter_type="",
                        corpus="uclass",
                    ))

    return unified


def standardise_fluencybank_session(
    aligned_csv: Path,
    file_id: str = "",
) -> list[dict]:
    """Standardise a FluencyBank session from parsed CHAT aligned CSV.

    FluencyBank CHAT transcripts are unannotated for disfluency type.
    The clean text from parse_chat.py serves as the intended reference.
    The raw_text column preserves some disfluency markers (filled pauses).
    """
    unified = []
    text = aligned_csv.read_bytes().replace(b"\x00", b"").decode("latin-1")
    for row in csv.DictReader(text.splitlines()):
            text = strip_chat_text(row["text"])
            if not text:
                continue

            start_s = float(row["start_ms"]) / 1000.0 if row["start_ms"] else None
            end_s = float(row["end_ms"]) / 1000.0 if row["end_ms"] else None

            unified.append(make_unified_record(
                file_id=file_id,
                start_s=start_s,
                end_s=end_s,
                speaker=row.get("speaker", ""),
                text_intended=text,
                text_surface=text,  # No separate surface annotation in FluencyBank
                stutter_type="",
                corpus="fluencybank",
            ))

    return unified


def standardise_librispeech_file(
    trans_line: str,
    file_id: str = "",
) -> list[dict]:
    """Standardise a single LibriSpeech transcript line.

    Format: 'UTTERANCE_ID WORD1 WORD2 WORD3...'
    """
    parts = trans_line.strip().split(" ", 1)
    if len(parts) < 2:
        return []

    utt_id = parts[0]
    text = strip_librispeech(parts[1])

    return [make_unified_record(
        file_id=file_id or utt_id,
        start_s=None,
        end_s=None,
        speaker="",
        text_intended=text,
        text_surface=text,  # Fluent speech — no disfluency
        stutter_type="fluent",
        corpus="librispeech",
    )]


# ── Write utility ────────────────────────────────────────────────────────────

def write_unified_csv(records: list[dict], output_path: Path) -> int:
    """Write unified records to a CSV file."""
    if not records:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=UNIFIED_FIELDNAMES)
        w.writeheader()
        w.writerows(records)
    return len(records)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Standardise transcripts across corpora")
    parser.add_argument("--corpus", choices=["slass", "uclass", "fluencybank", "librispeech", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("/Volumes/FATSPEECH/standardised"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.corpus in ("slass", "all"):
        print("Standardising SLASS...")
        _standardise_all_slass(args.output / "slass")

    if args.corpus in ("uclass", "all"):
        print("Standardising UCLASS...")
        _standardise_all_uclass(args.output / "uclass")

    if args.corpus in ("fluencybank", "all"):
        print("Standardising FluencyBank...")
        _standardise_all_fluencybank(args.output / "fluencybank")

    if args.corpus in ("librispeech", "all"):
        print("Standardising LibriSpeech...")
        _standardise_all_librispeech(args.output / "librispeech")


def _standardise_all_slass(output_dir: Path):
    """Standardise all SLASS curated subset files."""
    ann_root = Path("/Volumes/FATSPEECH/slass/processed/annotations")
    total_records = 0
    total_files = 0

    for group_dir in sorted(ann_root.iterdir()):
        if not group_dir.is_dir():
            continue

        # Find unique session IDs from orthographic files
        ortho_files = sorted(group_dir.glob("*_orthographic.csv"))
        for ortho_csv in ortho_files:
            stem = ortho_csv.stem.replace("_orthographic", "")
            stutter_csv = group_dir / f"{stem}_stutter.csv"
            type_csv = None
            # Type files have varying names: _type.csv or _types.csv
            for suffix in ("_type.csv", "_types.csv"):
                candidate = group_dir / f"{stem}{suffix}"
                if candidate.exists():
                    type_csv = candidate
                    break

            records = standardise_slass_session(
                ortho_csv=ortho_csv,
                stutter_csv=stutter_csv if stutter_csv.exists() else None,
                type_csv=type_csv,
                file_id=stem,
            )

            if records:
                out_path = output_dir / group_dir.name / f"{stem}.csv"
                write_unified_csv(records, out_path)
                total_records += len(records)
                total_files += 1

    print(f"  SLASS: {total_files} files, {total_records} records -> {output_dir}")


def _standardise_all_uclass(output_dir: Path):
    """Standardise all UCLASS transcript files."""
    aligned_dir = Path("/Volumes/FATSPEECH/uclass/processed/transcripts/aligned")
    flat_dir = Path("/Volumes/FATSPEECH/uclass/processed/transcripts/flat")
    total_records = 0
    total_files = 0

    # Find sessions with orthographic TextGrids
    processed = set()
    for f in sorted(aligned_dir.glob("*_word_orth.csv")):
        stem = f.stem.replace("_word_orth", "")
        # Look for matching stutter-coded TextGrid
        stutter_csv = aligned_dir / f"{stem}_word.csv"
        records = standardise_uclass_session(
            orth_csv=f,
            stutter_csv=stutter_csv if stutter_csv.exists() else None,
            file_id=stem,
        )
        if records:
            write_unified_csv(records, output_dir / f"{stem}.csv")
            total_records += len(records)
            total_files += 1
            processed.add(stem)

    # Flat ortho files not already covered
    for f in sorted(flat_dir.glob("*_ortho.csv")):
        stem = f.stem.replace("_ortho", "")
        if stem in processed:
            continue
        records = standardise_uclass_session(flat_ortho=f, file_id=stem)
        if records:
            write_unified_csv(records, output_dir / f"{stem}.csv")
            total_records += len(records)
            total_files += 1

    print(f"  UCLASS: {total_files} files, {total_records} records -> {output_dir}")


def _standardise_all_fluencybank(output_dir: Path):
    """Standardise all FluencyBank parsed CHAT files."""
    proc_dir = Path("/Volumes/FATSPEECH/fluencybank/processed")
    total_records = 0
    total_files = 0

    for corpus_dir in sorted(proc_dir.iterdir()):
        aligned_dir = corpus_dir / "aligned"
        if not aligned_dir.exists():
            continue
        for f in sorted(aligned_dir.glob("*.csv")):
            records = standardise_fluencybank_session(f, file_id=f.stem)
            if records:
                write_unified_csv(records, output_dir / corpus_dir.name / f"{f.stem}.csv")
                total_records += len(records)
                total_files += 1

    print(f"  FluencyBank: {total_files} files, {total_records} records -> {output_dir}")


def _standardise_all_librispeech(output_dir: Path):
    """Standardise all LibriSpeech transcript files."""
    ls_root = Path("/Volumes/FATSPEECH/librispeech/LibriSpeech")
    total_records = 0
    total_files = 0

    for split in ("test-clean", "test-other", "dev-clean", "dev-other"):
        split_dir = ls_root / split
        if not split_dir.exists():
            continue

        for trans_file in sorted(split_dir.rglob("*.trans.txt")):
            with open(trans_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id = line.split(" ", 1)[0]
                    records = standardise_librispeech_file(line, file_id=utt_id)
                    total_records += len(records)

            total_files += 1

        # Write one file per split for LibriSpeech
        all_records = []
        for trans_file in sorted(split_dir.rglob("*.trans.txt")):
            with open(trans_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.extend(standardise_librispeech_file(line))
        write_unified_csv(all_records, output_dir / f"{split}.csv")

    print(f"  LibriSpeech: {total_files} trans files, {total_records} records -> {output_dir}")


if __name__ == "__main__":
    main()
