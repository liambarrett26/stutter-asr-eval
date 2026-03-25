#!/usr/bin/env python3
"""
Parse CHILDES CHAT transcript files into standardised formats for the ASR
evaluation pipeline.

CHAT format reference: https://talkbank.org/manuals/CHAT.html

Produces two output types per file:
  1. Aligned CSV:  time_ms, duration_ms, speaker, text  (utterance-level)
  2. Flat CSV:     speaker, text  (full transcript, no timestamps)

Usage:
    # Parse a single file
    python src/data/parse_chat.py /path/to/file.cha

    # Parse all CHAT files in a zip
    python src/data/parse_chat.py /path/to/Voices-AWS.zip -o output_dir

    # Parse and filter to participant only (exclude investigator)
    python src/data/parse_chat.py /path/to/file.cha --speakers PAR PAR0 CHI
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import zipfile
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class CHATUtterance:
    speaker: str          # e.g. "PAR", "INV", "CHI", "MOT"
    text: str             # raw CHAT text (with coding)
    clean_text: str       # text stripped of CHAT markers
    start_ms: int | None  # millisecond timestamp (None if missing)
    end_ms: int | None
    mor: str = ""         # %mor tier
    gra: str = ""         # %gra tier


@dataclass
class CHATFile:
    filepath: str
    languages: list[str] = field(default_factory=list)
    participants: dict[str, str] = field(default_factory=dict)  # code -> role
    media: str = ""
    corpus: str = ""
    transcriber: str = ""
    types: list[str] = field(default_factory=list)
    speaker_ids: dict[str, dict] = field(default_factory=dict)  # code -> {age, gender, group, ...}
    utterances: list[CHATUtterance] = field(default_factory=list)


# ── CHAT text cleaning ───────────────────────────────────────────────────────

def clean_chat_text(text: str) -> str:
    """Strip CHAT coding markers to produce clean reference text.

    Removes:
      - Filled pause markers: &-um, &-uh, &-erm
      - Retracing markers: [/], [//], [///]
      - Overlap markers: [<], [>]
      - Error markers: [*]
      - Comment markers: [% ...]
      - Pause markers: (.), (..), (...)
      - Special markers: +", +, +//., +//.
      - Angle bracket groups for retracing: <word word> [/]
      - Actions: &=laughs, &=clears:throat
      - Postcodes: [+ ...]
      - Precodes: [- ...]
      - @g, @s, @l annotations on words

    Preserves:
      - The actual spoken words
      - Proper nouns
    """
    t = text

    # Remove CHAT bullet timestamps: \x15 digits_digits \x15
    t = re.sub(r"\x15\d+_\d+\x15", "", t)
    # Also catch any bare timestamps at end (digits_digits)
    t = re.sub(r"\s*\d+_\d+\s*$", "", t)

    # Remove actions: &=laughs, &=clears:throat
    t = re.sub(r"&=\S+", "", t)

    # Remove filled pauses: &-um, &-uh, &-erm, &-ah
    t = re.sub(r"&-\w+", "", t)

    # Remove other & markers (fragment markers): &word
    t = re.sub(r"&\+?\w+", "", t)

    # Remove retracing/overlap markers in brackets
    # <word word> [/] -> remove the bracketed part and the [/]
    t = re.sub(r"<[^>]+>\s*\[/+\]", "", t)
    # word [/] -> remove the [/] but keep the word (it was repeated)
    t = re.sub(r"\[/+\]", "", t)

    # Remove all square bracket annotations: [*], [+ ...], [- ...], [% ...], [=! ...], etc.
    t = re.sub(r"\[[^\]]*\]", "", t)

    # Remove angle brackets (used for overlap/retracing scope)
    t = re.sub(r"[<>]", "", t)

    # Remove pause markers
    t = re.sub(r"\(\.\.*\)", "", t)

    # Remove special utterance terminators and connectors
    t = re.sub(r"\+[\"/,.<>]+\.?", "", t)
    t = re.sub(r"\+\.\.\.", "", t)

    # Remove word-level annotations: @g, @s, @l, @c etc.
    t = re.sub(r"@[a-z]", "", t)

    # Remove CHAT punctuation markers that aren't real punctuation
    # Keep . ? ! but remove others like +//.
    t = re.sub(r"\+/?/?\.", "", t)

    # Remove xxx (unintelligible), yyy (phonological coding), www (untranscribed)
    t = re.sub(r"\b(xxx|yyy|www)\b", "", t)

    # Clean up whitespace
    t = re.sub(r"\s+", " ", t).strip()

    # Remove trailing punctuation-only remnants, then re-add period if needed
    t = re.sub(r"\s*[.!?;,]+\s*$", "", t).strip()

    return t


# ── CHAT parsing ─────────────────────────────────────────────────────────────

def parse_chat(lines: list[str], filepath: str = "") -> CHATFile:
    """Parse CHAT format lines into structured data."""
    chat = CHATFile(filepath=filepath)

    current_utt = None
    current_tier = None

    for line in lines:
        line = line.rstrip("\n\r")

        # Header lines
        if line.startswith("@"):
            _parse_header(line, chat)
            continue

        # Speaker utterance line: *SPK:\ttext
        if line.startswith("*"):
            # Save previous utterance
            if current_utt is not None:
                chat.utterances.append(current_utt)

            m = re.match(r"\*(\w+):\s*(.*)", line)
            if m:
                speaker = m.group(1)
                text = m.group(2).strip()

                # Extract timestamp: CHAT uses \x15 digits_digits \x15 delimiters
                start_ms, end_ms = None, None
                ts_match = re.search(r"\x15(\d+)_(\d+)\x15", text)
                if not ts_match:
                    ts_match = re.search(r"(\d+)_(\d+)\s*$", text)
                if ts_match:
                    start_ms = int(ts_match.group(1))
                    end_ms = int(ts_match.group(2))

                current_utt = CHATUtterance(
                    speaker=speaker,
                    text=text,
                    clean_text=clean_chat_text(text),
                    start_ms=start_ms,
                    end_ms=end_ms,
                )
                current_tier = "main"
            continue

        # Dependent tier lines: %mor:, %gra:, etc.
        if line.startswith("%") and current_utt is not None:
            m = re.match(r"%(\w+):\s*(.*)", line)
            if m:
                tier_name = m.group(1)
                tier_text = m.group(2).strip()
                if tier_name == "mor":
                    current_utt.mor = tier_text
                elif tier_name == "gra":
                    current_utt.gra = tier_text
                current_tier = tier_name
            continue

        # Continuation lines (start with \t)
        if line.startswith("\t") and current_utt is not None:
            continuation = line.strip()
            if current_tier == "main":
                current_utt.text += " " + continuation
                # Re-parse timestamp and clean text
                ts_match = re.search(r"\x15(\d+)_(\d+)\x15", current_utt.text)
                if not ts_match:
                    ts_match = re.search(r"(\d+)_(\d+)\s*$", current_utt.text)
                if ts_match:
                    current_utt.start_ms = int(ts_match.group(1))
                    current_utt.end_ms = int(ts_match.group(2))
                current_utt.clean_text = clean_chat_text(current_utt.text)
            elif current_tier == "mor":
                current_utt.mor += " " + continuation
            elif current_tier == "gra":
                current_utt.gra += " " + continuation

    # Don't forget the last utterance
    if current_utt is not None:
        chat.utterances.append(current_utt)

    return chat


def _parse_header(line: str, chat: CHATFile) -> None:
    """Parse a CHAT @ header line."""
    if line.startswith("@Languages:"):
        chat.languages = [l.strip() for l in line.split(":", 1)[1].split(",")]
    elif line.startswith("@Participants:"):
        parts = line.split(":", 1)[1].strip().split(",")
        for part in parts:
            tokens = part.strip().split()
            if len(tokens) >= 2:
                chat.participants[tokens[0]] = " ".join(tokens[1:])
    elif line.startswith("@Media:"):
        chat.media = line.split(":", 1)[1].strip()
    elif line.startswith("@Transcriber:"):
        chat.transcriber = line.split(":", 1)[1].strip()
    elif line.startswith("@Types:"):
        chat.types = [t.strip() for t in line.split(":", 1)[1].split(",")]
    elif line.startswith("@ID:"):
        # Format: lang|corpus|code|age|gender|group||role|||
        parts = line.split(":", 1)[1].strip().split("|")
        if len(parts) >= 8:
            code = parts[2]
            chat.speaker_ids[code] = {
                "language": parts[0],
                "corpus": parts[1],
                "age": parts[3],
                "gender": parts[4],
                "group": parts[5],
                "role": parts[7],
            }
            if not chat.corpus:
                chat.corpus = parts[1]


# ── Export functions ──────────────────────────────────────────────────────────

def to_aligned_csv(
    chat: CHATFile,
    output_path: Path,
    speakers: set[str] | None = None,
) -> int:
    """Export utterances with timestamps as aligned CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for utt in chat.utterances:
        if speakers and utt.speaker not in speakers:
            continue
        if not utt.clean_text:
            continue

        rows.append({
            "start_ms": utt.start_ms if utt.start_ms is not None else "",
            "end_ms": utt.end_ms if utt.end_ms is not None else "",
            "duration_ms": (utt.end_ms - utt.start_ms) if utt.start_ms is not None and utt.end_ms is not None else "",
            "speaker": utt.speaker,
            "text": utt.clean_text,
            "raw_text": utt.text,
        })

    if not rows:
        return 0

    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    return len(rows)


def to_flat_text(
    chat: CHATFile,
    speakers: set[str] | None = None,
) -> str:
    """Export clean text as a single string (for WER calculation)."""
    parts = []
    for utt in chat.utterances:
        if speakers and utt.speaker not in speakers:
            continue
        if utt.clean_text:
            parts.append(utt.clean_text)
    return " ".join(parts)


# ── Batch processing ─────────────────────────────────────────────────────────

def process_zip(
    zip_path: Path,
    output_dir: Path,
    speakers: set[str] | None = None,
) -> dict:
    """Process all CHAT files in a zip archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir = output_dir / "aligned"
    flat_dir = output_dir / "flat"
    aligned_dir.mkdir(exist_ok=True)
    flat_dir.mkdir(exist_ok=True)

    stats = {"files": 0, "utterances": 0, "with_timestamps": 0, "speakers": set()}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".cha"):
                continue

            with zf.open(name) as f:
                lines = TextIOWrapper(f, encoding="utf-8", errors="replace").readlines()

            chat = parse_chat(lines, filepath=name)

            # Determine output filename from path structure
            # e.g. Voices-AWS/interview/20f.cha -> Voices-AWS_interview_20f
            parts = Path(name).parts
            stem = "_".join(parts).replace(".cha", "")

            # Export aligned CSV
            n = to_aligned_csv(chat, aligned_dir / f"{stem}.csv", speakers)

            # Export flat text
            flat_text = to_flat_text(chat, speakers)
            if flat_text:
                flat_path = flat_dir / f"{stem}.txt"
                flat_path.write_text(flat_text, encoding="utf-8")

            stats["files"] += 1
            stats["utterances"] += n
            for utt in chat.utterances:
                stats["speakers"].add(utt.speaker)
                if utt.start_ms is not None:
                    stats["with_timestamps"] += 1

    stats["speakers"] = sorted(stats["speakers"])
    return stats


def process_all_fluencybank(
    raw_dir: Path = Path("/Volumes/FATSPEECH/fluencybank/raw"),
    output_dir: Path = Path("/Volumes/FATSPEECH/fluencybank/processed"),
):
    """Process all FluencyBank transcript zips."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define which speakers to extract for each corpus
    # PAR/PAR0 = adult participant, CHI = child, MOT = mother, INV = investigator
    corpus_speakers = {
        "Voices-AWS": {"PAR", "PAR0"},
        "Voices-CWS": {"CHI"},
        "Voices-AWC": {"PAR", "PAR0"},
        "UMD-CMU": None,  # Keep all speakers (child-parent conversation)
        "Hakim": {"CHI"},
        "Examples": {"PAR", "PAR0"},
        "VanZaalen": {"PAR", "PAR0"},
        "Brejon": {"CHI"},
    }

    all_stats = {}
    for corpus, speakers in corpus_speakers.items():
        zip_path = raw_dir / corpus / "transcripts" / f"{corpus}.zip"
        if not zip_path.exists():
            print(f"  SKIP {corpus}: zip not found")
            continue

        print(f"  Processing {corpus}...", end="", flush=True)
        corpus_out = output_dir / corpus
        stats = process_zip(zip_path, corpus_out, speakers)
        all_stats[corpus] = stats
        print(
            f" {stats['files']} files, {stats['utterances']} utterances "
            f"({stats['with_timestamps']} with timestamps)"
        )

    return all_stats


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Parse CHAT transcripts")
    parser.add_argument("input", type=Path, help="CHAT file or zip archive")
    parser.add_argument("-o", "--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--speakers",
        nargs="*",
        help="Speaker codes to extract (e.g. PAR CHI). Default: all.",
    )
    parser.add_argument(
        "--all-fluencybank",
        action="store_true",
        help="Process all FluencyBank zips from /Volumes/FATSPEECH/fluencybank/raw",
    )

    args = parser.parse_args()
    speakers = set(args.speakers) if args.speakers else None

    if args.all_fluencybank:
        print("Processing all FluencyBank corpora...")
        process_all_fluencybank()
        return

    if args.input.suffix == ".zip":
        stats = process_zip(args.input, args.output, speakers)
        print(f"Processed {stats['files']} files, {stats['utterances']} utterances")
    elif args.input.suffix == ".cha":
        lines = args.input.read_text(encoding="utf-8", errors="replace").split("\n")
        chat = parse_chat(lines, filepath=str(args.input))
        n = to_aligned_csv(chat, args.output / f"{args.input.stem}.csv", speakers)
        flat = to_flat_text(chat, speakers)
        print(f"Parsed {n} utterances, {len(flat)} chars of text")
        print(f"Speakers: {sorted(set(u.speaker for u in chat.utterances))}")
        print(f"Participants: {chat.participants}")
        if chat.utterances:
            print(f"Sample: {chat.utterances[0].clean_text[:100]}...")
    else:
        print(f"Unknown file type: {args.input.suffix}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
