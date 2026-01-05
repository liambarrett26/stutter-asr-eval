"""
Text preprocessing and alignment utilities.

Handles transcript normalization with options for:
- Surface transcription (preserving disfluencies)
- Intended speech (normalized fluent transcription)
"""

import re
from typing import Optional


def normalize_transcript(
    text: str,
    remove_disfluencies: bool = False,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """
    Normalize transcript text for evaluation.

    Args:
        text: Input transcript text.
        remove_disfluencies: If True, remove common disfluency markers
            (um, uh, filled pauses). For intended-speech evaluation.
        lowercase: Convert to lowercase.
        remove_punctuation: Remove punctuation marks.

    Returns:
        Normalized text string.
    """
    result = text

    if remove_disfluencies:
        # Remove common filled pauses
        disfluency_patterns = [
            r"\b(um+|uh+|er+|ah+|eh+|mm+|hm+)\b",  # Filled pauses
            r"\b(like|you know|i mean)\b",  # Discourse markers (optional)
            r"\[.*?\]",  # Bracketed annotations
            r"\(.*?\)",  # Parenthetical annotations
        ]
        for pattern in disfluency_patterns:
            result = re.sub(pattern, " ", result, flags=re.IGNORECASE)

    if lowercase:
        result = result.lower()

    if remove_punctuation:
        result = re.sub(r"[^\w\s]", " ", result)

    # Clean up whitespace
    result = re.sub(r"\s+", " ", result).strip()

    return result


def align_words(
    reference: str,
    hypothesis: str,
) -> list[dict]:
    """
    Align words between reference and hypothesis transcripts.

    Uses edit distance alignment to match words and identify
    substitutions, deletions, and insertions.

    Args:
        reference: Ground truth transcript.
        hypothesis: ASR model output.

    Returns:
        List of alignment dicts with keys:
        - 'ref_word': Reference word (or None for insertions)
        - 'hyp_word': Hypothesis word (or None for deletions)
        - 'operation': 'match', 'substitution', 'deletion', 'insertion'
    """
    import jiwer

    # Get alignment from jiwer
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    # Use jiwer's alignment
    output = jiwer.process_words(reference, hypothesis)

    alignments = []
    for chunk in output.alignments[0]:
        if chunk.type == "equal":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                alignments.append({
                    "ref_word": ref_words[chunk.ref_start_idx + i],
                    "hyp_word": hyp_words[chunk.hyp_start_idx + i],
                    "operation": "match",
                })
        elif chunk.type == "substitute":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                ref_idx = chunk.ref_start_idx + i
                hyp_idx = chunk.hyp_start_idx + i
                alignments.append({
                    "ref_word": ref_words[ref_idx] if ref_idx < len(ref_words) else None,
                    "hyp_word": hyp_words[hyp_idx] if hyp_idx < len(hyp_words) else None,
                    "operation": "substitution",
                })
        elif chunk.type == "delete":
            for i in range(chunk.ref_end_idx - chunk.ref_start_idx):
                alignments.append({
                    "ref_word": ref_words[chunk.ref_start_idx + i],
                    "hyp_word": None,
                    "operation": "deletion",
                })
        elif chunk.type == "insert":
            for i in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                alignments.append({
                    "ref_word": None,
                    "hyp_word": hyp_words[chunk.hyp_start_idx + i],
                    "operation": "insertion",
                })

    return alignments


def extract_stutter_events(
    transcript: str,
    annotation_format: str = "brackets",
) -> list[dict]:
    """
    Extract stutter event annotations from annotated transcript.

    Args:
        transcript: Annotated transcript with stutter markers.
        annotation_format: Format of annotations:
            - 'brackets': [prolongation] word [/prolongation]
            - 'curly': {PWR word}
            - 'sep28k': SEP-28k style annotations

    Returns:
        List of stutter event dicts with word and type.
    """
    events = []

    if annotation_format == "brackets":
        # Pattern: [type] word [/type]
        pattern = r"\[(\w+)\]\s*(\S+)\s*\[/\1\]"
        for match in re.finditer(pattern, transcript):
            events.append({
                "stutter_type": match.group(1),
                "word": match.group(2),
                "start_pos": match.start(),
                "end_pos": match.end(),
            })

    # TODO: Add other annotation formats as needed

    return events
