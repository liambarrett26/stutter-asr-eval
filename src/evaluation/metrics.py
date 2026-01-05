"""
ASR evaluation metrics for stuttered speech.

Includes standard metrics (WER, CER) and stuttering-specific metrics
for analyzing co-dependency between stutter types and error types.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jiwer


class ErrorType(Enum):
    """Types of ASR errors."""

    SUBSTITUTION = "substitution"
    DELETION = "deletion"
    INSERTION = "insertion"
    CORRECT = "correct"


class StutterType(Enum):
    """Types of stuttering dysfluencies."""

    FLUENT = "fluent"
    PROLONGATION = "prolongation"
    PART_WORD_REPETITION = "part_word_repetition"  # PWR
    WHOLE_WORD_REPETITION = "whole_word_repetition"  # WWR
    BLOCK = "block"
    INTERJECTION = "interjection"


@dataclass
class WERResult:
    """Word Error Rate calculation result."""

    wer: float
    """Word Error Rate (0-1 scale, can exceed 1)."""

    substitutions: int
    """Number of word substitutions."""

    deletions: int
    """Number of word deletions."""

    insertions: int
    """Number of word insertions."""

    hits: int
    """Number of correct words."""

    reference_length: int
    """Number of words in reference."""

    hypothesis_length: int
    """Number of words in hypothesis."""


@dataclass
class CERResult:
    """Character Error Rate calculation result."""

    cer: float
    """Character Error Rate (0-1 scale, can exceed 1)."""

    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_length: int
    hypothesis_length: int


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> WERResult:
    """
    Calculate Word Error Rate between reference and hypothesis.

    Args:
        reference: Ground truth transcription.
        hypothesis: ASR model output.
        normalize: If True, apply standard text normalization
            (lowercase, remove punctuation).

    Returns:
        WERResult with detailed error breakdown.
    """
    if normalize:
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemoveEmptyStrings(),
        ])
        reference = transformation(reference)
        hypothesis = transformation(hypothesis)

    measures = jiwer.compute_measures(reference, hypothesis)

    return WERResult(
        wer=measures["wer"],
        substitutions=measures["substitutions"],
        deletions=measures["deletions"],
        insertions=measures["insertions"],
        hits=measures["hits"],
        reference_length=len(reference.split()),
        hypothesis_length=len(hypothesis.split()),
    )


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> CERResult:
    """
    Calculate Character Error Rate between reference and hypothesis.

    Useful for analyzing partial words and character-level errors
    common in stuttered speech transcription.

    Args:
        reference: Ground truth transcription.
        hypothesis: ASR model output.
        normalize: If True, apply standard text normalization.

    Returns:
        CERResult with detailed error breakdown.
    """
    if normalize:
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])
        reference = transformation(reference)
        hypothesis = transformation(hypothesis)

    measures = jiwer.cer(reference, hypothesis, return_dict=True)

    return CERResult(
        cer=measures["cer"],
        substitutions=measures["substitutions"],
        deletions=measures["deletions"],
        insertions=measures["insertions"],
        hits=measures["hits"],
        reference_length=len(reference),
        hypothesis_length=len(hypothesis),
    )


def calculate_semantic_similarity(
    reference: str,
    hypothesis: str,
    model_name: str = "microsoft/deberta-xlarge-mnli",
) -> float:
    """
    Calculate semantic similarity using BERTScore.

    Captures meaning preservation beyond surface-level errors.
    Important for stuttered speech where surface transcription
    may differ but meaning is preserved.

    Args:
        reference: Ground truth transcription.
        hypothesis: ASR model output.
        model_name: HuggingFace model for BERTScore.

    Returns:
        F1 BERTScore (0-1 scale).
    """
    from bert_score import score

    P, R, F1 = score(
        [hypothesis],
        [reference],
        model_type=model_name,
        verbose=False,
    )

    return F1.item()


@dataclass
class StutterMetrics:
    """
    Stuttering-specific metrics for analyzing ASR performance.

    Analyzes co-dependency between stutter types and error types,
    as outlined in the research documentation.
    """

    # Overall metrics
    wer: float
    cer: float
    semantic_similarity: Optional[float] = None

    # Per-stutter-type WER (if annotations available)
    wer_by_stutter_type: Optional[dict[StutterType, float]] = None

    # Error type distribution
    error_distribution: Optional[dict[ErrorType, int]] = None

    # Co-dependency: P(error_type | stutter_type)
    error_given_stutter: Optional[dict[tuple[StutterType, ErrorType], float]] = None

    # Hallucination rate (words inserted that don't correspond to speech)
    hallucination_rate: Optional[float] = None

    @classmethod
    def from_aligned_transcripts(
        cls,
        reference: str,
        hypothesis: str,
        stutter_annotations: Optional[list[dict]] = None,
    ) -> "StutterMetrics":
        """
        Calculate stutter metrics from transcripts.

        Args:
            reference: Ground truth transcription.
            hypothesis: ASR model output.
            stutter_annotations: Optional list of word-level stutter annotations.
                Each dict should have 'word', 'stutter_type' keys.

        Returns:
            StutterMetrics with available metrics calculated.
        """
        wer_result = calculate_wer(reference, hypothesis)
        cer_result = calculate_cer(reference, hypothesis)

        error_distribution = {
            ErrorType.SUBSTITUTION: wer_result.substitutions,
            ErrorType.DELETION: wer_result.deletions,
            ErrorType.INSERTION: wer_result.insertions,
            ErrorType.CORRECT: wer_result.hits,
        }

        metrics = cls(
            wer=wer_result.wer,
            cer=cer_result.cer,
            error_distribution=error_distribution,
        )

        # TODO: Implement stutter-type specific analysis
        # when word-level stutter annotations are available

        return metrics
