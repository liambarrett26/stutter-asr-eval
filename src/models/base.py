"""
Base classes for ASR model implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TranscriptionResult:
    """Result from an ASR transcription."""

    text: str
    """The transcribed text."""

    audio_path: Path
    """Path to the source audio file."""

    model_name: str
    """Name of the model used for transcription."""

    # Optional fields for detailed analysis
    word_timestamps: Optional[list[dict]] = None
    """Word-level timestamps if available."""

    confidence: Optional[float] = None
    """Overall confidence score if available."""

    language: Optional[str] = None
    """Detected or specified language."""

    processing_time_seconds: Optional[float] = None
    """Time taken for transcription."""

    raw_response: Optional[dict] = field(default=None, repr=False)
    """Raw API/model response for debugging."""


class ASRModel(ABC):
    """Abstract base class for ASR model implementations."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.
            language: Optional language code (e.g., 'en', 'de').

        Returns:
            TranscriptionResult with the transcription and metadata.
        """
        pass

    def transcribe_batch(
        self,
        audio_paths: list[Path],
        language: Optional[str] = None,
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files.

        Default implementation processes sequentially.
        Subclasses may override for parallel processing.
        """
        return [self.transcribe(path, language) for path in audio_paths]

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory."""
        pass

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()
        return False
