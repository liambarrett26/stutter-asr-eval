"""
Dataset classes for stuttered speech corpora.

Supports loading from:
- UCLASS (UCL Archive of Stuttered Speech)
- FluencyBank
- SEP-28k (for stutter event labels)
- Custom datasets
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from src.evaluation.metrics import StutterType


class TranscriptType(Enum):
    """Type of transcript reference."""

    SURFACE = "surface"  # Include disfluencies as spoken
    INTENDED = "intended"  # Normalized fluent speech


@dataclass
class AudioSample:
    """A single audio sample with metadata."""

    audio_path: Path
    """Path to the audio file."""

    transcript: str
    """Reference transcript."""

    sample_id: str
    """Unique identifier for the sample."""

    transcript_type: TranscriptType = TranscriptType.SURFACE
    """Whether transcript includes disfluencies or is normalized."""

    speaker_id: Optional[str] = None
    """Speaker identifier for speaker-specific analysis."""

    severity: Optional[str] = None
    """Stuttering severity label (mild, moderate, severe)."""

    stutter_annotations: list[dict] = field(default_factory=list)
    """Word-level stutter type annotations."""

    duration_seconds: Optional[float] = None
    """Audio duration in seconds."""

    metadata: dict = field(default_factory=dict)
    """Additional metadata."""


class StutteredSpeechDataset:
    """
    Base class for stuttered speech datasets.

    Provides unified interface for loading different corpora.
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "test",
        transcript_type: TranscriptType = TranscriptType.SURFACE,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transcript_type = transcript_type
        self._samples: list[AudioSample] = []

    def load(self) -> None:
        """Load dataset samples. Override in subclasses."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[AudioSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> AudioSample:
        return self._samples[idx]

    def filter_by_severity(self, severity: str) -> list[AudioSample]:
        """Filter samples by stuttering severity."""
        return [s for s in self._samples if s.severity == severity]

    def filter_by_speaker(self, speaker_id: str) -> list[AudioSample]:
        """Filter samples by speaker."""
        return [s for s in self._samples if s.speaker_id == speaker_id]

    def get_speakers(self) -> list[str]:
        """Get list of unique speakers."""
        return list(set(s.speaker_id for s in self._samples if s.speaker_id))


class FluencyBankDataset(StutteredSpeechDataset):
    """
    FluencyBank dataset loader.

    FluencyBank is a key benchmark for stuttered speech ASR,
    used across nearly all recent research.
    """

    def load(self) -> None:
        """Load FluencyBank samples."""
        # TODO: Implement FluencyBank loading
        # Structure typically:
        # - Audio files in WAV format
        # - Transcripts with timing information
        # - Speaker and session metadata
        raise NotImplementedError(
            "FluencyBank loader not yet implemented. "
            "Requires access to FluencyBank corpus."
        )


class UCLASSDataset(StutteredSpeechDataset):
    """
    UCLASS (UCL Archive of Stuttered Speech) dataset loader.

    Limited word-level transcriptions available (~15 recordings).
    """

    def load(self) -> None:
        """Load UCLASS samples."""
        # TODO: Implement UCLASS loading
        raise NotImplementedError(
            "UCLASS loader not yet implemented. "
            "Requires access to UCLASS corpus from UCL."
        )


class CustomDataset(StutteredSpeechDataset):
    """
    Custom dataset loader for user-provided data.

    Expected directory structure:
    root_dir/
        audio/
            sample1.wav
            sample2.wav
            ...
        transcripts/
            sample1.txt
            sample2.txt
            ...
        metadata.json (optional)
    """

    def load(self) -> None:
        """Load custom dataset samples."""
        audio_dir = self.root_dir / "audio"
        transcript_dir = self.root_dir / "transcripts"

        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # Load metadata if available
        metadata_file = self.root_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            import json

            with open(metadata_file) as f:
                metadata = json.load(f)

        # Find all audio files
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        audio_files = [
            f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions
        ]

        for audio_file in sorted(audio_files):
            sample_id = audio_file.stem

            # Find corresponding transcript
            transcript = ""
            transcript_file = transcript_dir / f"{sample_id}.txt"
            if transcript_file.exists():
                transcript = transcript_file.read_text().strip()

            # Get sample metadata
            sample_meta = metadata.get(sample_id, {})

            self._samples.append(
                AudioSample(
                    audio_path=audio_file,
                    transcript=transcript,
                    sample_id=sample_id,
                    transcript_type=self.transcript_type,
                    speaker_id=sample_meta.get("speaker_id"),
                    severity=sample_meta.get("severity"),
                    duration_seconds=sample_meta.get("duration"),
                    metadata=sample_meta,
                )
            )
