"""
OpenAI Whisper model implementation.

Supports multiple Whisper variants:
- whisper (original OpenAI implementation)
- faster-whisper (CTranslate2 optimized)
- distil-whisper (distilled, faster variant)
"""

import time
from pathlib import Path
from typing import Literal, Optional

from .base import ASRModel, TranscriptionResult

WhisperSize = Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


class WhisperModel(ASRModel):
    """
    OpenAI Whisper ASR model.

    Uses the original OpenAI Whisper implementation.
    For faster inference, consider FasterWhisperModel.
    """

    def __init__(
        self,
        size: WhisperSize = "large-v3",
        device: Optional[str] = None,
    ):
        super().__init__(model_name=f"whisper-{size}")
        self.size = size
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the Whisper model."""
        import whisper

        self._model = whisper.load_model(self.size, device=self.device)
        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Whisper."""
        if not self._is_loaded:
            self.load()

        start_time = time.time()

        options = {}
        if language:
            options["language"] = language

        result = self._model.transcribe(str(audio_path), **options)

        processing_time = time.time() - start_time

        # Extract word timestamps if available
        word_timestamps = None
        if "segments" in result:
            word_timestamps = [
                {
                    "word": seg.get("text", "").strip(),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                }
                for seg in result["segments"]
            ]

        return TranscriptionResult(
            text=result["text"].strip(),
            audio_path=audio_path,
            model_name=self.model_name,
            word_timestamps=word_timestamps,
            language=result.get("language"),
            processing_time_seconds=processing_time,
            raw_response=result,
        )

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False

            # Force CUDA memory cleanup if using GPU
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


class FasterWhisperModel(ASRModel):
    """
    Faster-Whisper implementation using CTranslate2.

    Significantly faster than original Whisper with similar accuracy.
    Recommended for large-scale evaluation.
    """

    def __init__(
        self,
        size: WhisperSize = "large-v3",
        device: str = "auto",
        compute_type: str = "float16",
    ):
        super().__init__(model_name=f"faster-whisper-{size}")
        self.size = size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        """Load the Faster-Whisper model."""
        from faster_whisper import WhisperModel as FW

        self._model = FW(
            self.size,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Faster-Whisper."""
        if not self._is_loaded:
            self.load()

        start_time = time.time()

        options = {}
        if language:
            options["language"] = language

        segments, info = self._model.transcribe(str(audio_path), **options)

        # Collect all segments
        all_segments = list(segments)
        text = " ".join(seg.text.strip() for seg in all_segments)

        processing_time = time.time() - start_time

        word_timestamps = [
            {
                "word": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
            }
            for seg in all_segments
        ]

        return TranscriptionResult(
            text=text,
            audio_path=audio_path,
            model_name=self.model_name,
            word_timestamps=word_timestamps,
            language=info.language,
            processing_time_seconds=processing_time,
        )

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
