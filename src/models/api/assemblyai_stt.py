"""
AssemblyAI Speech-to-Text API wrapper.

AssemblyAI achieves consistently higher accuracy than Google Cloud STT
and handles diverse, challenging audio well - trained on real-world data
rather than clean laboratory datasets.
"""

import time
from pathlib import Path
from typing import Optional

from ..base import ASRModel, TranscriptionResult


class AssemblyAIModel(ASRModel):
    """
    AssemblyAI Speech-to-Text API wrapper.

    Strong performer on challenging audio including disfluent speech.
    Requires ASSEMBLYAI_API_KEY environment variable or explicit key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        speech_model: str = "best",
    ):
        super().__init__(model_name=f"assemblyai-{speech_model}")
        self.api_key = api_key
        self.speech_model = speech_model
        self._client = None

    def load(self) -> None:
        """Initialize the AssemblyAI client."""
        import os

        import assemblyai as aai

        key = self.api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if not key:
            raise ValueError(
                "AssemblyAI API key not found. Set ASSEMBLYAI_API_KEY "
                "environment variable or pass api_key explicitly."
            )

        aai.settings.api_key = key
        self._client = aai.Transcriber()
        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using AssemblyAI."""
        if not self._is_loaded:
            self.load()

        import assemblyai as aai

        start_time = time.time()

        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best
            if self.speech_model == "best"
            else aai.SpeechModel.nano,
            language_code=language or "en",
        )

        transcript = self._client.transcribe(str(audio_path), config=config)

        processing_time = time.time() - start_time

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        # Extract word timestamps
        word_timestamps = None
        if transcript.words:
            word_timestamps = [
                {
                    "word": word.text,
                    "start": word.start / 1000.0,  # Convert ms to seconds
                    "end": word.end / 1000.0,
                }
                for word in transcript.words
            ]

        return TranscriptionResult(
            text=transcript.text or "",
            audio_path=audio_path,
            model_name=self.model_name,
            word_timestamps=word_timestamps,
            confidence=transcript.confidence,
            language=language or "en",
            processing_time_seconds=processing_time,
        )

    def unload(self) -> None:
        """Clean up resources."""
        self._client = None
        self._is_loaded = False
