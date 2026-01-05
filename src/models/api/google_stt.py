"""
Google Cloud Speech-to-Text API wrapper.

Note: Literature indicates Google STT has poor performance on stuttered speech,
but included as an important baseline for commercial API comparison.
"""

import time
from pathlib import Path
from typing import Optional

from ..base import ASRModel, TranscriptionResult


class GoogleSTTModel(ASRModel):
    """
    Google Cloud Speech-to-Text API wrapper.

    Requires GOOGLE_APPLICATION_CREDENTIALS environment variable
    or explicit credentials file path.

    Note: Benchmarks show poor performance on disfluent speech,
    but important for commercial API comparison.
    """

    def __init__(
        self,
        model: str = "latest_long",
        credentials_path: Optional[str] = None,
    ):
        super().__init__(model_name=f"google-stt-{model}")
        self.model = model
        self.credentials_path = credentials_path
        self._client = None

    def load(self) -> None:
        """Initialize the Google Cloud Speech client."""
        from google.cloud import speech

        if self.credentials_path:
            self._client = speech.SpeechClient.from_service_account_file(
                self.credentials_path
            )
        else:
            self._client = speech.SpeechClient()

        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Google Cloud Speech-to-Text."""
        if not self._is_loaded:
            self.load()

        from google.cloud import speech

        start_time = time.time()

        # Read audio file
        with open(audio_path, "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language or "en-US",
            model=self.model,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        )

        response = self._client.recognize(config=config, audio=audio)

        processing_time = time.time() - start_time

        # Extract text and word timestamps
        text_parts = []
        word_timestamps = []

        for result in response.results:
            alternative = result.alternatives[0]
            text_parts.append(alternative.transcript)

            for word_info in alternative.words:
                word_timestamps.append({
                    "word": word_info.word,
                    "start": word_info.start_time.total_seconds(),
                    "end": word_info.end_time.total_seconds(),
                })

        return TranscriptionResult(
            text=" ".join(text_parts).strip(),
            audio_path=audio_path,
            model_name=self.model_name,
            word_timestamps=word_timestamps if word_timestamps else None,
            language=language or "en-US",
            processing_time_seconds=processing_time,
        )

    def unload(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            self._client = None
            self._is_loaded = False
