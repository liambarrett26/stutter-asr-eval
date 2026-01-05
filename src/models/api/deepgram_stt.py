"""
Deepgram Speech-to-Text API wrapper.

Deepgram Nova-3 is one of the top commercial ASR providers,
achieving <5% WER on standard benchmarks.
"""

import time
from pathlib import Path
from typing import Literal, Optional

from ..base import ASRModel, TranscriptionResult

DeepgramModel = Literal["nova-2", "nova-3", "enhanced", "base"]


class DeepgramModel(ASRModel):
    """
    Deepgram Speech-to-Text API wrapper.

    Top commercial performer with Nova-3 model.
    Requires DEEPGRAM_API_KEY environment variable or explicit key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: DeepgramModel = "nova-3",
    ):
        super().__init__(model_name=f"deepgram-{model}")
        self.api_key = api_key
        self.model = model
        self._client = None

    def load(self) -> None:
        """Initialize the Deepgram client."""
        import os

        from deepgram import DeepgramClient

        key = self.api_key or os.environ.get("DEEPGRAM_API_KEY")
        if not key:
            raise ValueError(
                "Deepgram API key not found. Set DEEPGRAM_API_KEY "
                "environment variable or pass api_key explicitly."
            )

        self._client = DeepgramClient(key)
        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Deepgram."""
        if not self._is_loaded:
            self.load()

        from deepgram import FileSource, PrerecordedOptions

        start_time = time.time()

        # Read audio file
        with open(audio_path, "rb") as f:
            buffer_data = f.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model=self.model,
            language=language or "en",
            punctuate=True,
            utterances=True,
            smart_format=True,
        )

        response = self._client.listen.rest.v("1").transcribe_file(
            payload, options
        )

        processing_time = time.time() - start_time

        # Extract transcript
        result = response.results
        if not result or not result.channels:
            return TranscriptionResult(
                text="",
                audio_path=audio_path,
                model_name=self.model_name,
                processing_time_seconds=processing_time,
            )

        channel = result.channels[0]
        text = channel.alternatives[0].transcript if channel.alternatives else ""

        # Extract word timestamps
        word_timestamps = None
        if channel.alternatives and channel.alternatives[0].words:
            word_timestamps = [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end,
                }
                for word in channel.alternatives[0].words
            ]

        confidence = (
            channel.alternatives[0].confidence if channel.alternatives else None
        )

        return TranscriptionResult(
            text=text.strip(),
            audio_path=audio_path,
            model_name=self.model_name,
            word_timestamps=word_timestamps,
            confidence=confidence,
            language=language or "en",
            processing_time_seconds=processing_time,
        )

    def unload(self) -> None:
        """Clean up resources."""
        self._client = None
        self._is_loaded = False
