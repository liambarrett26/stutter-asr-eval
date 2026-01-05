"""
Microsoft Azure Speech Services API wrapper.
"""

import time
from pathlib import Path
from typing import Optional

from ..base import ASRModel, TranscriptionResult


class AzureSTTModel(ASRModel):
    """
    Microsoft Azure Speech-to-Text API wrapper.

    Requires AZURE_SPEECH_KEY and AZURE_SPEECH_REGION environment variables
    or explicit configuration.
    """

    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: Optional[str] = None,
    ):
        super().__init__(model_name="azure-stt")
        self.subscription_key = subscription_key
        self.region = region
        self._speech_config = None

    def load(self) -> None:
        """Initialize the Azure Speech configuration."""
        import os

        import azure.cognitiveservices.speech as speechsdk

        key = self.subscription_key or os.environ.get("AZURE_SPEECH_KEY")
        region = self.region or os.environ.get("AZURE_SPEECH_REGION")

        if not key or not region:
            raise ValueError(
                "Azure Speech credentials not found. Set AZURE_SPEECH_KEY and "
                "AZURE_SPEECH_REGION environment variables or pass explicitly."
            )

        self._speech_config = speechsdk.SpeechConfig(
            subscription=key,
            region=region,
        )
        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Azure Speech-to-Text."""
        if not self._is_loaded:
            self.load()

        import azure.cognitiveservices.speech as speechsdk

        start_time = time.time()

        if language:
            self._speech_config.speech_recognition_language = language

        audio_config = speechsdk.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=audio_config,
        )

        # Use continuous recognition to get full transcription
        all_results = []
        done = False

        def handle_result(evt):
            all_results.append(evt.result.text)

        def handle_stopped(evt):
            nonlocal done
            done = True

        recognizer.recognized.connect(handle_result)
        recognizer.session_stopped.connect(handle_stopped)
        recognizer.canceled.connect(handle_stopped)

        recognizer.start_continuous_recognition()
        while not done:
            time.sleep(0.1)
        recognizer.stop_continuous_recognition()

        processing_time = time.time() - start_time

        return TranscriptionResult(
            text=" ".join(all_results).strip(),
            audio_path=audio_path,
            model_name=self.model_name,
            language=language,
            processing_time_seconds=processing_time,
        )

    def unload(self) -> None:
        """Clean up resources."""
        self._speech_config = None
        self._is_loaded = False
