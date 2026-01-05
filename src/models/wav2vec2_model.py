"""
Wav2Vec2 model implementation.

Facebook's wav2vec 2.0 is an important baseline for comparison,
as literature shows significant performance gaps vs Whisper on stuttered speech.
"""

import time
from pathlib import Path
from typing import Literal, Optional

from .base import ASRModel, TranscriptionResult

Wav2Vec2Variant = Literal[
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60-self",
    "facebook/wav2vec2-large-robust-ft-libri-960h",
]


class Wav2Vec2Model(ASRModel):
    """
    Wav2Vec2 ASR model using HuggingFace Transformers.

    Important baseline as literature shows ~30-55% WER on stuttered speech
    vs ~5-21% for Whisper (Boli dataset comparison).
    """

    def __init__(
        self,
        model_id: Wav2Vec2Variant = "facebook/wav2vec2-large-960h-lv60-self",
        device: Optional[str] = None,
    ):
        # Extract short name from model_id
        short_name = model_id.split("/")[-1]
        super().__init__(model_name=f"wav2vec2-{short_name}")
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load the Wav2Vec2 model and processor."""
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_id)

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = self._model.to(self.device)
        self._model.eval()
        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Wav2Vec2."""
        if not self._is_loaded:
            self.load()

        import librosa
        import torch

        start_time = time.time()

        # Load audio at 16kHz (required by wav2vec2)
        audio, sr = librosa.load(str(audio_path), sr=16000)

        # Process input
        inputs = self._processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            logits = self._model(**inputs).logits

        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        text = self._processor.batch_decode(predicted_ids)[0]

        processing_time = time.time() - start_time

        return TranscriptionResult(
            text=text.strip(),
            audio_path=audio_path,
            model_name=self.model_name,
            language=language or "en",
            processing_time_seconds=processing_time,
        )

    def unload(self) -> None:
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            self._is_loaded = False

            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
