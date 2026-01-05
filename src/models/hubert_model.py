"""
HuBERT model implementation.

HuBERT (Hidden-Unit BERT) is a self-supervised speech representation model
that learns discrete hidden units for speech. Important for comparison
as it represents a different pretraining approach than wav2vec2.
"""

import time
from pathlib import Path
from typing import Literal, Optional

from .base import ASRModel, TranscriptionResult

HuBERTVariant = Literal[
    "facebook/hubert-large-ls960-ft",
    "facebook/hubert-xlarge-ls960-ft",
]


class HuBERTModel(ASRModel):
    """
    HuBERT ASR model using HuggingFace Transformers.

    Self-supervised model with different pretraining approach than wav2vec2.
    Mentioned in project research docs as a key comparison model.
    """

    def __init__(
        self,
        model_id: HuBERTVariant = "facebook/hubert-large-ls960-ft",
        device: Optional[str] = None,
    ):
        short_name = model_id.split("/")[-1]
        super().__init__(model_name=f"hubert-{short_name}")
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Load the HuBERT model and processor."""
        import torch
        from transformers import HubertForCTC, Wav2Vec2Processor

        # HuBERT uses the same processor as wav2vec2
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        self._model = HubertForCTC.from_pretrained(self.model_id)

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
        """Transcribe audio using HuBERT."""
        if not self._is_loaded:
            self.load()

        import librosa
        import torch

        start_time = time.time()

        # Load audio at 16kHz
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
