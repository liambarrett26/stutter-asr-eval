"""
ASR model implementations and wrappers.

This module provides a unified interface for various ASR models:
- Open-source models (Whisper, wav2vec2, HuBERT)
- Commercial APIs (Google, Azure, Amazon, AssemblyAI, Deepgram)
"""

from .base import ASRModel, TranscriptionResult

__all__ = ["ASRModel", "TranscriptionResult"]
