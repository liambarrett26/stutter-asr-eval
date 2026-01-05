# stutter-asr-eval

Evaluating and fine-tuning state-of-the-art automatic speech recognition (ASR) systems for stuttered speech.

## Description

This repository evaluates state-of-the-art ASR models on stuttered British and American English speech, explores fine-tuning strategies to improve recognition accuracy for people who stutter (PWS), and investigates phonetic-level transcription capabilities. Current ASR systems are typically trained on fluent speech and struggle with dysfluencies such as repetitions, prolongations, and blocks. This project aims to benchmark performance gaps and develop methods to improve ASR accessibility for PWS.

## Overview

Speech recognition technology has made remarkable advances, but these systems often fail when encountering stuttered speech. This creates significant barriers for people who stutter in accessing voice-controlled technology, transcription services, and other speech-dependent applications. This project addresses three key questions:

1. **Baseline Performance**: How do current state-of-the-art ASR models perform on stuttered speech compared to fluent speech?
2. **Fine-tuning for Accuracy**: Can these models be fine-tuned to better recognize stuttered speech patterns including whole-word repetitions, part-word repetitions, prolongations, and blocks?
3. **Phonetic Transcription**: Can ASR models be adapted to produce phonetic-level transcriptions that capture the specific articulation patterns in stuttered speech?

## Features

- Comprehensive evaluation of multiple SoTA ASR models (Whisper, Wav2Vec2, etc.)
- WER, CER, and stuttering-specific metrics calculation
- Support for British and American English stuttered speech corpora
- Fine-tuning pipelines for improved dysfluency recognition
- Phonetic-level transcription evaluation
- Comparative analysis tools and visualization

## Project Status

🚧 Under active development
