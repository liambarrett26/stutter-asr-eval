"""
Amazon Transcribe API wrapper.
"""

import time
import uuid
from pathlib import Path
from typing import Optional

from ..base import ASRModel, TranscriptionResult


class AmazonTranscribeModel(ASRModel):
    """
    Amazon Transcribe API wrapper.

    Requires AWS credentials configured via environment variables
    or AWS credentials file.

    Note: Good accuracy for pre-recorded audio, but slower than
    specialized providers.
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        s3_bucket: Optional[str] = None,
    ):
        super().__init__(model_name="amazon-transcribe")
        self.region_name = region_name
        self.s3_bucket = s3_bucket
        self._transcribe_client = None
        self._s3_client = None

    def load(self) -> None:
        """Initialize AWS clients."""
        import boto3

        self._transcribe_client = boto3.client(
            "transcribe",
            region_name=self.region_name,
        )
        self._s3_client = boto3.client(
            "s3",
            region_name=self.region_name,
        )
        self._is_loaded = True

    def transcribe(
        self,
        audio_path: Path,
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio using Amazon Transcribe."""
        if not self._is_loaded:
            self.load()

        import json

        import boto3

        start_time = time.time()

        # Generate unique job name
        job_name = f"stutter-eval-{uuid.uuid4().hex[:8]}"

        # Upload to S3 if bucket provided
        if self.s3_bucket:
            s3_key = f"audio/{audio_path.name}"
            self._s3_client.upload_file(
                str(audio_path),
                self.s3_bucket,
                s3_key,
            )
            media_uri = f"s3://{self.s3_bucket}/{s3_key}"
        else:
            raise ValueError(
                "S3 bucket required for Amazon Transcribe. "
                "Provide s3_bucket parameter."
            )

        # Start transcription job
        self._transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={"MediaFileUri": media_uri},
            MediaFormat=audio_path.suffix.lstrip("."),
            LanguageCode=language or "en-US",
        )

        # Wait for completion
        while True:
            response = self._transcribe_client.get_transcription_job(
                TranscriptionJobName=job_name
            )
            status = response["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                break
            elif status == "FAILED":
                raise RuntimeError(
                    f"Transcription failed: "
                    f"{response['TranscriptionJob'].get('FailureReason')}"
                )

            time.sleep(2)

        # Get transcript
        transcript_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]

        # Download transcript
        import urllib.request

        with urllib.request.urlopen(transcript_uri) as resp:
            transcript_data = json.loads(resp.read().decode())

        text = transcript_data["results"]["transcripts"][0]["transcript"]

        # Extract word timestamps
        word_timestamps = []
        if "items" in transcript_data["results"]:
            for item in transcript_data["results"]["items"]:
                if item["type"] == "pronunciation":
                    word_timestamps.append({
                        "word": item["alternatives"][0]["content"],
                        "start": float(item.get("start_time", 0)),
                        "end": float(item.get("end_time", 0)),
                    })

        processing_time = time.time() - start_time

        # Clean up S3 file and job
        try:
            self._s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
            self._transcribe_client.delete_transcription_job(
                TranscriptionJobName=job_name
            )
        except Exception:
            pass  # Best effort cleanup

        return TranscriptionResult(
            text=text.strip(),
            audio_path=audio_path,
            model_name=self.model_name,
            word_timestamps=word_timestamps if word_timestamps else None,
            language=language or "en-US",
            processing_time_seconds=processing_time,
        )

    def unload(self) -> None:
        """Clean up resources."""
        self._transcribe_client = None
        self._s3_client = None
        self._is_loaded = False
