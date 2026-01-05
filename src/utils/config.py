"""
Configuration management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for a single ASR model."""

    name: str
    type: str  # 'whisper', 'wav2vec2', 'hubert', 'api'
    variant: Optional[str] = None
    device: str = "auto"
    api_key_env: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    metrics: list[str] = field(default_factory=lambda: ["wer", "cer"])
    normalize_text: bool = True
    calculate_semantic: bool = False
    semantic_model: str = "microsoft/deberta-xlarge-mnli"


@dataclass
class Config:
    """Main configuration class."""

    models: list[ModelConfig] = field(default_factory=list)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data_dir: Path = Path("data")
    output_dir: Path = Path("results")

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        models = [ModelConfig(**m) for m in data.get("models", [])]
        eval_config = EvaluationConfig(**data.get("evaluation", {}))

        return cls(
            models=models,
            evaluation=eval_config,
            data_dir=Path(data.get("data_dir", "data")),
            output_dir=Path(data.get("output_dir", "results")),
        )


def load_config(path: Optional[Path] = None) -> Config:
    """
    Load configuration from file or return defaults.

    Args:
        path: Path to YAML config file. If None, uses defaults.

    Returns:
        Config object.
    """
    if path is None:
        return Config()

    return Config.from_yaml(path)
