"""Configuration dataclasses for experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from models.generator import GeneratorConfig
from models.discriminator import DiscriminatorConfig
from models.detector import DetectorConfig
from utils.logging import LoggingConfig


@dataclass
class DataConfig:
    dataset_id: str = "balabit"
    sequence_length: int = 64
    batch_size: int = 64
    num_workers: int = 4
    use_generated_negatives: bool = False
    max_gestures: Optional[int] = None
    cache_enabled: bool = True
    cache_dir: str = "data/processed"
    replay_path: Optional[str] = None
    replay_sample_ratio: float = 1.0
    eval_dataset_ids: Optional[list[str]] = None
    split: Optional[str] = "train"
    user_filter: Optional[list[str]] = None
    cross_eval_manifest: Optional[str] = None
    normalize_sequences: bool = True
    normalize_features: bool = True
    feature_mode: str = "neuromotor"


@dataclass
class GanTrainingConfig:
    epochs: int = 100
    lr_generator: float = 1e-4
    lr_discriminator: float = 4e-4
    beta1: float = 0.5
    beta2: float = 0.9
    gradient_penalty_weight: float = 10.0
    discriminator_steps: int = 5
    log_interval: int = 100
    sample_interval: int = 500
    replay_buffer_path: str = "checkpoints/gan/replay_buffer.pt"
    replay_samples_per_epoch: int = 256
    replay_buffer_max_size: int = 5000
    co_train_detector: bool = False
    detector_update_every: int = 0
    detector_config_path: Optional[str] = None
    detector_epochs_per_update: int = 1
    reconstruction_epochs: int = 0
    reconstruction_loss_weight: float = 1.0
    adversarial_type: str = "wgan"


@dataclass
class DetectorTrainingConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    log_interval: int = 100
    eval_interval: int = 5


@dataclass
class GanModelConfig:
    architecture: str = "tcn"
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig


@dataclass
class GanExperimentConfig:
    experiment_name: str
    seed: int
    data: DataConfig
    model: GanModelConfig
    training: GanTrainingConfig
    logging: LoggingConfig


@dataclass
class DetectorModelConfig:
    detector: DetectorConfig


@dataclass
class DetectorExperimentConfig:
    experiment_name: str
    seed: int
    data: DataConfig
    model: DetectorModelConfig
    training: DetectorTrainingConfig
    logging: LoggingConfig
