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
    canonicalize_path: bool = False
    canonicalize_duration: bool = False
    sampling_rate: Optional[float | str] = None
    min_events: int = 5
    include_goal_geometry: bool = False
    use_click_boundaries: bool = False
    click_button: Optional[str] = "left"
    direction_buckets: Optional[list[int]] = None
    rotate_to_buckets: bool = False
    min_path_length: float = 0.0
    min_path_length: float = 0.0
    feature_reservoir_size: Optional[int] = None
    prefetch_factor: int = 2


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
    metric_log_interval: Optional[int] = None
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
    sigma_eval_enabled: bool = False
    sigma_eval_interval: int = 1
    sigma_eval_samples: int = 512
    sigma_eval_dataset_id: Optional[str] = None
    sigma_eval_max_gestures: int = 4096
    sigma_eval_make_plots: bool = True
    absolute_coordinates: bool = False
    generator_init_path: Optional[str] = None
    curvature_match_weight: float = 0.0
    lateral_match_weight: float = 0.0
    direction_match_weight: float = 0.0
    warmup_encoder_hidden_dim: int = 128
    warmup_encoder_layers: int = 1
    warmup_encoder_dropout: float = 0.0
    warmup_noise_std: float = 0.0
    warmup_latent_normalize: bool = True
    warmup_kl_weight: float = 0.1
    reset_generator_after_warmup: bool = False
    cold_start_epochs: int = 0
    sigma_freeze_upper: Optional[float] = None
    sigma_freeze_lower: Optional[float] = None
    label_smoothing_real: float = 1.0
    label_smoothing_fake: float = 0.0
    r1_gamma: float = 0.0
    feature_workers: Optional[int] = None
    sigma_feature_device: str = "auto"
    adaptive_freeze_enabled: bool = False
    adaptive_freeze_target: float = 0.6
    adaptive_freeze_margin: float = 0.05
    adaptive_freeze_warmup: int = 50
    adaptive_freeze_smoothing: float = 0.01
    adaptive_freeze_cooldown: int = 0
    adaptive_freeze_freeze_generator: bool = True
    fake_batch_ratio: float = 0.5
    fake_batch_ratio_start: float = 0.2
    fake_batch_ratio_warmup: int = 50
    ada_enabled: bool = False
    ada_target: float = 0.6
    ada_interval: int = 16
    ada_rate: float = 0.05
    ada_p_init: float = 0.0
    ada_p_max: float = 0.9
    adaptive_lr_enabled: bool = False
    adaptive_lr_target: float = 0.75
    adaptive_lr_warmup: int = 0
    adaptive_lr_error_gain: float = 2.0
    adaptive_lr_derivative_gain: float = 1.0
    adaptive_lr_min: float = 1e-6
    adaptive_lr_max: float = 2e-4
    diffusion_eval_enabled: bool = False
    diffusion_eval_checkpoint: Optional[str] = None
    diffusion_eval_samples: int = 512
    diffusion_eval_steps: int = 50
    diffusion_eval_eta: float = 0.0
    diffusion_eval_seq_len: Optional[int] = None
    diffusion_eval_interval: int = 1
    diffusion_eval_log_dir: str = "diffusion_eval"


@dataclass
class DetectorTrainingConfig:
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    log_interval: int = 100
    eval_interval: int = 5


@dataclass
class GanModelConfig:
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig
    architecture: str = "tcn"


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
