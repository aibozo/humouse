#!/usr/bin/env python3
"""Profile a limited run of GAN training with torch.profiler."""
from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, schedule as profiler_schedule
from omegaconf import DictConfig, OmegaConf

from data.dataset import GestureDataset
from train.train_gan import (
    _build_experiment_config,
    _prepare_dataset_and_dataloader,
    _position_stats_from_dataset,
    _select_feature_device,
    _training_loop,
)
from train.config_schemas import GanExperimentConfig
from utils.logging import CSVMetricLogger


class LimitedDataLoader:
    """Wrap a DataLoader to expose only a fixed number of batches per epoch."""

    def __init__(self, loader: torch.utils.data.DataLoader, max_batches: int):
        self._loader = loader
        self._max_batches = max_batches
        self.dataset = loader.dataset
        self.batch_size = getattr(loader, "batch_size", None)

    def __iter__(self):
        if self._max_batches <= 0:
            return iter(self._loader)
        return itertools.islice(iter(self._loader), self._max_batches)

    def __len__(self):  # pragma: no cover - simple helper
        base_len = len(self._loader)
        return min(self._max_batches, base_len) if self._max_batches > 0 else base_len


def _load_cfg(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if isinstance(cfg, DictConfig) and "experiment" in cfg:
        cfg = cfg["experiment"]
    return cfg


def _prepare_experiment(cfg: DictConfig) -> tuple[GanExperimentConfig, dict]:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)
    return _build_experiment_config(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GAN training for a few steps.")
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML (e.g., conf/experiment/train_gan.yaml)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to run inside the profiler")
    parser.add_argument("--batches", type=int, default=64, help="Maximum batches per epoch to profile (0 = full epoch)")
    parser.add_argument("--trace", type=Path, help="Optional path to export Chrome trace JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/profile"), help="Directory for profiler artefacts")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    parser.add_argument("--active-steps", type=int, default=1, help="Number of steps to record during the active profiler window")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    cfg = _load_cfg(args.config)
    experiment_cfg, cfg_dict = _prepare_experiment(cfg)

    # Lighten expensive hooks for profiling runs.
    experiment_cfg.training.epochs = args.epochs
    experiment_cfg.training.sigma_eval_enabled = False
    experiment_cfg.training.reconstruction_epochs = 0
    experiment_cfg.training.detector_update_every = 0
    if args.batches > 0:
        bump = args.batches + 1
        experiment_cfg.training.log_interval = max(experiment_cfg.training.log_interval, bump)
        experiment_cfg.training.sample_interval = max(experiment_cfg.training.sample_interval, bump)
    experiment_cfg.training.replay_samples_per_epoch = 0
    experiment_cfg.training.feature_workers = None

    experiment_cfg.data.num_workers = max(1, int(experiment_cfg.data.num_workers or 4))
    if experiment_cfg.data.max_gestures is not None:
        experiment_cfg.data.max_gestures = min(experiment_cfg.data.max_gestures, 1024)
    experiment_cfg.logging.checkpoint_dir = str(args.output_dir)
    Path(experiment_cfg.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    dataset, dataloader = _prepare_dataset_and_dataloader(experiment_cfg)
    if args.batches > 0:
        dataloader = LimitedDataLoader(dataloader, args.batches)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sigma_feature_device = _select_feature_device(experiment_cfg.training.sigma_feature_device, device)

    if experiment_cfg.model.architecture == "tcn":
        from models.generator import ConditionalGenerator
        from models.discriminator import GestureDiscriminator

        generator = ConditionalGenerator(experiment_cfg.model.generator).to(device)
        discriminator = GestureDiscriminator(experiment_cfg.model.discriminator).to(device)
    elif experiment_cfg.model.architecture == "lstm":
        from models.gan_lstm import LSTMDiscriminator, LSTMGenerator

        generator = LSTMGenerator(experiment_cfg.model.generator).to(device)
        discriminator = LSTMDiscriminator(experiment_cfg.model.discriminator).to(device)
    elif experiment_cfg.model.architecture == "seq2seq":
        from models.seq2seq import Seq2SeqGenerator
        from models.gan_lstm import LSTMDiscriminator

        generator = Seq2SeqGenerator(experiment_cfg.model.generator).to(device)
        discriminator = LSTMDiscriminator(experiment_cfg.model.discriminator).to(device)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported architecture: {experiment_cfg.model.architecture}")

    position_stats = None
    if experiment_cfg.training.absolute_coordinates:
        pos_mean, pos_std = _position_stats_from_dataset(dataset, 2)
        position_stats = (pos_mean[:, :, :2], pos_std[:, :, :2])

    _metrics_path = Path(experiment_cfg.logging.checkpoint_dir) / "profile_metrics.csv"
    metrics_logger = CSVMetricLogger(_metrics_path)

    logging.info("Profiling on device=%s for epochs=%d, batches=%d", device, experiment_cfg.training.epochs, args.batches)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    active_steps = max(1, args.active_steps)
    trace_handler = None
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)

        def _trace_export(prof):  # pragma: no cover - simple IO callback
            prof.export_chrome_trace(str(args.trace))

        trace_handler = _trace_export

    with torch.profiler.profile(
        activities=activities,
        schedule=profiler_schedule(wait=0, warmup=1, active=active_steps, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=False,
        profile_memory=False,
    ) as prof:
        summary = _training_loop(
            experiment_cfg,
            generator,
            discriminator,
            dataset,
            dataloader,
            device,
            sigma_feature_device,
            run=None,
            metrics_logger=metrics_logger,
            position_stats=position_stats,
            profiler=prof,
        )
        # Ensure the profiler advances to flush any buffered events.
        prof.step()
        try:
            cpu_summary = prof.key_averages(group_by_input_shape=False).table(
                sort_by="self_cpu_time_total", row_limit=20
            )
            logging.info("Top CPU operators:\n%s", cpu_summary)
            if torch.cuda.is_available():
                gpu_summary = prof.key_averages(group_by_input_shape=False).table(
                    sort_by="self_cuda_time_total", row_limit=20
                )
                logging.info("Top CUDA operators:\n%s", gpu_summary)
        except Exception as exc:  # pragma: no cover - diagnostic path
            logging.warning("Failed to summarise profiler results: %s", exc)

    logging.info("Training summary: g_loss=%.4f | d_loss=%.4f", summary.get("g_loss", 0.0), summary.get("d_loss", 0.0))
    logging.info("Profiler run complete. Re-launch with --trace to capture a Chrome trace if needed.")
if __name__ == "__main__":
    main()
