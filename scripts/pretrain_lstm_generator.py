"""Lightweight LSTM generator pretraining on real gestures.

The script trains the paper-mode LSTM generator with a simple reconstruction
objective (predict accumulated (x, y) trajectories) so we can inspect the
resulting checkpoint and generated traces before adversarial training kicks in.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from data.dataset import GestureDataset, GestureDatasetConfig
from models.gan_lstm import LSTMGenerator
from train.config_schemas import GeneratorConfig


def _deltas_to_positions(sequence: torch.Tensor) -> torch.Tensor:
    """Cumulative sum of Δx/Δy to obtain absolute positions."""
    return torch.cumsum(sequence[..., :2], dim=1)


def _positions_to_deltas(positions: torch.Tensor) -> torch.Tensor:
    first = positions[:, :1, :]
    diffs = positions[:, 1:, :] - positions[:, :-1, :]
    return torch.cat([first, diffs], dim=1)


def _compute_position_stats(dataset: GestureDataset) -> tuple[torch.Tensor, torch.Tensor]:
    sum_vec = torch.zeros(2, dtype=torch.float64)
    sum_sq = torch.zeros(2, dtype=torch.float64)
    count = 0
    for seq, _, _ in dataset.samples:
        pos = torch.cumsum(seq[:, :2], dim=0).double()
        sum_vec += pos.sum(dim=0)
        sum_sq += (pos**2).sum(dim=0)
        count += pos.shape[0]
    mean = (sum_vec / max(count, 1)).float()
    var = (sum_sq / max(count, 1) - mean.double() ** 2).clamp_min(1e-6)
    std = var.sqrt().float()
    return mean, std


def _apply_output_rescale(generator: LSTMGenerator, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> None:
    with torch.no_grad():
        std = std.to(device)
        mean = mean.to(device)
        weight = generator.output.weight.data  # shape (2, hidden)
        bias = generator.output.bias.data
        weight.mul_(std.view(-1, 1))
        bias.mul_(std).add_(mean)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain the LSTM generator with a reconstruction loss.")
    parser.add_argument("--dataset", default="balabit", help="Dataset identifier")
    parser.add_argument("--sequence-length", type=int, default=200, help="Resampled gesture length")
    parser.add_argument("--sampling-rate", type=float, default=200.0, help="Resampling rate (Hz)")
    parser.add_argument("--max-gestures", type=int, default=4096, help="Maximum gestures to load")
    parser.add_argument("--epochs", type=int, default=10, help="Pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate")
    parser.add_argument("--device", default="auto", help="Training device (cpu, cuda, auto)")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pretrain/lstm_generator.pt"), help="Checkpoint path to save")
    parser.add_argument("--samples-out", type=Path, default=Path("checkpoints/pretrain/pretrain_samples.npz"), help="Optional NPZ of generated samples")
    parser.add_argument("--seed", type=int, default=2021, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataset_cfg = GestureDatasetConfig(
        dataset_id=args.dataset,
        sequence_length=args.sequence_length,
        sampling_rate=args.sampling_rate,
        max_gestures=args.max_gestures,
        cache_enabled=False,
        feature_mode="sigma_lognormal",
        normalize_sequences=False,
        normalize_features=False,
        canonicalize_path=False,
        canonicalize_duration=False,
    )
    dataset = GestureDataset(dataset_cfg)
    if len(dataset) == 0:
        raise RuntimeError("Dataset produced zero gestures; check configuration")

    pos_mean, pos_std = _compute_position_stats(dataset)
    pos_mean = pos_mean.view(1, 1, 2)
    pos_std = pos_std.view(1, 1, 2)

    class WarmupDataset(Dataset):
        def __init__(self, base):
            self.base = base

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, idx: int):
            seq, _, _ = self.base[idx]
            return idx, seq

    warmup_ds = WarmupDataset(dataset)
    dataloader = DataLoader(warmup_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    gen_cfg = GeneratorConfig(
        latent_dim=100,
        condition_dim=0,
        hidden_dim=128,
        target_len=args.sequence_length,
        num_layers=2,
        use_spectral_norm=False,
        output_dim=2,
        activation_xy="linear",
        activation_dt="none",
    )
    generator = LSTMGenerator(gen_cfg).to(device)
    latent_table = torch.nn.Embedding(len(dataset), gen_cfg.latent_dim).to(device)
    optimizer = optim.Adam(
        list(generator.parameters()) + list(latent_table.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    criterion = nn.L1Loss()

    generator.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for indices, sequences in dataloader:
            sequences = sequences.to(device)
            target_positions = _deltas_to_positions(sequences)
            target_norm = (target_positions - pos_mean.to(device)) / pos_std.to(device)
            z = latent_table(indices.to(device))
            cond = torch.zeros(sequences.size(0), gen_cfg.condition_dim, device=device)

            preds = generator(z, cond)
            loss = criterion(preds, target_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences.size(0)
            total_samples += sequences.size(0)

        avg_loss = total_loss / max(1, total_samples)
        print(f"Epoch {epoch:02d} | recon_loss={avg_loss:.6f}")

    _apply_output_rescale(generator, pos_mean.squeeze(0).squeeze(0), pos_std.squeeze(0).squeeze(0), device)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": generator.state_dict(),
            "config": gen_cfg.__dict__,
            "position_mean": pos_mean.cpu().numpy().squeeze(),
            "position_std": pos_std.cpu().numpy().squeeze(),
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint to {args.checkpoint}")

    generator.eval()
    with torch.no_grad():
        sample_count = min(64, len(dataset))
        sample_indices = torch.arange(sample_count, device=device)
        preds = generator(latent_table(sample_indices), torch.zeros(sample_count, gen_cfg.condition_dim, device=device)).cpu().numpy()
        deltas = _positions_to_deltas(torch.from_numpy(preds)).numpy()
        dt = np.full((deltas.shape[0], deltas.shape[1], 1), 1.0 / args.sampling_rate if args.sampling_rate else 1.0, dtype=np.float32)
        sequences = np.concatenate([deltas.astype(np.float32), dt.astype(np.float32)], axis=-1)
        args.samples_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.samples_out, sequences=sequences)
        print(f"Wrote sample NPZ to {args.samples_out}")


if __name__ == "__main__":
    main()
