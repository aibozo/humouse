"""Sampling utilities for diffusion-based mouse trajectory models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from diffusion.models import UNet1D, UNet1DConfig
from diffusion.noise import DiffusionScheduleConfig, build_schedule, eps_from_v, x0_from_eps, x0_from_v


class DiffusionSampler:
    """DDIM sampler for trajectory diffusion models."""

    def __init__(
        self,
        model: UNet1D,
        schedule,
        *,
        device: torch.device | str = "cpu",
        cond_dim: int = 0,
        in_channels: int = 2,
        self_condition: bool = False,
        objective: str = "v",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.schedule = schedule
        self.cond_dim = cond_dim
        self.in_channels = in_channels
        self.self_condition = self_condition
        self.objective = (objective or "v").lower()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        seq_len: int,
        *,
        cond: Optional[torch.Tensor] = None,
        steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if n <= 0 or seq_len <= 0:
            raise ValueError("'n' and 'seq_len' must be positive integers.")
        if steps <= 0:
            raise ValueError("'steps' must be positive.")

        steps = min(steps, self.schedule.timesteps)
        device = self.device
        x = torch.randn((n, seq_len, self.in_channels), device=device, generator=generator)
        cond_tensor = self._prepare_condition(cond, n)
        self_cond = None

        ddim_timesteps = self._select_timesteps(steps)
        prev_timesteps = torch.cat([ddim_timesteps[1:], torch.tensor([-1], device=device)])

        for t, t_prev in zip(ddim_timesteps, prev_timesteps):
            t_batch = torch.full((n,), int(t.item()), dtype=torch.long, device=device)
            model_in = x.permute(0, 2, 1)
            model_out = self.model(model_in, t_batch, cond=cond_tensor, mask=None, self_cond=self_cond)
            model_out = model_out.permute(0, 2, 1)

            alpha_t, sigma_t = self.schedule.coefficients(t_batch, device=device)
            if self.objective == "epsilon":
                eps = model_out
                x0 = x0_from_eps(x, eps, alpha_t, sigma_t)
            else:
                v_pred = model_out
                x0 = x0_from_v(x, v_pred, alpha_t, sigma_t)
                eps = eps_from_v(v_pred, x0, alpha_t, sigma_t)
            if self.self_condition:
                self_cond = x0.permute(0, 2, 1)

            if t_prev.item() < 0:
                x = x0
                break

            alpha_prev, _ = self.schedule.coefficients(t_prev.unsqueeze(0), device=device)
            alpha_prev = alpha_prev.expand_as(alpha_t)
            a_t = alpha_t ** 2
            a_prev = torch.clamp(alpha_prev ** 2, min=1e-8)
            if eta > 0.0:
                sigma_eta_vec = eta * torch.sqrt(torch.clamp(((1 - a_prev) / (1 - a_t)) * (1 - a_t / a_prev), min=0.0))
            else:
                sigma_eta_vec = torch.zeros_like(a_prev)
            reshape_dims = (-1,) + (1,) * (x.ndim - 1)
            sigma_eta = sigma_eta_vec.view(reshape_dims)
            coeff_eps = torch.sqrt(torch.clamp(1 - a_prev - sigma_eta_vec ** 2, min=0.0)).view(reshape_dims)
            noise = torch.randn_like(x, generator=generator) if eta > 0.0 else torch.zeros_like(x)
            mean = alpha_prev.view(reshape_dims) * x0 + coeff_eps * eps
            x = mean + sigma_eta * noise

        return x

    def _prepare_condition(self, cond: Optional[torch.Tensor], n: int) -> Optional[torch.Tensor]:
        if self.cond_dim == 0:
            return None
        if cond is None:
            return torch.zeros((n, self.cond_dim), device=self.device)
        cond = cond.to(self.device)
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if cond.size(0) == 1 and n > 1:
            cond = cond.expand(n, -1)
        if cond.size(0) != n:
            raise ValueError("Condition batch size must match number of samples.")
        if cond.size(1) > self.cond_dim:
            cond = cond[:, : self.cond_dim]
        elif cond.size(1) < self.cond_dim:
            pad = self.cond_dim - cond.size(1)
            cond = F.pad(cond, (0, pad))
        return cond

    def _select_timesteps(self, steps: int) -> torch.Tensor:
        device = self.device
        if steps == self.schedule.timesteps:
            indices = torch.arange(self.schedule.timesteps - 1, -1, -1, device=device)
        else:
            linspace = torch.linspace(self.schedule.timesteps - 1, 0, steps, device=device)
            indices = torch.round(linspace).long()
            indices = torch.unique_consecutive(indices)
            if indices[-1].item() != 0:
                indices = torch.cat([indices, torch.tensor([0], device=device)])
        return indices


def load_sampler_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: Optional[torch.device | str] = None,
    objective: Optional[str] = None,
) -> DiffusionSampler:
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config")
    if not config:
        raise ValueError("Checkpoint does not contain configuration metadata required for sampling.")
    model_cfg_dict = config.get("model")
    diffusion_cfg_dict = config.get("diffusion")
    if not model_cfg_dict or not diffusion_cfg_dict:
        raise ValueError("Checkpoint config missing 'model' or 'diffusion' entries.")

    training_cfg_dict = config.get("training", {})
    obj = objective or training_cfg_dict.get("objective", "v")

    model_cfg = UNet1DConfig(**model_cfg_dict)
    model = UNet1D(model_cfg)
    ema_state = checkpoint.get("ema")
    if ema_state and "shadow" in ema_state:
        model.load_state_dict(ema_state["shadow"])
    else:
        model.load_state_dict(checkpoint["model"])

    diff_cfg = DiffusionScheduleConfig(**diffusion_cfg_dict)
    schedule = build_schedule(diff_cfg, device=device)

    sampler = DiffusionSampler(
        model,
        schedule,
        device=device,
        cond_dim=model_cfg.cond_dim,
        in_channels=model_cfg.in_channels,
        self_condition=model_cfg.self_condition,
        objective=obj,
    )
    return sampler


def generate_diffusion_samples(
    n: int,
    seq_len: int,
    *,
    cond: Optional[torch.Tensor] = None,
    checkpoint_path: Optional[str | Path] = None,
    model: Optional[UNet1D] = None,
    schedule=None,
    steps: int = 50,
    eta: float = 0.0,
    device: Optional[torch.device | str] = None,
    generator: Optional[torch.Generator] = None,
    objective: str = "v",
) -> torch.Tensor:
    if checkpoint_path is None and (model is None or schedule is None):
        raise ValueError("Provide either a checkpoint_path or both model and schedule.")

    if checkpoint_path is not None:
        sampler = load_sampler_from_checkpoint(checkpoint_path, device=device, objective=objective)
    else:
        device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sampler = DiffusionSampler(
            model,
            schedule,
            device=device,
            cond_dim=model.cfg.cond_dim,
            in_channels=model.cfg.in_channels,
            self_condition=model.cfg.self_condition,
            objective=objective,
        )
    return sampler.sample(n, seq_len, cond=cond, steps=steps, eta=eta, generator=generator)


def _parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(description="Sample mouse trajectories from a diffusion model.")
    parser.add_argument("checkpoint", type=str, help="Path to diffusion checkpoint.")
    parser.add_argument("--n", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length of generated trajectories.")
    parser.add_argument("--steps", type=int, default=50, help="Number of DDIM steps.")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity parameter for DDIM.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu or cuda).")
    parser.add_argument("--out", type=str, default=None, help="Optional path to save generated tensors.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNG.")
    parser.add_argument("--plot", action="store_true", help="Display generated trajectories using matplotlib.")
    parser.add_argument("--plot-out", type=str, default=None, help="Optional path to save a trajectory plot image.")
    return parser.parse_args(argv)


def _plot_samples(samples: torch.Tensor, *, show: bool = False, output_path: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting trajectories") from exc

    samples = samples.detach().cpu()
    num = min(samples.size(0), 16)
    if num == 0:
        print("No samples to plot.")
        return
    cols = min(4, num)
    rows = math.ceil(num / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols]
        if idx >= num:
            ax.axis("off")
            continue
        seq = samples[idx]
        if seq.size(-1) >= 2:
            positions = torch.cumsum(seq[:, :2], dim=0)
            ax.plot(positions[:, 0], positions[:, 1], "-o", markersize=2)
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)
    device = torch.device(args.device) if args.device else None
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device) if args.seed is not None else None
    sampler = load_sampler_from_checkpoint(args.checkpoint, device=device)
    samples = sampler.sample(args.n, args.seq_len, steps=args.steps, eta=args.eta, generator=generator)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples.cpu(), out_path)
        print(f"Saved samples to {out_path}")
    else:
        print(f"Generated samples tensor with shape {tuple(samples.shape)}")

    if args.plot or args.plot_out:
        _plot_samples(samples, show=args.plot, output_path=args.plot_out)


if __name__ == "__main__":
    main()
