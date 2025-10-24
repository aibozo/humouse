"""UNet-style backbone for trajectory diffusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class UNet1DConfig:
    """Configuration for the 1D UNet used in diffusion training."""

    in_channels: int = 2
    out_channels: int = 2
    base_channels: int = 128
    channel_mults: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.0)
    num_res_blocks: int = 2
    dropout: float = 0.1
    norm_groups: int = 32
    use_attention: bool = True
    attn_heads: int = 4
    self_condition: bool = False
    cond_dim: int = 0
    time_embed_dim: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.channel_mults, tuple):
            self.channel_mults = tuple(self.channel_mults)


class UNet1D(nn.Module):
    """Time-conditionable UNet with optional self-conditioning and FiLM conditioning."""

    def __init__(self, cfg: UNet1DConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.self_condition = cfg.self_condition
        self.cond_dim = cfg.cond_dim

        time_embed_dim = cfg.time_embed_dim or cfg.base_channels * 4
        self.time_embed_dim = time_embed_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        if cfg.cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cfg.cond_dim, time_embed_dim * 2),
                nn.SiLU(),
                nn.Linear(time_embed_dim * 2, time_embed_dim),
            )
        else:
            self.cond_mlp = None

        input_channels = cfg.in_channels + (cfg.in_channels if cfg.self_condition else 0)
        self.input_proj = nn.Conv1d(input_channels, cfg.base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels_per_stage: list[list[int]] = []
        self.stage_out_channels: list[int] = []
        in_ch = cfg.base_channels

        for idx, mult in enumerate(cfg.channel_mults):
            stage_blocks = nn.ModuleList()
            stage_skips: list[int] = []
            out_ch = int(cfg.base_channels * mult)
            for _ in range(cfg.num_res_blocks):
                block = ResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    embed_dim=time_embed_dim,
                    dropout=cfg.dropout,
                    norm_groups=cfg.norm_groups,
                    cond_enabled=cfg.cond_dim > 0,
                )
                stage_blocks.append(block)
                stage_skips.append(out_ch)
                in_ch = out_ch
            self.stage_out_channels.append(in_ch)
            self.down_blocks.append(stage_blocks)
            self.skip_channels_per_stage.append(stage_skips)
            if idx != len(cfg.channel_mults) - 1:
                self.downsamples.append(Downsample(in_ch))
            else:
                self.downsamples.append(None)

        self.mid_block1 = ResidualBlock(
            in_channels=in_ch,
            out_channels=in_ch,
            embed_dim=time_embed_dim,
            dropout=cfg.dropout,
            norm_groups=cfg.norm_groups,
            cond_enabled=cfg.cond_dim > 0,
        )
        if cfg.use_attention:
            self.mid_attn = TemporalSelfAttention(in_ch, cfg.attn_heads)
        else:
            self.mid_attn = nn.Identity()
        self.mid_block2 = ResidualBlock(
            in_channels=in_ch,
            out_channels=in_ch,
            embed_dim=time_embed_dim,
            dropout=cfg.dropout,
            norm_groups=cfg.norm_groups,
            cond_enabled=cfg.cond_dim > 0,
        )

        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_idx in reversed(range(len(cfg.channel_mults))):
            stage_blocks = nn.ModuleList()
            stage_skips = self.skip_channels_per_stage[stage_idx]
            out_ch = self.stage_out_channels[stage_idx]
            for skip_channels in reversed(stage_skips):
                block = ResidualBlock(
                    in_channels=in_ch + skip_channels,
                    out_channels=out_ch,
                    embed_dim=time_embed_dim,
                    dropout=cfg.dropout,
                    norm_groups=cfg.norm_groups,
                    cond_enabled=cfg.cond_dim > 0,
                )
                stage_blocks.append(block)
                in_ch = out_ch
            self.up_blocks.append(stage_blocks)
            if stage_idx > 0:
                self.upsamples.append(Upsample(in_ch))
            else:
                self.upsamples.append(None)

        self.out_norm = group_norm_or_instance(in_ch, cfg.norm_groups)
        self.out_act = nn.SiLU()
        self.out_proj = nn.Conv1d(in_ch, cfg.out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] input tensor.
            timesteps: [B] integer timesteps.
            cond: optional conditioning tensor [B, cond_dim].
            mask: optional [B, T] mask (1=valid).
            self_cond: optional [B, C, T] previous prediction for self-conditioning.
        """
        if x.dim() != 3:
            raise ValueError("Input tensor must have shape [B, C, T].")
        if timesteps.dim() != 1 or timesteps.shape[0] != x.shape[0]:
            raise ValueError("Timesteps must be a 1D tensor matching batch size.")

        if self.self_condition:
            if self_cond is None:
                self_cond = torch.zeros_like(x)
            if self_cond.shape != x.shape:
                raise ValueError("self_cond must match shape of x.")
            input_tensor = torch.cat([x, self_cond], dim=1)
        else:
            input_tensor = x

        temb = self.time_mlp(self.time_embedding(timesteps))
        cond_emb = self.cond_mlp(cond) if (self.cond_mlp is not None and cond is not None) else None

        h = self.input_proj(input_tensor)
        skip_stack: list[torch.Tensor] = []

        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h, temb, cond_emb)
                skip_stack.append(h)
            if downsample is not None:
                h = downsample(h)

        h = self.mid_block1(h, temb, cond_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb, cond_emb)

        for stage_blocks, upsample in zip(self.up_blocks, self.upsamples):
            for block in stage_blocks:
                if not skip_stack:
                    raise RuntimeError("Skip stack depleted before completion of up path.")
                skip = skip_stack.pop()
                if skip.shape[-1] != h.shape[-1]:
                    if skip.shape[-1] > h.shape[-1]:
                        skip = skip[..., : h.shape[-1]]
                    else:
                        h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
                h = torch.cat([h, skip], dim=1)
                h = block(h, temb, cond_emb)
            if upsample is not None:
                h = upsample(h)

        if skip_stack:
            raise RuntimeError("Skip stack not fully consumed; architecture mismatch.")

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_proj(h)

        if mask is not None:
            if mask.dim() != 2 or mask.shape != (x.shape[0], x.shape[-1]):
                raise ValueError("Mask must have shape [B, T].")
            h = h * mask.unsqueeze(1)
        return h


class ResidualBlock(nn.Module):
    """Residual block with FiLM modulation from time/conditioning embeddings."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        dropout: float,
        norm_groups: int,
        cond_enabled: bool,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = group_norm_or_instance(in_channels, norm_groups)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = group_norm_or_instance(out_channels, norm_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(embed_dim, in_channels * 2)
        self.cond_proj = nn.Linear(embed_dim, in_channels * 2) if cond_enabled else None
        self.time_proj_out = nn.Linear(embed_dim, out_channels * 2)
        self.cond_proj_out = nn.Linear(embed_dim, out_channels * 2) if cond_enabled else None
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cond_emb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        scale_shift = self.time_proj(temb)
        if self.cond_proj is not None and cond_emb is not None:
            scale_shift = scale_shift + self.cond_proj(cond_emb)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        scale_shift_out = self.time_proj_out(temb)
        if self.cond_proj_out is not None and cond_emb is not None:
            scale_shift_out = scale_shift_out + self.cond_proj_out(cond_emb)
        scale_out, shift_out = scale_shift_out.chunk(2, dim=1)
        h = h * (1 + scale_out.unsqueeze(-1)) + shift_out.unsqueeze(-1)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.skip(x) + h


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=max(1, num_heads), batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        h = x.permute(0, 2, 1)  # B, T, C
        h_norm = self.norm(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)
        h = h + attn_out
        return h.permute(0, 2, 1)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal position embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Time embedding dimension must be even.")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.dim // 2
        if half_dim == 0:
            raise ValueError("Time embedding dimension must be at least 2.")
        exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
        exponent = exponent / max(half_dim - 1, 1)
        freqs = torch.exp(-math.log(10000.0) * exponent)
        timesteps = timesteps.float().unsqueeze(1)
        angles = timesteps * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return emb


def group_norm_or_instance(num_channels: int, max_groups: int) -> nn.Module:
    for groups in reversed(range(1, max_groups + 1)):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)
