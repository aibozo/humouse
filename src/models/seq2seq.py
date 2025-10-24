"""Seq2Seq LSTM generator and encoder for warm-up autoencoding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .generator import GeneratorConfig


class Seq2SeqEncoder(nn.Module):
    """LSTM encoder that produces Gaussian latent parameters."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # sequence: (B, T, input_dim)
        _, (hidden, _) = self.lstm(sequence)
        last = hidden[-1]
        mu = self.mu(last)
        logvar = self.logvar(last)
        return mu, logvar


class Seq2SeqGenerator(nn.Module):
    """Autoregressive LSTM decoder that supports teacher forcing."""

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.output_dim = config.output_dim
        self.context_dim = config.latent_dim + config.condition_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = max(1, config.num_layers)

        self.hidden_init = nn.Linear(self.context_dim, self.hidden_dim * self.num_layers)
        self.cell_init = nn.Linear(self.context_dim, self.hidden_dim * self.num_layers)

        self.input_proj = nn.Linear(self.output_dim + self.context_dim, self.hidden_dim)
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, self.output_dim)

    def _prepare_context(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond is None or cond.numel() == 0:
            if self.config.condition_dim > 0:
                cond = z.new_zeros(z.size(0), self.config.condition_dim)
            else:
                cond = z.new_zeros(z.size(0), 0)
        context = torch.cat([z, cond], dim=-1)
        return context

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        *,
        teacher_forcing: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        context = self._prepare_context(z, cond)
        batch_size = context.size(0)
        total_steps = steps or self.config.target_len
        if teacher_forcing is not None:
            total_steps = min(total_steps, teacher_forcing.size(1))

        h0 = self.hidden_init(context).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = self.cell_init(context).view(self.num_layers, batch_size, self.hidden_dim)
        hidden = (h0, c0)

        outputs = []
        prev = z.new_zeros(batch_size, self.output_dim)
        for t in range(total_steps):
            if teacher_forcing is not None:
                prev = teacher_forcing[:, t, :]
            decoder_input = torch.cat([prev, context], dim=-1)
            proj = self.input_proj(decoder_input).unsqueeze(1)
            out, hidden = self.lstm(proj, hidden)
            step_out = self.output(out.squeeze(1))
            outputs.append(step_out)
            prev = step_out

        if steps is not None and steps > total_steps:
            # continue autoregressively for remaining steps
            for _ in range(total_steps, steps):
                decoder_input = torch.cat([prev, context], dim=-1)
                proj = self.input_proj(decoder_input).unsqueeze(1)
                out, hidden = self.lstm(proj, hidden)
                step_out = self.output(out.squeeze(1))
                outputs.append(step_out)
                prev = step_out

        output = torch.stack(outputs, dim=1)
        return output

    def sample(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        *,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        return self.forward(z, cond, teacher_forcing=None, steps=steps)


__all__ = ["Seq2SeqEncoder", "Seq2SeqGenerator"]
