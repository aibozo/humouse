"""Utility CLI scripts for the 1-layer mouse GAN project."""

from .aggregate_checkpoints import aggregate_checkpoints, AggregationResult  # noqa: F401

__all__ = ["aggregate_checkpoints", "AggregationResult"]
