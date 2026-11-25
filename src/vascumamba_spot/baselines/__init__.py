"""Unified baseline architectures and spotting/recognition adapters for identical LOSO splits."""

from vascumamba_spot.baselines.registry import build_baseline, list_baselines

__all__ = ["build_baseline", "list_baselines"]
