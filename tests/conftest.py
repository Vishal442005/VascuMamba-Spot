"""Pytest session configuration for deterministic single-thread CPU execution."""

import torch


def pytest_configure() -> None:
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
