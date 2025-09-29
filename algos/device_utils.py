"""
Utility helpers for selecting the appropriate torch device.
"""

from __future__ import annotations

import torch


def select_device(prefer_gpu: bool = True) -> str:
    """Return ``'cuda'`` when a GPU is available, otherwise ``'cpu'``.

    Parameters
    ----------
    prefer_gpu: bool
        When False, always returns ``'cpu'`` even if CUDA is available.
    """
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"
