"""Shared device selection for scripts, notebooks, and model wrappers."""

import os

import torch


def resolve_device(device=None, config=None):
    """Resolve explicit > MOLFLAE_DEVICE > runtime.device > auto.

    Auto selects the current CUDA device when available, otherwise CPU.
    Explicit unavailable devices raise instead of silently falling back.
    """
    if device is None:
        device = os.environ.get("MOLFLAE_DEVICE")
    if device is None:
        device = (config or {}).get("runtime", {}).get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use a CUDA environment with GPU access, or select cpu.")
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} is unavailable ({torch.cuda.device_count()} visible devices).")
        selected = torch.device("cuda", index)
    elif selected.type != "cpu":
        raise ValueError(f"Unsupported device {selected}; use auto, cpu, cuda, or cuda:N.")
    return selected
