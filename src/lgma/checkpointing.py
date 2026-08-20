from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


def load_full_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = None,
    mmap: bool = False,
) -> Any:
    """Load a trusted trainer checkpoint across old and new PyTorch releases."""
    kwargs: dict[str, object] = {"map_location": map_location}
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" in parameters:
        kwargs["weights_only"] = False
    if mmap and "mmap" in parameters:
        kwargs["mmap"] = True
    return torch.load(path, **kwargs)
