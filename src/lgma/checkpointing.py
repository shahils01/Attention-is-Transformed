from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch


def load_full_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str | None = None,
) -> Any:
    """Load a trusted trainer checkpoint across old and new PyTorch releases."""
    kwargs: dict[str, object] = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)
