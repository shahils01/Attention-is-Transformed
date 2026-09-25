from __future__ import annotations

import torch


class StaticKVCache:
    """Inference-only K/V buffer that grows logically without copying past tokens.

    Storage is allocated on the first append, using the projection's shape,
    device, and dtype. Each subsequent append writes only the new token(s).
    """

    def __init__(self, batch_size: int, max_length: int) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")
        self.batch_size = batch_size
        self.max_length = max_length
        self.length = 0
        self._key: torch.Tensor | None = None
        self._value: torch.Tensor | None = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            raise RuntimeError("StaticKVCache requires torch.no_grad() or torch.inference_mode()")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("keys and values must have shape [batch, heads, time, dim]")
        if key.shape[0] != self.batch_size or value.shape[0] != self.batch_size:
            raise ValueError("cache batch size does not match the current projection")
        if key.shape[2] != value.shape[2] or key.shape[2] <= 0:
            raise ValueError("keys and values must have the same positive sequence length")
        end = self.length + key.shape[2]
        if end > self.max_length:
            raise ValueError(f"cache length {end} exceeds max_length {self.max_length}")
        if self._key is None:
            self._key = key.new_empty((key.shape[0], key.shape[1], self.max_length, key.shape[3]))
            self._value = value.new_empty(
                (value.shape[0], value.shape[1], self.max_length, value.shape[3])
            )
        assert self._value is not None
        for storage, current in ((self._key, key), (self._value, value)):
            if storage.shape[:2] + storage.shape[3:] != current.shape[:2] + current.shape[3:]:
                raise ValueError("cached projection shape is incompatible with the new projection")
            if storage.device != current.device or storage.dtype != current.dtype:
                raise ValueError("cached projection device or dtype has changed")
        self._key[:, :, self.length:end, :].copy_(key)
        self._value[:, :, self.length:end, :].copy_(value)
        self.length = end
        return self[0], self[1]

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._key is None or self._value is None:
            raise ValueError("cache has no stored tokens yet")
        if index == 0:
            return self._key[:, :, : self.length, :]
        if index == 1:
            return self._value[:, :, : self.length, :]
        raise IndexError(index)

    def __iter__(self):
        yield self[0]
        yield self[1]

    def __len__(self) -> int:
        return 2

    @property
    def allocated_bytes(self) -> int:
        if self._key is None or self._value is None:
            return 0
        return sum(t.numel() * t.element_size() for t in (self._key, self._value))

    def reset(self) -> None:
        """Reuse the allocated storage for another sequence of the same shape."""
        self.length = 0
