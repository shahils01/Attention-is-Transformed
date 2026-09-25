"""Disable the optional Transformer Engine integration for BERT experiments.

DeltaAI's CUDA 13 Python module currently ships incompatible/broken
Transformer Engine components. PEFT only checks whether this module exposes a
``pytorch`` attribute; this deliberately empty shim makes that optional path
unavailable while leaving stock PyTorch and SDPA untouched.
"""

__all__: list[str] = []
