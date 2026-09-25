"""Compact loader adapter preserving the reference's explicit data RNG."""
import torch
from lgma.synthetic import SyntheticBatch
from ospool.train_tinystories_compat import install_ospool_patches


def compact_batch(encoded, batch_size, seq_len, device="cpu", generator=None):
    if encoded.numel() <= seq_len + 1:
        raise ValueError("encoded text is too short")
    starts = torch.randint(0, encoded.numel() - seq_len - 1,
                           (batch_size,), generator=generator)
    positions = starts[:, None] + torch.arange(seq_len)[None, :]
    return SyntheticBatch(
        input_ids=encoded[positions].to(device=device, dtype=torch.long),
        targets=encoded[positions + 1].to(device=device, dtype=torch.long),
    )


def install():
    training = install_ospool_patches()
    training.make_lm_batch = compact_batch
    return training


if __name__ == "__main__":
    install().main()
