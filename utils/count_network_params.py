from __future__ import annotations

import argparse

from lgma.transformer import TinyTransformerLM


def count_parameters(model: TinyTransformerLM, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def build_mha(vocab_size: int) -> TinyTransformerLM:
    return TinyTransformerLM(
        vocab_size=vocab_size,
        attention_type="mha",
        d_model=1024,
        num_layers=12,
        num_heads=16,
        head_dim=64,
        base_dim=64,
        value_dim=64,
        context_length=1024,
        dropout=0.1,
    )


def build_lgma(vocab_size: int) -> TinyTransformerLM:
    return TinyTransformerLM(
        vocab_size=vocab_size,
        attention_type="lgma_multibase",
        d_model=1024,
        num_layers=12,
        num_heads=16,
        num_base_heads=2,
        head_dim=64,
        base_dim=64,
        value_dim=64,
        num_generators=16,
        metric_beta=0.25,
        context_length=1024,
        dropout=0.1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count parameters for the MHA and LGMA TinyTransformerLM configs."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=1,
        help="Tokenizer vocabulary size. Use the TinyStories char vocab size for exact totals.",
    )
    args = parser.parse_args()

    models = {
        "mha": build_mha(args.vocab_size),
        "lgma_multibase": build_lgma(args.vocab_size),
    }

    for name, model in models.items():
        trainable = count_parameters(model, trainable_only=True)
        total = count_parameters(model, trainable_only=False)
        attention = count_parameters(model.first_attention, trainable_only=True)
        print(f"{name}:")
        print(f"  trainable parameters: {trainable:,}")
        print(f"  total parameters:     {total:,}")
        print(f"  first attention layer: {attention:,}")


if __name__ == "__main__":
    main()
