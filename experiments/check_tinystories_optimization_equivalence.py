from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lgma.checkpointing import load_full_checkpoint
from lgma.transformer import TinyTransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a checkpoint's original execution path with the optimized "
            "GT-MHA path before resuming a long training run."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument(
        "--fuse_base_qkv", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fold_value_transform_into_output",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sdpa_gqa_mode", choices=["auto", "native", "expand"], default="auto"
    )
    parser.add_argument(
        "--check_backward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also compare gradients from one deterministic loss evaluation",
    )
    return parser.parse_args()


def precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def tolerances(precision: str, atol: float | None, rtol: float | None) -> tuple[float, float]:
    default = 1e-5 if precision == "fp32" else 2e-2
    return (default if atol is None else atol, default if rtol is None else rtol)


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    difference = (candidate - reference).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "reference_max_abs": float(reference.abs().max()),
    }


def selected_attention_gradients(model: TinyTransformerLM) -> dict[str, torch.Tensor]:
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not name.startswith("blocks.0.attn.") or parameter.grad is None:
            continue
        gradients[name] = parameter.grad.detach().float().cpu().clone()
    return gradients


def run_model(
    *,
    config: dict[str, object],
    model_state: dict[str, torch.Tensor],
    vocab_size: int,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    precision: str,
    check_backward: bool,
) -> tuple[torch.Tensor, float, dict[str, torch.Tensor]]:
    model = TinyTransformerLM(vocab_size=vocab_size, **config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    with precision_context(device, precision):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    gradients: dict[str, torch.Tensor] = {}
    if check_backward:
        loss.backward()
        gradients = selected_attention_gradients(model)
    return logits.detach().cpu(), float(loss.detach()), gradients


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.context_length <= 0:
        raise SystemExit("batch_size and context_length must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")

    device = torch.device(args.device)
    checkpoint = load_full_checkpoint(args.checkpoint, map_location="cpu")
    saved_config = checkpoint.get("model_config")
    model_state = checkpoint.get("model_state")
    if not isinstance(saved_config, dict) or not isinstance(model_state, dict):
        raise SystemExit("checkpoint must contain model_config and model_state")
    token_weight = model_state.get("token_embedding.weight")
    if not torch.is_tensor(token_weight):
        raise SystemExit("checkpoint model_state is missing token_embedding.weight")
    vocab_size = int(token_weight.shape[0])

    trained_context = int(saved_config["context_length"])
    if args.context_length > trained_context:
        raise SystemExit(f"context_length must not exceed the trained length {trained_context}")

    reference_config = dict(saved_config)
    reference_config.update(
        {
            "fuse_base_qkv": False,
            "fold_value_transform_into_output": False,
            "sdpa_gqa_mode": "auto",
        }
    )
    candidate_config = dict(saved_config)
    candidate_config.update(
        {
            "fuse_base_qkv": args.fuse_base_qkv,
            "fold_value_transform_into_output": args.fold_value_transform_into_output,
            "sdpa_gqa_mode": args.sdpa_gqa_mode,
        }
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    input_ids = torch.randint(
        vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        generator=generator,
    )
    targets = torch.randint(
        vocab_size,
        (args.batch_size, args.context_length),
        device=device,
        generator=generator,
    )

    reference_logits, reference_loss, reference_gradients = run_model(
        config=reference_config,
        model_state=model_state,
        vocab_size=vocab_size,
        input_ids=input_ids,
        targets=targets,
        device=device,
        precision=args.precision,
        check_backward=args.check_backward,
    )
    candidate_logits, candidate_loss, candidate_gradients = run_model(
        config=candidate_config,
        model_state=model_state,
        vocab_size=vocab_size,
        input_ids=input_ids,
        targets=targets,
        device=device,
        precision=args.precision,
        check_backward=args.check_backward,
    )

    atol, rtol = tolerances(args.precision, args.atol, args.rtol)
    logits_close = torch.allclose(reference_logits.float(), candidate_logits.float(), atol=atol, rtol=rtol)
    gradient_results: dict[str, dict[str, float | bool]] = {}
    gradients_close = True
    if args.check_backward:
        if reference_gradients.keys() != candidate_gradients.keys():
            gradients_close = False
        for name in sorted(reference_gradients.keys() & candidate_gradients.keys()):
            result = compare_tensors(reference_gradients[name], candidate_gradients[name])
            result["close"] = torch.allclose(
                reference_gradients[name], candidate_gradients[name], atol=atol, rtol=rtol
            )
            gradients_close = gradients_close and bool(result["close"])
            gradient_results[name] = result

    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", 0)),
        "precision": args.precision,
        "atol": atol,
        "rtol": rtol,
        "candidate": {
            "fuse_base_qkv": args.fuse_base_qkv,
            "fold_value_transform_into_output": args.fold_value_transform_into_output,
            "sdpa_gqa_mode": args.sdpa_gqa_mode,
        },
        "reference_loss": reference_loss,
        "candidate_loss": candidate_loss,
        "loss_abs_difference": abs(candidate_loss - reference_loss),
        "logits": {**compare_tensors(reference_logits, candidate_logits), "close": logits_close},
        "gradients_close": gradients_close if args.check_backward else None,
        "gradients": gradient_results,
        "passed": logits_close and (gradients_close or not args.check_backward),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
