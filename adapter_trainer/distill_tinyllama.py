from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adapter_trainer.accounting import build_replacement_summary, format_replacement_summary
from adapter_trainer.lgma_llama_attention import LlamaLgmaAttention


def import_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "adapter_trainer requires transformers. Install with:\n"
            "  pip install transformers datasets accelerate"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TinyLlama LGMA teacher-student adapter trainer")
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--calibration_text_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3"],
        default="stage1",
        help="stage1 trains per-layer adapters; stage2/stage3 train swapped whole-model adapters.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer ids, inclusive range like 0-3, or all.",
    )
    parser.add_argument("--adapter_dir", type=Path, default=None)
    parser.add_argument("--attention_variant", default="lgma_residual")
    parser.add_argument("--generator_type", choices=["full", "diagonal", "symmetric"], default="full")
    parser.add_argument("--num_generators", type=int, default=4)
    parser.add_argument("--qk_num_base_heads", type=int, default=2)
    parser.add_argument("--value_num_base_heads", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps_per_layer", type=int, default=200)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--cosine_weight", type=float, default=0.1)
    parser.add_argument("--hidden_mse_weight", type=float, default=1.0)
    parser.add_argument("--logit_kl_weight", type=float, default=1.0)
    parser.add_argument("--lm_loss_weight", type=float, default=0.1)
    parser.add_argument("--kl_temperature", type=float, default=2.0)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Forwarded to Hugging Face model/tokenizer loading.",
    )
    parser.add_argument(
        "--train_output_projections",
        action="store_true",
        help="For stage3, also train adapter o_proj parameters.",
    )
    parser.add_argument(
        "--train_norms",
        action="store_true",
        help="For stage3, also train model normalization parameters.",
    )
    return parser.parse_args()


def dtype_from_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def autocast_context(device: str, precision: str):
    if device.startswith("cuda") and precision in {"bf16", "fp16"}:
        return torch.autocast(device_type="cuda", dtype=dtype_from_precision(precision))
    return nullcontext()


def parse_layers(spec: str, num_layers: int) -> list[int]:
    if spec == "all":
        return list(range(num_layers))
    layers: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", maxsplit=1)
            start, end = int(start_s), int(end_s)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    ordered = sorted(layers)
    if not ordered or ordered[0] < 0 or ordered[-1] >= num_layers:
        raise ValueError(f"invalid --layers={spec!r} for model with {num_layers} layers")
    return ordered


class TokenBatcher:
    def __init__(
        self,
        tokenizer: Any,
        text_path: Path,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> None:
        text = text_path.read_text(encoding="utf-8")
        tokenized = tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
        if len(tokenized) < sequence_length + 1:
            raise ValueError(
                f"calibration_text_file has {len(tokenized)} tokens; need at least {sequence_length + 1}"
            )
        self.tokens = torch.tensor(tokenized, dtype=torch.long)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device
        self.offset = 0

    def next_batch(self) -> dict[str, torch.Tensor]:
        max_start = self.tokens.numel() - self.sequence_length - 1
        starts = []
        for _ in range(self.batch_size):
            starts.append(self.offset % max_start)
            self.offset += self.sequence_length
        input_ids = torch.stack(
            [self.tokens[start : start + self.sequence_length] for start in starts],
            dim=0,
        )
        labels = torch.stack(
            [self.tokens[start + 1 : start + self.sequence_length + 1] for start in starts],
            dim=0,
        )
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.to(self.device),
            "labels": labels.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }


def freeze(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("expected a Hugging Face Llama-style model with model.layers")


def get_self_attention(layer: nn.Module) -> nn.Module:
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    raise ValueError("expected decoder layer to expose self_attn")


def create_student_for_layer(
    teacher_attention: nn.Module,
    args: argparse.Namespace,
    *,
    layer_idx: int,
) -> LlamaLgmaAttention:
    return LlamaLgmaAttention.from_teacher_attention(
        teacher_attention,
        qk_num_base_heads=args.qk_num_base_heads,
        value_num_base_heads=args.value_num_base_heads,
        num_generators=args.num_generators,
        attention_variant=args.attention_variant,
        generator_type=args.generator_type,
        layer_idx=layer_idx,
    )


def print_startup_summaries(
    teacher_model: nn.Module,
    students: dict[int, LlamaLgmaAttention],
    layers: Iterable[int],
) -> None:
    decoder_layers = get_decoder_layers(teacher_model)
    print("=" * 80)
    print("TinyLlama LGMA replacement summary")
    print("=" * 80)
    for layer_idx in layers:
        teacher_attention = get_self_attention(decoder_layers[layer_idx])
        summary = build_replacement_summary(
            teacher_attention,
            students[layer_idx],
            layer_idx=layer_idx,
        )
        print(format_replacement_summary(summary))
    print("=" * 80, flush=True)


def adapter_path(output_dir: Path, layer_idx: int) -> Path:
    return output_dir / f"layer_{layer_idx:03d}_adapter.pt"


def save_adapter(
    output_dir: Path,
    layer_idx: int,
    student: LlamaLgmaAttention,
    args: argparse.Namespace,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "layer_idx": layer_idx,
            "step": step,
            "config": asdict(student.config),
            "state_dict": student.state_dict(),
            "trainer_args": vars(args),
        },
        adapter_path(output_dir, layer_idx),
    )


def load_adapter(path: Path, map_location: torch.device | str) -> LlamaLgmaAttention:
    checkpoint = torch.load(path, map_location=map_location)
    from adapter_trainer.lgma_llama_attention import LlamaLgmaConfig

    student = LlamaLgmaAttention(LlamaLgmaConfig(**checkpoint["config"]))
    student.load_state_dict(checkpoint["state_dict"])
    return student


def capture_attention_io(
    teacher_model: nn.Module,
    layer_idx: int,
    batch: dict[str, torch.Tensor],
    *,
    precision: str,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any], torch.Tensor]:
    decoder_layers = get_decoder_layers(teacher_model)
    attention = get_self_attention(decoder_layers[layer_idx])
    capture: dict[str, Any] = {}

    def pre_hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            raise RuntimeError("could not capture attention hidden_states")
        capture["hidden_states"] = hidden_states.detach()
        filtered = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in kwargs.items()
            if key
            in {
                "attention_mask",
                "position_ids",
                "cache_position",
                "position_embeddings",
            }
        }
        filtered["output_attentions"] = False
        filtered["use_cache"] = False
        capture["kwargs"] = filtered

    def post_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        attn_output = output[0] if isinstance(output, tuple) else output
        capture["attn_output"] = attn_output.detach()

    pre_handle = attention.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = attention.register_forward_hook(post_hook)
    try:
        with torch.no_grad(), autocast_context(device, precision):
            teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
    finally:
        pre_handle.remove()
        post_handle.remove()
    return capture["hidden_states"], capture["kwargs"], capture["attn_output"]


def attention_distillation_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    *,
    cosine_weight: float,
) -> torch.Tensor:
    mse = F.mse_loss(student_output.float(), teacher_output.float())
    if cosine_weight <= 0:
        return mse
    cosine = 1.0 - F.cosine_similarity(
        student_output.float().flatten(0, 1),
        teacher_output.float().flatten(0, 1),
        dim=-1,
    ).mean()
    return mse + cosine_weight * cosine


def train_stage1(
    teacher_model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    layers: list[int],
) -> None:
    device = torch.device(args.device)
    batcher = TokenBatcher(
        tokenizer,
        args.calibration_text_file,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    decoder_layers = get_decoder_layers(teacher_model)
    students = {
        layer_idx: create_student_for_layer(
            get_self_attention(decoder_layers[layer_idx]),
            args,
            layer_idx=layer_idx,
        ).to(device=device, dtype=dtype_from_precision(args.precision))
        for layer_idx in layers
    }
    print_startup_summaries(teacher_model, students, layers)

    for layer_idx in layers:
        student = students[layer_idx]
        student.train()
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print(f"[stage1] training layer={layer_idx} for {args.steps_per_layer} steps", flush=True)
        for step in range(1, args.steps_per_layer + 1):
            batch = batcher.next_batch()
            hidden_states, attention_kwargs, teacher_output = capture_attention_io(
                teacher_model,
                layer_idx,
                batch,
                precision=args.precision,
                device=args.device,
            )
            hidden_states = hidden_states.to(device=device, dtype=dtype_from_precision(args.precision))
            teacher_output = teacher_output.to(device=device, dtype=dtype_from_precision(args.precision))
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args.device, args.precision):
                student_output = student(hidden_states, **attention_kwargs)[0]
                loss = attention_distillation_loss(
                    student_output,
                    teacher_output,
                    cosine_weight=args.cosine_weight,
                )
            loss.backward()
            optimizer.step()
            if step == 1 or step % args.log_every == 0:
                print(
                    f"[stage1] layer={layer_idx} step={step} loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
            if args.save_every > 0 and step % args.save_every == 0:
                save_adapter(args.output_dir, layer_idx, student, args, step)
        save_adapter(args.output_dir, layer_idx, student, args, args.steps_per_layer)


def replace_model_attention(
    model: nn.Module,
    args: argparse.Namespace,
    layers: list[int],
    *,
    device: torch.device,
) -> dict[int, LlamaLgmaAttention]:
    decoder_layers = get_decoder_layers(model)
    students: dict[int, LlamaLgmaAttention] = {}
    for layer_idx in layers:
        path = adapter_path(args.adapter_dir or args.output_dir, layer_idx)
        if path.exists():
            student = load_adapter(path, map_location=device)
        else:
            student = create_student_for_layer(
                get_self_attention(decoder_layers[layer_idx]),
                args,
                layer_idx=layer_idx,
            )
        student.to(device=device, dtype=dtype_from_precision(args.precision))
        decoder_layers[layer_idx].self_attn = student
        students[layer_idx] = student
    return students


def set_recovery_trainable(model: nn.Module, args: argparse.Namespace) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LlamaLgmaAttention):
            for name, param in module.named_parameters():
                if args.stage == "stage3" and not args.train_output_projections and name.startswith("o_proj"):
                    continue
                param.requires_grad_(True)
    if args.stage == "stage3" and args.train_norms:
        for name, param in model.named_parameters():
            if "norm" in name:
                param.requires_grad_(True)


def kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(temperature, 1e-6)
    return F.kl_div(
        F.log_softmax(student_logits.float() / t, dim=-1),
        F.softmax(teacher_logits.float() / t, dim=-1),
        reduction="batchmean",
    ) * (t * t)


def train_recovery_stage(
    teacher_model: nn.Module,
    tokenizer: Any,
    args: argparse.Namespace,
    layers: list[int],
) -> None:
    AutoModelForCausalLM, _ = import_transformers()
    device = torch.device(args.device)
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_from_precision(args.precision),
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    ).to(device)
    students = replace_model_attention(student_model, args, layers, device=device)
    print_startup_summaries(teacher_model, students, layers)
    set_recovery_trainable(student_model, args)
    trainable = [param for param in student_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    batcher = TokenBatcher(
        tokenizer,
        args.calibration_text_file,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    student_model.train()
    teacher_model.eval()
    for step in range(1, args.steps + 1):
        batch = batcher.next_batch()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast_context(args.device, args.precision):
            teacher_outputs = teacher_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                output_hidden_states=True,
            )
        with autocast_context(args.device, args.precision):
            student_outputs = student_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
                output_hidden_states=True,
            )
            loss = args.logit_kl_weight * kl_loss(
                student_outputs.logits,
                teacher_outputs.logits.detach(),
                args.kl_temperature,
            )
            if args.hidden_mse_weight > 0:
                loss = loss + args.hidden_mse_weight * F.mse_loss(
                    student_outputs.hidden_states[-1].float(),
                    teacher_outputs.hidden_states[-1].detach().float(),
                )
            if args.lm_loss_weight > 0 and student_outputs.loss is not None:
                loss = loss + args.lm_loss_weight * student_outputs.loss
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(f"[{args.stage}] step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            save_swapped_adapters(args.output_dir, student_model, layers, args, step)
    save_swapped_adapters(args.output_dir, student_model, layers, args, args.steps)


def save_swapped_adapters(
    output_dir: Path,
    model: nn.Module,
    layers: list[int],
    args: argparse.Namespace,
    step: int,
) -> None:
    decoder_layers = get_decoder_layers(model)
    for layer_idx in layers:
        attention = get_self_attention(decoder_layers[layer_idx])
        if not isinstance(attention, LlamaLgmaAttention):
            continue
        save_adapter(output_dir, layer_idx, attention, args, step)


def save_run_config(output_dir: Path, args: argparse.Namespace, layers: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["calibration_text_file"] = str(payload["calibration_text_file"])
    payload["output_dir"] = str(payload["output_dir"])
    payload["adapter_dir"] = None if payload["adapter_dir"] is None else str(payload["adapter_dir"])
    payload["layers_resolved"] = layers
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    AutoModelForCausalLM, AutoTokenizer = import_transformers()
    dtype = dtype_from_precision(args.precision)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    ).to(device)
    freeze(teacher_model)
    layers = parse_layers(args.layers, len(get_decoder_layers(teacher_model)))
    save_run_config(args.output_dir, args, layers)
    if args.stage == "stage1":
        train_stage1(teacher_model, tokenizer, args, layers)
    else:
        train_recovery_stage(teacher_model, tokenizer, args, layers)


if __name__ == "__main__":
    main()
