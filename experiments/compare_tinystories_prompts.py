from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from tinystories_runtime import (
    DEFAULT_STOP_SEQUENCE,
    generate_text,
    load_tinystories_generation_checkpoint,
    precision_context,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "benchmarks" / "tinystories_100_prompts.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "tinystories_lgma_mha_100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired TinyStories continuations from named checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Named checkpoint as name=/path/to/checkpoint.pt. Repeat once per model.",
    )
    parser.add_argument("--prompts_file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop_sequence", default=DEFAULT_STOP_SEQUENCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit("--checkpoint must be formatted as name=/path/to/checkpoint.pt")
    name, raw_path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit("checkpoint name must not be empty")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise SystemExit(f"checkpoint does not exist: {path}")
    return name, path


def load_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"prompts file does not exist: {path}")
    prompts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid JSON on {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"prompt on {path}:{line_number} must be a JSON object")
        prompt_id = item.get("id")
        theme = item.get("theme")
        prompt = item.get("prompt")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise SystemExit(f"prompt on {path}:{line_number} has no string id")
        if prompt_id in seen_ids:
            raise SystemExit(f"duplicate prompt id: {prompt_id}")
        if not isinstance(theme, str) or not theme:
            raise SystemExit(f"prompt {prompt_id} has no string theme")
        if not isinstance(prompt, str) or not prompt:
            raise SystemExit(f"prompt {prompt_id} has no story text")
        seen_ids.add(prompt_id)
        prompts.append(item)
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def make_run_id(
    prompts_path: Path,
    checkpoint_specs: list[tuple[str, Path]],
    *,
    device: str,
    precision: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    stop_sequence: str,
) -> str:
    payload = {
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "checkpoints": [
            (name, str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            for name, path in checkpoint_specs
        ],
        "device": device,
        "precision": precision,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
        "stop_sequence": stop_sequence,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"invalid existing result on {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"existing result on {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def render_markdown(
    path: Path,
    prompts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    model_names: list[str],
    run_id: str,
) -> None:
    current = {
        (str(row.get("model")), str(row.get("prompt_id"))): row
        for row in rows
        if row.get("run_id") == run_id
    }
    completed = sum((name, prompt["id"]) in current for prompt in prompts for name in model_names)
    lines = [
        "# TinyStories LGMA vs. MHA prompt comparison",
        "",
        f"Run ID: `{run_id}`  ",
        f"Completed pairs: {completed}/{len(prompts) * len(model_names)}",
        "",
    ]
    for prompt in prompts:
        prompt_id = str(prompt["id"])
        lines.extend(
            [
                f"## {prompt_id}: {prompt['theme']}",
                "",
                "**Prompt**",
                "",
                str(prompt["prompt"]),
                "",
                f"**Evaluation focus:** {prompt.get('evaluation_notes', '')}",
                "",
            ]
        )
        if prompt.get("target_age"):
            lines.extend([f"**Target age:** {prompt['target_age']}", ""])
        for model_name in model_names:
            row = current.get((model_name, prompt_id))
            lines.extend([f"### {model_name}", ""])
            if row is None:
                lines.extend(["_Not generated yet._", ""])
            else:
                lines.extend(
                    [
                        f"Checkpoint step: {row['checkpoint_step']}; seed: {row['seed']}",
                        "",
                        str(row["completion"]),
                        "",
                    ]
                )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise SystemExit("--max_new_tokens must be positive")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if args.top_k < 0:
        raise SystemExit("--top_k must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device} requested, but CUDA is not available")


def main() -> None:
    args = parse_args()
    validate_args(args)
    prompts = load_prompts(args.prompts_file)
    checkpoints = [parse_checkpoint_spec(spec) for spec in args.checkpoint]
    model_names = [name for name, _ in checkpoints]
    if len(set(model_names)) != len(model_names):
        raise SystemExit("checkpoint names must be unique")

    run_id = make_run_id(
        args.prompts_file,
        checkpoints,
        device=args.device,
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        stop_sequence=args.stop_sequence,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "completions.jsonl"
    report_path = args.output_dir / "comparison.md"
    rows = read_results(results_path)
    completed = {
        (str(row.get("model")), str(row.get("prompt_id")))
        for row in rows
        if row.get("run_id") == run_id
    }
    device = torch.device(args.device)

    print(json.dumps({"event": "comparison_started", "run_id": run_id, "prompts": len(prompts)}))
    for model_name, checkpoint_path in checkpoints:
        pending = [prompt for prompt in prompts if (model_name, str(prompt["id"])) not in completed]
        if not pending:
            print(json.dumps({"event": "model_already_complete", "model": model_name}))
            continue
        print(
            json.dumps(
                {
                    "event": "loading_checkpoint",
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "pending_prompts": len(pending),
                }
            )
        )
        sys.stdout.flush()
        model, tokenizer, config, checkpoint_step = load_tinystories_generation_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            data_path=args.data_path,
            val_data_path=args.val_data_path,
        )
        too_long = [
            str(prompt["id"])
            for prompt in pending
            if len(prompt["prompt"]) > int(config["context_length"])
        ]
        if too_long:
            raise SystemExit(
                "prompts exceed the checkpoint context window and would be left-truncated: "
                + ", ".join(too_long)
            )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        with results_path.open("a", encoding="utf-8", buffering=1) as output:
            for prompt_index, prompt in enumerate(prompts):
                prompt_id = str(prompt["id"])
                if (model_name, prompt_id) in completed:
                    continue
                sample_seed = args.seed + prompt_index
                torch.manual_seed(sample_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(sample_seed)
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with torch.inference_mode(), precision_context(device, args.precision):
                    completion = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=str(prompt["prompt"]),
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k or None,
                        device=device,
                        stop_sequence=args.stop_sequence or None,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                row = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step,
                    "attention_type": config["attention_type"],
                    "parameters": parameters,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "theme": prompt["theme"],
                    "prompt": prompt["prompt"],
                    "evaluation_notes": prompt.get("evaluation_notes"),
                    "target_age": prompt.get("target_age"),
                    "seed": sample_seed,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "stop_sequence": args.stop_sequence,
                    "device": str(device),
                    "precision": args.precision,
                    "completion": completion,
                    "generated_characters": len(completion),
                    "elapsed_seconds": elapsed,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                rows.append(row)
                completed.add((model_name, prompt_id))
                print(
                    json.dumps(
                        {
                            "event": "prompt_complete",
                            "model": model_name,
                            "prompt_id": prompt_id,
                            "elapsed_seconds": round(elapsed, 3),
                        }
                    )
                )
                sys.stdout.flush()
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        render_markdown(report_path, prompts, rows, model_names, run_id)

    render_markdown(report_path, prompts, rows, model_names, run_id)
    print(
        json.dumps(
            {
                "event": "comparison_complete",
                "run_id": run_id,
                "results": str(results_path),
                "report": str(report_path),
            }
        )
    )


if __name__ == "__main__":
    main()
