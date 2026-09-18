from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
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
DEFAULT_ALL_OUTPUT_DIR = ROOT / "outputs" / "tinystories_all_checkpoints_100"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TinyStories continuations from named or discovered checkpoints."
    )
    checkpoint_source = parser.add_mutually_exclusive_group(required=True)
    checkpoint_source.add_argument(
        "--checkpoint",
        action="append",
        help="Named checkpoint as name=/path/to/checkpoint.pt. Repeat once per model.",
    )
    checkpoint_source.add_argument(
        "--checkpoint_dir",
        type=Path,
        help="Discover and run every checkpoint under this directory.",
    )
    parser.add_argument(
        "--checkpoint_glob",
        default="**/checkpoint*.pt",
        help="Glob used with --checkpoint_dir (default: **/checkpoint*.pt).",
    )
    parser.add_argument(
        "--skip_checkpoint_name",
        action="append",
        default=[],
        help=(
            "Discovered or explicitly named checkpoint to skip for this invocation. "
            "Repeat as needed; skipped checkpoints remain in the run manifest/run ID."
        ),
    )
    parser.add_argument("--prompts_file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--val_data_path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--max_new_tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--samples_per_prompt",
        type=int,
        default=1,
        help="Independent generations per checkpoint/prompt (default: 1).",
    )
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


def slugify_checkpoint_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise SystemExit(f"could not derive a checkpoint name from {value!r}")
    return slug


def discover_checkpoints(root: Path, pattern: str) -> list[tuple[str, Path]]:
    root = root.expanduser()
    if not root.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {root}")
    paths = sorted(
        (path for path in root.glob(pattern) if path.is_file()),
        key=lambda path: str(path.relative_to(root)).lower(),
    )
    if not paths:
        raise SystemExit(f"no checkpoints matching {pattern!r} found under {root}")

    base_names: list[str] = []
    for path in paths:
        relative = path.relative_to(root)
        parent_text = relative.parent.name if relative.parent != Path(".") else ""
        base_names.append(slugify_checkpoint_name(parent_text or path.stem))
    duplicate_bases = {name for name in base_names if base_names.count(name) > 1}

    checkpoints: list[tuple[str, Path]] = []
    for base_name, path in zip(base_names, paths):
        name = base_name
        if base_name in duplicate_bases:
            name = f"{base_name}__{slugify_checkpoint_name(path.stem)}"
        checkpoints.append((name, path))
    names = [name for name, _ in checkpoints]
    if len(names) != len(set(names)):
        raise SystemExit("discovered checkpoint paths do not produce unique names")
    return checkpoints


def resolve_checkpoints(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.checkpoint_dir is not None:
        return discover_checkpoints(args.checkpoint_dir, args.checkpoint_glob)
    return [parse_checkpoint_spec(spec) for spec in args.checkpoint]


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
    samples_per_prompt: int = 1,
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
        "samples_per_prompt": samples_per_prompt,
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


def write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def checkpoint_output_dir(output_dir: Path, model_name: str) -> Path:
    return output_dir / "checkpoints" / model_name


def sync_checkpoint_results(
    output_dir: Path,
    rows: list[dict[str, Any]],
    model_name: str,
    run_id: str,
) -> Path:
    path = checkpoint_output_dir(output_dir, model_name) / "completions.jsonl"
    selected = [
        row
        for row in rows
        if row.get("run_id") == run_id and row.get("model") == model_name
    ]
    write_rows_atomic(path, selected)
    return path


def write_run_manifest(
    path: Path,
    *,
    run_id: str,
    prompts_path: Path,
    checkpoints: list[tuple[str, Path]],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "prompts_file": str(prompts_path),
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "models": [name for name, _ in checkpoints],
        "checkpoints": [
            {"model": name, "path": str(checkpoint.resolve())}
            for name, checkpoint in checkpoints
        ],
        "generation": {
            "device": args.device,
            "precision": args.precision,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "seed": args.seed,
            "samples_per_prompt": args.samples_per_prompt,
            "stop_sequence": args.stop_sequence,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_markdown(
    path: Path,
    prompts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    model_names: list[str],
    run_id: str,
    samples_per_prompt: int = 1,
) -> None:
    current = {
        (
            str(row.get("model")),
            str(row.get("prompt_id")),
            int(row.get("sample_index", 0)),
        ): row
        for row in rows
        if row.get("run_id") == run_id
    }
    completed = sum(
        (name, prompt["id"], sample_index) in current
        for prompt in prompts
        for name in model_names
        for sample_index in range(samples_per_prompt)
    )
    lines = [
        "# TinyStories checkpoint prompt comparison",
        "",
        f"Run ID: `{run_id}`  ",
        "Completed generations: "
        f"{completed}/{len(prompts) * len(model_names) * samples_per_prompt}",
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
            lines.extend([f"### {model_name}", ""])
            for sample_index in range(samples_per_prompt):
                row = current.get((model_name, prompt_id, sample_index))
                if samples_per_prompt > 1:
                    lines.extend([f"#### Sample {sample_index + 1}", ""])
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
    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples_per_prompt must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {args.device} requested, but CUDA is not available")


def main() -> None:
    args = parse_args()
    if args.checkpoint_dir is not None and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_ALL_OUTPUT_DIR
    validate_args(args)
    prompts = load_prompts(args.prompts_file)
    checkpoints = resolve_checkpoints(args)
    model_names = [name for name, _ in checkpoints]
    if len(set(model_names)) != len(model_names):
        raise SystemExit("checkpoint names must be unique")
    unknown_skips = sorted(set(args.skip_checkpoint_name).difference(model_names))
    if unknown_skips:
        raise SystemExit(
            "--skip_checkpoint_name did not match a checkpoint: "
            + ", ".join(unknown_skips)
        )
    skipped_models = set(args.skip_checkpoint_name)

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
        samples_per_prompt=args.samples_per_prompt,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "completions.jsonl"
    report_path = args.output_dir / "comparison.md"
    manifest_path = args.output_dir / "run_manifest.json"
    write_run_manifest(
        manifest_path,
        run_id=run_id,
        prompts_path=args.prompts_file,
        checkpoints=checkpoints,
        args=args,
    )
    rows = read_results(results_path)
    completed = {
        (
            str(row.get("model")),
            str(row.get("prompt_id")),
            int(row.get("sample_index", 0)),
        )
        for row in rows
        if row.get("run_id") == run_id
    }
    device = torch.device(args.device)

    print(
        json.dumps(
            {
                "event": "comparison_started",
                "run_id": run_id,
                "prompts": len(prompts),
                "samples_per_prompt": args.samples_per_prompt,
                "checkpoints": len(checkpoints),
                "models": model_names,
            }
        )
    )
    for model_name, checkpoint_path in checkpoints:
        if model_name in skipped_models:
            print(json.dumps({"event": "model_skipped", "model": model_name}))
            continue
        model_results_path = sync_checkpoint_results(
            args.output_dir, rows, model_name, run_id
        )
        pending = [
            (prompt, sample_index)
            for prompt in prompts
            for sample_index in range(args.samples_per_prompt)
            if (model_name, str(prompt["id"]), sample_index) not in completed
        ]
        if not pending:
            print(json.dumps({"event": "model_already_complete", "model": model_name}))
            render_markdown(
                checkpoint_output_dir(args.output_dir, model_name) / "report.md",
                prompts,
                rows,
                [model_name],
                run_id,
                args.samples_per_prompt,
            )
            continue
        print(
            json.dumps(
                {
                    "event": "loading_checkpoint",
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "pending_generations": len(pending),
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
            for prompt, _ in pending
            if len(prompt["prompt"]) > int(config["context_length"])
        ]
        if too_long:
            raise SystemExit(
                "prompts exceed the checkpoint context window and would be left-truncated: "
                + ", ".join(too_long)
            )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        with (
            results_path.open("a", encoding="utf-8", buffering=1) as output,
            model_results_path.open("a", encoding="utf-8", buffering=1) as model_output,
        ):
            for sample_index in range(args.samples_per_prompt):
                for prompt_index, prompt in enumerate(prompts):
                    prompt_id = str(prompt["id"])
                    result_key = (model_name, prompt_id, sample_index)
                    if result_key in completed:
                        continue
                    # The schedule is identical across checkpoints. Sample zero retains
                    # the historical per-prompt seeds used by one-sample runs.
                    sample_seed = args.seed + sample_index * len(prompts) + prompt_index
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
                        "schema_version": 2,
                        "run_id": run_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model": model_name,
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_step": checkpoint_step,
                        "attention_type": config["attention_type"],
                        "parameters": parameters,
                        "prompt_index": prompt_index,
                        "prompt_id": prompt_id,
                        "sample_index": sample_index,
                        "sample_number": sample_index + 1,
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
                    model_output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    model_output.flush()
                    rows.append(row)
                    completed.add(result_key)
                    print(
                        json.dumps(
                            {
                                "event": "prompt_complete",
                                "model": model_name,
                                "prompt_id": prompt_id,
                                "sample_number": sample_index + 1,
                                "elapsed_seconds": round(elapsed, 3),
                            }
                        )
                    )
                    sys.stdout.flush()
        del model, tokenizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        render_markdown(
            report_path, prompts, rows, model_names, run_id, args.samples_per_prompt
        )
        render_markdown(
            checkpoint_output_dir(args.output_dir, model_name) / "report.md",
            prompts,
            rows,
            [model_name],
            run_id,
            args.samples_per_prompt,
        )

    render_markdown(report_path, prompts, rows, model_names, run_id, args.samples_per_prompt)
    print(
        json.dumps(
            {
                "event": "comparison_complete",
                "run_id": run_id,
                "results": str(results_path),
                "report": str(report_path),
                "manifest": str(manifest_path),
                "checkpoint_outputs": str(args.output_dir / "checkpoints"),
            }
        )
    )


if __name__ == "__main__":
    main()
