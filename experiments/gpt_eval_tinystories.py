from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from tinystories_runtime import generate_text, load_tinystories_checkpoint


DEFAULT_PROMPTS = [
    "Alice was so tired when she got back home so she went",
    "Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked",
    'Alice and Jack walked up the street and met a girl in a red dress. The girl said to them, "Hi, I\'m Jane. What are your names?"',
]

JUDGE_SYSTEM_PROMPT = """You are grading a student's completion of a children's story.
The student was given the beginning of a story and wrote the text after ***.
Grade only the student's completion, while considering whether it fits the beginning.
Return only valid JSON with these keys:
grammar, creativity, consistency, plot, age_group, assessment.
grammar, creativity, consistency, and plot must be integers from 1 to 10.
age_group must be one of: A, B, C, D, E, F.
A means 3 or under, B means 4-5, C means 6-7, D means 8-9, E means 10-12, F means 13-16.
assessment must be one concise sentence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TinyStories-style GPT-Eval for one or more checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Named checkpoint as name=/path/to/checkpoint.pt. Repeat for each model.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Training text path. Defaults to the path saved in each checkpoint.",
    )
    parser.add_argument(
        "--val_data_path",
        type=Path,
        default=None,
        help="Validation text path. Defaults to the path saved in each checkpoint.",
    )
    parser.add_argument(
        "--prompts_file",
        type=Path,
        default=None,
        help="Optional text file with one prompt per line. Defaults to three paper-style prompts.",
    )
    parser.add_argument("--samples_per_prompt", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/tinystories_gpt_eval"))
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Call the OpenAI API to grade completions. Without this, only completions are generated.",
    )
    parser.add_argument(
        "--judge_model",
        default="gpt-4.1",
        help="OpenAI model used as judge. Override this if your account uses a different model.",
    )
    parser.add_argument("--judge_sleep", type=float, default=0.0, help="Seconds to sleep between judge calls.")
    return parser.parse_args()


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit("--checkpoint must be formatted as name=/path/to/checkpoint.pt")
    name, path = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise SystemExit("checkpoint name must not be empty")
    return name, Path(path).expanduser()


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def truncate_repeating_ngrams(text: str, n: int = 4) -> tuple[str, bool]:
    words = text.split()
    seen: set[tuple[str, ...]] = set()
    for index in range(0, len(words) - n + 1):
        ngram = tuple(words[index : index + n])
        if ngram in seen:
            return " ".join(words[:index]).rstrip(), True
        seen.add(ngram)
    return text, False


def make_judge_input(prompt: str, completion: str) -> str:
    return f"""The following exercise tests language ability and creativity.
The student is given the beginning of a story. The student needs to complete it into a full story.
The symbol *** marks the separator between the prescribed beginning and the student's completion.

{prompt}***{completion}

Grade the student's completion in terms of grammar, creativity, consistency with the story beginning, and whether the plot makes sense."""


def extract_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"judge response did not contain JSON: {text}")
    return json.loads(text[start : end + 1])


def judge_completion(client: Any, judge_model: str, prompt: str, completion: str) -> dict[str, Any]:
    response = client.responses.create(
        model=judge_model,
        input=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": make_judge_input(prompt, completion)},
        ],
    )
    return extract_json(response.output_text)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "prompt_index",
        "sample_index",
        "grammar",
        "creativity",
        "consistency",
        "plot",
        "age_group",
        "repetition_truncated",
        "assessment",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        summary[model] = {}
        for field in ["grammar", "creativity", "consistency", "plot"]:
            values = [float(row[field]) for row in model_rows if row.get(field) is not None]
            summary[model][field] = statistics.mean(values) if values else float("nan")
        summary[model]["total"] = statistics.mean(
            [summary[model][field] for field in ["grammar", "creativity", "consistency", "plot"]]
        )
        summary[model]["repetition_rate"] = statistics.mean(
            [1.0 if row.get("repetition_truncated") else 0.0 for row in model_rows]
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples_per_prompt must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max_new_tokens must be positive")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but CUDA is not available")

    prompts = load_prompts(args.prompts_file)
    checkpoints = [parse_checkpoint_spec(spec) for spec in args.checkpoint]
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if args.judge:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("Install the OpenAI Python package first: pip install openai") from exc
        client = OpenAI()

    rows: list[dict[str, Any]] = []
    for model_name, checkpoint_path in checkpoints:
        model, tokenizer, _, _, _, step = load_tinystories_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            data_path=args.data_path,
            val_data_path=args.val_data_path,
        )
        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(args.samples_per_prompt):
                torch.manual_seed(args.seed + prompt_index * 1000 + sample_index)
                completion = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    device=device,
                )
                completion, repetition_truncated = truncate_repeating_ngrams(completion)
                row: dict[str, Any] = {
                    "model": model_name,
                    "checkpoint": str(checkpoint_path),
                    "step": step,
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "prompt": prompt,
                    "completion": completion,
                    "repetition_truncated": repetition_truncated,
                }
                if client is not None:
                    scores = judge_completion(client, args.judge_model, prompt, completion)
                    row.update(scores)
                    if args.judge_sleep > 0:
                        time.sleep(args.judge_sleep)
                rows.append(row)
                print(json.dumps({key: row.get(key) for key in ["model", "prompt_index", "sample_index"]}))
                sys.stdout.flush()

    completions_path = args.output_dir / "completions.jsonl"
    write_jsonl(completions_path, rows)
    print(f"wrote {completions_path}")

    if args.judge:
        scores_path = args.output_dir / "scores.csv"
        summary_path = args.output_dir / "summary.json"
        write_scores_csv(scores_path, rows)
        summary = summarize_scores(rows)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"wrote {scores_path}")
        print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
