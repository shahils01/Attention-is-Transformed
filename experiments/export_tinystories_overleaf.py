from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


METRICS = ("grammar", "consistency", "creativity", "plot")
MODEL_LABELS = {
    "collaborative_mha": "Collaborative MHA",
    "gqa": "GQA",
    "gt_mha_exact": "GT-MHA Exact",
    "gt_mha_quad": "GT-MHA Quadratic",
    "gt_mha_residual": "GT-MHA Residual",
    "mha": "MHA",
    "mqa": "MQA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export scored TinyStories prompt comparisons as Overleaf LaTeX tables."
    )
    parser.add_argument(
        "comparisons",
        nargs="+",
        type=Path,
        help="One or more prompt_comparison_*.json files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--title",
        default="TinyStories Generated-Text Comparison",
        help="Standalone document title.",
    )
    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Write only table bodies for \\input{} into an existing paper.",
    )
    return parser.parse_args()


def latex_escape(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    lines = str(value).splitlines() or [""]
    escaped_lines = ["".join(replacements.get(char, char) for char in line) for line in lines]
    # A paragraph break can escape a fragile alignment context in some paper
    # templates. Explicit in-cell line breaks preserve all three table columns.
    return r"\newline ".join(escaped_lines)


def read_comparison(path: Path) -> dict[str, Any]:
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read comparison JSON {path}: {exc}") from exc
    if not isinstance(item, dict):
        raise SystemExit(f"comparison must be a JSON object: {path}")
    for key in ("prompt_id", "theme", "prompt", "model_order", "continuations"):
        if key not in item:
            raise SystemExit(f"comparison {path} is missing {key}")
    model_order = item["model_order"]
    continuations = item["continuations"]
    if not isinstance(model_order, list) or not isinstance(continuations, dict):
        raise SystemExit(f"comparison {path} has malformed model data")
    for model in model_order:
        candidate = continuations.get(model)
        if not isinstance(candidate, dict) or not isinstance(candidate.get("answer_text"), str):
            raise SystemExit(f"comparison {path} is missing answer_text for {model}")
        scores = candidate.get("scores")
        if not isinstance(scores, dict):
            raise SystemExit(f"comparison {path} is missing scores for {model}")
        missing = [metric for metric in METRICS if metric not in scores]
        if missing:
            raise SystemExit(
                f"comparison {path}/{model} is missing rubric-v2 scores: {', '.join(missing)}"
            )
    return item


def score_cell(candidate: dict[str, Any]) -> str:
    scores = candidate["scores"]
    lines = [
        rf"\textbf{{{metric.title()}:}} {int(scores[metric])}/10"
        for metric in METRICS
    ]
    average = float(candidate.get("average_score", sum(scores[m] for m in METRICS) / 4))
    lines.append(rf"\textbf{{Average:}} {average:.2f}/10")
    return r"\newline ".join(lines)


def safe_label(prompt_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9:-]+", "-", prompt_id).strip("-") or "prompt"


def render_table(item: dict[str, Any], index: int) -> str:
    prompt_id = str(item["prompt_id"])
    theme = str(item["theme"]).replace("_", " ").title()
    caption = f"TinyStories completions for {theme} ({prompt_id})"
    lines = [
        r"\begin{longtable}{@{}>{\RaggedRight\arraybackslash}p{0.14\linewidth}"
        r">{\RaggedRight\arraybackslash}p{0.62\linewidth}"
        r">{\RaggedRight\arraybackslash}p{0.18\linewidth}@{}}",
        rf"\caption{{{latex_escape(caption)}}}\label{{tab:tinystories-{safe_label(prompt_id)}}}\\",
        r"\toprule",
        r"\textbf{Model} & \textbf{Generated text} & \textbf{Scores} \\",
        r"\midrule",
        r"\endfirsthead",
        rf"\multicolumn{{3}}{{c}}{{\tablename\ \thetable{{}} -- continued from previous page}}\\",
        r"\toprule",
        r"\textbf{Model} & \textbf{Generated text} & \textbf{Scores} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{3}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
        rf"\textbf{{Theme}} & {latex_escape(theme)} & \\",
        rf"\textbf{{Prompt}} & {latex_escape(item['prompt'])} & \\",
        r"\midrule",
    ]
    continuations = item["continuations"]
    for model_index, model in enumerate(item["model_order"]):
        candidate = continuations[model]
        model_label = MODEL_LABELS.get(str(model), str(model).replace("_", " ").title())
        model_cell = rf"\textbf{{{latex_escape(model_label)}}}"
        step = candidate.get("checkpoint_step")
        if step is not None:
            model_cell += rf"\newline Step {int(step):,}"
        lines.append(
            f"{model_cell} & {latex_escape(candidate['answer_text'])} & "
            + score_cell(candidate)
            + r" \\"
        )
        if model_index + 1 < len(item["model_order"]):
            lines.append(r"\midrule")
    lines.extend([r"\end{longtable}", ""])
    if index > 0:
        lines.insert(0, r"\clearpage")
    return "\n".join(lines)


def render_document(items: list[dict[str, Any]], title: str) -> str:
    preamble = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=0.55in]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{ragged2e}",
        r"\usepackage{microtype}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\emergencystretch}{2em}",
        r"\begin{document}",
        r"\small",
        rf"\section*{{{latex_escape(title)}}}",
        "",
    ]
    tables = [render_table(item, index) for index, item in enumerate(items)]
    return "\n".join([*preamble, *tables, r"\end{document}", ""])


def render_fragment(items: list[dict[str, Any]]) -> str:
    header = [
        "% Required packages in the main document preamble:",
        r"% \usepackage{array,booktabs,longtable,ragged2e,microtype}",
        "% Insert this file directly in the document body. Do not wrap it in",
        "% table, table*, figure, resizebox, adjustbox, or center environments.",
        "% Recommended table settings:",
        r"% \setlength{\tabcolsep}{4pt}",
        r"% \renewcommand{\arraystretch}{1.08}",
        r"\begingroup",
        r"\small",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\setlength{\emergencystretch}{2em}",
        "",
    ]
    tables = [render_table(item, index) for index, item in enumerate(items)]
    return "\n".join([*header, *tables, r"\endgroup", ""])


def main() -> None:
    args = parse_args()
    items = [read_comparison(path) for path in args.comparisons]
    rendered = render_fragment(items) if args.fragment else render_document(items, args.title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "overleaf_tables_exported",
                "prompts": [item["prompt_id"] for item in items],
                "tables": len(items),
                "fragment": args.fragment,
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
