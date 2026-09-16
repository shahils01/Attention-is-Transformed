import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from export_tinystories_overleaf import (  # noqa: E402
    latex_escape,
    read_comparison,
    render_document,
    render_fragment,
)


def test_latex_escape_handles_table_characters_and_paragraphs():
    assert latex_escape("A&B_50%\nnext") == r"A\&B\_50\%\newline next"


def test_render_document_includes_prompt_models_and_four_scores(tmp_path):
    comparison = {
        "prompt_id": "ts_test",
        "theme": "cause_and_effect_reasoning",
        "prompt": "Why did the ice melt?",
        "model_order": ["mha", "gt_mha_exact"],
        "continuations": {
            model: {
                "checkpoint_step": 10,
                "answer_text": f"{model} said: Sun & ice.",
                "scores": {
                    "grammar": 8,
                    "consistency": 7,
                    "creativity": 6,
                    "plot": 7,
                },
                "average_score": 7.0,
            }
            for model in ("mha", "gt_mha_exact")
        },
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison), encoding="utf-8")

    rendered = render_document([read_comparison(path)], "Examples")

    assert r"\begin{longtable}" in rendered
    assert "Cause And Effect Reasoning" in rendered
    assert "GT-MHA Exact" in rendered
    assert r"Sun \& ice" in rendered
    assert r"\textbf{Plot:} 7/10" in rendered
    assert r"\textbf{Average:} 7.00/10" in rendered
    assert r"p{0.18\linewidth}" in rendered
    assert r"\textbf{Prompt} & Why did the ice melt? & \\" in rendered

    fragment = render_fragment([comparison])
    assert r"\begin{longtable}" in fragment
    assert r"\documentclass" not in fragment
