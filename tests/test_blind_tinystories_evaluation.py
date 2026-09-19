import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_blind_tinystories import (  # noqa: E402
    METRICS,
    candidate_comparison_rows,
    canonical_hash,
    is_retryable_judge_error,
    judge_blind_item,
    prepare_blind_rows,
    resolve_judge_connection,
    score_summary,
    unblind_records,
)


THEMES = [
    "simple_story_continuation",
    "character_object_consistency",
    "cause_and_effect_reasoning",
    "multi_character_interactions",
    "emotional_development",
    "moral_lesson_completion",
    "long_range_detail_retention",
    "age_appropriate_language",
]


def synthetic_pairs():
    pairs = {}
    prompt_index = 0
    for theme in THEMES:
        for theme_index in range(4):
            prompt_id = f"p{prompt_index:03d}"
            common = {
                "run_id": "run1",
                "prompt_id": prompt_id,
                "prompt_index": prompt_index,
                "theme": theme,
                "prompt": f"Story prompt {prompt_index}",
                "evaluation_notes": f"Evaluate {theme}.",
                "target_age": "8-10" if theme == "age_appropriate_language" else None,
                "checkpoint_step": 100,
                "seed": prompt_index,
            }
            pairs[prompt_id] = {
                "lgma": {
                    **common,
                    "model": "lgma",
                    "attention_type": "lgma_quad",
                    "checkpoint": "/private/lgma.pt",
                    "completion": f"First generated story text {prompt_index}.",
                },
                "mha": {
                    **common,
                    "model": "mha",
                    "attention_type": "mha",
                    "checkpoint": "/private/mha.pt",
                    "completion": f"Second generated story text {prompt_index}.",
                },
            }
            prompt_index += 1
    return pairs


def score(value):
    return {
        "grammar": value,
        "creativity": value,
        "consistency": value,
        "theme_alignment": value,
        "overall": value,
        "assessment": f"This candidate receives {value}.",
    }


def test_prepare_creates_balanced_blind_rows_and_private_identity_map():
    blind, mapping = prepare_blind_rows(
        synthetic_pairs(),
        ("lgma", "mha"),
        source_run_id="run1",
        count=30,
        seed=17,
    )

    assert len(blind) == len(mapping) == 30
    assert sorted(Counter(row["theme"] for row in blind).values()) == [3, 3, 4, 4, 4, 4, 4, 4]
    assert len({row["prompt_id"] for row in blind}) == 30
    assert all("model" not in row and "checkpoint" not in row for row in blind)
    assert all("lgma" not in json.dumps(row).lower() for row in blind)
    assert all("mha" not in json.dumps(row).lower() for row in blind)
    assert {
        item[candidate]["model"]
        for item in mapping
        for candidate in ("candidate_a", "candidate_b")
    } == {"lgma", "mha"}
    assert all(
        item["blind_record_sha256"] == canonical_hash(blind[index])
        for index, item in enumerate(mapping)
    )


def test_openai_judge_receives_only_blind_content_and_parses_scores():
    calls = []

    class Response:
        id = "response_1"
        output_text = json.dumps(
            {"candidate_a": score(8), "candidate_b": score(6)}
        )

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return Response()

    class Client:
        responses = Responses()

    blind_item = {
        "blind_id": "blind_001",
        "prompt_id": "p001",
        "theme": "consistency",
        "prompt": "A red ball rolled",
        "evaluation_notes": "Keep the ball red.",
        "target_age": None,
        "candidate_a": " under a chair.",
        "candidate_b": " and became blue.",
    }
    parsed, response = judge_blind_item(Client(), blind_item, "judge-model")

    assert parsed["candidate_a"]["overall"] == 8
    assert parsed["candidate_b"]["overall"] == 6
    assert response.id == "response_1"
    rendered_call = json.dumps(calls[0]).lower()
    assert "lgma" not in rendered_call
    assert "mha" not in rendered_call
    assert calls[0]["text"]["format"]["strict"] is True


def test_rcd_openai_provider_uses_gateway_and_rcd_key():
    assert resolve_judge_connection("rcd-openai") == (
        "RCD_LLM_API_KEY",
        "https://llm.rcd.clemson.edu/openai/v1",
    )


def test_permanent_http_errors_are_not_retried():
    class AuthenticationError(Exception):
        status_code = 401

    class RateLimitError(Exception):
        status_code = 429

    assert not is_retryable_judge_error(AuthenticationError())
    assert is_retryable_judge_error(RateLimitError())


def test_unblind_restores_models_and_builds_paired_summary():
    blind, mapping = prepare_blind_rows(
        synthetic_pairs(),
        ("lgma", "mha"),
        source_run_id="run1",
        count=8,
        seed=3,
    )
    scores = []
    for blind_row, mapping_row in zip(blind, mapping):
        score_by_candidate = {}
        for candidate in ("candidate_a", "candidate_b"):
            model = mapping_row[candidate]["model"]
            score_by_candidate[candidate] = score(9 if model == "lgma" else 7)
        scores.append(
            {
                "blind_id": blind_row["blind_id"],
                "blind_record_sha256": canonical_hash(blind_row),
                "judge_model": "judge-model",
                **score_by_candidate,
            }
        )

    unblinded, model_order = unblind_records(blind, mapping, scores)
    summary = score_summary(unblinded, model_order)

    assert len(unblinded) == 16
    assert Counter(row["model"] for row in unblinded) == {"lgma": 8, "mha": 8}
    assert all(row["overall"] == (9 if row["model"] == "lgma" else 7) for row in unblinded)
    assert summary["by_model"]["lgma"]["overall"]["mean"] == 9
    assert summary["by_model"]["mha"]["overall"]["mean"] == 7
    assert summary["paired_differences"]["overall"]["mean_first_minus_second"] == 2
    assert summary["overall_pairwise_wins"] == {"lgma": 8, "mha": 0, "ties": 0}
    assert set(summary["metrics"]) == set(METRICS)

    comparisons = candidate_comparison_rows(unblinded)
    assert len(comparisons) == 8
    assert {
        (row["candidate_a_model"], row["candidate_b_model"])
        for row in comparisons
    }.issubset({("lgma", "mha"), ("mha", "lgma")})
    assert all(row["overall_winner"].endswith("(lgma)") for row in comparisons)
