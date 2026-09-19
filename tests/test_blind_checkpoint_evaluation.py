import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from evaluate_blind_checkpoints import (  # noqa: E402
    METRICS,
    judge_blind_item,
    multiple_prompt_comparisons,
    prepare_blind_rows,
    score_summary,
    single_prompt_comparison,
    unblind_command,
    unblind_records,
    validate_score,
)
from evaluate_blind_tinystories import canonical_hash  # noqa: E402


MODELS = ["gqa", "gt_mha_quad", "mha"]


def synthetic_matrix():
    matrix = {}
    for prompt_index in range(9):
        prompt_id = f"p{prompt_index:02d}"
        theme = f"theme_{prompt_index % 3}"
        matrix[prompt_id] = {}
        for model in MODELS:
            matrix[prompt_id][model] = {
                "run_id": "run1",
                "model": model,
                "checkpoint": f"/private/{model}.pt",
                "checkpoint_step": 100,
                "attention_type": model,
                "parameters": 123,
                "prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "theme": theme,
                "prompt": f"Story prompt {prompt_index}",
                "evaluation_notes": "Continue coherently.",
                "target_age": None,
                "seed": prompt_index,
                "completion": (
                    f"Generated text {prompt_index}, variant {MODELS.index(model)}."
                ),
            }
    return matrix


def score(value):
    return {
        "grammar": value,
        "consistency": value,
        "creativity": value,
        "plot": value,
        "assessment": f"This candidate receives {value}.",
    }


def test_prepare_balances_candidate_positions_and_hides_models():
    blind, mapping = prepare_blind_rows(
        synthetic_matrix(), MODELS, source_run_id="run1", count=9, seed=17
    )

    assert len(blind) == len(mapping) == 9
    position_counts = Counter(
        (candidate["candidate_id"], candidate["model"])
        for row in mapping
        for candidate in row["candidates"]
    )
    assert set(position_counts.values()) == {3}
    assert all("model" not in json.dumps(row).lower() for row in blind)
    assert all(
        model not in json.dumps(row).lower() for row in blind for model in MODELS
    )
    assert all(
        row["blind_record_sha256"] == canonical_hash(blind[index])
        for index, row in enumerate(mapping)
    )


def test_dynamic_judge_schema_covers_every_candidate():
    calls = []

    class Response:
        id = "response_1"
        output_text = json.dumps(
            {"candidate_a": score(8), "candidate_b": score(7), "candidate_c": score(6)}
        )

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return Response()

    class Client:
        responses = Responses()

    blind_item = {
        "blind_id": "blind_001",
        "prompt_id": "p01",
        "theme": "simple",
        "prompt": "Once there was a fox.",
        "evaluation_notes": "Continue.",
        "target_age": None,
        "candidates": [
            {"candidate_id": "candidate_a", "completion": "It ran home."},
            {"candidate_id": "candidate_b", "completion": "It found a kite."},
            {"candidate_id": "candidate_c", "completion": "It helped a hen."},
        ],
    }
    parsed, response = judge_blind_item(Client(), blind_item, "gpt-5.6-sol", "medium")

    assert response.id == "response_1"
    assert set(parsed) == {"candidate_a", "candidate_b", "candidate_c"}
    assert calls[0]["reasoning"] == {"effort": "medium"}
    schema = calls[0]["text"]["format"]["schema"]
    assert schema["required"] == ["candidate_a", "candidate_b", "candidate_c"]
    candidate_schema = schema["properties"]["candidate_a"]
    assert candidate_schema["required"] == [*METRICS, "assessment"]
    assert "plot" in candidate_schema["properties"]
    rendered = json.dumps(calls[0]).lower()
    assert "gt_mha" not in rendered


def test_unblind_restores_all_checkpoints_and_ranks_them():
    blind, mapping = prepare_blind_rows(
        synthetic_matrix(), MODELS, source_run_id="run1", count=9, seed=3
    )
    model_scores = {"gqa": 6, "gt_mha_quad": 9, "mha": 7}
    scores = []
    for blind_row, mapping_row in zip(blind, mapping):
        identities = {
            item["candidate_id"]: item["model"] for item in mapping_row["candidates"]
        }
        scores.append(
            {
                "blind_id": blind_row["blind_id"],
                "blind_record_sha256": canonical_hash(blind_row),
                "judge_model": "gpt-5.6-sol",
                "scores": {
                    candidate_id: score(model_scores[model])
                    for candidate_id, model in identities.items()
                },
            }
        )

    unblinded, model_order = unblind_records(blind, mapping, scores)
    summary = score_summary(unblinded, model_order)

    assert len(unblinded) == 27
    assert Counter(row["model"] for row in unblinded) == {
        "gqa": 9,
        "gt_mha_quad": 9,
        "mha": 9,
    }
    assert summary["ranking"] == ["gt_mha_quad", "mha", "gqa"]
    assert summary["by_model"]["gt_mha_quad"]["mean_score"] == 9
    assert set(summary["metrics"]) == set(METRICS)

    example = single_prompt_comparison(unblinded, model_order, "p00")
    assert example["prompt_id"] == "p00"
    assert example["model_order"] == MODELS
    assert set(example["continuations"]) == set(MODELS)
    assert example["continuations"]["gt_mha_quad"]["answer_text"].startswith(
        "Generated text 0"
    )
    assert example["continuations"]["gt_mha_quad"]["scores"] == {
        "grammar": 9,
        "consistency": 9,
        "creativity": 9,
        "plot": 9,
    }
    assert example["continuations"]["gt_mha_quad"]["average_score"] == 9

    examples = multiple_prompt_comparisons(unblinded, model_order, 5, "p03")
    assert len(examples) == 5
    assert examples[0]["prompt_id"] == "p03"
    assert len({item["prompt_id"] for item in examples}) == 5
    assert all(len(item["continuations"]) == len(MODELS) for item in examples)


def test_creativity_is_conditioned_on_consistency():
    valid = {
        "grammar": 8,
        "consistency": 6,
        "creativity": 6,
        "plot": 7,
        "assessment": "Imaginative development remains consistent with the supplied story.",
    }
    assert validate_score(valid, "candidate_a")["creativity"] == 6

    invalid = {**valid, "creativity": 7}
    try:
        validate_score(invalid, "candidate_a")
    except ValueError as exc:
        assert "must not exceed consistency" in str(exc)
    else:
        raise AssertionError("expected inconsistent creativity score to be rejected")


def test_five_samples_are_blinded_separately_and_report_prompt_variance():
    matrix = {}
    for prompt_index in range(2):
        prompt_id = f"p{prompt_index:02d}"
        matrix[prompt_id] = {}
        for sample_index in range(5):
            matrix[prompt_id][sample_index] = {}
            for model in MODELS:
                matrix[prompt_id][sample_index][model] = {
                    "model": model,
                    "checkpoint": f"/private/{model}.pt",
                    "checkpoint_step": 100,
                    "attention_type": model,
                    "parameters": 123,
                    "prompt_index": prompt_index,
                    "prompt_id": prompt_id,
                    "sample_index": sample_index,
                    "sample_number": sample_index + 1,
                    "theme": "theme",
                    "prompt": f"Story prompt {prompt_index}",
                    "evaluation_notes": "Continue coherently.",
                    "target_age": None,
                    "seed": sample_index * 2 + prompt_index,
                    "completion": f"{model} completion {prompt_index}/{sample_index}.",
                }

    blind, mapping = prepare_blind_rows(
        matrix, MODELS, source_run_id="run1", count=0, seed=4
    )
    assert len(blind) == 10
    assert Counter(row["sample_number"] for row in blind) == {
        1: 2,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
    }

    scores = []
    for blind_row, mapping_row in zip(blind, mapping):
        identities = {
            item["candidate_id"]: item["model"] for item in mapping_row["candidates"]
        }
        scores.append(
            {
                "blind_id": blind_row["blind_id"],
                "blind_record_sha256": canonical_hash(blind_row),
                "scores": {
                    candidate_id: score(blind_row["sample_number"])
                    for candidate_id in identities
                },
            }
        )

    unblinded, model_order = unblind_records(blind, mapping, scores)
    summary = score_summary(unblinded, model_order)
    stats = summary["by_prompt_model"]["p00"]["gqa"]

    assert summary["samples_per_prompt"] == 5
    assert stats["count"] == 5
    assert stats["mean_score"] == 3
    assert stats["score_sample_variance"] == 2.5
    assert stats["grammar"]["sample_variance"] == 2.5


def test_unblind_command_writes_requested_csvs_and_prompt_examples(tmp_path):
    blind, mapping = prepare_blind_rows(
        synthetic_matrix(), MODELS, source_run_id="run1", count=3, seed=5
    )
    scores = []
    for blind_row, mapping_row in zip(blind, mapping):
        identities = {
            item["candidate_id"]: item["model"] for item in mapping_row["candidates"]
        }
        scores.append(
            {
                "blind_id": blind_row["blind_id"],
                "blind_record_sha256": canonical_hash(blind_row),
                "scores": {
                    candidate_id: score(MODELS.index(model) + 6)
                    for candidate_id, model in identities.items()
                },
            }
        )

    def write_jsonl(path, rows):
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    blind_path = tmp_path / "blind.jsonl"
    mapping_path = tmp_path / "mapping.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    output_dir = tmp_path / "reports"
    write_jsonl(blind_path, blind)
    write_jsonl(mapping_path, mapping)
    write_jsonl(scores_path, scores)

    unblind_command(
        SimpleNamespace(
            blind_file=blind_path,
            mapping_file=mapping_path,
            scores_file=scores_path,
            output_dir=output_dir,
            num_example_prompts=2,
            example_prompt_id=None,
            example_output=None,
            examples_dir=None,
            overwrite=False,
        )
    )

    assert (output_dir / "detailed.csv").is_file()
    assert (output_dir / "leaderboard.csv").read_text(encoding="utf-8") == (
        output_dir / "summary.csv"
    ).read_text(encoding="utf-8")
    prompt_manifest = json.loads(
        (output_dir / "prompt_comparisons" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert prompt_manifest["count"] == 2
