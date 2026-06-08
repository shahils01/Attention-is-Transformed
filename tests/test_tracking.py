from pathlib import Path

from lgma.tracking import flatten_metrics, init_wandb_run, json_safe, parse_wandb_tags


def test_flatten_metrics_keeps_only_scalar_metric_values():
    payload = {
        "step": 1,
        "loss": 0.5,
        "nested": {"accuracy": 0.75, "label": "skip"},
        "text": "skip",
    }
    assert flatten_metrics(payload) == {
        "step": 1,
        "loss": 0.5,
        "nested/accuracy": 0.75,
    }


def test_wandb_disabled_or_missing_project_does_not_require_import():
    assert init_wandb_run(project=None) is None
    assert init_wandb_run(project="unused", mode="disabled") is None


def test_parse_wandb_tags_and_json_safe_paths():
    assert parse_wandb_tags("a, b,,c") == ["a", "b", "c"]
    assert json_safe({"path": Path("x/y")}) == {"path": "x/y"}
