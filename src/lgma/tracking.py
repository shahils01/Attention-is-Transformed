from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_wandb_tags(tags: str | None) -> list[str] | None:
    if tags is None or tags.strip() == "":
        return None
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def flatten_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float, bool)):
            flattened[name] = value
    return flattened


def init_wandb_run(
    *,
    project: str | None,
    entity: str | None = None,
    name: str | None = None,
    group: str | None = None,
    tags: str | None = None,
    mode: str = "online",
    output_dir: Path | None = None,
    config: dict[str, Any] | None = None,
):
    if project is None or mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed. Install it with `pip install wandb` or run without "
            "`--wandb_project`."
        ) from exc

    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=parse_wandb_tags(tags),
        mode=mode,
        dir=str(output_dir) if output_dir is not None else None,
        config=json_safe(config or {}),
    )


def log_wandb(run, payload: dict[str, Any], step: int | None = None) -> None:
    if run is None:
        return
    metrics = flatten_metrics(payload)
    if not metrics:
        return
    run.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is not None:
        run.finish()
