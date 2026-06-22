from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from summarize_tinystories_runs import read_jsonl, summarize_run


def test_summarize_run_reports_normalized_runtime_axes(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            [
                '{"step": 1, "validation_loss": 0.31, "tokens_per_step": 1000, '
                '"tokens_per_second": 4000, "world_size": 4, "peak_memory_bytes": 1073741824}',
                '{"step": 2, "validation_loss": 0.264, "tokens_per_step": 1000, '
                '"tokens_per_second": 8000, "world_size": 4, "gpu_hours": 0.5, '
                '"peak_memory_bytes": 2147483648}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_run(
        "run",
        read_jsonl(metrics_path),
        {"world_size": 4},
        thresholds=[0.27, 0.265],
    )

    assert summary["tokens_seen"] == 2000
    assert summary["peak_memory_gib"] == 2
    assert summary["tokens_per_second_per_gpu_median"] == 1500
    assert summary["step_to_val_0p27"] == 2
    assert summary["gpu_hours_to_val_0p265"] == 0.5
