import json
import math
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from aggregate_paper_results import (  # noqa: E402
    aggregate_groups,
    query_slurm_jobs,
    summarize_manifest_run,
)
from benchmark_tinystories_inference import benchmark_context  # noqa: E402
from tinystories_runtime import evaluate_sequential_loss  # noqa: E402
from lgma.transformer import TinyTransformerLM  # noqa: E402


def tiny_model(context_length=8):
    torch.manual_seed(0)
    model = TinyTransformerLM(
        vocab_size=7,
        d_model=8,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        attention_type="mha",
        context_length=context_length,
        dropout=0.0,
    )
    model.eval()
    return model


def test_sequential_evaluation_is_deterministic_and_covers_each_target_once():
    model = tiny_model()
    encoded = torch.arange(20, dtype=torch.long) % 7

    first = evaluate_sequential_loss(
        model,
        encoded,
        batch_size=2,
        seq_len=8,
        device=torch.device("cpu"),
    )
    second = evaluate_sequential_loss(
        model,
        encoded,
        batch_size=1,
        seq_len=8,
        device=torch.device("cpu"),
    )

    assert first["evaluated_tokens"] == 19
    assert first["windows"] == 3
    assert math.isclose(first["loss"], second["loss"], rel_tol=1e-7)
    assert math.isclose(first["perplexity"], math.exp(first["loss"]), rel_tol=1e-7)


def test_inference_benchmark_labels_uncached_decode():
    result = benchmark_context(
        tiny_model(),
        torch.arange(20, dtype=torch.long) % 7,
        context_length=4,
        batch_size=1,
        decode_tokens=2,
        warmup=0,
        repeats=2,
        device=torch.device("cpu"),
        precision="fp32",
    )

    assert result["prefill"]["tokens_per_second"] > 0
    assert result["decode"]["tokens_per_second"] > 0
    assert result["decode"]["mode"] == "full_context_recompute_no_kv_cache"
    assert result["attention_accounting"]["attention_score_flops"] > 0


def test_slurm_aggregation_uses_allocated_gpu_count(monkeypatch):
    stdout = "123|train|COMPLETED|3600|billing=4,cpu=32,gres/gpu=4,mem=256G|0:0|\n"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=stdout, stderr="")

    monkeypatch.setattr("aggregate_paper_results.subprocess.run", fake_run)
    summary = query_slurm_jobs(["123"])

    assert summary["slurm_elapsed_hours"] == 1
    assert summary["slurm_gpu_hours"] == 4
    assert summary["slurm_jobs_recorded"] == 1


def test_manifest_aggregation_combines_training_eval_and_inference(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        '{"step": 100, "tokens_seen": 1000, "validation_loss": 0.4, '
        '"tokens_per_second_per_gpu": 200, "peak_memory_gib": 2}\n',
        encoding="utf-8",
    )
    (run_dir / "final_report.json").write_text(
        json.dumps(
            {
                "world_size": 4,
                "parameters": 1000,
                "model_config": {"attention_type": "mha"},
                "attention_accounting": {
                    "total_parameters": 100,
                    "qkv_parameters": 80,
                    "generator_parameters": 0,
                    "kv_cache_bytes_per_token_per_layer": 64,
                    "attention_score_flops": 512,
                },
            }
        ),
        encoding="utf-8",
    )
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(
        json.dumps(
            {
                "loss": 0.35,
                "perplexity": math.exp(0.35),
                "bits_per_character": 0.35 / math.log(2),
                "evaluated_tokens": 999,
                "source": "test.txt",
            }
        ),
        encoding="utf-8",
    )
    inference_path = tmp_path / "inference.json"
    inference_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "context_length": 8,
                        "prefill": {"tokens_per_second": 300, "median_ms": 2},
                        "decode": {
                            "tokens_per_second": 20,
                            "median_ms_per_token": 50,
                            "mode": "full_context_recompute_no_kv_cache",
                        },
                        "attention_accounting": {
                            "kv_cache_bytes_per_token_per_layer": 64
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    entry = {
        "name": "mha_seed0",
        "group": "mha",
        "seed": 0,
        "run_dir": str(run_dir),
        "full_eval_json": str(eval_path),
        "inference_json": str(inference_path),
    }

    summary = summarize_manifest_run(
        entry,
        tmp_path,
        skip_slurm=True,
        fetch_wandb=False,
    )
    groups = aggregate_groups([summary])

    assert summary["attention_type"] == "mha"
    assert summary["full_test_loss"] == 0.35
    assert summary["prefill_tokens_per_second"] == 300
    assert summary["kv_cache_bytes_per_token_per_layer"] == 64
    assert groups[0]["group"] == "mha"
    assert groups[0]["seeds"] == 1
