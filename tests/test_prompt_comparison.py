import json
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from compare_tinystories_prompts import (  # noqa: E402
    discover_checkpoints,
    load_prompts,
    main,
    make_run_id,
    render_markdown,
)
from lgma.transformer import TinyTransformerLM  # noqa: E402


def test_benchmark_has_100_unique_balanced_ascii_prompts():
    path = ROOT / "benchmarks" / "tinystories_100_prompts.jsonl"
    prompts = load_prompts(path)

    assert len(prompts) == 100
    assert len({prompt["id"] for prompt in prompts}) == 100
    assert all(prompt["prompt"].isascii() for prompt in prompts)
    assert max(len(prompt["prompt"]) for prompt in prompts) <= 512
    assert sorted(Counter(prompt["theme"] for prompt in prompts).values()) == [
        12,
        12,
        12,
        12,
        13,
        13,
        13,
        13,
    ]


def test_run_id_changes_with_decoding_configuration(tmp_path):
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text('{"id":"p1","theme":"simple","prompt":"Once"}\n')
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    specs = [("mha", checkpoint)]

    first = make_run_id(
        prompts_path,
        specs,
        device="cpu",
        precision="fp32",
        max_new_tokens=100,
        temperature=0.8,
        top_k=20,
        seed=0,
        stop_sequence="stop",
    )
    second = make_run_id(
        prompts_path,
        specs,
        device="cpu",
        precision="fp32",
        max_new_tokens=101,
        temperature=0.8,
        top_k=20,
        seed=0,
        stop_sequence="stop",
    )

    assert first != second


def test_checkpoint_discovery_is_sorted_and_names_nested_folders(tmp_path):
    (tmp_path / "MHA").mkdir()
    (tmp_path / "GT-MHA quad").mkdir()
    mha = tmp_path / "MHA" / "checkpoint_step_20.pt"
    gt_mha = tmp_path / "GT-MHA quad" / "checkpoint_step_10.pt"
    mha.touch()
    gt_mha.touch()

    assert discover_checkpoints(tmp_path, "**/checkpoint_step_*.pt") == [
        ("gt_mha_quad", gt_mha),
        ("mha", mha),
    ]


def test_checkpoint_discovery_includes_final_checkpoints(tmp_path):
    (tmp_path / "Identity").mkdir()
    final = tmp_path / "Identity" / "checkpoint_final.pt"
    final.touch()

    assert discover_checkpoints(tmp_path, "**/checkpoint*.pt") == [
        ("identity", final)
    ]


def test_markdown_report_pairs_models_by_prompt(tmp_path):
    output = tmp_path / "comparison.md"
    prompts = [
        {
            "id": "p1",
            "theme": "consistency",
            "prompt": "A red ball rolled",
            "evaluation_notes": "Keep the ball red.",
        }
    ]
    rows = [
        {
            "run_id": "run1",
            "model": "mha",
            "prompt_id": "p1",
            "checkpoint_step": 10,
            "seed": 0,
            "completion": " under the chair.",
        }
    ]

    render_markdown(output, prompts, rows, ["lgma", "mha"], "run1")
    rendered = output.read_text()

    assert "Completed generations: 1/2" in rendered
    assert "### lgma\n\n_Not generated yet._" in rendered
    assert "### mha" in rendered
    assert "under the chair" in rendered


def test_comparison_runs_and_resumes_with_generation_only_loader(tmp_path, monkeypatch):
    train_path = tmp_path / "train.txt"
    val_path = tmp_path / "val.txt"
    train_path.write_text("abcabcabc", encoding="utf-8")
    val_path.write_text("cbacba", encoding="utf-8")
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        json.dumps(
            {
                "id": "p1",
                "theme": "simple",
                "prompt": "ab",
                "evaluation_notes": "Continue.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = {
        "d_model": 8,
        "num_layers": 1,
        "num_heads": 2,
        "head_dim": 4,
        "attention_type": "mha",
        "context_length": 8,
        "dropout": 0.0,
    }
    model = TinyTransformerLM(vocab_size=3, **config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "step": 7,
            "model_config": config,
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "args": {"data_path": str(train_path), "val_data_path": str(val_path)},
        },
        checkpoint_path,
    )
    output_dir = tmp_path / "results"
    argv = [
        "compare_tinystories_prompts.py",
        "--checkpoint",
        f"tiny={checkpoint_path}",
        "--prompts_file",
        str(prompts_path),
        "--device",
        "cpu",
        "--max_new_tokens",
        "2",
        "--samples_per_prompt",
        "5",
        "--output_dir",
        str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    main()
    main()

    rows = [json.loads(line) for line in (output_dir / "completions.jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert {row["checkpoint_step"] for row in rows} == {7}
    assert {row["prompt_id"] for row in rows} == {"p1"}
    assert {row["model"] for row in rows} == {"tiny"}
    assert [row["sample_number"] for row in rows] == [1, 2, 3, 4, 5]
    assert "Completed generations: 5/5" in (output_dir / "comparison.md").read_text()
    assert json.loads((output_dir / "run_manifest.json").read_text())["models"] == ["tiny"]
    individual = output_dir / "checkpoints" / "tiny" / "completions.jsonl"
    assert len(individual.read_text().splitlines()) == 5
    assert "Completed generations: 5/5" in (
        output_dir / "checkpoints" / "tiny" / "report.md"
    ).read_text()
