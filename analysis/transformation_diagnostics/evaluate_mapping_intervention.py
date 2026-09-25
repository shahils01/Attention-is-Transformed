#!/usr/bin/env python3
"""Full-validation counterfactual mapping evaluation for TinyStories GT-MHA.

The trained residual checkpoint is held fixed.  Only the runtime mapping used
to construct the query--key and/or value transformations is changed from
``residual`` to ``exp``.  Every variant is evaluated on the same complete set
of validation stories, and uncertainty is computed by paired story bootstrap.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = Path(
    "/work/hdd/bifg/arai3/lgma_jobs/tinystories_ci_full_20260915"
)
DATA_ROOT = Path("/work/hdd/bifg/arai3/lgma_data/tinystories")
CHECKPOINT = REFERENCE_ROOT / "GT-MHA residual.pt"
EXPECTED_CHECKPOINT_SHA256 = (
    "a3b36ada91bb42b22b6df4da2b36be5cd3333f5a473e857c326aa06323ae06f2"
)
BOOTSTRAP_SEED = 731
BOOTSTRAP_REPLICATES = 2000

VARIANTS = (
    ("residual_qk_residual_v", "Residual QK / residual value", "residual", "residual"),
    ("exp_qk_residual_v", "Exponential QK / residual value", "exp", "residual"),
    ("residual_qk_exp_v", "Residual QK / exponential value", "residual", "exp"),
    ("exp_qk_exp_v", "Exponential QK / exponential value", "exp", "exp"),
)
BASELINE_KEY = VARIANTS[0][0]


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_data() -> tuple[list[str], list[int], list[str], dict[str, int], str]:
    train_path = DATA_ROOT / "TinyStoriesV2-GPT4-train.txt"
    validation_path = DATA_ROOT / "TinyStoriesV2-GPT4-valid.txt"
    chars: set[str] = set()
    with train_path.open(encoding="utf-8") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            chars.update(chunk)
    validation_text = validation_path.read_text(encoding="utf-8")
    chars.update(validation_text)
    vocabulary = sorted(chars)
    stoi = {character: index for index, character in enumerate(vocabulary)}
    stories = validation_text.split("<|endoftext|>")
    story_ids = [index for index, story in enumerate(stories) if len(story) > 1]
    validation_sha256 = hashlib.sha256(validation_text.encode()).hexdigest()
    return stories, story_ids, vocabulary, stoi, validation_sha256


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(17029)

    sys.path.insert(0, str(REFERENCE_ROOT / "source"))
    from lgma.attention import LieGeneratedMetricAttention  # noqa: PLC0415
    from lgma.transformer import TinyTransformerLM  # noqa: PLC0415

    observed_checkpoint_sha256 = sha256(CHECKPOINT)
    if observed_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch: "
            f"{observed_checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
        )

    stories, story_ids, vocabulary, stoi, validation_sha256 = build_validation_data()
    reference_manifest = json.loads((REFERENCE_ROOT / "manifest.json").read_text())
    if validation_sha256 != reference_manifest["validation_sha256"]:
        raise RuntimeError("Validation-corpus hash differs from the established evaluation")
    if story_ids != reference_manifest["story_ids"]:
        raise RuntimeError("Validation story identifiers differ from the established evaluation")
    if vocabulary != reference_manifest["vocabulary"]:
        raise RuntimeError("Vocabulary differs from the established evaluation")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    config = dict(checkpoint["model_config"])
    if config["context_length"] != 512:
        raise RuntimeError(f"Unexpected context length: {config['context_length']}")
    if config["num_base_heads"] != 4 or config["num_generators"] != 8:
        raise RuntimeError("Checkpoint does not match the paper GT-MHA configuration")
    model = TinyTransformerLM(vocab_size=len(vocabulary), **config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    checkpoint_step = checkpoint.get("step")
    del checkpoint
    gc.collect()
    model = model.to("cuda").eval()

    attention_modules = [
        module
        for module in model.modules()
        if isinstance(module, LieGeneratedMetricAttention)
    ]
    if len(attention_modules) != config["num_layers"]:
        raise RuntimeError(
            f"Found {len(attention_modules)} GT-MHA modules; expected {config['num_layers']}"
        )
    if any(module.metric_mode != "residual" for module in attention_modules):
        raise RuntimeError("Checkpoint QK mapping is not residual")
    if any(module.value_transform_mode != "residual" for module in attention_modules):
        raise RuntimeError("Checkpoint value mapping is not residual")

    original_transform_methods = [
        (module.compute_metrics, module.compute_value_transforms)
        for module in attention_modules
    ]

    def configure_mapping(qk_mode: str, value_mode: str) -> None:
        """Set mapping modes and cache only exact transforms for fixed-weight evaluation."""
        for module, (compute_metrics, compute_values) in zip(
            attention_modules, original_transform_methods
        ):
            module.compute_metrics = compute_metrics
            module.compute_value_transforms = compute_values
            module.metric_mode = qk_mode
            module.value_transform_mode = value_mode

        # Exact transforms depend only on fixed model parameters.  Precomputing
        # them avoids repeating matrix exponentials for every context window.
        # Residual pathways remain untouched so the baseline exactly reproduces
        # the established evaluation implementation.
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for module, (compute_metrics, compute_values) in zip(
                attention_modules, original_transform_methods
            ):
                if qk_mode == "exp":
                    cached_metrics = compute_metrics().detach()
                    module.compute_metrics = lambda cached=cached_metrics: cached
                if value_mode == "exp":
                    cached_values = compute_values().detach()
                    module.compute_value_transforms = lambda cached=cached_values: cached

    def loss_for(text: str) -> tuple[float, int]:
        encoded = torch.tensor(
            [stoi[character] for character in text],
            dtype=torch.long,
            device="cuda",
        )
        total = 0.0
        count = 0
        context_length = int(config["context_length"])
        for position in range(0, len(text) - 1, context_length):
            target_count = min(context_length, len(text) - 1 - position)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(encoded[position : position + target_count].unsqueeze(0))
                loss = F.cross_entropy(
                    logits.float().reshape(-1, len(vocabulary)),
                    encoded[position + 1 : position + target_count + 1],
                    reduction="sum",
                )
            total += loss.item()
            count += target_count
        return total, count

    protocol = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": observed_checkpoint_sha256,
        "checkpoint_step": checkpoint_step,
        "validation_sha256": validation_sha256,
        "story_count": len(story_ids),
        "mapping_intervention": (
            "Frozen trained residual checkpoint; runtime query-key and value mappings "
            "changed independently to exact matrix exponentiation"
        ),
        "evaluation": reference_manifest["protocol"],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "variants": [
            {"key": key, "label": label, "qk_mode": qk, "value_mode": value}
            for key, label, qk, value in VARIANTS
        ],
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    results: dict[str, dict[str, object]] = {}
    per_story_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, label, qk_mode, value_mode in VARIANTS:
        configure_mapping(qk_mode, value_mode)
        first = loss_for(stories[story_ids[0]])
        second = loss_for(stories[story_ids[0]])
        if not np.isclose(first[0], second[0], rtol=1e-6):
            raise RuntimeError(f"Repeatability check failed for {key}: {first} vs {second}")

        torch.cuda.synchronize()
        start = time.perf_counter()
        rows: list[dict[str, float | int]] = []
        for offset, story_id in enumerate(story_ids):
            nll_sum, token_count = loss_for(stories[story_id])
            if not math.isfinite(nll_sum) or token_count != len(stories[story_id]) - 1:
                raise RuntimeError(f"Invalid evaluation result for {key}, story {story_id}")
            rows.append(
                {"story_id": story_id, "nll_sum": nll_sum, "tokens": token_count}
            )
            if (offset + 1) % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "variant": key,
                            "stories_done": offset + 1,
                            "total_stories": len(story_ids),
                            "seconds": time.perf_counter() - start,
                        }
                    ),
                    flush=True,
                )
                (ROOT / f"{key}_partial.json").write_text(json.dumps(rows))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        (ROOT / f"{key}_per_story.json").write_text(json.dumps(rows, indent=2))
        losses = np.asarray([row["nll_sum"] for row in rows], dtype=np.float64)
        counts = np.asarray([row["tokens"] for row in rows], dtype=np.int64)
        mean_nll = float(losses.sum() / counts.sum())
        results[key] = {
            "label": label,
            "qk_mode": qk_mode,
            "value_mode": value_mode,
            "nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "evaluated_tokens": int(counts.sum()),
            "eval_seconds": elapsed,
            "tokens_per_second": int(counts.sum()) / elapsed,
            "repeatability_passed": True,
        }
        per_story_arrays[key] = (losses, counts)
        (ROOT / "progress_summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({key: results[key]}, indent=2), flush=True)

    baseline_losses, baseline_counts = per_story_arrays[BASELINE_KEY]
    if any(not np.array_equal(counts, baseline_counts) for _, counts in per_story_arrays.values()):
        raise RuntimeError("Token counts differ across counterfactual variants")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_nll: dict[str, list[np.ndarray]] = {key: [] for key, *_ in VARIANTS}
    for _ in range(20):
        indices = rng.integers(0, len(story_ids), (100, len(story_ids)))
        denominator = baseline_counts[indices].sum(axis=1)
        for key, (losses, _) in per_story_arrays.items():
            bootstrap_nll[key].append(losses[indices].sum(axis=1) / denominator)

    baseline_bootstrap = np.concatenate(bootstrap_nll[BASELINE_KEY])
    for key, *_ in VARIANTS:
        samples = np.concatenate(bootstrap_nll[key])
        ci = np.quantile(samples, [0.025, 0.975])
        results[key]["nll_ci95"] = ci.tolist()
        results[key]["ppl_ci95"] = np.exp(ci).tolist()
        if key != BASELINE_KEY:
            difference = samples - baseline_bootstrap
            ppl_difference = np.exp(samples) - np.exp(baseline_bootstrap)
            results[key]["delta_nll_vs_residual"] = (
                float(results[key]["nll"]) - float(results[BASELINE_KEY]["nll"])
            )
            results[key]["delta_nll_ci95"] = np.quantile(
                difference, [0.025, 0.975]
            ).tolist()
            results[key]["delta_ppl_vs_residual"] = (
                float(results[key]["perplexity"])
                - float(results[BASELINE_KEY]["perplexity"])
            )
            results[key]["delta_ppl_ci95"] = np.quantile(
                ppl_difference, [0.025, 0.975]
            ).tolist()

    reference_residual = json.loads(
        (REFERENCE_ROOT / "summary.json").read_text()
    )["GT-MHA residual"]
    baseline_difference = abs(
        float(results[BASELINE_KEY]["nll"]) - float(reference_residual["nll"])
    )
    if baseline_difference > 1e-8:
        raise RuntimeError(
            "Residual baseline does not reproduce the established evaluation: "
            f"absolute NLL difference {baseline_difference}"
        )

    output = {
        "status": "MAPPING_INTERVENTION_COMPLETE",
        "protocol": protocol,
        "results": results,
        "baseline_replication_absolute_nll_error": baseline_difference,
        "uncertainty": (
            "2000 paired story-cluster percentile bootstrap replicates; fixed checkpoint; "
            "not training-seed variability"
        ),
    }
    (ROOT / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(output["status"], flush=True)


if __name__ == "__main__":
    main()
