#!/usr/bin/env bash
# GPU smoke test for loading and training the selected Identity GT-MHA checkpoint.
#SBATCH --job-name=id-glue-smoke
#SBATCH --account=bicn-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=/work/hdd/bicn/anayak2/logs/%x_%j.out
#SBATCH --error=/work/hdd/bicn/anayak2/logs/%x_%j.err

set -euo pipefail
PROJECT_DIR=/work/hdd/bicn/anayak2
REPO_DIR=${PROJECT_DIR}/repos/Attention-is-Transformed-bert-identity
VENV_DIR=${PROJECT_DIR}/venvs/lgma-clean
CHECKPOINT_DIR=${PROJECT_DIR}/lgma_runs/bert_base_gt_mha_identity_qk_b4g8h12_random_seed42

module load python/miniforge3_pytorch/2.12.0
source "${VENV_DIR}/bin/activate"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/compat/no_transformer_engine:${REPO_DIR}/src:${PYTHONPATH:-}"

python - <<'PY'
import torch
import transformer_engine
from transformers import Trainer
import peft
from lgma.bert import load_bert_sequence_classifier

assert not hasattr(transformer_engine, "pytorch")

checkpoint = "/work/hdd/bicn/anayak2/lgma_runs/bert_base_gt_mha_identity_qk_b4g8h12_random_seed42"
model, audit = load_bert_sequence_classifier(
    checkpoint,
    num_labels=3,
    attention_type="gt_mha_identity",
    num_base_heads=4,
    num_generators=8,
    use_sdpa=True,
    fuse_base_qkv=True,
    sdpa_gqa_mode="native",
    enforce_paper_gt_mha=False,
)
attention = model.bert.encoder.layer[0].attention.self.gt_attention
assert attention.metric_mode == "identity"
assert attention.value_transform_mode == "residual"
assert not hasattr(attention, "generators")
assert hasattr(attention, "value_generators")
model.cuda().train()
input_ids = torch.randint(0, model.config.vocab_size, (2, 16), device="cuda")
attention_mask = torch.ones_like(input_ids)
labels = torch.tensor([0, 1], device="cuda")
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
loss.backward()
assert torch.isfinite(loss)
print({"status": "SMOKE_OK", "layers_replaced": len(audit), "loss": float(loss)})
PY
