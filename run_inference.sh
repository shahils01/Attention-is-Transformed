#!/usr/bin/env bash

# Run the complete TinyStories checkpoint evaluation pipeline:
#   inference -> blind-set preparation -> blind judging -> unblinding
#
# Activate the desired HPC Python environment before launching this script, or
# set VENV_DIR to a virtual environment that should be activated here.

set -Eeuo pipefail

trap 'echo "ERROR: run_inference.sh failed at line ${LINENO}." >&2' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"

if [[ -n "${VENV_DIR:-}" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_DIR}/ckpts/TinyStories-v2 Checkpoints}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-**/checkpoint*.pt}"
PROMPTS_FILE="${PROMPTS_FILE:-${REPO_DIR}/benchmarks/tinystories_100_prompts.jsonl}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data/tinystories}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/TinyStoriesV2-GPT4-train.txt}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/TinyStoriesV2-GPT4-valid.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/tinystories_all_checkpoints_100_x5_1600}"
BLIND_DIR="${BLIND_DIR:-${OUTPUT_DIR}/blind_eval}"

DEVICE="${DEVICE:-cuda}"
PRECISION="${PRECISION:-bf16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1600}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-0}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-5}"
NUM_PROMPTS="${NUM_PROMPTS:-0}"

JUDGE_PROVIDER="${JUDGE_PROVIDER:-rcd-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.6-sol}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-medium}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-RCD_LLM_API_KEY}"
JUDGE_REQUEST_DELAY="${JUDGE_REQUEST_DELAY:-0}"

COMPLETIONS_FILE="${OUTPUT_DIR}/completions.jsonl"
MANIFEST_FILE="${OUTPUT_DIR}/run_manifest.json"
BLIND_FILE="${BLIND_DIR}/blind.jsonl"
MAPPING_FILE="${BLIND_DIR}/blind_mapping.jsonl"
GUIDANCE_FILE="${BLIND_DIR}/judge_instructions.md"
SCORES_FILE="${BLIND_DIR}/blind_scores.jsonl"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory does not exist: $1" >&2
    exit 2
  fi
}

require_dir "${REPO_DIR}"
require_dir "${CHECKPOINT_DIR}"
require_file "${PROMPTS_FILE}"
require_file "${TRAIN_DATA}"
require_file "${VAL_DATA}"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9+ is required"'

mkdir -p "${OUTPUT_DIR}" "${BLIND_DIR}"
cd "${REPO_DIR}"

echo "[1/4] Generating checkpoint completions (resumes completed generations)"
"${PYTHON_BIN}" -u experiments/compare_tinystories_prompts.py \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --checkpoint_glob "${CHECKPOINT_GLOB}" \
  --prompts_file "${PROMPTS_FILE}" \
  --data_path "${TRAIN_DATA}" \
  --val_data_path "${VAL_DATA}" \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_k "${TOP_K}" \
  --seed "${SEED}" \
  --samples_per_prompt "${SAMPLES_PER_PROMPT}" \
  --output_dir "${OUTPUT_DIR}"

echo "[2/4] Preparing anonymous blind-evaluation records"
blind_artifacts=0
for path in "${BLIND_FILE}" "${MAPPING_FILE}" "${GUIDANCE_FILE}"; do
  if [[ -e "${path}" ]]; then
    blind_artifacts=$((blind_artifacts + 1))
  fi
done

if [[ "${blind_artifacts}" -eq 0 ]]; then
  if [[ -s "${SCORES_FILE}" ]]; then
    echo "A score file exists without its blind-set files: ${SCORES_FILE}" >&2
    echo "Choose a new OUTPUT_DIR/BLIND_DIR or move the stale score file." >&2
    exit 2
  fi
  "${PYTHON_BIN}" -u experiments/evaluate_blind_checkpoints.py prepare \
    --completions "${COMPLETIONS_FILE}" \
    --manifest "${MANIFEST_FILE}" \
    --num-prompts "${NUM_PROMPTS}" \
    --seed "${SEED}" \
    --output-dir "${BLIND_DIR}"
elif [[ "${blind_artifacts}" -eq 3 ]]; then
  "${PYTHON_BIN}" - \
    "${MANIFEST_FILE}" \
    "${MAPPING_FILE}" \
    "${COMPLETIONS_FILE}" \
    "${NUM_PROMPTS}" \
    "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
mapping_path = Path(sys.argv[2])
completions_path = Path(sys.argv[3])
requested_prompts = int(sys.argv[4])
requested_seed = int(sys.argv[5])
manifest_run_id = json.loads(manifest_path.read_text(encoding="utf-8"))["run_id"]
mapping_rows = [
    json.loads(line)
    for line in mapping_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
mapping_run_ids = {row["source_run_id"] for row in mapping_rows}
if mapping_run_ids != {manifest_run_id}:
    raise SystemExit(
        "Existing blind files belong to a different generation run. "
        "Choose a new OUTPUT_DIR/BLIND_DIR or remove those blind artifacts."
    )
if {row["selection_seed"] for row in mapping_rows} != {requested_seed}:
    raise SystemExit(
        "Existing blind files use a different selection seed. "
        "Choose a new OUTPUT_DIR/BLIND_DIR or remove those blind artifacts."
    )
selected_prompt_count = len({row["prompt_id"] for row in mapping_rows})
if requested_prompts == 0:
    completion_prompt_ids = {
        row["prompt_id"]
        for line in completions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row.get("run_id") == manifest_run_id
    }
    requested_prompts = len(completion_prompt_ids)
if selected_prompt_count != requested_prompts:
    raise SystemExit(
        f"Existing blind files contain {selected_prompt_count} prompts, but "
        f"this run requests {requested_prompts}. Choose a new OUTPUT_DIR/BLIND_DIR "
        "or remove those blind artifacts."
    )
print(f"Blind set already prepared for run {manifest_run_id}; reusing it.")
PY
else
  echo "Blind evaluation directory is incomplete. Expected all or none of:" >&2
  echo "  ${BLIND_FILE}" >&2
  echo "  ${MAPPING_FILE}" >&2
  echo "  ${GUIDANCE_FILE}" >&2
  exit 2
fi

if [[ -z "${!JUDGE_API_KEY_ENV:-}" ]]; then
  echo "Set ${JUDGE_API_KEY_ENV} before blind judging." >&2
  exit 2
fi

echo "[3/4] Blind judging (resumes records already present in the score file)"
"${PYTHON_BIN}" -u experiments/evaluate_blind_checkpoints.py judge \
  --blind-file "${BLIND_FILE}" \
  --output-file "${SCORES_FILE}" \
  --provider "${JUDGE_PROVIDER}" \
  --model "${JUDGE_MODEL}" \
  --api-key-env "${JUDGE_API_KEY_ENV}" \
  --reasoning-effort "${JUDGE_REASONING_EFFORT}" \
  --request-delay "${JUDGE_REQUEST_DELAY}"

echo "[4/4] Unblinding scores and writing aggregate reports"
"${PYTHON_BIN}" -u experiments/evaluate_blind_checkpoints.py unblind \
  --blind-file "${BLIND_FILE}" \
  --mapping-file "${MAPPING_FILE}" \
  --scores-file "${SCORES_FILE}" \
  --output-dir "${BLIND_DIR}" \
  --overwrite

echo "Pipeline complete. Leaderboard: ${BLIND_DIR}/leaderboard.csv"
