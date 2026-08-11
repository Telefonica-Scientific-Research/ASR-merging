#!/bin/bash
#SBATCH --job-name=voxtral_2opt_bias
#SBATCH --output=./slurm_logs/voxtral_2opt_bias_%j.out
#SBATCH --error=./slurm_logs/voxtral_2opt_bias_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00

# 2-option positional-bias study — OOTB Voxtral-Mini-3B-2507
#
# Evaluates 4 bias-mitigation configs in parallel (1 GPU each) on the 413
# 2-option questions extracted from the challenge Phase-2 eval set.
#
# CONFIGS (1 per GPU):
#   GPU 0  baseline       No ICL, no debias.  Baseline A-bias measurement.
#   GPU 1  icl2_2opt      --few-shot-count 2  (slots [0],[1] = 2-option ICL shots)
#   GPU 2  icl4_2opt4opt  --few-shot-count 4  (slots [0],[1] 2-opt + [2],[3] 4-opt shots)
#   GPU 3  debias         --debias-cyclic-permutation  (2 rotations → cancels A-slot prior)
#
# All configs use:
#   - transcript-augmented prompts (--transcript-dir data/transcripts)
#   - question-timestamp cropping  (--eval-crop-from-question-refs --collar 30s)
#   - fp16 (same as production OOTB runs)
#
# Usage:
#   sbatch asr_scripts/run_voxtral_2opt_bias_study.sh
#
# Output directory:
#   experiments/voxtral_ootb_2opt_bias_study_<JOBID>/
#     baseline/   icl2_2opt/   icl4_2opt4opt/   debias/
#   Each subdir contains forgetting_eval_predictions.jsonl + metrics.

set -euo pipefail

MODEL_ID="${MODEL_ID:-mistralai/Voxtral-Mini-3B-2507}"
EVAL_AUDIO_ROOT="${EVAL_AUDIO_ROOT:-data/mlc26_task2/mlc-slm-2nd-eval}"
TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-data/transcripts}"
JSONL_2OPT="${JSONL_2OPT:-data/mlc26_task2/challenge_eval_2opt_only.jsonl}"

echo "=========================================="
echo "MareNostrum 5 - Voxtral 2-option bias study"
echo "Base model : ${MODEL_ID}"
echo "Dataset    : ${JSONL_2OPT}"
echo "Configs    : baseline | icl2_2opt | icl4_2opt4opt | debias"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
date
echo "=========================================="

module purge
module load singularity

export PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export REPO_DIR="/opt/ASR-merging"
export CACHE_DIR="/gpfs/scratch/ehpc628/models"
export TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"
export MLC_DATA_DIR="/gpfs/scratch/ehpc494/jls/datasets"

RUN_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}/tmp"
OUTPUT_BASE="experiments/voxtral_ootb_2opt_bias_study_${SLURM_JOB_ID:-manual}"

REPO_HOST="${SANDBOX_DIR}/opt/ASR-merging"

mkdir -p ./slurm_logs
mkdir -p "$TMP_ROOT"
mkdir -p "${REPO_HOST}/${OUTPUT_BASE}"

# Verify input JSONL exists
if [[ ! -f "${REPO_HOST}/${JSONL_2OPT}" ]]; then
  echo "ERROR: 2-option JSONL not found: ${REPO_HOST}/${JSONL_2OPT}"
  exit 1
fi

# -----------------------------------------------------------------
# Config definitions
# -----------------------------------------------------------------
CONFIG_NAMES=("baseline" "icl2_2opt" "icl4_2opt4opt" "debias" "calibration")
CONFIG_FLAGS=(
  ""
  "--few-shot-count 2"
  "--few-shot-count 4"
  "--debias-cyclic-permutation"
  "--calibrate-prior"
)

# Common flags for all configs
COMMON_FLAGS="
  --model-id ${MODEL_ID}
  --tasks jsonl_audio_mc
  --jsonl-path ${JSONL_2OPT}
  --audio-root ${EVAL_AUDIO_ROOT}
  --jsonl-max-questions-per-audio 0
  --max-samples-per-task 0
  --split test
  --no-use-bf16
  --use-fp16
  --prompt-language en
  --transcript-dir ${TRANSCRIPT_DIR}
  --eval-crop-from-question-refs
  --eval-crop-collar-seconds 30
  --eval-random-crop-seconds 0
  --prediction-only
"
# Collapse to a single line so it expands safely inside bash -c heredoc strings
# (multi-line expansion breaks \ line-continuations in SHARD_CMD).
COMMON_FLAGS=$(echo "$COMMON_FLAGS" | tr '\n' ' ')

# -----------------------------------------------------------------
# Launch configs (wave scheduler: up to 4 parallel, then next wave)
# -----------------------------------------------------------------
declare -a PIDS=()
declare -a LOG_FILES=()

next_idx=0
total=${#CONFIG_NAMES[@]}
running=0
declare -a gpu_pids=(-1 -1 -1 -1)
declare -a gpu_cfgs=("" "" "" "")

launch_config() {
  local gpu="$1"
  local idx="$2"
  local NAME="${CONFIG_NAMES[$idx]}"
  local EXTRA="${CONFIG_FLAGS[$idx]}"
  local OUTPUT_REL="${OUTPUT_BASE}/${NAME}"
  local OUTPUT_ABS="${REPO_HOST}/${OUTPUT_REL}"
  local LOG_FILE="${SLURM_SUBMIT_DIR}/slurm_logs/voxtral_2opt_bias_${NAME}_${SLURM_JOB_ID:-manual}.log"

  mkdir -p "$OUTPUT_ABS"
  LOG_FILES[$idx]="$LOG_FILE"

  local SHARD_CMD="
set -euo pipefail
export HF_HOME=${CACHE_DIR}
export HF_HUB_CACHE=${CACHE_DIR}
export TRANSFORMERS_CACHE=${CACHE_DIR}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=${TORCH_EXT_DIR}
export PYTHONUNBUFFERED=1
export TMPDIR=${TMP_ROOT}/${NAME}
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${gpu}
mkdir -p \${TMPDIR}
cd ${REPO_DIR}
. /opt/asrenv/bin/activate
echo '[${NAME}] Starting on GPU ${gpu}...'
python -m asr_merging.voxtral_forgetting_eval \\
  ${COMMON_FLAGS} \\
  --output-dir ${OUTPUT_REL} \\
  ${EXTRA}
echo '[${NAME}] Done.'
"

  singularity exec \
    --nv \
    --writable \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $TORCH_EXT_DIR:$TORCH_EXT_DIR \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind $TMP_ROOT:$TMP_ROOT \
    --bind /gpfs:/gpfs \
    --bind $MLC_DATA_DIR:/mnt/mirror1/datasets/MLC-SLM_Workshop_2026 \
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "$SHARD_CMD" > "$LOG_FILE" 2>&1 &

  gpu_pids[$gpu]=$!
  gpu_cfgs[$gpu]=$idx
  running=$((running + 1))
  echo "Launched config '${NAME}' on GPU ${gpu} (pid ${gpu_pids[$gpu]}), log: ${LOG_FILE}"
}

# Initial fill
for gpu in 0 1 2 3; do
  if [[ "$next_idx" -ge "$total" ]]; then break; fi
  launch_config "$gpu" "$next_idx"
  next_idx=$((next_idx + 1))
done

# Dynamic scheduling
FAILED=0
while [[ "$running" -gt 0 ]]; do
  progress=0
  for gpu in 0 1 2 3; do
    local_pid="${gpu_pids[$gpu]}"
    if [[ "$local_pid" == "-1" ]]; then continue; fi
    if ! kill -0 "$local_pid" 2>/dev/null; then
      cfg_idx="${gpu_cfgs[$gpu]}"
      if wait "$local_pid"; then
        echo "[OK]   ${CONFIG_NAMES[$cfg_idx]}"
      else
        echo "[FAIL] ${CONFIG_NAMES[$cfg_idx]}"
        FAILED=$((FAILED + 1))
      fi
      gpu_pids[$gpu]=-1
      gpu_cfgs[$gpu]=""
      running=$((running - 1))
      progress=1
      if [[ "$next_idx" -lt "$total" ]]; then
        launch_config "$gpu" "$next_idx"
        next_idx=$((next_idx + 1))
      fi
    fi
  done
  if [[ "$progress" -eq 0 ]]; then sleep 5; fi
done

# -----------------------------------------------------------------
# Quick label-distribution summary
# -----------------------------------------------------------------
echo ""
echo "============== LABEL DISTRIBUTION SUMMARY (2-option questions) =============="
for NAME in "${CONFIG_NAMES[@]}"; do
  PRED_FILE="${REPO_HOST}/${OUTPUT_BASE}/${NAME}/forgetting_eval_predictions.jsonl"
  if [[ -f "$PRED_FILE" ]]; then
    echo ""
    echo "--- ${NAME} ---"
    python3 - "$PRED_FILE" << 'PYEOF'
import json, sys
from collections import Counter
preds = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
total = len(preds)
cnt = Counter(p.get("pred_choice","?") for p in preds)
raw_cnt = Counter(p.get("yn_pred_raw") or p.get("pred_choice","?") for p in preds)
for lbl in sorted(cnt):
    print(f"  {lbl}: {cnt[lbl]:4d}  ({100*cnt[lbl]/total:.1f}%)")
if any(p.get("yn_pred_raw") for p in preds):
    print(f"  Raw Yes/No: { {k:v for k,v in raw_cnt.items()} }")
print(f"  Total: {total}")
PYEOF
  else
    echo "--- ${NAME}: predictions file not found ---"
  fi
done

echo ""
echo "Results: ${REPO_HOST}/${OUTPUT_BASE}/"
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

if [[ "$FAILED" -ne 0 ]]; then
  echo "$FAILED config(s) failed."
  exit 1
fi
