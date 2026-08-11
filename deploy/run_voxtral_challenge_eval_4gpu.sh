#!/bin/bash
#SBATCH --job-name=voxtral_challenge_eval
#SBATCH --output=./slurm_logs/voxtral_challenge_eval_%j.out
#SBATCH --error=./slurm_logs/voxtral_challenge_eval_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Usage:
#   MODEL_ADAPTER_PATH=experiments/voxtral_mcq_connector_only_4gpu_TIMESTAMP/final_model \
#   sbatch deploy/run_voxtral_challenge_eval_4gpu.sh
#
# Optional env vars:
#   EVAL_CROP=1        (default) crop audio from question timestamp refs (collar=30s)
#   EVAL_CROP=0        use full audio — no cropping
#   TRANSCRIPT_DIR=data/transcripts   inject ASR transcripts into prompts (default)
#   TRANSCRIPT_DIR=    disable transcript hints
#   DEBIAS_CYCLIC=1    position-debias MCQ via cyclic option permutation (default 0)
#
# Output (hyp.txt + per-shard dirs) lands next to the final_model:
#   experiments/EXPERIMENT/challenge_hyp_{crop|full}[_debias].txt
#   experiments/EXPERIMENT/challenge_eval_{crop|full}[_debias]_shard_00_of_04/hyp.txt  ...

set -euo pipefail

if [[ -z "${MODEL_ADAPTER_PATH:-}" ]]; then
  echo "ERROR: MODEL_ADAPTER_PATH is not set."
  echo "Usage: MODEL_ADAPTER_PATH=experiments/.../final_model sbatch deploy/run_voxtral_challenge_eval_4gpu.sh"
  exit 1
fi

NUM_SHARDS=4
EVAL_AUDIO_ROOT="${EVAL_AUDIO_ROOT:-data/mlc26_task2/mlc-slm-2nd-eval}"
# Pass TRANSCRIPTION_HINT_FORMAT=1 to inject lang:en\n[TRANSCRIBE]\n prefix inside [INST].
# Must match the flag used during training (--use-transcription-hint-format in voxtral_train_MCQ.py).
TRANSCRIPTION_HINT_FORMAT="${TRANSCRIPTION_HINT_FORMAT:-0}"
PROMPT_LANGUAGE="${PROMPT_LANGUAGE:-en}"
# Set TRANSCRIPT_DIR to the directory containing ASR .txt files to inject transcripts into prompts.
# Leave empty (default) to run without transcript hints.
TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-data/transcripts}"
# EVAL_CROP=1 (default): crop audio to ±30s around question timestamp references.
# EVAL_CROP=0: pass full audio to the model (no cropping).
EVAL_CROP="${EVAL_CROP:-1}"
# DEBIAS_CYCLIC=1: position-debias MCQ predictions via cyclic option permutation
# (N forward passes/question, cancels the per-slot "A" prior). Default 0 = off.
DEBIAS_CYCLIC="${DEBIAS_CYCLIC:-0}"
EVAL_MODE_TAG=$([ "${EVAL_CROP}" == "1" ] && echo "crop" || echo "full")
if [[ "${DEBIAS_CYCLIC}" == "1" ]]; then
  EVAL_MODE_TAG="${EVAL_MODE_TAG}_debias"
fi

# Derive experiment dir from the adapter path (strip trailing /final_model if present)
EXPERIMENT_DIR="${MODEL_ADAPTER_PATH%/final_model}"
EXPERIMENT_DIR="${EXPERIMENT_DIR%/}"

echo "=========================================="
echo "MareNostrum 5 - Voxtral Challenge Eval (4-GPU single job)"
echo "Model/adapter: ${MODEL_ADAPTER_PATH}"
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Transcription hint format: ${TRANSCRIPTION_HINT_FORMAT} (prompt language: ${PROMPT_LANGUAGE})"
echo "Transcript dir: ${TRANSCRIPT_DIR:-<none>}"
echo "Audio mode    : ${EVAL_MODE_TAG} (EVAL_CROP=${EVAL_CROP})"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
date
echo "=========================================="

module purge
module load singularity

export PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SCRATCH_DIR="/gpfs/scratch/ehpc628/jls/ehpc628XXX"
export REPO_DIR="/opt/ASR-merging"
export CACHE_DIR="/gpfs/scratch/ehpc628/models"
export TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"
export MLC_DATA_DIR="/gpfs/scratch/ehpc494/jls/datasets"
RUN_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}/tmp"

mkdir -p ./slurm_logs
mkdir -p "$TMP_ROOT"

# Repo root on the host (container mounts SANDBOX_DIR as writable root, repo lives at opt/ASR-merging inside)
REPO_HOST="${SANDBOX_DIR}/opt/ASR-merging"

# Verify shard JSONL files exist before spending GPU time
SHARD_DIR="${REPO_HOST}/data/mlc26_task2/challenge_eval_shards"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_FILE="${SHARD_DIR}/shard_$(printf '%02d' $i)_of_$(printf '%02d' $NUM_SHARDS).jsonl"
  if [[ ! -f "$SHARD_FILE" ]]; then
    echo "ERROR: Shard file not found: $SHARD_FILE"
    exit 1
  fi
done

# Build per-shard CMD and launch one singularity process per GPU in background
declare -a PIDS=()
declare -a SHARD_LOG_FILES=()
declare -a SHARD_OUTPUT_DIRS=()

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_TAG="shard_$(printf '%02d' $i)_of_$(printf '%02d' $NUM_SHARDS)"
  SHARD_JSONL="data/mlc26_task2/challenge_eval_shards/${SHARD_TAG}.jsonl"
  SHARD_OUTPUT_REL="${EXPERIMENT_DIR}/challenge_eval_${EVAL_MODE_TAG}_${SHARD_TAG}"
  SHARD_OUTPUT_ABS="${REPO_HOST}/${SHARD_OUTPUT_REL}"
  SHARD_LOG="${SLURM_SUBMIT_DIR}/slurm_logs/voxtral_challenge_eval_shard${i}_${SLURM_JOB_ID}.log"

  mkdir -p "$SHARD_OUTPUT_ABS"
  SHARD_OUTPUT_DIRS+=("$SHARD_OUTPUT_ABS")
  SHARD_LOG_FILES+=("$SHARD_LOG")

  # Compute hint flag outside the CMD string so it expands correctly.
  HINT_FLAG=""
  if [[ "${TRANSCRIPTION_HINT_FORMAT}" == "1" ]]; then
    HINT_FLAG="--use-transcription-hint-format"
  fi

  TRANSCRIPT_FLAG=""
  if [[ -n "${TRANSCRIPT_DIR:-}" ]]; then
    TRANSCRIPT_FLAG="--transcript-dir ${TRANSCRIPT_DIR}"
  fi

  CROP_FLAG=""
  if [[ "${EVAL_CROP}" == "1" ]]; then
    CROP_FLAG="--eval-crop-from-question-refs --eval-crop-collar-seconds 30 --eval-random-crop-seconds 0"
  fi

  DEBIAS_FLAG=""
  if [[ "${DEBIAS_CYCLIC}" == "1" ]]; then
    DEBIAS_FLAG="--debias-cyclic-permutation"
  fi

  SHARD_CMD="
set -euo pipefail
export HF_HOME=${CACHE_DIR}
export HF_HUB_CACHE=${CACHE_DIR}
export TRANSFORMERS_CACHE=${CACHE_DIR}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=${TORCH_EXT_DIR}
export PYTHONUNBUFFERED=1
export TMPDIR=${TMP_ROOT}/shard${i}
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${i}
mkdir -p \${TMPDIR}
cd ${REPO_DIR}
. /opt/asrenv/bin/activate
echo '[shard ${i}] Starting eval on GPU ${i}...'
python -m asr_merging.voxtral_forgetting_eval \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --adapter-path ${MODEL_ADAPTER_PATH} \
  --tasks jsonl_audio_mc \
  --jsonl-path ${SHARD_JSONL} \
  --audio-root ${EVAL_AUDIO_ROOT} \
  --jsonl-max-questions-per-audio 0 \
  --max-samples-per-task 0 \
  --split test \
  --no-use-bf16 \
  --use-fp16 \
  --output-dir ${SHARD_OUTPUT_REL} \
  --prompt-language ${PROMPT_LANGUAGE} \
  ${HINT_FLAG} \
  ${TRANSCRIPT_FLAG} \
  ${CROP_FLAG} \
  ${DEBIAS_FLAG} \
  --prediction-only
echo '[shard ${i}] Done.'
"

  singularity exec \
    --nv \
    --writable \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $TORCH_EXT_DIR:$TORCH_EXT_DIR \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind $SCRATCH_DIR:$SCRATCH_DIR \
    --bind /gpfs:/gpfs \
    --bind $MLC_DATA_DIR:/mnt/mirror1/datasets/MLC-SLM_Workshop_2026 \
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "$SHARD_CMD" > "$SHARD_LOG" 2>&1 &

  PIDS+=($!)
  echo "Launched shard ${i}/GPU${i} (pid ${PIDS[$i]}), log: $SHARD_LOG"
done

# Wait for all 4 shards and collect exit codes
echo ""
echo "Waiting for all ${NUM_SHARDS} shards to finish..."
FAILED=0
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  if wait "${PIDS[$i]}"; then
    echo "Shard ${i}: OK"
  else
    echo "Shard ${i}: FAILED (exit $?)"
    echo "--- last 20 lines of ${SHARD_LOG_FILES[$i]} ---"
    tail -20 "${SHARD_LOG_FILES[$i]}" 2>/dev/null || true
    FAILED=$((FAILED + 1))
  fi
done

# Print shard logs to main job output for visibility
echo ""
echo "=== Shard log tails ==="
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "--- shard ${i} ---"
  grep -E "processed|Done|error|Error" "${SHARD_LOG_FILES[$i]}" 2>/dev/null | tail -5 || true
done

if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/${NUM_SHARDS} shards failed. Aborting merge."
  exit 1
fi

# Merge shard hyp.txt files in order → EXPERIMENT_DIR/challenge_hyp_{crop|full}.txt
FINAL_HYP="${REPO_HOST}/${EXPERIMENT_DIR}/challenge_hyp_${EVAL_MODE_TAG}.txt"
echo ""
echo "Merging shard hyp.txt files → ${FINAL_HYP}"

HYP_FILES=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_HYP="${SHARD_OUTPUT_DIRS[$i]}/hyp.txt"
  if [[ ! -f "$SHARD_HYP" ]]; then
    echo "ERROR: Missing hyp.txt for shard ${i}: $SHARD_HYP"
    exit 1
  fi
  LINES=$(wc -l < "$SHARD_HYP")
  echo "  Shard ${i}: ${LINES} lines — $SHARD_HYP"
  HYP_FILES+=("$SHARD_HYP")
done

cat "${HYP_FILES[@]}" > "$FINAL_HYP"
TOTAL=$(wc -l < "$FINAL_HYP")
UNIQUE=$(sort -u "$FINAL_HYP" | wc -l)
echo ""
echo "Merged ${TOTAL} predictions → $FINAL_HYP"

# When cyclic debiasing is active, also merge the per-shard detail JSONL files
# (same predictions completed with per-permutation logprobs) in shard order.
if [[ "${DEBIAS_CYCLIC}" == "1" ]]; then
  FINAL_DETAIL="${REPO_HOST}/${EXPERIMENT_DIR}/challenge_hyp_${EVAL_MODE_TAG}_detail.jsonl"
  DETAIL_FILES=()
  MISSING_DETAIL=0
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    SHARD_DETAIL="${SHARD_OUTPUT_DIRS[$i]}/hyp_debias_detail.jsonl"
    if [[ ! -f "$SHARD_DETAIL" ]]; then
      echo "WARNING: Missing debias detail for shard ${i}: $SHARD_DETAIL"
      MISSING_DETAIL=$((MISSING_DETAIL + 1))
      continue
    fi
    DETAIL_FILES+=("$SHARD_DETAIL")
  done
  if [[ ${#DETAIL_FILES[@]} -gt 0 ]]; then
    cat "${DETAIL_FILES[@]}" > "$FINAL_DETAIL"
    DETAIL_TOTAL=$(wc -l < "$FINAL_DETAIL")
    echo "Merged ${DETAIL_TOTAL} debias detail rows → $FINAL_DETAIL"
  fi
fi

if [[ "$TOTAL" -ne "$UNIQUE" ]]; then
  echo "WARNING: Only ${UNIQUE} unique lines out of ${TOTAL} — possible duplicate question_ids."
else
  echo "Sanity OK: all ${TOTAL} predictions are unique."
fi

echo ""
echo "Submit this file to the challenge scorer:"
echo "  ${FINAL_HYP}"
echo ""
echo "=========================================="
echo "Challenge eval finished at $(date)"
echo "=========================================="
