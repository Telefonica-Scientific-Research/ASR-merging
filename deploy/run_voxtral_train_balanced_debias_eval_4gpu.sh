#!/bin/bash
#SBATCH --job-name=voxtral_ootb_train_eval
#SBATCH --output=./slurm_logs/voxtral_ootb_train_eval_%j.out
#SBATCH --error=./slurm_logs/voxtral_ootb_train_eval_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# OOTB (base, NO LoRA) Voxtral eval on the BALANCED FULL TRAINING data — WITH gold labels.
# Same recipe as the OOTB challenge eval (transcript-augmented prompt + question-ref crop +
# cyclic option debias), but run on mlcslm_2nd_dev_qa_successed_opensource_balanced.jsonl
# (150 audio rows, 4500 questions, exactly 25%/25%/25%/25% A/B/C/D, nested schema).
#
# WHY: on labeled, label-balanced data we can MEASURE (not guess) the accuracy effect of
#   - raw (rotation-0) prediction      vs
#   - cyclic position-debiased prediction
# The balanced prior means there is no answer-marginal to exploit, so any accuracy gain is
# pure positional-bias removal (the prior-agnostic mechanism that DOES transfer to the LB).
#
# IMPORTANT CAVEAT: do NOT read an "optimal label-prior shift" off this run and copy it to the
# LB. Balanced data wants a shift toward uniform; the hidden LB is A-heavy (~45% A). The thing
# that transfers is the cyclic-debias accuracy delta, not a distribution shift.
#
# Usage:
#   sbatch deploy/run_voxtral_train_balanced_debias_eval_4gpu.sh
#
# Optional env vars (defaults match the OOTB challenge recipe):
#   EVAL_CROP=1            (default) crop audio from question timestamp refs (collar=30s)
#   EVAL_CROP=0            use full audio — no cropping
#   TRANSCRIPT_DIR=data/transcripts   (default) inject ASR transcripts into prompts
#   TRANSCRIPT_DIR=        disable transcript hints
#   DEBIAS_CYCLIC=1        (default) position-debias MCQ via cyclic option permutation
#   DEBIAS_CYCLIC=0        plain greedy decoding
#   OUTPUT_DIR=experiments/voxtral_ootb_train_balanced_debias  (default)
#   MODEL_ID=mistralai/Voxtral-Mini-3B-2507                    (default) base model
#
# Output:
#   ${OUTPUT_DIR}/train_balanced_hyp_{crop|full}[_debias].txt
#   ${OUTPUT_DIR}/train_balanced_hyp_{crop|full}[_debias]_detail.jsonl   (when DEBIAS_CYCLIC=1)
#   per-shard dirs hold predictions.jsonl with is_correct (accuracy is computed: gold present).

set -euo pipefail

NUM_SHARDS=4
MODEL_ID="${MODEL_ID:-mistralai/Voxtral-Mini-3B-2507}"
# Training/dev audio root: nested file paths are mlc-slm-2nd-dev/... which resolve here.
EVAL_AUDIO_ROOT="${EVAL_AUDIO_ROOT:-data/mlc26_task2}"
TRANSCRIPTION_HINT_FORMAT="${TRANSCRIPTION_HINT_FORMAT:-0}"
PROMPT_LANGUAGE="${PROMPT_LANGUAGE:-en}"
TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-data/transcripts}"
EVAL_CROP="${EVAL_CROP:-1}"
DEBIAS_CYCLIC="${DEBIAS_CYCLIC:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/voxtral_ootb_train_balanced_debias}"

# Nested balanced-training shards (built once from the balanced full file, round-robin by row).
TRAIN_SHARD_REL="data/mlc26_task2/train_balanced_eval_shards"

EVAL_MODE_TAG=$([ "${EVAL_CROP}" == "1" ] && echo "crop" || echo "full")
if [[ "${DEBIAS_CYCLIC}" == "1" ]]; then
  EVAL_MODE_TAG="${EVAL_MODE_TAG}_debias"
fi

EXPERIMENT_DIR="${OUTPUT_DIR%/}"

echo "=========================================="
echo "MareNostrum 5 - Voxtral OOTB eval on BALANCED FULL TRAIN data (4-GPU, NO adapter, WITH gold)"
echo "Base model    : ${MODEL_ID}  (out-of-the-box, no LoRA)"
echo "Output dir    : ${EXPERIMENT_DIR}"
echo "Audio root    : ${EVAL_AUDIO_ROOT}"
echo "Transcript dir: ${TRANSCRIPT_DIR:-<none>}"
echo "Shard source  : ${TRAIN_SHARD_REL} (nested, balanced 25/25/25/25)"
echo "Audio mode    : ${EVAL_MODE_TAG} (EVAL_CROP=${EVAL_CROP}, DEBIAS_CYCLIC=${DEBIAS_CYCLIC})"
echo "Accuracy      : COMPUTED (no --prediction-only; correct_answer gold present)"
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

REPO_HOST="${SANDBOX_DIR}/opt/ASR-merging"

# Verify shard JSONL files exist before spending GPU time
SHARD_DIR="${REPO_HOST}/${TRAIN_SHARD_REL}"
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_FILE="${SHARD_DIR}/shard_$(printf '%02d' $i)_of_$(printf '%02d' $NUM_SHARDS).jsonl"
  if [[ ! -f "$SHARD_FILE" ]]; then
    echo "ERROR: Shard file not found: $SHARD_FILE"
    exit 1
  fi
done

declare -a PIDS=()
declare -a SHARD_LOG_FILES=()
declare -a SHARD_OUTPUT_DIRS=()

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_TAG="shard_$(printf '%02d' $i)_of_$(printf '%02d' $NUM_SHARDS)"
  SHARD_JSONL="${TRAIN_SHARD_REL}/${SHARD_TAG}.jsonl"
  SHARD_OUTPUT_REL="${EXPERIMENT_DIR}/train_balanced_${EVAL_MODE_TAG}_${SHARD_TAG}"
  SHARD_OUTPUT_ABS="${REPO_HOST}/${SHARD_OUTPUT_REL}"
  SHARD_LOG="${SLURM_SUBMIT_DIR}/slurm_logs/voxtral_ootb_train_eval_shard${i}_${SLURM_JOB_ID}.log"

  mkdir -p "$SHARD_OUTPUT_ABS"
  SHARD_OUTPUT_DIRS+=("$SHARD_OUTPUT_ABS")
  SHARD_LOG_FILES+=("$SHARD_LOG")

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

  # NOTE: no --adapter-path → BASE model (OOTB). NO --prediction-only → gold is read,
  # is_correct is computed, accuracy is reported (debiased pred vs correct_answer).
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
echo '[shard ${i}] Starting OOTB train-balanced eval on GPU ${i}...'
python -m asr_merging.voxtral_forgetting_eval \
  --model-id ${MODEL_ID} \
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
  ${DEBIAS_FLAG}
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

echo ""
echo "=== Shard log tails (accuracy lines) ==="
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "--- shard ${i} ---"
  grep -iE "accuracy|acc=|correct|processed|Done|error" "${SHARD_LOG_FILES[$i]}" 2>/dev/null | tail -8 || true
done

if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/${NUM_SHARDS} shards failed. Aborting merge."
  exit 1
fi

# Merge shard hyp.txt files in order
FINAL_HYP="${REPO_HOST}/${EXPERIMENT_DIR}/train_balanced_hyp_${EVAL_MODE_TAG}.txt"
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
echo "Merged ${TOTAL} predictions → $FINAL_HYP"

# Merge per-shard debias detail (for offline raw-vs-debias-vs-shift accuracy analysis)
if [[ "${DEBIAS_CYCLIC}" == "1" ]]; then
  FINAL_DETAIL="${REPO_HOST}/${EXPERIMENT_DIR}/train_balanced_hyp_${EVAL_MODE_TAG}_detail.jsonl"
  DETAIL_FILES=()
  for i in $(seq 0 $((NUM_SHARDS - 1))); do
    SHARD_DETAIL="${SHARD_OUTPUT_DIRS[$i]}/hyp_debias_detail.jsonl"
    if [[ -f "$SHARD_DETAIL" ]]; then
      DETAIL_FILES+=("$SHARD_DETAIL")
    else
      echo "WARNING: Missing debias detail for shard ${i}: $SHARD_DETAIL"
    fi
  done
  if [[ ${#DETAIL_FILES[@]} -gt 0 ]]; then
    cat "${DETAIL_FILES[@]}" > "$FINAL_DETAIL"
    echo "Merged $(wc -l < "$FINAL_DETAIL") debias detail rows → $FINAL_DETAIL"
  fi
fi

echo ""
echo "=========================================="
echo "OOTB balanced-train eval finished at $(date)"
echo "Per-shard predictions.jsonl carry is_correct (debiased accuracy)."
echo "Use the detail JSONL to compute raw-vs-debias accuracy and shift sweeps offline."
echo "=========================================="
