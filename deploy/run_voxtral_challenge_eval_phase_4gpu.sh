#!/bin/bash
#SBATCH --job-name=voxtral_challenge_phase
#SBATCH --output=./slurm_logs/voxtral_challenge_phase_%j.out
#SBATCH --error=./slurm_logs/voxtral_challenge_phase_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00

# Evaluate a trained adapter (or OOTB base model) on the challenge Phase 1 or Phase 2 dataset,
# using either the original questions or the NLLB-translated English questions.
#
# Usage:
#   MODEL_ADAPTER_PATH=experiments/.../final_model \
#   PHASE=2 \
#   USE_NLLB=0 \
#   DEBIAS_CYCLIC=1 \
#   TRANSCRIPT_DIR=data/transcripts \
#   sbatch deploy/run_voxtral_challenge_eval_phase_4gpu.sh
#
# Variables:
#   MODEL_ADAPTER_PATH  path to adapter inside container (relative to /opt/ASR-merging)
#                       Leave empty (OOTB mode) to run the base model without any adapter.
#   PHASE               1 or 2  (default: 1)
#   USE_NLLB            1 = NLLB-translated English questions
#                       0 = original questions (may be non-English)  (default: 1)
#   DEBIAS_CYCLIC       1 = position-debias MCQ via cyclic option permutation  (default: 0)
#   CALIBRATE_PRIOR     1 = prior-calibrated prediction (Zhao et al. 2021): subtract
#                           log P(label|null) from log P(label|real) for each question.
#                           Null prompt keeps question+audio but empties all option texts.
#                           Works for N=2,3,4 options; costs exactly 2 fwd passes/question.  (default: 0)
#   TRANSCRIPT_DIR      path to ASR transcript .txt files (default: empty = no transcripts)
#   EVAL_AUDIO_ROOT     audio root inside container (default: data/mlc26_task2/mlc-slm-2nd-eval)
#   PROMPT_LANGUAGE     language passed to --prompt-language (default: en)
#
# Output:
#   experiments/<EXPERIMENT>/challenge_phase<PHASE>[_orig|_nllb][_debias]_hyp.txt

set -euo pipefail

# MODEL_ADAPTER_PATH is optional — empty = OOTB (base model, no adapter).
MODEL_ADAPTER_PATH="${MODEL_ADAPTER_PATH:-}"
MODEL_ID="${MODEL_ID:-mistralai/Voxtral-Mini-3B-2507}"

PHASE="${PHASE:-1}"
USE_NLLB="${USE_NLLB:-1}"
DEBIAS_CYCLIC="${DEBIAS_CYCLIC:-0}"
CALIBRATE_PRIOR="${CALIBRATE_PRIOR:-0}"
EVAL_CROP="${EVAL_CROP:-1}"
CROP_COLLAR_SECONDS="${CROP_COLLAR_SECONDS:-30}"
TRANSCRIPT_DIR="${TRANSCRIPT_DIR:-}"
NUM_SHARDS=4
EVAL_AUDIO_ROOT="${EVAL_AUDIO_ROOT:-data/mlc26_task2/mlc-slm-2nd-eval}"
PROMPT_LANGUAGE="${PROMPT_LANGUAGE:-en}"
TRANSCRIPTION_HINT_FORMAT="${TRANSCRIPTION_HINT_FORMAT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
FEW_SHOT_COUNT="${FEW_SHOT_COUNT:-0}"
ICL_MULTIMODAL="${ICL_MULTIMODAL:-0}"
ICL_MULTIMODAL_NONEN="${ICL_MULTIMODAL_NONEN:-0}"

if [[ "${PHASE}" != "1" && "${PHASE}" != "2" ]]; then
  echo "ERROR: PHASE must be 1 or 2, got: ${PHASE}"
  exit 1
fi

# Select input JSONL
if [[ "${USE_NLLB}" == "1" ]]; then
  JSONL_BASENAME="task2_phase${PHASE}_questions_options_nllb_en.jsonl"
  PHASE_TAG="phase${PHASE}_nllb"
else
  JSONL_BASENAME="task2_phase${PHASE}_questions_options.jsonl"
  PHASE_TAG="phase${PHASE}_orig"
fi
if [[ "${EVAL_CROP}" != "1" ]]; then
  PHASE_TAG="${PHASE_TAG}_full"
elif [[ "${CROP_COLLAR_SECONDS}" != "30" ]]; then
  PHASE_TAG="${PHASE_TAG}_c${CROP_COLLAR_SECONDS}"
fi
if [[ "${DEBIAS_CYCLIC}" == "1" ]]; then
  PHASE_TAG="${PHASE_TAG}_debias"
fi
if [[ "${CALIBRATE_PRIOR}" == "1" ]]; then
  PHASE_TAG="${PHASE_TAG}_calibrated"
fi
if [[ -z "${TRANSCRIPT_DIR:-}" ]]; then
  PHASE_TAG="${PHASE_TAG}_notx"
fi
if [[ "${MAX_NEW_TOKENS}" != "16" ]]; then
  PHASE_TAG="${PHASE_TAG}_mnt${MAX_NEW_TOKENS}"
fi
if [[ "${FEW_SHOT_COUNT}" != "0" ]]; then
  PHASE_TAG="${PHASE_TAG}_icl${FEW_SHOT_COUNT}"
fi
if [[ "${ICL_MULTIMODAL}" == "1" ]]; then
  PHASE_TAG="${PHASE_TAG}_mm"
fi
if [[ "${ICL_MULTIMODAL_NONEN}" == "1" ]]; then
  PHASE_TAG="${PHASE_TAG}_nonen"
fi

# Derive experiment dir: from adapter path, or fixed OOTB dir if no adapter.
if [[ -n "${MODEL_ADAPTER_PATH}" ]]; then
  EXPERIMENT_DIR="${MODEL_ADAPTER_PATH%/final_model}"
  EXPERIMENT_DIR="${EXPERIMENT_DIR%/}"
else
  # Encode model name in dir when non-default (e.g. 24B)
  _MODEL_SLUG=$(echo "${MODEL_ID}" | sed 's|mistralai/||;s|/|-|g;s|[^A-Za-z0-9_-]||g' | tr '[:upper:]' '[:lower:]')
  if [[ "${MODEL_ID}" == "mistralai/Voxtral-Mini-3B-2507" ]]; then
    EXPERIMENT_DIR="experiments/voxtral_ootb_phase${PHASE}_eval"
  else
    EXPERIMENT_DIR="experiments/voxtral_ootb_${_MODEL_SLUG}_phase${PHASE}_eval"
  fi
fi

echo "=========================================="
echo "MareNostrum 5 - Voxtral Challenge Eval (4-GPU)"
echo "Phase: ${PHASE}  NLLB: ${USE_NLLB}  crop: ${EVAL_CROP}  collar: ${CROP_COLLAR_SECONDS}s  debias: ${DEBIAS_CYCLIC}  calibrate: ${CALIBRATE_PRIOR}  tag: ${PHASE_TAG}"
echo "Input JSONL: data/mlc26_task2/${JSONL_BASENAME}"
echo "Model/adapter: ${MODEL_ADAPTER_PATH:-<OOTB base model, no adapter>}"
echo "Experiment dir: ${EXPERIMENT_DIR}"
echo "Transcript dir : ${TRANSCRIPT_DIR:-<none>}"
echo "Prompt language: ${PROMPT_LANGUAGE}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
date
echo "=========================================="

module purge
module load singularity

export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export CACHE_DIR="/gpfs/scratch/ehpc628/models"
export TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"
export MLC_DATA_DIR="/gpfs/scratch/ehpc494/jls/datasets"
export SCRATCH_DIR="/gpfs/scratch/ehpc628/jls/ehpc628XXX"
export REPO_DIR="/opt/ASR-merging"

RUN_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}/tmp"

mkdir -p ./slurm_logs
mkdir -p "$TMP_ROOT"

REPO_HOST="${SANDBOX_DIR}/opt/ASR-merging"

# ---- Verify input JSONL exists ----
INPUT_JSONL="${REPO_HOST}/data/mlc26_task2/${JSONL_BASENAME}"
if [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "ERROR: Input JSONL not found: ${INPUT_JSONL}"
  exit 1
fi
TOTAL_RECORDS=$(wc -l < "${INPUT_JSONL}")
echo "Input: ${INPUT_JSONL} (${TOTAL_RECORDS} records)"

# ---- Dynamically shard the selected JSONL into NUM_SHARDS parts ----
SHARD_DIR="${REPO_HOST}/data/mlc26_task2/challenge_${PHASE_TAG}_shards_${SLURM_JOB_ID:-manual}"
mkdir -p "${SHARD_DIR}"

echo "Sharding ${TOTAL_RECORDS} records into ${NUM_SHARDS} shards → ${SHARD_DIR}"
python3 - <<PYEOF
import json, math, os

input_path = "${INPUT_JSONL}"
shard_dir  = "${SHARD_DIR}"
n_shards   = ${NUM_SHARDS}
n_shards_fmt = f"{n_shards:02d}"

records = [line for line in open(input_path) if line.strip()]
total   = len(records)
base    = total // n_shards
rem     = total % n_shards

idx = 0
for s in range(n_shards):
    size   = base + (1 if s < rem else 0)
    chunk  = records[idx : idx + size]
    idx   += size
    fname  = os.path.join(shard_dir, f"shard_{s:02d}_of_{n_shards_fmt}.jsonl")
    with open(fname, "w") as f:
        f.writelines(chunk)
    print(f"  shard {s:02d}: {len(chunk)} records → {fname}")

print(f"Sharding complete: {total} records → {n_shards} shards")
PYEOF

echo "Sharding done."
echo ""

# ---- Launch one singularity process per GPU ----
declare -a PIDS=()
declare -a SHARD_LOG_FILES=()
declare -a SHARD_OUTPUT_DIRS=()

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_TAG_FILE="shard_$(printf '%02d' $i)_of_$(printf '%02d' $NUM_SHARDS)"
  SHARD_JSONL_ABS="${SHARD_DIR}/${SHARD_TAG_FILE}.jsonl"
  # Path as seen inside container (SHARD_DIR is under /gpfs, bound via --bind /gpfs:/gpfs)
  SHARD_JSONL_CONT="${SHARD_JSONL_ABS}"

  SHARD_OUTPUT_REL="${EXPERIMENT_DIR}/challenge_${PHASE_TAG}_eval_${SHARD_TAG_FILE}"
  SHARD_OUTPUT_ABS="${REPO_HOST}/${SHARD_OUTPUT_REL}"
  SHARD_LOG="${SLURM_SUBMIT_DIR}/slurm_logs/voxtral_challenge_phase_shard${i}_${SLURM_JOB_ID}.log"

  mkdir -p "${SHARD_OUTPUT_ABS}"
  SHARD_OUTPUT_DIRS+=("${SHARD_OUTPUT_ABS}")
  SHARD_LOG_FILES+=("${SHARD_LOG}")

  HINT_FLAG=""
  if [[ "${TRANSCRIPTION_HINT_FORMAT}" == "1" ]]; then
    HINT_FLAG="--use-transcription-hint-format"
  fi

  ADAPTER_FLAG=""
  if [[ -n "${MODEL_ADAPTER_PATH}" ]]; then
    ADAPTER_FLAG="--adapter-path ${MODEL_ADAPTER_PATH}"
  fi

  TRANSCRIPT_FLAG=""
  if [[ -n "${TRANSCRIPT_DIR:-}" ]]; then
    TRANSCRIPT_FLAG="--transcript-dir ${TRANSCRIPT_DIR}"
  fi

  CROP_FLAG=""
  if [[ "${EVAL_CROP}" == "1" ]]; then
    CROP_FLAG="--eval-crop-from-question-refs --eval-crop-collar-seconds ${CROP_COLLAR_SECONDS} --eval-random-crop-seconds 0"
  fi

  DEBIAS_FLAG=""
  if [[ "${DEBIAS_CYCLIC}" == "1" ]]; then
    DEBIAS_FLAG="--debias-cyclic-permutation"
  fi

  CALIBRATE_FLAG=""
  if [[ "${CALIBRATE_PRIOR}" == "1" ]]; then
    CALIBRATE_FLAG="--calibrate-prior"
  fi

  ICL_MM_FLAG=""
  if [[ "${ICL_MULTIMODAL}" == "1" ]]; then
    ICL_MM_FLAG="--icl-multimodal"
  fi
  if [[ "${ICL_MULTIMODAL_NONEN}" == "1" ]]; then
    ICL_MM_FLAG="${ICL_MM_FLAG} --icl-multimodal-nonen"
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
echo '[shard ${i}] Starting eval on GPU ${i} ...'
python -m asr_merging.voxtral_forgetting_eval \
  --model-id ${MODEL_ID} \
  ${ADAPTER_FLAG} \
  --tasks jsonl_audio_mc \
  --jsonl-path ${SHARD_JSONL_CONT} \
  --audio-root ${EVAL_AUDIO_ROOT} \
  --jsonl-max-questions-per-audio 0 \
  --max-samples-per-task 0 \
  --split test \
  --no-use-bf16 \
  --use-fp16 \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --few-shot-count ${FEW_SHOT_COUNT} \
  --output-dir ${SHARD_OUTPUT_REL} \
  --prompt-language ${PROMPT_LANGUAGE} \
  ${HINT_FLAG} \
  ${TRANSCRIPT_FLAG} \
  ${CROP_FLAG} \
  ${DEBIAS_FLAG} \
  ${CALIBRATE_FLAG} \
  ${ICL_MM_FLAG} \
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
    $SANDBOX_DIR \
    bash -c "$SHARD_CMD" > "$SHARD_LOG" 2>&1 &

  PIDS+=($!)
  echo "Launched shard ${i} / GPU${i} (pid ${PIDS[$i]}), log: $SHARD_LOG"
done

# ---- Wait for all shards ----
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
echo "=== Shard log tails ==="
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  echo "--- shard ${i} ---"
  grep -E "processed|Done|error|Error" "${SHARD_LOG_FILES[$i]}" 2>/dev/null | tail -5 || true
done

if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/${NUM_SHARDS} shards failed. Aborting merge."
  exit 1
fi

# ---- Merge shard hyp.txt → single submission file ----
FINAL_HYP="${REPO_HOST}/${EXPERIMENT_DIR}/challenge_${PHASE_TAG}_hyp.txt"
mkdir -p "${REPO_HOST}/${EXPERIMENT_DIR}"
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

if [[ "$TOTAL" -ne "$UNIQUE" ]]; then
  echo "WARNING: Only ${UNIQUE} unique lines out of ${TOTAL} — possible duplicate question_ids."
else
  echo "Sanity OK: all ${TOTAL} predictions are unique."
fi

# ---- Clean up temp shard files ----
echo ""
echo "Cleaning up temp shards: ${SHARD_DIR}"
rm -rf "${SHARD_DIR}"

echo ""
echo "Submit this file to the challenge scorer:"
echo "  ${FINAL_HYP}"
echo ""
echo "=========================================="
echo "Challenge phase eval finished at $(date)"
echo "=========================================="
