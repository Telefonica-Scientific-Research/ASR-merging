#!/bin/bash
#SBATCH --job-name=voxtral_eval26
#SBATCH --output=./slurm_logs/voxtral_eval26_%j.out
#SBATCH --error=./slurm_logs/voxtral_eval26_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Evaluate 4 models in parallel (1 per GPU) on the clean eval_26files_challenge_repr.jsonl
# held-out split (37 questions, 0 overlap with training data).
#
# Models evaluated:
#   GPU0: R6_base    (voxtral_mcq_nllb_translated_4gpu_20260620_112000)
#   GPU1: cl_bs16_a  (voxtral_mcq_cl_bs16_a_ft_4gpu_20260621_123553)
#   GPU2: cl_fms_shuf (voxtral_mcq_cl_fms_shuf_ft_4gpu_20260621_113708)
#   GPU3: cl_nllb_safe (voxtral_mcq_cl_nllb_safe_ft_4gpu_20260621_112919)
#
# Output: experiments/{exp}/eval26/mcq_eval_{metrics,predictions}.json[l]

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 - Voxtral MCQ eval26 (4 GPU)"
echo "Eval set: eval_26files_challenge_repr.jsonl (37 Q, clean held-out)"
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
export SCRATCH_DIR="/gpfs/scratch/ehpc628/jls/ehpc628XXX"
export REPO_DIR="/opt/ASR-merging"

RUN_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}/tmp"

mkdir -p ./slurm_logs
mkdir -p "$TMP_ROOT"

# Common eval args shared by all 4 eval processes
COMMON_ARGS="
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --model-mode adapter \
  --eval-jsonl data/mlc26_task2/organisers_balanced.jsonl \
  --audio-root data/mlc26_task2 \
  --eval-batch-size 1 \
  --no-use-bf16 \
  --use-fp16 \
  --prompt-language en \
  --crop-from-question-refs \
  --crop-collar-seconds 30 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --do-eval
"

# R6 base adapter (used by cl_bs16 models as base-adapter-path)
R6_BASE="experiments/voxtral_mcq_nllb_translated_4gpu_20260620_112000/final_model"
# corr-batch base adapter (used by cl_fms_shuf and cl_nllb_safe)
CORR_BASE="experiments/voxtral_mcq_nllb150_r32a64_duallr_chat_tmpl_4gpu_20260621_091853/final_model"

declare -a LABELS=(
  "R6_base"
  "cl_bs16_a"
  "cl_fms_shuf"
  "cl_nllb_safe"
)
declare -a ADAPTER_ARGS=(
  "--adapter-path ${R6_BASE}"
  "--base-adapter-path ${R6_BASE} --adapter-path experiments/voxtral_mcq_cl_bs16_a_ft_4gpu_20260621_123553/final_model"
  "--base-adapter-path ${CORR_BASE} --adapter-path experiments/voxtral_mcq_cl_fms_shuf_ft_4gpu_20260621_113708/final_model"
  "--base-adapter-path ${CORR_BASE} --adapter-path experiments/voxtral_mcq_cl_nllb_safe_ft_4gpu_20260621_112919/final_model"
)
declare -a OUTPUT_DIRS=(
  "experiments/voxtral_mcq_nllb_translated_4gpu_20260620_112000/eval26"
  "experiments/voxtral_mcq_cl_bs16_a_ft_4gpu_20260621_123553/eval26"
  "experiments/voxtral_mcq_cl_fms_shuf_ft_4gpu_20260621_113708/eval26"
  "experiments/voxtral_mcq_cl_nllb_safe_ft_4gpu_20260621_112919/eval26"
)

declare -a PIDS=()
declare -a LOG_FILES=()

for i in 0 1 2 3; do
  LOG="${SLURM_SUBMIT_DIR}/slurm_logs/voxtral_eval26_gpu${i}_${SLURM_JOB_ID:-manual}.log"
  LOG_FILES+=("$LOG")

  EVAL_CMD="
set -euo pipefail
export HF_HOME=${CACHE_DIR}
export HF_HUB_CACHE=${CACHE_DIR}
export TRANSFORMERS_CACHE=${CACHE_DIR}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=${TORCH_EXT_DIR}
export PYTHONUNBUFFERED=1
export TMPDIR=${TMP_ROOT}/gpu${i}
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=${i}
mkdir -p \${TMPDIR}
cd ${REPO_DIR}
. /opt/asrenv/bin/activate
echo '[GPU${i}] Starting eval: ${LABELS[$i]}'
python -m asr_merging.voxtral_train_MCQ \
  ${COMMON_ARGS} \
  ${ADAPTER_ARGS[$i]} \
  --output-dir ${OUTPUT_DIRS[$i]}
echo '[GPU${i}] Done: ${LABELS[$i]}'
"

  singularity exec \
    --nv \
    --writable \
    --bind "$CACHE_DIR:$CACHE_DIR" \
    --bind "$TORCH_EXT_DIR:$TORCH_EXT_DIR" \
    --bind "$SANDBOX_DIR:$SANDBOX_DIR" \
    --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
    --bind /gpfs:/gpfs \
    --pwd "$SANDBOX_DIR" \
    "$SANDBOX_DIR" \
    bash -c "$EVAL_CMD" > "$LOG" 2>&1 &

  PIDS+=($!)
  echo "Launched GPU${i}: ${LABELS[$i]} (pid ${PIDS[$i]}), log: $LOG"
done

echo ""
echo "Waiting for all 4 eval processes to finish..."
FAILED=0
for i in 0 1 2 3; do
  if wait "${PIDS[$i]}"; then
    echo "GPU${i} (${LABELS[$i]}): OK"
  else
    echo "GPU${i} (${LABELS[$i]}): FAILED"
    echo "--- last 30 lines of ${LOG_FILES[$i]} ---"
    grep -c "" "${LOG_FILES[$i]}" 2>/dev/null && \
      awk 'END{s=NR-29; if(s<1)s=1; for(i=s;i<=NR;i++) print lines[i]} {lines[NR]=$0}' "${LOG_FILES[$i]}" 2>/dev/null || true
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "EVAL26 RESULTS SUMMARY"
echo "Eval set: eval_26files_challenge_repr.jsonl (37 Q, clean)"
echo "============================================================"

REPO_HOST="${SANDBOX_DIR}/opt/ASR-merging"
for i in 0 1 2 3; do
  METRICS="${REPO_HOST}/${OUTPUT_DIRS[$i]}/mcq_eval_metrics.json"
  if [[ -f "$METRICS" ]]; then
    ACC=$(python3 -c "import json; d=json.load(open('${METRICS}')); print(f\"{d['accuracy']*100:.1f}% ({d['n_correct']}/{d['n_total']})\")" 2>/dev/null || echo "parse error")
    echo "  ${LABELS[$i]}: ${ACC}"
  else
    echo "  ${LABELS[$i]}: NO METRICS FILE (${METRICS})"
  fi
done
echo "============================================================"

if [[ $FAILED -gt 0 ]]; then
  echo "ERROR: ${FAILED}/4 evals failed."
  exit 1
fi

echo ""
echo "All 4 evals completed. Results in:"
for i in 0 1 2 3; do
  echo "  ${REPO_HOST}/${OUTPUT_DIRS[$i]}/"
done
echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
