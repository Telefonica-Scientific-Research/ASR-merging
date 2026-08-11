#!/bin/bash
#SBATCH --job-name=voxtral_cl_nllb_safe
#SBATCH --output=./slurm_logs/voxtral_cl_nllb_safe_%j.out
#SBATCH --error=./slurm_logs/voxtral_cl_nllb_safe_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Curriculum Learning Phase 2 — Option C: NLLB safe (EN×2 + NLLB×8).
#
# Same cross-lingual signal as Option B (NLLB upsample) but with doubled EN
# to reduce catastrophic forgetting.  EN share rises from 11% → 20%.
#
# Experiment matrix for CL Phase 2:
#   A  native         EN×1 + native non-EN                  same-lang only
#   B  nllb_up        EN×1 + NLLB×8                         cross-lang only, 11% EN
#   C  nllb_safe      EN×2 + NLLB×8                         cross-lang only, 20% EN  ← this script
#   D  full_mix       EN×1 + native non-EN + NLLB×8         both,            10% EN
#   E  full_mix_safe  EN×2 + native non-EN + NLLB×8         both,            19% EN
#
# Dataset: data/mlc26_task2/mlcslm_2nd_dev_qa_150files_cl_nllb_safe.jsonl
#   - 1,500 records  (EN×2=300 + NLLB×8=1200)
#   - 45,000 questions
#   - EN share: 20%
#   - Steps (1 epoch, batch=32): ~1,406
#
# Usage:
#   BASE_ADAPTER_PATH=experiments/voxtral_mcq_nllb150_r32a64_duallr_chat_tmpl_4gpu_<ts>/final_model \
#   sbatch deploy/run_voxtral_mcq_cl_nllb_safe_finetune_4gpu.sh

set -euo pipefail

if [[ -z "${BASE_ADAPTER_PATH:-}" ]]; then
  echo "ERROR: BASE_ADAPTER_PATH is not set."
  echo "Usage: BASE_ADAPTER_PATH=experiments/.../final_model sbatch $0"
  exit 1
fi

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "Voxtral MCQ — CL Phase 2: NLLB safe (EN×2 + NLLB×8)"
echo "=========================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node(s):          $SLURM_JOB_NODELIST"
echo "Base adapter:     $BASE_ADAPTER_PATH"
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
RUN_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}"
export TMP_ROOT="$RUN_ROOT/tmp"

mkdir -p ./slurm_logs
mkdir -p "$CACHE_DIR"
mkdir -p "$TORCH_EXT_DIR"
mkdir -p "$TMP_ROOT"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=ib0
export KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-0}"

export CMD="
set -euo pipefail

debug_hold_on_failure() {
  exit_code=\$?
  if [[ \$exit_code -ne 0 && \"${KEEP_ALIVE_ON_FAILURE}\" == \"1\" ]]; then
    echo '[DEBUG] Training failed (exit='\"\$exit_code\"'). Keeping allocation alive.'
    while true; do sleep 60; done
  fi
  exit \$exit_code
}
trap debug_hold_on_failure EXIT

export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=$TORCH_EXT_DIR
export PYTHONUNBUFFERED=1
export TMPDIR=$TMP_ROOT
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

nvidia-smi
cd $REPO_DIR
. /opt/asrenv/bin/activate
python -V

MASTER_PORT=\${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}
echo Using torchrun master port: \${MASTER_PORT}
TORCHRUN_LOG_DIR=${TMP_ROOT}/torchrun_logs
mkdir -p \${TORCHRUN_LOG_DIR}

# Effective batch size: 4 GPUs x 1 x grad_accum=8 = 32
# 45,000q / 32 = ~1,406 steps for 1 epoch (early stopping with patience=5 expected)
torchrun --standalone --nproc_per_node=4 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_MCQ \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --model-mode baseline \
  --base-adapter-path ${BASE_ADAPTER_PATH} \
  --train-jsonl data/mlc26_task2/mlcslm_2nd_dev_qa_150files_cl_nllb_safe.jsonl \
  --eval-jsonl  data/mlc26_task2/organisers_balanced.jsonl \
  --test-jsonl  data/mlc26_task2/organisers.jsonl \
  --audio-root data/mlc26_task2 \
  --do-train \
  --do-eval \
  --num-epochs 1 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 8 \
  --lora-r 32 \
  --lora-alpha 64 \
  --encoder-connector-learning-rate 5e-6 \
  --llm-learning-rate 2e-5 \
  --use-chat-template-for-training \
  --no-use-bf16 \
  --use-fp16 \
  --output-root experiments \
  --experiment-name voxtral_mcq_cl_nllb_safe_ft_4gpu \
  --timestamped-exp-dir \
  --prompt-language en \
  --crop-from-question-refs \
  --crop-collar-seconds 30 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --eval-steps 25 \
  --save-steps 25 \
  --logging-steps 3 \
  --early-stopping-patience 5 \
  2>&1

echo 'CL Phase 2 NLLB safe fine-tuning completed successfully'
trap - EXIT
"

echo "Starting Singularity container..."
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
    bash -c "$CMD"

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
