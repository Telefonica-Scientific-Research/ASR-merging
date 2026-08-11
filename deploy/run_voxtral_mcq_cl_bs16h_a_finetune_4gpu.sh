#!/bin/bash
#SBATCH --job-name=voc_bs16h_nat
#SBATCH --output=./slurm_logs/voc_bs16h_nat_%j.out
#SBATCH --error=./slurm_logs/voc_bs16h_nat_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# CL Phase 2 (bs16 halved-LR) — Option A: EN×1 + native non-EN
#
# Identical to bs16 batch but lr halved: 5e-5 → 2.5e-5
# Base: voxtral_mcq_nllb_translated_4gpu_20260620_112000/final_model
# Hyperparams: lr=2.5e-5, r=16/a=32, grad_accum=4 (batch=16), no chat template
# Dataset: data/mlc26_task2/mlcslm_2nd_dev_qa_150files_cl_native.jsonl
#   - 263 records, 7890 questions, EN share 57%
#   - Steps (1 epoch, batch=16): ~493

set -euo pipefail

if [[ -z "${BASE_ADAPTER_PATH:-}" ]]; then
  BASE_ADAPTER_PATH="experiments/voxtral_mcq_nllb_translated_4gpu_20260620_112000/final_model"
fi

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "Voxtral MCQ — CL bs16-halfLR Phase 2: Option A"
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

mkdir -p ./slurm_logs "$CACHE_DIR" "$TORCH_EXT_DIR" "$TMP_ROOT"

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
    echo '[DEBUG] Training failed. Keeping allocation alive.'
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

# Effective batch size: 4 GPUs x 1 x grad_accum=4 = 16
# 7890q / 16 = ~493 steps for 1 epoch
torchrun --standalone --nproc_per_node=4 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_MCQ \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --model-mode baseline \
  --base-adapter-path ${BASE_ADAPTER_PATH} \
  --train-jsonl data/mlc26_task2/mlcslm_2nd_dev_qa_150files_cl_native.jsonl \
  --eval-jsonl  data/mlc26_task2/organisers_balanced.jsonl \
  --test-jsonl  data/mlc26_task2/organisers.jsonl \
  --audio-root data/mlc26_task2 \
  --do-train \
  --do-eval \
  --num-epochs 1 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 4 \
  --learning-rate 2.5e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --no-use-bf16 \
  --use-fp16 \
  --output-root experiments \
  --experiment-name voxtral_mcq_cl_bs16h_a_ft_4gpu \
  --timestamped-exp-dir \
  --prompt-language en \
  --crop-from-question-refs \
  --crop-collar-seconds 30 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --eval-steps 25 \
  --save-steps 25 \
  --logging-steps 3 \
  --early-stopping-patience 3 \
  2>&1

echo 'CL bs16-halfLR Phase 2 Option A completed successfully'
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
