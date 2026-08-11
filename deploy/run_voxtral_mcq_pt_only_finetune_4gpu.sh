#!/bin/bash
#SBATCH --job-name=voc_pt_only
#SBATCH --output=./slurm_logs/voc_pt_only_%j.out
#SBATCH --error=./slurm_logs/voc_pt_only_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Per-language LoRA from OOTB Voxtral — PT (Portuguese, Portuguese_Brazil)
#
# Base: mistralai/Voxtral-Mini-3B-2507 (out-of-the-box, no pre-trained adapter)
# Hyperparams: lr=5e-5, r=16/a=32, grad_accum=4 (batch=16), no chat template
# Train: train_124files_pt_only.jsonl — 15 records, 450 questions
# Eval:  eval_26files_pt_only.jsonl  — 2 records, 60 questions (zero overlap)
# Steps: 450q / batch16 = ~28 steps/epoch; 10 epochs max
set -euo pipefail
echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "Voxtral MCQ — PT-only fine-tuning (OOTB base)"
echo "=========================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Node(s):          $SLURM_JOB_NODELIST"
echo "Base model:       mistralai/Voxtral-Mini-3B-2507 (OOTB, no adapter)"
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
torchrun --standalone --nproc_per_node=4 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_MCQ \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --model-mode baseline \
  --train-jsonl data/mlc26_task2/train_124files_pt_only.jsonl \
  --eval-jsonl  data/mlc26_task2/eval_26files_pt_only.jsonl \
  --test-jsonl  data/mlc26_task2/eval_26files_pt_only.jsonl \
  --audio-root data/mlc26_task2 \
  --do-train \
  --do-eval \
  --num-epochs 10 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 4 \
  --learning-rate 5e-5 \
  --lora-r 16 \
  --lora-alpha 32 \
  --no-use-bf16 \
  --use-fp16 \
  --output-root experiments \
  --experiment-name voxtral_mcq_pt_only_ft_4gpu \
  --timestamped-exp-dir \
  --prompt-language en \
  --crop-from-question-refs \
  --crop-collar-seconds 30 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --eval-steps 3 \
  --save-steps 3 \
  --logging-steps 1 \
  --early-stopping-patience 5 \
  --best-ckpt-strategy min_generalization_gap \
  2>&1
echo 'PT-only fine-tuning completed successfully'
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
