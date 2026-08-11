#!/bin/bash
# Usage: sbatch --export=LORA_R=8,LORA_ALPHA=16 run_voxtral_mcq_lora_sweep.sh
#   or via the launcher: deploy/launch_voxtral_mcq_lora_sweep.sh
#SBATCH --job-name=voxtral_mcq_lora_sweep
#SBATCH --output=./slurm_logs/voxtral_mcq_lora_r%x_a%y_%j.out
#SBATCH --error=./slurm_logs/voxtral_mcq_lora_r%x_a%y_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

# LoRA hyperparameters — set via --export or environment
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "Voxtral MCQ LoRA sweep — r=${LORA_R} alpha=${LORA_ALPHA}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "QoS: $SLURM_JOB_QOS"
echo "Tasks per node: $SLURM_TASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
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
export KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-1}"

export CMD="
set -euo pipefail

debug_hold_on_failure() {
  exit_code=\$?
  if [[ \$exit_code -ne 0 && \"${KEEP_ALIVE_ON_FAILURE}\" == \"1\" ]]; then
    echo '[DEBUG] Training command failed (exit='\"\$exit_code\"').'
    echo '[DEBUG] Keeping allocation alive for node/container debugging.'
    echo '[DEBUG] Node: ${SLURM_JOB_NODELIST}'
    echo '[DEBUG] Attach example: ssh ${SLURM_JOB_NODELIST}'
    echo '[DEBUG] Then: singularity shell --nv --no-home --writable ${PATH_SINGULARITY}'
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

if [[ ! -f /opt/asrenv/bin/activate ]]; then
  echo '/opt/asrenv is missing.'
  exit 2
fi

. /opt/asrenv/bin/activate
python -V

python - <<'PY'
import importlib, os
from pathlib import Path
required = ['torch','transformers','datasets','peft','accelerate','sentencepiece',
            'tiktoken','tokenizers','jiwer','numpy','pandas','silero_vad']
missing = [m for m in required if not __import__('importlib').util.find_spec(m)]
if missing:
    raise SystemExit('Missing modules: ' + ', '.join(missing))
cache_root = Path(os.environ.get('HF_HUB_CACHE', '.'))
snapshots_root = cache_root / 'models--mistralai--Voxtral-Mini-3B-2507' / 'snapshots'
if not snapshots_root.exists():
    raise SystemExit('Missing model cache under ' + str(snapshots_root))
print('Offline preflight OK')
PY

MASTER_PORT=\${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}
echo Using torchrun master port: \${MASTER_PORT}
TORCHRUN_LOG_DIR=${TMP_ROOT}/torchrun_logs
mkdir -p \${TORCHRUN_LOG_DIR}

# Effective batch: 4 GPUs x 1 x 8 accum = 32
torchrun --standalone --nproc_per_node=4 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_MCQ \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --model-mode baseline \
  --train-jsonl data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_balanced.jsonl \
  --eval-jsonl data/mlc26_task2/organisers_balanced.jsonl \
  --test-jsonl data/mlc26_task2/organisers.jsonl \
  --audio-root data/mlc26_task2 \
  --do-train \
  --do-eval \
  --num-epochs 3 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 8 \
  --learning-rate 5e-5 \
  --no-use-bf16 \
  --use-fp16 \
  --lora-r ${LORA_R} \
  --lora-alpha ${LORA_ALPHA} \
  --output-root experiments \
  --experiment-name voxtral_mcq_balanced_r${LORA_R}_a${LORA_ALPHA} \
  --prompt-language en \
  --crop-from-question-refs \
  --crop-collar-seconds 30 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --eval-steps 25 \
  --logging-steps 3 \
  --early-stopping-patience 3 \
  2>&1

echo 'Voxtral MCQ LoRA sweep r=${LORA_R} alpha=${LORA_ALPHA} completed successfully'

trap - EXIT
"

echo "Starting Singularity container..."
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, scale=$(echo "scale=2; ${LORA_ALPHA}/${LORA_R}" | bc)"
echo ""

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
