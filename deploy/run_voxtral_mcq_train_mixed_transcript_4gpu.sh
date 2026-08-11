#!/bin/bash
#SBATCH --job-name=voxtral_mixed_trx_4gpu
#SBATCH --output=./slurm_logs/voxtral_mixed_trx_4gpu_%j.out
#SBATCH --error=./slurm_logs/voxtral_mixed_trx_4gpu_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# R6-base + mixed dataset (original + NLLB) + transcript augmentation
# ────────────────────────────────────────────────────────────────────
# Same R6 hyperparams (r=16, α=32, lr=5e-5, 3 epochs, early-stop=3) as
# voxtral_mcq_nllb_transcript but trained on the mixed JSONL which contains
# both the NLLB-translated English questions AND the original cross-lingual
# audio-question pairs:
#
#   mlcslm_2nd_dev_qa_successed_opensource_mixed.jsonl
#     263 entries, 7890 questions, 149 unique audio files (~1.76× per audio)
#     A:35%  B:40%  C:21%  D:2%  (same distribution as nllb-only)
#
# All 149 audio files have transcripts in data/transcripts/ (100% coverage).
# Transcripts are injected into every MCQ prompt during training and eval.

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "ASR-merging Voxtral MCQ Train 4-GPU — R6-base + mixed dataset + transcript"
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

# Container and storage paths
export PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SCRATCH_DIR="/gpfs/scratch/ehpc628/jls/ehpc628XXX"

# Repo path inside the sandbox
export REPO_DIR="/opt/ASR-merging"

# Caches and temp roots
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

# H100 network/GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-0}"

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
  echo '/opt/asrenv is missing. Build it first:'
  exit 2
fi

. /opt/asrenv/bin/activate
python -V

# Preflight: verify model cache and transcripts
python - <<'PY'
import importlib, os
from pathlib import Path

required = ['torch','transformers','datasets','peft','accelerate',
            'sentencepiece','tiktoken','tokenizers','jiwer','numpy',
            'pandas','silero_vad']
missing = []
for mod in required:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
if missing:
    raise SystemExit('Missing Python modules: ' + ', '.join(missing))

cache_root = Path(os.environ.get('HF_HUB_CACHE', '.'))
snapshots_root = cache_root / 'models--mistralai--Voxtral-Mini-3B-2507' / 'snapshots'
if not snapshots_root.exists():
    raise SystemExit('Missing HF cache for Voxtral-Mini-3B-2507 under ' + str(snapshots_root))
snapshots = [p for p in snapshots_root.iterdir() if p.is_dir()]
if not snapshots:
    raise SystemExit('No model snapshots found under ' + str(snapshots_root))
latest = sorted(snapshots)[-1]
for name in ['config.json', 'preprocessor_config.json']:
    if not (latest / name).exists():
        raise SystemExit('Incomplete snapshot: missing ' + name)

transcript_dir = Path('data/transcripts')
n_tx = len(list(transcript_dir.glob('*.txt'))) if transcript_dir.is_dir() else 0
if n_tx == 0:
    raise SystemExit('No transcript .txt files found in data/transcripts/')
print(f'Preflight OK — model cache and {n_tx} transcripts found')
PY

MASTER_PORT=\${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}
echo Using torchrun master port: \${MASTER_PORT}
TORCHRUN_LOG_DIR=${TMP_ROOT}/torchrun_logs
mkdir -p \${TORCHRUN_LOG_DIR}

# Effective batch size: 4 GPUs x 1 per-GPU batch x 4 grad_accum = 16
# R6 hyperparams: r=16, α=32, lr=5e-5, 3 epochs
# min_generalization_gap strategy + early-stopping-patience 0 → save_total_limit=None (keep all LoRA ckpts)
torchrun --standalone --nproc_per_node=4 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_MCQ \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --model-mode baseline \
  --train-jsonl data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_mixed.jsonl \
  --eval-jsonl data/mlc26_task2/organisers_balanced_mixed.jsonl \
  --test-jsonl data/mlc26_task2/organisers_mixed.jsonl \
  --audio-root data/mlc26_task2 \
  --do-train \
  --do-eval \
  --num-epochs 2 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 4 \
  --learning-rate 5e-5 \
  --no-use-bf16 \
  --use-fp16 \
  --output-root experiments \
  --experiment-name voxtral_mcq_mixed_transcript_4gpu \
  --timestamped-exp-dir \
  --prompt-language en \
  --crop-from-question-refs \
  --crop-collar-seconds 30 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --eval-steps 25 \
  --save-steps 25 \
  --logging-steps 3 \
  --early-stopping-patience 0 \
  --best-ckpt-strategy min_generalization_gap \
  --transcript-dir data/transcripts \
  2>&1

echo 'Voxtral MCQ mixed + transcript 4-GPU training completed successfully'

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
