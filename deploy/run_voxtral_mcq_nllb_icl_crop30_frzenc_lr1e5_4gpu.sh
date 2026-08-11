#!/bin/bash
#SBATCH --job-name=voxtral_icl_fz_lr1e5_4gpu
#SBATCH --output=./slurm_logs/voxtral_icl_fz_lr1e5_4gpu_%j.out
#SBATCH --error=./slurm_logs/voxtral_icl_fz_lr1e5_4gpu_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# R6-base + ICL rotation, no transcripts, frozen encoder+connector, lr=1e-5
# ────────────────────────────────────────────────────────────────
# Same as run_voxtral_mcq_nllb_icl_crop30_4gpu.sh but with
# --freeze-encoder-connector: the audio encoder and multi-modal projector
# are frozen; only the LLM LoRA (r=16, α=32) is updated.
# Same lr=5e-5 as the full-model variant.

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "ASR-merging Voxtral MCQ Train 4-GPU — R6 + ICL rotation, frozen enc+conn (crop 30s)"
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
# MLC audio data — JSONL paths are relative to this root inside the container
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
  echo '  uv venv /opt/asrenv --python /usr/bin/python3.12'
  echo '  . /opt/asrenv/bin/activate'
  echo '  uv pip install -r requirements-hpc.txt'
  exit 2
fi

. /opt/asrenv/bin/activate
python -V

python - <<'PY'
import importlib
import os
from pathlib import Path

required = [
  'torch',
  'transformers',
  'datasets',
  'peft',
  'accelerate',
  'sentencepiece',
  'tiktoken',
  'tokenizers',
  'jiwer',
  'numpy',
  'pandas',
  'silero_vad',
]
missing = []
for mod in required:
  try:
    importlib.import_module(mod)
  except Exception:
    missing.append(mod)

if missing:
  raise SystemExit(
    'Missing Python modules in /opt/asrenv (offline node cannot install): '
    + ', '.join(missing)
  )

cache_root = Path(os.environ.get('HF_HUB_CACHE', '.'))
repo_cache = cache_root / 'models--mistralai--Voxtral-Mini-3B-2507'
snapshots_root = repo_cache / 'snapshots'
if not snapshots_root.exists():
  raise SystemExit(
    'Missing local HF cache for mistralai/Voxtral-Mini-3B-2507 under '
    + str(snapshots_root)
  )

snapshots = [p for p in snapshots_root.iterdir() if p.is_dir()]
if not snapshots:
  raise SystemExit('No model snapshots found under ' + str(snapshots_root))

latest = sorted(snapshots)[-1]
required_files = ['config.json', 'preprocessor_config.json']
missing_files = [name for name in required_files if not (latest / name).exists()]
if missing_files:
  raise SystemExit(
    'Cached snapshot is incomplete at ' + str(latest) + '. Missing: ' + ', '.join(missing_files)
    + '. Run deploy/prefetch_voxtral_cache.sh on a login node with internet first.'
  )

tokenizer_candidates = ['tokenizer.json', 'tokenizer.model', 'spiece.model', 'tekken.json']
if not any((latest / name).exists() for name in tokenizer_candidates):
  raise SystemExit(
    'Cached snapshot has no tokenizer artifacts (' + ', '.join(tokenizer_candidates) + ') at ' + str(latest)
  )

# Check ICL audio clips are present
icl_dir = Path('data/mlc26_task2/icl_examples')
n_icl = len(list(icl_dir.glob('tr_s*.wav'))) if icl_dir.is_dir() else 0
if n_icl < 120:
  raise SystemExit(
    f'Expected 120 ICL training audio clips in data/mlc26_task2/icl_examples/ '
    f'(tr_s*.wav) but found {n_icl}. '
    'Re-run the ICL clip extraction script to regenerate them.'
  )
print(f'Offline preflight OK — model cache and {n_icl} ICL clips found')
PY

MASTER_PORT=\${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}
echo Using torchrun master port: \${MASTER_PORT}
TORCHRUN_LOG_DIR=${TMP_ROOT}/torchrun_logs
mkdir -p \${TORCHRUN_LOG_DIR}

# Effective batch size: 4 GPUs x 1 per-GPU batch x 4 grad_accum = 16
# (r=16, α=32, lr=5e-5, 3 epochs, early-stop patience=3, enc+conn frozen)
# ICL rotation: 20 sets × 6 shots, one set randomly chosen per batch.
torchrun --standalone --nproc_per_node=4 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_MCQ \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --model-mode baseline \
  --train-jsonl data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl \
  --eval-jsonl data/mlc26_task2/organisers_balanced_en_translated.jsonl \
  --test-jsonl data/mlc26_task2/organisers_en_translated.jsonl \
  --audio-root data/mlc26_task2 \
  --do-train \
  --do-eval \
  --num-epochs 3 \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --grad-accum-steps 4 \
  --learning-rate 1e-5 \
  --no-use-bf16 \
  --use-fp16 \
  --output-root experiments \
  --experiment-name voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_4gpu \
  --freeze-encoder-connector \
  --timestamped-exp-dir \
  --prompt-language en \
  --use-chat-template-for-training \
  --crop-from-question-refs \
  --crop-collar-seconds 30 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --train-icl-audio-dir data/mlc26_task2/icl_examples \
  --train-icl-n-shots 6 \
  --eval-steps 25 \
  --save-steps 25 \
  --logging-steps 3 \
  --early-stopping-patience 3 \
  --best-ckpt-strategy min_generalization_gap \
  2>&1

echo 'Voxtral MCQ R6 + ICL rotation (frozen enc+conn) lr=1e-5 4-GPU training completed successfully'

trap - EXIT
"

echo "Starting Singularity container..."
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
echo "Sandbox directory: $SANDBOX_DIR"
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
