#!/bin/bash
#SBATCH --job-name=voxtral_nllb_trx_c5_4gpu
#SBATCH --output=./slurm_logs/voxtral_nllb_trx_c5_4gpu_%j.out
#SBATCH --error=./slurm_logs/voxtral_nllb_trx_c5_4gpu_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# R6-base + transcript augmentation
# ─────────────────────────────────
# Identical training config to R6 (voxtral_mcq_nllb_translated_4gpu_20260620_112000,
# LB 0.7382) but with --transcript-dir data/transcripts so that every MCQ prompt is
# prefixed with the ASR transcript of the session audio:
#
#   Transcript:
#   {full transcript text}
#
#   Choose the most suitable answer …  (MCQ body)
#
# Both training collator and eval inference load transcripts from data/transcripts/
# using the audio-file stem as the lookup key ({stem}.txt).
#
# Requires transcript job to have finished first:
#   ls data/transcripts/*.txt | wc -l   # expect ~300

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "ASR-merging Voxtral MCQ Train 4-GPU — R6-base + transcript (NLLB-translated) — crop collar 5s"
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

# Preflight: verify transcript files are present.
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

# Check transcript files exist
transcript_dir = Path('data/transcripts')
n_transcripts = len(list(transcript_dir.glob('*.txt'))) if transcript_dir.is_dir() else 0
if n_transcripts == 0:
  raise SystemExit(
    'No transcript .txt files found in data/transcripts/. '
    'Wait for the transcription job (deploy/run_voxtral_transcribe_all_4gpu.sh) to finish.'
  )
print(f'Offline preflight OK — model cache and {n_transcripts} transcripts found')
PY

MASTER_PORT=\${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}
echo Using torchrun master port: \${MASTER_PORT}
TORCHRUN_LOG_DIR=${TMP_ROOT}/torchrun_logs
mkdir -p \${TORCHRUN_LOG_DIR}

# Effective batch size: 4 GPUs x 1 per-GPU batch x 4 grad_accum = 16
# (same as R6: r=16, α=32, lr=5e-5, 3 epochs, early-stop patience=3)
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
  --learning-rate 5e-5 \
  --no-use-bf16 \
  --use-fp16 \
  --output-root experiments \
  --experiment-name voxtral_mcq_nllb_transcript_collar5_4gpu \
  --timestamped-exp-dir \
  --prompt-language en \
  --crop-from-question-refs \
  --crop-collar-seconds 5 \
  --random-crop-seconds 300 \
  --audio-prompt-cache-size 16 \
  --eval-steps 25 \
  --save-steps 25 \
  --logging-steps 3 \
  --early-stopping-patience 3 \
  --transcript-dir data/transcripts \
  2>&1

echo 'Voxtral MCQ R6-base + transcript 4-GPU training completed successfully'

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
