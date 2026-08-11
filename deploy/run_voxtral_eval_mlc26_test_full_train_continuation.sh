#!/bin/bash
#SBATCH --job-name=voxtral_asr_wer_mlc26_test
#SBATCH --output=./slurm_logs/voxtral_asr_wer_mlc26_test_%j.out
#SBATCH --error=./slurm_logs/voxtral_asr_wer_mlc26_test_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "ASR-merging Voxtral Router Eval (mlc25 test + mlc26 NEW test, sequential)"
echo "Adapter: mlc25_mlc26_continue (full train continuation)"
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
export KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-1}"

export CMD="
set -euo pipefail

debug_hold_on_failure() {
  exit_code=\$?
  if [[ \$exit_code -ne 0 && \"${KEEP_ALIVE_ON_FAILURE}\" == \"1\" ]]; then
    echo '[DEBUG] Eval command failed (exit='\"\$exit_code\"').'
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

# Offline preflight: dependencies and model cache must already exist.
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

# Must be cached already when compute nodes have no internet.
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
    + '. Run deploy/prefetch_voxtral_cache.sh on a login node with internet first.'
  )

print('Offline preflight: dependencies and model cache files OK')
PY

MASTER_PORT=\${MASTER_PORT:-$((29500 + (${SLURM_JOB_ID:-0} % 1000)))}
echo Using torchrun master port: \${MASTER_PORT}
TORCHRUN_LOG_DIR=${TMP_ROOT}/torchrun_logs
mkdir -p \${TORCHRUN_LOG_DIR}

torchrun --standalone --nproc_per_node=1 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_router \
  --config-json configuration/mlc26_eval_mlc25_test_full_train_continuation.json

echo '--- mlc25 test eval done; starting mlc26 new-language test eval ---'

torchrun --standalone --nproc_per_node=1 --master_port \${MASTER_PORT} \
  --log-dir \${TORCHRUN_LOG_DIR} --redirects 3 --tee 1 --local-ranks-filter 0 \
  -m asr_merging.voxtral_train_router \
  --config-json configuration/voxtral_eval_mlc26_test_full_train_continuation.json

echo 'Voxtral eval completed successfully'

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
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "$CMD"

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
