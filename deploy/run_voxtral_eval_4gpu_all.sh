#!/bin/bash
#SBATCH --job-name=voxtral_eval_4gpu_all
#SBATCH --output=./slurm_logs/voxtral_eval_4gpu_all_%j.out
#SBATCH --error=./slurm_logs/voxtral_eval_4gpu_all_%j.err
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
echo "ASR-merging Voxtral Router - 4 parallel evals"
echo "GPU0: mlc25-test  / continue_from_current"
echo "GPU1: mlc25-test  / full_train_continuation"
echo "GPU2: mlc26-dev   / continue_from_current"
echo "GPU3: mlc26-dev   / full_train_continuation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "QoS: $SLURM_JOB_QOS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
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

export KEEP_ALIVE_ON_FAILURE="${KEEP_ALIVE_ON_FAILURE:-1}"

export CMD="
set -uo pipefail

debug_hold_on_failure() {
  exit_code=\$?
  if [[ \$exit_code -ne 0 && \"${KEEP_ALIVE_ON_FAILURE}\" == \"1\" ]]; then
    echo '[DEBUG] One or more evals failed.'
    echo '[DEBUG] Keeping allocation alive for debugging.'
    echo '[DEBUG] Node: ${SLURM_JOB_NODELIST}'
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
  echo '/opt/asrenv is missing.'
  exit 2
fi
. /opt/asrenv/bin/activate
python -V

LOG_DIR=${TMP_ROOT}/eval_logs
mkdir -p \${LOG_DIR}

echo
echo Starting 4 parallel evals...
echo GPU0: mlc25-test / continue_from_current -\> \${LOG_DIR}/gpu0.log
echo GPU1: mlc25-test / full_train_continuation -\> \${LOG_DIR}/gpu1.log
echo GPU2: mlc26-dev  / continue_from_current -\> \${LOG_DIR}/gpu2.log
echo GPU3: mlc26-dev  / full_train_continuation -\> \${LOG_DIR}/gpu3.log
echo

CUDA_VISIBLE_DEVICES=0 python -m asr_merging.voxtral_eval_router \
  --config-json configuration/voxtral_eval_mlc25_test_continue_from_current.json \
  > \${LOG_DIR}/gpu0.log 2>&1 &
pid0=\$!

CUDA_VISIBLE_DEVICES=1 python -m asr_merging.voxtral_eval_router \
  --config-json configuration/voxtral_eval_mlc25_test_full_train_continuation.json \
  > \${LOG_DIR}/gpu1.log 2>&1 &
pid1=\$!

CUDA_VISIBLE_DEVICES=2 python -m asr_merging.voxtral_eval_router \
  --config-json configuration/voxtral_eval_mlc26_dev_continue_from_current.json \
  > \${LOG_DIR}/gpu2.log 2>&1 &
pid2=\$!

CUDA_VISIBLE_DEVICES=3 python -m asr_merging.voxtral_eval_router \
  --config-json configuration/voxtral_eval_mlc26_dev_full_train_continuation.json \
  > \${LOG_DIR}/gpu3.log 2>&1 &
pid3=\$!

ec0=0; ec1=0; ec2=0; ec3=0
set +e
wait \${pid0}; ec0=\$?
wait \${pid1}; ec1=\$?
wait \${pid2}; ec2=\$?
wait \${pid3}; ec3=\$?
set -e

echo
echo ============================================================
echo GPU0 LOG [mlc25-test / continue_from_current]
echo ============================================================
cat \${LOG_DIR}/gpu0.log
echo
echo ============================================================
echo GPU1 LOG [mlc25-test / full_train_continuation]
echo ============================================================
cat \${LOG_DIR}/gpu1.log
echo
echo ============================================================
echo GPU2 LOG [mlc26-dev / continue_from_current]
echo ============================================================
cat \${LOG_DIR}/gpu2.log
echo
echo ============================================================
echo GPU3 LOG [mlc26-dev / full_train_continuation]
echo ============================================================
cat \${LOG_DIR}/gpu3.log
echo
echo ============================================================
echo Exit codes: gpu0=\${ec0} gpu1=\${ec1} gpu2=\${ec2} gpu3=\${ec3}
echo ============================================================

if [[ \${ec0} -ne 0 || \${ec1} -ne 0 || \${ec2} -ne 0 || \${ec3} -ne 0 ]]; then
  echo One or more evals failed.
  exit 1
fi

echo All 4 evals completed successfully.
trap - EXIT
"

echo "Starting Singularity container..."
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
