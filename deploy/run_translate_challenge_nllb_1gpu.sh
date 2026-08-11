#!/bin/bash
#SBATCH --job-name=nllb_challenge
#SBATCH --output=./slurm_logs/nllb_challenge_%j.out
#SBATCH --error=./slurm_logs/nllb_challenge_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "NLLB-200 Translation: challenge Phase1 & Phase2 -> English"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "=========================================="

module purge
module load singularity

# Container and storage paths
export PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export REPO_DIR="/opt/ASR-merging"

export CACHE_DIR="/gpfs/scratch/ehpc628/models"
export TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"

RUN_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}"
export TMP_ROOT="$RUN_ROOT/tmp"

mkdir -p ./slurm_logs
mkdir -p "$TMP_ROOT"

export CUDA_VISIBLE_DEVICES=0

# NLLB model snapshot path
NLLB_MODEL="${CACHE_DIR}/models--facebook--nllb-200-distilled-1.3B/snapshots/7be3e24664b38ce1cac29b8aeed6911aa0cf0576"

export CMD="
set -euo pipefail

export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_UNSAFE_ALLOW_LEGACY_WEIGHTS=1
export TORCH_EXTENSIONS_DIR=$TORCH_EXT_DIR
export PYTHONUNBUFFERED=1
export TMPDIR=$TMP_ROOT
export CUDA_VISIBLE_DEVICES=0

. /opt/asrenv/bin/activate
python -V

cd $REPO_DIR

echo ''
echo '--- Translating challenge Phase 1 JSONL ---'
python -m asr_merging.translate_challenge_nllb \
  --input  data/mlc26_task2/task2_phase1_questions_options.jsonl \
  --output data/mlc26_task2/task2_phase1_questions_options_nllb_en.jsonl \
  --model  ${NLLB_MODEL} \
  --batch-size 64

echo ''
echo '--- Translating challenge Phase 2 JSONL ---'
python -m asr_merging.translate_challenge_nllb \
  --input  data/mlc26_task2/task2_phase2_questions_options.jsonl \
  --output data/mlc26_task2/task2_phase2_questions_options_nllb_en.jsonl \
  --model  ${NLLB_MODEL} \
  --batch-size 64

echo ''
echo 'All challenge translation jobs completed successfully.'
"

echo "Starting Singularity container..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "NLLB model: $NLLB_MODEL"
echo ""

singularity exec \
    --nv \
    --writable \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $TORCH_EXT_DIR:$TORCH_EXT_DIR \
    --bind $TMP_ROOT:$TMP_ROOT \
    $SANDBOX_DIR \
    bash -c "$CMD"

echo "Singularity finished."
