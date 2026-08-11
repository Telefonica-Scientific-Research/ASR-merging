#!/bin/bash
#SBATCH --job-name=qwen3_translate
#SBATCH --output=./slurm_logs/qwen3_translate_%j.out
#SBATCH --error=./slurm_logs/qwen3_translate_%j.err
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
echo "Qwen3-4B Translation: training JSONL -> English"
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

export CUDA_VISIBLE_DEVICES=0

QWEN3_MODEL="${CACHE_DIR}/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"

export CMD="
set -euo pipefail

export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=$TORCH_EXT_DIR
export PYTHONUNBUFFERED=1
export TMPDIR=$TMP_ROOT
export CUDA_VISIBLE_DEVICES=0

. /opt/asrenv/bin/activate
python -V

cd $REPO_DIR

echo ''
echo '--- Translating training JSONL (Qwen3-4B) ---'
python -m asr_merging.translate_to_english_qwen3 \
  --input  data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource.jsonl \
  --output data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_en_translated_qwen3.jsonl \
  --model  ${QWEN3_MODEL} \
  --batch-size 16

echo ''
echo '--- Translating organisers eval JSONL (Qwen3-4B) ---'
python -m asr_merging.translate_to_english_qwen3 \
  --input  data/mlc26_task2/organisers.jsonl \
  --output data/mlc26_task2/organisers_en_translated_qwen3.jsonl \
  --model  ${QWEN3_MODEL} \
  --batch-size 16

echo ''
echo '--- Translating organisers_balanced eval JSONL (Qwen3-4B) ---'
python -m asr_merging.translate_to_english_qwen3 \
  --input  data/mlc26_task2/organisers_balanced.jsonl \
  --output data/mlc26_task2/organisers_balanced_en_translated_qwen3.jsonl \
  --model  ${QWEN3_MODEL} \
  --batch-size 16

echo ''
echo 'All Qwen3-4B translation jobs completed successfully.'
"

echo "Starting Singularity container..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Qwen3 model: $QWEN3_MODEL"
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
