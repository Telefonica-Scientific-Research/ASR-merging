#!/bin/bash
# Single-shard 1-GPU challenge eval for Voxtral MCQ.
# Submit via launch_voxtral_challenge_eval.sh (sets SHARD_IDX, NUM_SHARDS, MODEL_ADAPTER_PATH).
# Can also be submitted directly:
#   sbatch --export=SHARD_IDX=0,NUM_SHARDS=4,MODEL_ADAPTER_PATH=experiments/voxtral_mcq_balanced_jsonl_4gpu/final_model \
#          deploy/run_voxtral_challenge_eval_shard.sh
#SBATCH --job-name=voxtral_challenge_eval
#SBATCH --output=./slurm_logs/voxtral_challenge_eval_s%a_%j.out
#SBATCH --error=./slurm_logs/voxtral_challenge_eval_s%a_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

# Shard config — set via --export
SHARD_IDX="${SHARD_IDX:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MODEL_ADAPTER_PATH="${MODEL_ADAPTER_PATH:-experiments/voxtral_mcq_balanced_jsonl_4gpu/final_model}"
# Audio root inside the container for the challenge eval set.
EVAL_AUDIO_ROOT="${EVAL_AUDIO_ROOT:-data/mlc26_task2/mlc-slm-2nd-eval}"

SHARD_TAG="shard_$(printf '%02d' $SHARD_IDX)_of_$(printf '%02d' $NUM_SHARDS)"
SHARD_JSONL="data/mlc26_task2/challenge_eval_shards/${SHARD_TAG}.jsonl"
EXPERIMENT_NAME="voxtral_challenge_eval_${SHARD_TAG}"

echo "=========================================="
echo "MareNostrum 5 - Voxtral Challenge Eval"
echo "Shard ${SHARD_IDX}/${NUM_SHARDS} — ${SHARD_JSONL}"
echo "Model/adapter: ${MODEL_ADAPTER_PATH}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
date

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
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}/tmp"

mkdir -p ./slurm_logs
mkdir -p "$TMP_ROOT"

export CUDA_VISIBLE_DEVICES=0

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
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

nvidia-smi

cd $REPO_DIR

. /opt/asrenv/bin/activate
python -V

echo 'Starting eval for shard ${SHARD_IDX}/${NUM_SHARDS} ...'
python -m asr_merging.voxtral_forgetting_eval \
  --model-id mistralai/Voxtral-Mini-3B-2507 \
  --adapter-path ${MODEL_ADAPTER_PATH} \
  --tasks jsonl_audio_mc \
  --jsonl-path ${SHARD_JSONL} \
  --audio-root ${EVAL_AUDIO_ROOT} \
  --jsonl-max-questions-per-audio 0 \
  --max-samples-per-task 0 \
  --split test \
  --no-use-bf16 \
  --use-fp16 \
  --output-dir experiments/challenge_eval_shards/${EXPERIMENT_NAME} \
  --prediction-only

echo 'Shard ${SHARD_IDX} eval complete.'
echo 'Output dir: experiments/challenge_eval_shards/${EXPERIMENT_NAME}'
"

echo "Starting Singularity container for shard ${SHARD_IDX}..."
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
echo "Shard ${SHARD_IDX} finished at $(date)"
echo "=========================================="
