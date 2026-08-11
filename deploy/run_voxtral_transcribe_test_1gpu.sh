#!/bin/bash
#SBATCH --job-name=voxtral_transcribe_test
#SBATCH --output=./slurm_logs/voxtral_transcribe_test_%j.out
#SBATCH --error=./slurm_logs/voxtral_transcribe_test_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Transcription quality test — single Spanish session (5094_002_phone.wav, 20.5 min)
# Uses: mlc25_mlc26_continue final_model (WER 12.1% multilingual ASR)
# Output: transcript .txt + .json written to $RUN_ROOT/transcripts/

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 — Voxtral ASR Transcription Test"
echo "Target: Spanish/5094_002_phone.wav (20.5 min)"
echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_JOB_NODELIST"
echo "Partition   : $SLURM_JOB_PARTITION"
echo "GPUs        : $SLURM_GPUS_ON_NODE"
echo "Started at  : $(date)"
echo "=========================================="

module purge
module load singularity

# --- Paths ------------------------------------------------------------------
export PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export REPO_DIR="/opt/ASR-merging"
export CACHE_DIR="/gpfs/scratch/ehpc628/models"
export TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"
export MLC_DATA_DIR="/gpfs/scratch/ehpc494/jls/datasets"

RUN_TAG="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/transcribe_test_${RUN_TAG}"
export TMP_ROOT="$RUN_ROOT/tmp"

mkdir -p ./slurm_logs
mkdir -p "$CACHE_DIR"
mkdir -p "$TORCH_EXT_DIR"
mkdir -p "$TMP_ROOT"
mkdir -p "$RUN_ROOT/transcripts"

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

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo

cd $REPO_DIR

if [[ ! -f /opt/asrenv/bin/activate ]]; then
  echo '[ERROR] /opt/asrenv not found'
  exit 2
fi
. /opt/asrenv/bin/activate
python -V

echo
echo '--- Starting transcription ---'
echo

AUDIO_PATH=\"data/mlc26_task2/mlc-slm-2nd-dev/Spanish/5094_002_phone.wav\"
CHECKPOINT=\"experiments/mlc25_mlc26_continue_4gpu_fixed_bothsides_20260530_120641/final_model\"
OUTPUT_DIR=\"$REPO_DIR/data/transcripts\"

python -m asr_merging.transcribe_sessions \
  --audio-path  \${AUDIO_PATH} \
  --checkpoint-path \${CHECKPOINT} \
  --output-dir  \${OUTPUT_DIR} \
  --language    en \
  --max-new-tokens 8192 \
  --mode        chat

echo
echo '--- Transcript content ---'
cat \${OUTPUT_DIR}/5094_002_phone.txt || echo '(file not found)'

echo
echo '--- Metadata ---'
cat \${OUTPUT_DIR}/5094_002_phone.json || echo '(file not found)'
"

echo "Starting Singularity container..."
singularity exec \
    --nv \
    --writable \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $TORCH_EXT_DIR:$TORCH_EXT_DIR \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind /gpfs:/gpfs \
    --bind $MLC_DATA_DIR:/mnt/mirror1/datasets/MLC-SLM_Workshop_2026 \
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "$CMD"

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "Transcript output: $RUN_ROOT/transcripts/"
echo "=========================================="
