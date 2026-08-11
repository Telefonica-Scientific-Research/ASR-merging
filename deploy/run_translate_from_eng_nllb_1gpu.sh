#!/bin/bash
#SBATCH --job-name=nllb_en_to_lang
#SBATCH --output=./slurm_logs/nllb_en_to_lang_%j.out
#SBATCH --error=./slurm_logs/nllb_en_to_lang_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Translate the English training JSONL to a target language using NLLB-200.
#
# Usage:
#   TARGET_LANG_CODE=por_Latn sbatch deploy/run_translate_from_eng_nllb_1gpu.sh
#
# TARGET_LANG_CODE must be a valid NLLB BCP-47 code:
#   por_Latn  fra_Latn  spa_Latn  rus_Cyrl  vie_Latn
#   tur_Latn  tgl_Latn  deu_Latn  ita_Latn
#
# Input:  data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl
# Output: data/mlc26_task2/mlcslm_2nd_dev_qa_en_to_<lang>.jsonl
#         where <lang> is the 3-letter prefix of TARGET_LANG_CODE (e.g. "por")

set -euo pipefail

if [[ -z "${TARGET_LANG_CODE:-}" ]]; then
  echo "ERROR: TARGET_LANG_CODE is not set."
  echo "Usage: TARGET_LANG_CODE=por_Latn sbatch $0"
  echo "Supported codes: por_Latn fra_Latn spa_Latn rus_Cyrl vie_Latn tur_Latn tgl_Latn deu_Latn ita_Latn"
  exit 1
fi

# Derive short suffix from NLLB code (first 3 chars before underscore)
LANG_SHORT="${TARGET_LANG_CODE%%_*}"

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "NLLB-200 Translation: English -> ${TARGET_LANG_CODE} (${LANG_SHORT})"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "=========================================="

module purge
module load singularity

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

NLLB_MODEL="${CACHE_DIR}/models--facebook--nllb-200-distilled-1.3B/snapshots/7be3e24664b38ce1cac29b8aeed6911aa0cf0576"

INPUT_JSONL="data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl"
OUTPUT_JSONL="data/mlc26_task2/mlcslm_2nd_dev_qa_en_to_${LANG_SHORT}.jsonl"

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
echo '--- Translating: ${SOURCE_NLLB:-eng_Latn} -> ${TARGET_LANG_CODE} ---'
echo 'Input:  ${INPUT_JSONL}'
echo 'Output: ${OUTPUT_JSONL}'
echo ''

python -m asr_merging.translate_from_english_nllb \
  --input  ${INPUT_JSONL} \
  --output ${OUTPUT_JSONL} \
  --target-lang ${TARGET_LANG_CODE} \
  --model  ${NLLB_MODEL} \
  --batch-size 64

echo ''
echo 'Translation completed: ${TARGET_LANG_CODE} -> ${OUTPUT_JSONL}'
"

echo "Starting Singularity container..."
echo "Target language: $TARGET_LANG_CODE ($LANG_SHORT)"
echo "Input:  $INPUT_JSONL"
echo "Output: $OUTPUT_JSONL"
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

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
