#!/bin/bash
# Submit Phase-2 challenge eval for the 4 non-balanced ICL-crop30 fine-tuned
# Voxtral-Mini-3B-2507 models.
#
# Config (mirrors best-LB OOTB system):
#   PHASE=2, NLLB=0, ICL_MULTIMODAL=1, FEW_SHOT_COUNT=6
#   (4 four-option + 2 two-option shots, English only)
#   EVAL_CROP=1, CROP_COLLAR_SECONDS=30, MAX_NEW_TOKENS=128
#
# Run from the singularity_containers/ directory:
#   bash flower_speech_llm/opt/ASR-merging/deploy/submit_voxtral_3b_icl_crop30_icl6mm_phase2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE_SCRIPT="${SCRIPT_DIR}/run_voxtral_challenge_eval_phase_4gpu.sh"

declare -A MODELS=(
  ["frzenc_5e5"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_4gpu_20260628_004742/final_model"
  ["frzenc_1e5"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_4gpu_20260628_004740/final_model"
  ["frzenc_only_5e5"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_only_4gpu_20260628_004842/final_model"
  ["frzenc_only_1e5"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_4gpu_20260628_005302/final_model"
)

for TAG in frzenc_5e5 frzenc_1e5 frzenc_only_5e5 frzenc_only_1e5; do
  ADAPTER="${MODELS[$TAG]}"
  JID=$(
    MODEL_ID="mistralai/Voxtral-Mini-3B-2507" \
    MODEL_ADAPTER_PATH="${ADAPTER}" \
    PHASE="2" \
    USE_NLLB="0" \
    DEBIAS_CYCLIC="0" \
    CALIBRATE_PRIOR="0" \
    TRANSCRIPT_DIR="" \
    MAX_NEW_TOKENS="128" \
    FEW_SHOT_COUNT="6" \
    ICL_MULTIMODAL="1" \
    EVAL_CROP="1" \
    CROP_COLLAR_SECONDS="30" \
    sbatch "${PHASE_SCRIPT}" | awk '{print $NF}'
  )
  echo "  [${TAG}] → job ${JID}  (adapter: ${ADAPTER})"
done
