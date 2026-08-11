#!/bin/bash
# Submit Phase-2 challenge eval for the 4 balanced early checkpoints
# (step-125 and step-150 of frzenc_lr1e5_bal and frzenc_only_lr1e5_bal).
#
# Config: PHASE=2, NLLB=0, ICL_MULTIMODAL=1, FEW_SHOT_COUNT=6,
#         EVAL_CROP=1, CROP_COLLAR_SECONDS=30, MAX_NEW_TOKENS=128
#
# Run from the ASR-merging repo root:
#   bash deploy/submit_voxtral_3b_icl_crop30_bal_ckpts_phase2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE_SCRIPT="${SCRIPT_DIR}/run_voxtral_challenge_eval_phase_4gpu.sh"

declare -A MODELS=(
  ["frzenc_lr1e5_bal_s125"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_bal_4gpu_20260628_004749/checkpoint-125"
  ["frzenc_only_lr1e5_bal_s125"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_bal_4gpu_20260628_010215/checkpoint-125"
  ["frzenc_lr1e5_bal_s150"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_bal_4gpu_20260628_004749/checkpoint-150"
  ["frzenc_only_lr1e5_bal_s150"]="experiments/voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_bal_4gpu_20260628_010215/checkpoint-150"
)

for TAG in frzenc_lr1e5_bal_s125 frzenc_only_lr1e5_bal_s125 frzenc_lr1e5_bal_s150 frzenc_only_lr1e5_bal_s150; do
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
