#!/bin/bash
# Ablation: 6 multimodal ICL shots + prior calibration.
# Tests whether calibration still reduces A-bias on top of balanced audio ICL
# (icl6mm already presents A/B/C/D once each, which may have already reduced
# label-position prior compared to the text-only icl2 baseline).
#
# Run from the singularity_containers/ directory:
#   bash flower_speech_llm/opt/ASR-merging/deploy/submit_voxtral_24b_notx_mnt128_icl6mm_calib_phase2.sh
#
# NOTE: calibration doubles forward passes (~2× longer than icl6mm).
# Tag: challenge_phase2_orig_calibrated_notx_mnt128_icl6_mm

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE_SCRIPT="${SCRIPT_DIR}/run_voxtral_challenge_eval_phase_4gpu.sh"

if [[ ! -f "${PHASE_SCRIPT}" ]]; then
  echo "ERROR: phase script not found: ${PHASE_SCRIPT}"
  exit 1
fi

JID=$(
  MODEL_ID="mistralai/Voxtral-Small-24B-2507" \
  MODEL_ADAPTER_PATH="" \
  PHASE="2" \
  USE_NLLB="0" \
  DEBIAS_CYCLIC="0" \
  CALIBRATE_PRIOR="1" \
  TRANSCRIPT_DIR="" \
  MAX_NEW_TOKENS="128" \
  FEW_SHOT_COUNT="6" \
  ICL_MULTIMODAL="1" \
  ICL_MULTIMODAL_NONEN="0" \
  EVAL_CROP="1" \
  CROP_COLLAR_SECONDS="30" \
  sbatch "${PHASE_SCRIPT}" | awk '{print $NF}'
)

echo "Submitted → SLURM job ${JID}"
echo "Logs:      slurm_logs/voxtral_challenge_phase_${JID}.{out,err}"
echo "Tag:       challenge_phase2_orig_calibrated_notx_mnt128_icl6_mm"
echo "Output:    flower_speech_llm/opt/ASR-merging/experiments/voxtral_ootb_voxtral-small-24b-2507_phase2_eval/"
echo ""
echo "Monitor:   squeue -j ${JID}"
