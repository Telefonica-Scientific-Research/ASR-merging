#!/bin/bash
# Ablation: 6 multimodal ICL shots with one non-English audio clip (Vietnamese)
# replacing the 4-opt-D English_Indian slot.  Tests whether explicitly showing
# the model a non-English audio + English answer example improves accuracy on
# the ~40% non-English questions in the Phase 2 eval.
#
# Run from the singularity_containers/ directory:
#   bash flower_speech_llm/opt/ASR-merging/deploy/submit_voxtral_24b_notx_mnt128_icl6mm_nonen_phase2.sh
#
# Tag: challenge_phase2_orig_notx_mnt128_icl6_mm_nonen

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
  CALIBRATE_PRIOR="0" \
  TRANSCRIPT_DIR="" \
  MAX_NEW_TOKENS="128" \
  FEW_SHOT_COUNT="6" \
  ICL_MULTIMODAL="1" \
  ICL_MULTIMODAL_NONEN="1" \
  EVAL_CROP="1" \
  CROP_COLLAR_SECONDS="30" \
  sbatch "${PHASE_SCRIPT}" | awk '{print $NF}'
)

echo "Submitted → SLURM job ${JID}"
echo "Logs:      slurm_logs/voxtral_challenge_phase_${JID}.{out,err}"
echo "Tag:       challenge_phase2_orig_notx_mnt128_icl6_mm_nonen"
echo "Output:    flower_speech_llm/opt/ASR-merging/experiments/voxtral_ootb_voxtral-small-24b-2507_phase2_eval/"
echo ""
echo "Monitor:   squeue -j ${JID}"
