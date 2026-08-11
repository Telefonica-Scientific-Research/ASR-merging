#!/bin/bash
# Ablation: 4 multimodal ICL shots (4-opt balanced A/B/C/D) for ALL question
# types, including 2-option questions (non-adaptive pool selection).
# Baseline: icl6mm uses type-adaptive selection (4 shots for 4-opt, 2 for 2-opt).
# This ablation tests whether 2-opt-specific shots add value over a single
# unified 4-shot balanced set.
#
# Run from the singularity_containers/ directory:
#   bash flower_speech_llm/opt/ASR-merging/deploy/submit_voxtral_24b_notx_mnt128_icl4mm_phase2.sh
#
# Tag: challenge_phase2_orig_notx_mnt128_icl4_mm

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
  FEW_SHOT_COUNT="4" \
  ICL_MULTIMODAL="1" \
  ICL_MULTIMODAL_NONEN="0" \
  EVAL_CROP="1" \
  CROP_COLLAR_SECONDS="30" \
  sbatch "${PHASE_SCRIPT}" | awk '{print $NF}'
)

echo "Submitted → SLURM job ${JID}"
echo "Logs:      slurm_logs/voxtral_challenge_phase_${JID}.{out,err}"
echo "Tag:       challenge_phase2_orig_notx_mnt128_icl4_mm"
echo "Output:    flower_speech_llm/opt/ASR-merging/experiments/voxtral_ootb_voxtral-small-24b-2507_phase2_eval/"
echo ""
echo "Monitor:   squeue -j ${JID}"
