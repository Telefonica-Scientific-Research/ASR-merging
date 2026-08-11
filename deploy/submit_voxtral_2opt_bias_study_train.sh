#!/bin/bash
# 2-option positional-bias study on the TRAINING set (490 questions).
# Same 5 conditions as the dev-set study (job 42565690), but using
# train_2opt_only.jsonl so we can compare train vs dev A-bias.
#
# Run from the singularity_containers/ directory:
#   bash flower_speech_llm/opt/ASR-merging/deploy/submit_voxtral_2opt_bias_study_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE_SCRIPT="${SCRIPT_DIR}/run_voxtral_2opt_bias_study.sh"

if [[ ! -f "${PHASE_SCRIPT}" ]]; then
  echo "ERROR: bias study script not found: ${PHASE_SCRIPT}"
  exit 1
fi

JID=$(
  JSONL_2OPT="data/mlc26_task2/train_2opt_only.jsonl" \
  EVAL_AUDIO_ROOT="data/mlc26_task2/mlc-slm-2nd-dev" \
  sbatch "${PHASE_SCRIPT}" | awk '{print $NF}'
)

echo "Submitted → SLURM job ${JID}"
echo "Logs:      slurm_logs/voxtral_2opt_bias_${JID}.{out,err}"
echo "Output:    flower_speech_llm/opt/ASR-merging/experiments/voxtral_ootb_2opt_bias_study_${JID}/"
echo ""
echo "Monitor:   squeue -j ${JID}"
