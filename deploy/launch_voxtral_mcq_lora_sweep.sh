#!/bin/bash
# Submits all LoRA r/alpha sweep jobs for Voxtral MCQ training.
# Run from: /home/tele/tele574778/projects_new/jls/singularity_containers/
#
# Sweep design:
#   Tier 1 — capacity sweep (alpha=2*r, scale fixed at 2.0):
#     r=8  / alpha=16   (low capacity)
#     r=16 / alpha=32   (default)
#     r=32 / alpha=64   (high capacity)
#   Tier 2 — scale effect at r=16:
#     r=16 / alpha=16   (scale=1.0)
#     r=16 / alpha=64   (scale=4.0)

set -euo pipefail

SCRIPT="flower_speech_llm/opt/ASR-merging/deploy/run_voxtral_mcq_lora_sweep.sh"

declare -a COMBOS=(
  "8 16"    # capacity: low,   scale: 2.0
  "16 32"   # capacity: default, scale: 2.0
  "32 64"   # capacity: high,  scale: 2.0
  "16 16"   # capacity: default, scale: 1.0
  "16 64"   # capacity: default, scale: 4.0
)

for combo in "${COMBOS[@]}"; do
  r=$(echo $combo | awk '{print $1}')
  alpha=$(echo $combo | awk '{print $2}')
  scale=$(echo "scale=1; $alpha / $r" | bc)
  jid=$(sbatch --export=LORA_R=${r},LORA_ALPHA=${alpha} \
               --job-name="mcq_r${r}_a${alpha}" \
               --output="./slurm_logs/voxtral_mcq_r${r}_a${alpha}_%j.out" \
               --error="./slurm_logs/voxtral_mcq_r${r}_a${alpha}_%j.err" \
               "$SCRIPT" | awk '{print $NF}')
  echo "Submitted r=${r} alpha=${alpha} (scale=${scale}x) → job ${jid}"
done

echo ""
echo "All 5 sweep jobs submitted. Monitor with:"
echo "  squeue -u \$USER | grep mcq"
