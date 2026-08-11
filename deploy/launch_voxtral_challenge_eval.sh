#!/bin/bash
# Launch 4 parallel 1-GPU challenge eval shards, then merge hyp.txt files.
#
# Usage (from singularity_containers/ dir):
#   bash flower_speech_llm/opt/ASR-merging/deploy/launch_voxtral_challenge_eval.sh
#
# Override model/adapter:
#   MODEL_ADAPTER_PATH=experiments/voxtral_mcq_lora_r16_a32/final_model \
#   bash flower_speech_llm/opt/ASR-merging/deploy/launch_voxtral_challenge_eval.sh
#
# The shard JSONL files are expected to already exist under:
#   flower_speech_llm/opt/ASR-merging/data/mlc26_task2/challenge_eval_shards/
# Run the split script first if needed:
#   python flower_speech_llm/opt/ASR-merging/asr_merging/scripts/split_challenge_jsonl.py \
#     --jsonl-path flower_speech_llm/opt/ASR-merging/data/mlc26_task2/task2_phase1_questions_options.jsonl \
#     --output-dir flower_speech_llm/opt/ASR-merging/data/mlc26_task2/challenge_eval_shards \
#     --num-shards 4

set -euo pipefail

NUM_SHARDS=4
MODEL_ADAPTER_PATH="${MODEL_ADAPTER_PATH:-experiments/voxtral_mcq_balanced_jsonl_4gpu/final_model}"
EVAL_AUDIO_ROOT="${EVAL_AUDIO_ROOT:-data/mlc26_task2/mlc-slm-2nd-eval}"
SCRIPT="$(cd "$(dirname "$0")" && pwd)/run_voxtral_challenge_eval_shard.sh"
SANDBOX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SHARD_DIR="${SANDBOX_DIR}/data/mlc26_task2/challenge_eval_shards"

echo "Voxtral challenge eval — ${NUM_SHARDS} shards"
echo "Model/adapter: ${MODEL_ADAPTER_PATH}"
echo ""

# Verify shard files exist
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_TAG="shard_$(printf '%02d' $i)_of_$(printf '%02d' $NUM_SHARDS)"
  SHARD_FILE="${SHARD_DIR}/${SHARD_TAG}.jsonl"
  if [[ ! -f "$SHARD_FILE" ]]; then
    echo "ERROR: Shard file not found: $SHARD_FILE"
    echo "Run the split script first:"
    echo "  python ${SANDBOX_DIR}/asr_merging/scripts/split_challenge_jsonl.py \\"
    echo "    --jsonl-path ${SANDBOX_DIR}/data/mlc26_task2/task2_phase1_questions_options.jsonl \\"
    echo "    --output-dir ${SHARD_DIR} --num-shards ${NUM_SHARDS}"
    exit 1
  fi
done

# Submit shard jobs
declare -a JOB_IDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  JID=$(sbatch \
    --export=SHARD_IDX=${i},NUM_SHARDS=${NUM_SHARDS},MODEL_ADAPTER_PATH=${MODEL_ADAPTER_PATH},EVAL_AUDIO_ROOT=${EVAL_AUDIO_ROOT} \
    --job-name="challenge_eval_s${i}" \
    --output="./slurm_logs/voxtral_challenge_eval_s${i}_%j.out" \
    --error="./slurm_logs/voxtral_challenge_eval_s${i}_%j.err" \
    "$SCRIPT" | awk '{print $NF}')
  JOB_IDS+=("$JID")
  echo "Submitted shard ${i}/${NUM_SHARDS} → job ${JID}"
done

JOB_LIST=$(IFS=,; echo "${JOB_IDS[*]}")
MODEL_SLUG=$(echo "$MODEL_ADAPTER_PATH" | tr '/' '_')

echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep challenge"
echo ""
echo "Once all 4 shard jobs finish, run the merge locally:"
echo "  bash flower_speech_llm/opt/ASR-merging/deploy/merge_voxtral_challenge_hyp.sh"
echo ""
echo "Or wait automatically:"
echo "  squeue -j ${JOB_LIST} --noheader | while [ \$(squeue -j ${JOB_LIST} --noheader | wc -l) -gt 0 ]; do sleep 60; done"
echo "  bash flower_speech_llm/opt/ASR-merging/deploy/merge_voxtral_challenge_hyp.sh"
echo ""
echo "Final hyp.txt will be at:"
echo "  ${SANDBOX_DIR}/experiments/challenge_eval_final/hyp_${MODEL_SLUG}.txt"
