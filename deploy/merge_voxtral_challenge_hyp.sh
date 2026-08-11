#!/bin/bash
# Merge hyp.txt files from all challenge eval shards into a single submission file.
# Run on the login node once all shard jobs have completed:
#   bash flower_speech_llm/opt/ASR-merging/deploy/merge_voxtral_challenge_hyp.sh
#
# Override defaults:
#   NUM_SHARDS=4 MODEL_ADAPTER_PATH=experiments/... bash deploy/merge_voxtral_challenge_hyp.sh

set -euo pipefail

NUM_SHARDS="${NUM_SHARDS:-4}"
MODEL_ADAPTER_PATH="${MODEL_ADAPTER_PATH:-experiments/voxtral_mcq_balanced_jsonl_4gpu/final_model}"
SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm/opt/ASR-merging"
FINAL_DIR="${SANDBOX_DIR}/experiments/challenge_eval_final"

mkdir -p "$FINAL_DIR"
MODEL_SLUG=$(echo "$MODEL_ADAPTER_PATH" | tr '/' '_')
FINAL_HYP="${FINAL_DIR}/hyp_${MODEL_SLUG}.txt"

echo "Merging ${NUM_SHARDS} shard hyp.txt files..."
echo "Model/adapter: ${MODEL_ADAPTER_PATH}"

# Collect shard hyp.txt files in shard-index order
HYP_FILES=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_TAG="shard_$(printf '%02d' $i)_of_$(printf '%02d' $NUM_SHARDS)"
  SHARD_HYP="${SANDBOX_DIR}/experiments/challenge_eval_shards/voxtral_challenge_eval_${SHARD_TAG}/hyp.txt"
  if [[ ! -f "$SHARD_HYP" ]]; then
    echo "ERROR: Missing shard hyp.txt: $SHARD_HYP"
    exit 1
  fi
  HYP_FILES+=("$SHARD_HYP")
  echo "  Shard ${i}: $(wc -l < "$SHARD_HYP") lines — $SHARD_HYP"
done

# Concatenate in order — preserves original JSONL ordering
cat "${HYP_FILES[@]}" > "$FINAL_HYP"

TOTAL=$(wc -l < "$FINAL_HYP")
echo ""
echo "Merged ${TOTAL} predictions → ${FINAL_HYP}"

# Sanity: check for duplicates
UNIQ=$(sort -u "$FINAL_HYP" | wc -l)
if [[ "$TOTAL" != "$UNIQ" ]]; then
  echo "WARNING: ${TOTAL} lines but only ${UNIQ} unique — possible duplicate predictions!"
else
  echo "Sanity OK: all ${TOTAL} predictions are unique."
fi

echo ""
echo "Submit to challenge scorer:"
echo "  ${FINAL_HYP}"
