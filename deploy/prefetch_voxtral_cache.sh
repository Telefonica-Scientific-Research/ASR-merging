#!/bin/bash
set -euo pipefail

# Run this on a login node WITH internet access.
# It fills HF cache so compute nodes can run with HF_HUB_OFFLINE=1.

SBOX="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
CACHE_DIR="${CACHE_DIR:-/gpfs/scratch/ehpc628/models}"
MODEL_ID="${MODEL_ID:-mistralai/Voxtral-Mini-3B-2507}"

if [[ ! -d "$SBOX" ]]; then
  echo "Singularity sandbox not found: $SBOX"
  exit 1
fi

mkdir -p "$CACHE_DIR"

echo "Prefetching model/tokenizer cache for: $MODEL_ID"
echo "Cache dir: $CACHE_DIR"

singularity exec --nv --no-home --writable "$SBOX" bash -lc "
set -euo pipefail
. /opt/asrenv/bin/activate
export HF_HOME='$CACHE_DIR'
export HF_HUB_CACHE='$CACHE_DIR'
export TRANSFORMERS_CACHE='$CACHE_DIR'
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

python - <<'PY'
from pathlib import Path
from transformers import VoxtralProcessor, AutoProcessor, VoxtralForConditionalGeneration

model_id = '$MODEL_ID'

# Fetch processor/tokenizer artifacts.
VoxtralProcessor.from_pretrained(model_id, local_files_only=False)
VoxtralProcessor.from_pretrained(model_id, use_fast=False, local_files_only=False)
AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=False, local_files_only=False)

# Fetch model config/weights metadata; actual weight shards are lazily fetched as needed.
VoxtralForConditionalGeneration.from_pretrained(model_id, local_files_only=False, device_map='cpu')

print('Prefetch complete')
PY
"

echo "Done. You can now submit offline jobs with HF_HUB_OFFLINE=1 on compute nodes."
