#!/bin/bash
#SBATCH --job-name=voxtral_transcribe_all
#SBATCH --output=./slurm_logs/voxtral_transcribe_all_%j.out
#SBATCH --error=./slurm_logs/voxtral_transcribe_all_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Transcribe all unique FULL SESSION audio files:
#   - ~150 dev (training) sessions: paths read from JSONL 'path' fields
#   - ~151 challenge eval sessions: resolved from challenge JSONL 'session_id' fields
#     via _relative_audio_paths_from_session_id() (same logic as the challenge eval script)
#
# NOTE: mlc-slm-2nd-eval also contains thousands of per-question audio clips.
# We deliberately exclude those (rglob is NOT used); only session-level files are needed.
#
# Strategy:
#   - 4 GPUs, 1 Python process per GPU, each takes every 4th file (round-robin)
#   - File list is sorted dev-first so training files get priority within each shard
#   - language=en for all files (verified better WER in multilingual setting)
#   - Files >29 min are split automatically by transcribe_sessions.py
#
# Timing estimate (based on 41.7× RT observed on H100):
#   ~301 files × avg 21.2 min = ~107h audio / 4 GPUs / 41.7× RT ≈ 38 min wall time
#
# Output: $REPO_DIR/data/transcripts/{stem}.txt  +  .json

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 — Voxtral ASR Batch Transcription"
echo "Files: ~150 dev + ~151 challenge = ~301 unique session files (from JSONL paths)"
echo "GPUs: 4 parallel, 1 file at a time per GPU"
echo "Language: en (verified best for multilingual)"
echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_JOB_NODELIST"
echo "Started at  : $(date)"
echo "=========================================="

module purge
module load singularity

# --- Paths ------------------------------------------------------------------
export PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export REPO_DIR="/opt/ASR-merging"
export CACHE_DIR="/gpfs/scratch/ehpc628/models"
export TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"
export MLC_DATA_DIR="/gpfs/scratch/ehpc494/jls/datasets"

RUN_TAG="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/transcribe_all_${RUN_TAG}/tmp"

mkdir -p ./slurm_logs "$CACHE_DIR" "$TORCH_EXT_DIR" "$TMP_ROOT"

# --- Build ordered file list (dev first = training priority) ----------------
# Generated inside the container so paths match container view.
# Uses JSONL path fields (dev) and session_id→path mapping (eval) so we only
# transcribe the ~150 dev session files + ~151 challenge session files, NOT
# the thousands of per-question audio clips that also live in mlc-slm-2nd-eval.
export BUILD_LIST_CMD="
set -euo pipefail
. /opt/asrenv/bin/activate

cd $REPO_DIR
python3 - <<'PYEOF'
from pathlib import Path
import json, glob, soundfile as sf
from asr_merging.voxtral_forgetting_eval import _relative_audio_paths_from_session_id

out_path = Path('data/transcripts/.file_list_all.txt')
out_path.parent.mkdir(parents=True, exist_ok=True)

data_root = Path('data/mlc26_task2')
dev_base  = data_root / 'mlc-slm-2nd-dev'
eval_base = data_root / 'mlc-slm-2nd-eval'

# --- Dev files: collect unique paths from all training/eval JSONL files -----
dev_paths = set()
for fn in sorted(data_root.glob('*.jsonl')):
    with open(fn) as fh:
        for line in fh:
            p = json.loads(line).get('path')
            if p:
                dev_paths.add(p)
dev_files = sorted(dev_base / Path(p).name if '/' not in p else dev_base.parent / p
                   for p in dev_paths)
# Resolve relative to repo root (paths are like mlc-slm-2nd-dev/English_American/x.wav)
dev_files = sorted(data_root / p for p in dev_paths)
dev_files = [f for f in dev_files if f.exists()]

# --- Eval files: derive session audio paths from challenge JSONL session_ids ------
challenge_jsonls = [
    'data/mlc26_task2/task2_phase1_questions_options.jsonl',
    'data/mlc26_task2/task2_phase2_questions_options.jsonl',
]
eval_session_ids = set()
for fn in challenge_jsonls:
    p = Path(fn)
    if not p.exists():
        continue
    with open(p) as fh:
        for line in fh:
            sid = json.loads(line).get('session_id')
            if sid:
                eval_session_ids.add(sid)

eval_files = []
dev_stems = {f.stem for f in dev_files}
for sid in sorted(eval_session_ids):
    for rel in _relative_audio_paths_from_session_id(sid):
        cand = eval_base / rel
        if cand.exists() and cand.stem not in dev_stems:
            eval_files.append(cand)
            break

all_files = dev_files + eval_files  # training/dev files first
with open(out_path, 'w') as fh:
    for f in all_files:
        fh.write(str(f) + '\n')

total_dur = sum(sf.info(str(f)).duration for f in all_files)
print(f'File list written: {len(all_files)} files  ({total_dur/3600:.1f}h audio)  -> {out_path}')
print(f'  Dev (training): {len(dev_files)}')
print(f'  Eval-only (challenge): {len(eval_files)}')
print(f'  Session IDs with no resolved file: {len(eval_session_ids) - len(eval_files)}')
PYEOF
"

echo "Building file list..."
singularity exec \
    --nv --writable \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind /gpfs:/gpfs \
    --bind $MLC_DATA_DIR:/mnt/mirror1/datasets/MLC-SLM_Workshop_2026 \
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "$BUILD_LIST_CMD"

echo "File list ready. Launching 4 GPU workers..."
echo ""

# --- Per-GPU transcription command ------------------------------------------
make_gpu_cmd() {
  local GPU_ID=$1
  cat <<CMDEOF
set -euo pipefail

export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCH_EXTENSIONS_DIR=$TORCH_EXT_DIR
export PYTHONUNBUFFERED=1
export TMPDIR=$TMP_ROOT
export CUDA_VISIBLE_DEVICES=${GPU_ID}

. /opt/asrenv/bin/activate
cd $REPO_DIR

echo "[GPU ${GPU_ID}] Starting shard ${GPU_ID}/4"
python -m asr_merging.transcribe_sessions \
  --file-list     data/transcripts/.file_list_all.txt \
  --checkpoint-path experiments/mlc25_mlc26_continue_4gpu_fixed_bothsides_20260530_120641/final_model \
  --output-dir    data/transcripts \
  --language      en \
  --max-new-tokens 8192 \
  --max-chunk-minutes 29 \
  --skip-existing \
  --shard-index   ${GPU_ID} \
  --num-shards    4
echo "[GPU ${GPU_ID}] Shard ${GPU_ID} done."
CMDEOF
}

# --- Launch 4 parallel workers ----------------------------------------------
for GPU_ID in 0 1 2 3; do
  CMD=$(make_gpu_cmd $GPU_ID)
  singularity exec \
      --nv --writable \
      --bind $CACHE_DIR:$CACHE_DIR \
      --bind $TORCH_EXT_DIR:$TORCH_EXT_DIR \
      --bind $SANDBOX_DIR:$SANDBOX_DIR \
      --bind /gpfs:/gpfs \
      --bind $MLC_DATA_DIR:/mnt/mirror1/datasets/MLC-SLM_Workshop_2026 \
      --pwd $SANDBOX_DIR \
      $PATH_SINGULARITY \
      bash -c "$CMD" \
    > "./slurm_logs/voxtral_transcribe_all_${SLURM_JOB_ID}_gpu${GPU_ID}.log" 2>&1 &
  echo "  GPU ${GPU_ID}: PID $!"
done

echo ""
echo "Waiting for all 4 GPU workers to finish..."
wait
echo ""
echo "All workers done."

# --- Summary ----------------------------------------------------------------
singularity exec \
    --nv --writable \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind /gpfs:/gpfs \
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "
. /opt/asrenv/bin/activate
cd $REPO_DIR
python3 - <<'PYEOF'
from pathlib import Path
import json

out = Path('data/transcripts')
txts = sorted(out.glob('*.txt'))
jsons = sorted(out.glob('*.json'))

total_words = 0
errors = []
for jf in jsons:
    try:
        d = json.loads(jf.read_text())
        if 'error' in d:
            errors.append(jf.stem)
        else:
            total_words += d.get('word_count', 0)
    except Exception:
        pass

print(f'=== Transcription summary ===')
print(f'Transcripts written : {len(txts)}')
print(f'Total words         : {total_words:,}')
print(f'Errors              : {len(errors)}')
if errors:
    for e in errors: print(f'  ERROR: {e}')
PYEOF
"

echo ""
echo "=========================================="
echo "Job finished at $(date)"
echo "Transcripts: $SANDBOX_DIR/opt/ASR-merging/data/transcripts/"
echo "GPU logs   : ./slurm_logs/voxtral_transcribe_all_${SLURM_JOB_ID}_gpu*.log"
echo "=========================================="
