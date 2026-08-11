#!/bin/bash
#SBATCH --job-name=voxtral_transcribe_retry
#SBATCH --output=./slurm_logs/voxtral_transcribe_retry_%j.out
#SBATCH --error=./slurm_logs/voxtral_transcribe_retry_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Second-pass transcription for files where the first round produced
# hallucination loops (5-gram repetition detected in the existing .txt).
#
# Strategy:
#   - Detect looped transcripts by scanning data/transcripts/*.json for
#     audio_path, then checking the corresponding .txt for 5-gram loops.
#   - Re-transcribe ONLY those files using 1-minute chunks + 2-second overlap.
#     Short chunks prevent the filler-word attractor from filling 8192 tokens;
#     2s overlap gives the model audio context at each chunk boundary.
#   - 4 GPUs, round-robin sharding; no --skip-existing (overwrites bad files).
#
# Timing estimate (based on 41.7× RT on H100):
#   ~88 files × 21 min avg = ~1848 min audio / 4 GPUs / 35× RT ≈ 13 min wall
#   (latency per chunk is higher for 1-min clips; budget ~45 min wall time)
#
# Output: same data/transcripts/{stem}.txt + .json (overwrites looped files)

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 — Voxtral ASR Retry Transcription (1-min chunks, 2s overlap)"
echo "Target  : looped transcripts in data/transcripts/"
echo "GPUs    : 4 parallel, 1-min chunks, 2s overlap"
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
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/transcribe_retry_${RUN_TAG}/tmp"

mkdir -p ./slurm_logs "$CACHE_DIR" "$TORCH_EXT_DIR" "$TMP_ROOT"

# --- Build file list: only audio files whose transcript is looped -----------
export BUILD_LIST_CMD="
set -euo pipefail
. /opt/asrenv/bin/activate

cd $REPO_DIR
python3 - <<'PYEOF'
import json
from collections import Counter
from pathlib import Path

# Sliding-window density check: same logic as _load_transcript in voxtral_train_MCQ.py.
# A real loop fills a 150-word window; a natural repeated phrase appears <=2x per window.
NGRAM, WIN, MIN_REPS = 5, 150, 5

def is_looped(text: str) -> bool:
    words = text.split()
    if len(words) < NGRAM + 1:
        return False
    step = max(1, WIN // 3)
    for ws in range(0, max(1, len(words) - WIN + 1), step):
        ww = words[ws: ws + WIN]
        ng = [' '.join(ww[i:i+NGRAM]) for i in range(len(ww) - NGRAM)]
        cnt = Counter(ng)
        if cnt and cnt.most_common(1)[0][1] > MIN_REPS:
            return True
    return False

out_path = Path('data/transcripts/.file_list_retry.txt')
transcript_dir = Path('data/transcripts')

looped = []
total_jsons = 0
for jf in sorted(transcript_dir.glob('*.json')):
    total_jsons += 1
    try:
        meta = json.loads(jf.read_text())
    except Exception:
        continue
    audio_path = meta.get('audio_path', '')
    if not audio_path:
        continue
    txt_path = transcript_dir / (jf.stem + '.txt')
    if not txt_path.exists():
        continue
    text = txt_path.read_text(encoding='utf-8').strip()
    if is_looped(text):
        # Verify the audio file actually exists
        if Path(audio_path).exists():
            looped.append(audio_path)
        else:
            print(f'  WARNING: audio not found: {audio_path}')

with open(out_path, 'w') as fh:
    for p in looped:
        fh.write(p + '\n')

print(f'Scanned {total_jsons} JSON files; found {len(looped)} looped transcripts.')
print(f'Retry file list written -> {out_path}')
for p in looped:
    print(f'  {Path(p).name}')
PYEOF
"

echo "Scanning existing transcripts for hallucination loops..."
singularity exec \
    --nv --writable \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind /gpfs:/gpfs \
    --bind $MLC_DATA_DIR:/mnt/mirror1/datasets/MLC-SLM_Workshop_2026 \
    --pwd $SANDBOX_DIR \
    $PATH_SINGULARITY \
    bash -c "$BUILD_LIST_CMD"

echo "Retry file list ready. Launching 4 GPU workers..."
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

echo "[GPU ${GPU_ID}] Starting retry shard ${GPU_ID}/4"
python -m asr_merging.transcribe_sessions \
  --file-list            data/transcripts/.file_list_retry.txt \
  --checkpoint-path      experiments/mlc25_mlc26_continue_4gpu_fixed_bothsides_20260530_120641/final_model \
  --output-dir           data/transcripts \
  --language             en \
  --max-new-tokens       300 \
  --max-chunk-minutes    1 \
  --chunk-overlap-seconds 2 \
  --shard-index          ${GPU_ID} \
  --num-shards           4
echo "[GPU ${GPU_ID}] Retry shard ${GPU_ID} done."
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
    > "./slurm_logs/voxtral_transcribe_retry_${SLURM_JOB_ID}_gpu${GPU_ID}.log" 2>&1 &
  echo "  GPU ${GPU_ID}: PID $!"
done

echo ""
echo "Waiting for all 4 GPU workers to finish..."
wait
echo ""
echo "All workers done."

# --- Post-run summary -------------------------------------------------------
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
import json
from collections import Counter
from pathlib import Path

NGRAM, WIN, MIN_REPS = 5, 150, 5

def is_looped(text):
    words = text.split()
    if len(words) < NGRAM + 1:
        return False
    step = max(1, WIN // 3)
    for ws in range(0, max(1, len(words) - WIN + 1), step):
        ww = words[ws: ws + WIN]
        ng = [' '.join(ww[i:i+NGRAM]) for i in range(len(ww) - NGRAM)]
        cnt = Counter(ng)
        if cnt and cnt.most_common(1)[0][1] > MIN_REPS:
            return True
    return False

retry_list = Path('data/transcripts/.file_list_retry.txt')
if not retry_list.exists():
    print('No retry list found.')
else:
    audio_paths = [l.strip() for l in retry_list.read_text().splitlines() if l.strip()]
    still_looped = []
    fixed = []
    missing = []
    for ap in audio_paths:
        stem = Path(ap).stem
        txt = Path('data/transcripts') / (stem + '.txt')
        if not txt.exists():
            missing.append(stem)
            continue
        text = txt.read_text(encoding='utf-8').strip()
        if is_looped(text):
            still_looped.append(stem)
        else:
            wc = len(text.split())
            fixed.append((stem, wc))
    print()
    print('=== Retry Summary ===')
    print(f'Total retried  : {len(audio_paths)}')
    print(f'Fixed (no loop): {len(fixed)}')
    print(f'Still looped   : {len(still_looped)}')
    print(f'Missing output : {len(missing)}')
    if still_looped:
        print()
        print('Still looped:')
        for s in still_looped:
            print(f'  {s}')
    if missing:
        print()
        print('Missing output (likely errored):')
        for s in missing:
            print(f'  {s}')
    print('=====================')
PYEOF
"

echo ""
echo "Retry transcription complete. Check slurm_logs/voxtral_transcribe_retry_${SLURM_JOB_ID}_gpu*.log"
echo "Finished at: $(date)"
