#!/bin/bash
#SBATCH --job-name=voxtral_fmt_cmp
#SBATCH --output=./slurm_logs/voxtral_fmt_cmp_%j.out
#SBATCH --error=./slurm_logs/voxtral_fmt_cmp_%j.err
#SBATCH --account=ehpc628
#SBATCH -A ehpc628
#SBATCH --qos=acc_ehpc
#SBATCH --partition=acc
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

set -euo pipefail

echo "=========================================="
echo "MareNostrum 5 - Acceleration Partition"
echo "Format comparison eval: chat_template vs transcription_request"
echo "Model: v2_mixed final_model"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"

module purge
module load singularity

export PATH_SINGULARITY="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SANDBOX_DIR="/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm"
export SCRATCH_DIR="/gpfs/scratch/ehpc628/jls/ehpc628XXX"
export REPO_DIR="/opt/ASR-merging"
export CACHE_DIR="/gpfs/scratch/ehpc628/models"
export TORCH_EXT_DIR="/gpfs/scratch/ehpc628/jls/torch_extensions"
export MLC_DATA_DIR="/gpfs/scratch/ehpc494/jls/datasets"
RUN_TAG="${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)"
export TMP_ROOT="/gpfs/scratch/ehpc628/jls/asr_merging_runs/${RUN_TAG}/tmp"

mkdir -p ./slurm_logs
mkdir -p "$CACHE_DIR"
mkdir -p "$TMP_ROOT"

export CMD="
set -euo pipefail

export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export TMPDIR=$TMP_ROOT
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0

nvidia-smi

cd $REPO_DIR

. /opt/asrenv/bin/activate
python -V

python - <<'PY'
import sys, json, torch, random
sys.path.insert(0, '/opt/ASR-merging')
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor
from asr_merging.voxtral_train_router import _resolve_pretrained_source, _offline_aware_from_pretrained_kwargs
from peft import PeftModel
import soundfile as sf

random.seed(42)

MODEL_ID  = 'mistralai/Voxtral-Mini-3B-2507'
ADAPTER   = '/opt/ASR-merging/experiments/voxtral_mcq_v2_mixed_bs32_4gpu_20260620_151902/final_model'
EVAL_FILE = '/opt/ASR-merging/data/mlc26_task2/eval_26files_challenge_repr.jsonl'
AUDIO_DIR = '/opt/ASR-merging/data/mlc26_task2'   # paths in jsonl include mlc-slm-2nd-dev/
N_SAMPLE  = None   # None = all questions (1110 total)

src = _resolve_pretrained_source(MODEL_ID)
processor = VoxtralProcessor.from_pretrained(src)
tok = processor.tokenizer
kw = _offline_aware_from_pretrained_kwargs({'torch_dtype': torch.float16, 'device_map': {'': 0}})
base = VoxtralForConditionalGeneration.from_pretrained(src, **kw)
model = PeftModel.from_pretrained(base, ADAPTER, is_trainable=False)
model.eval()

# Build question list
records = []
with open(EVAL_FILE) as f:
    for line in f:
        r = json.loads(line)
        audio_path = AUDIO_DIR + '/' + r['path']
        lang = r.get('language', '')
        for q in r.get('questions', []):
            opts = ' '.join(f\"{o['label']}. {o['text']}\" for o in q.get('options', []))
            prompt_text = q['question_stem'] + '\n' + opts
            records.append({
                'audio_path': audio_path,
                'prompt_text': prompt_text,
                'gold': q.get('correct_answer', ''),
                'language': lang,
            })

if N_SAMPLE is not None:
    records = random.sample(records, min(N_SAMPLE, len(records)))

print(f'Evaluating {len(records)} questions ...')

def eval_format(rec):
    \"\"\"Current eval format: question inside [INST] via apply_chat_template.\"\"\"
    conv = [{'role': 'user', 'content': [
        {'type': 'audio', 'path': rec['audio_path']},
        {'type': 'text',  'text': rec['prompt_text']},
    ]}]
    inputs = processor.apply_chat_template(conv, tokenize=True, return_tensors='pt', return_dict=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

def train_format(rec):
    \"\"\"Training format: apply_transcription_request + question appended after [/INST].\"\"\"
    audio, sr = sf.read(rec['audio_path'], dtype='float32', always_2d=False)
    prompt = processor.apply_transcription_request(
        language='en', model_id=MODEL_ID,
        audio=[audio], format=['WAV'], sampling_rate=sr, return_tensors='pt')
    # Append question tokens exactly as done by VoxtralMCQCollator in training
    q_ids = tok(rec['prompt_text'] + '\n-> ', add_special_tokens=False, return_tensors='pt')['input_ids']
    input_ids = torch.cat([prompt['input_ids'], q_ids], dim=1)
    attn_mask  = torch.cat([prompt['attention_mask'], torch.ones_like(q_ids)], dim=1)
    inputs = {
        'input_ids':       input_ids.to(model.device),
        'attention_mask':  attn_mask.to(model.device),
        'input_features':  prompt['input_features'].to(model.device),
    }
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    return tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

def label(pred):
    for c in pred:
        if c in 'ABCD': return c
    return '?'

n_e = n_t = 0
rows = []
for i, rec in enumerate(records):
    ep = label(eval_format(rec))
    tp = label(train_format(rec))
    g  = rec['gold']
    n_e += (ep == g)
    n_t += (tp == g)
    rows.append({'language': rec['language'], 'eval_pred': ep, 'train_pred': tp, 'gold': g})
    if (i+1) % 50 == 0:
        print(f'  [{i+1:4d}/{len(records)}]  eval_fmt={n_e/(i+1):.4f}  train_fmt={n_t/(i+1):.4f}')

print()
print(f'=== FINAL RESULTS ({len(records)} questions) ===')
print(f'  eval format  (chat_template):         {n_e}/{len(records)} = {n_e/len(records):.4f}')
print(f'  train format (transcription_request): {n_t}/{len(records)} = {n_t/len(records):.4f}')

# Per-language breakdown
from collections import defaultdict
lang_e = defaultdict(lambda: [0,0])
lang_t = defaultdict(lambda: [0,0])
for row in rows:
    lang_e[row['language']][0] += (row['eval_pred'] == row['gold'])
    lang_e[row['language']][1] += 1
    lang_t[row['language']][0] += (row['train_pred'] == row['gold'])
    lang_t[row['language']][1] += 1

print()
print(f'=== PER-LANGUAGE (eval_fmt vs train_fmt) ===')
for lang in sorted(lang_e):
    ce, ne = lang_e[lang]; ct, nt = lang_t[lang]
    print(f'  {lang:30s}  eval={ce/ne:.3f} ({ce}/{ne})  train={ct/nt:.3f} ({ct}/{nt})')

# Save predictions
import os
out_dir = '/opt/ASR-merging/experiments/ablation_format_comparison'
os.makedirs(out_dir, exist_ok=True)
with open(f'{out_dir}/predictions.jsonl', 'w') as f:
    for row in rows:
        f.write(json.dumps(row) + '\n')
with open(f'{out_dir}/metrics.json', 'w') as f:
    json.dump({
        'eval_format_accuracy':  n_e/len(records),
        'train_format_accuracy': n_t/len(records),
        'n_total': len(records),
    }, f, indent=2)
print(f'\nSaved to {out_dir}/')
PY

echo 'Format comparison eval completed successfully'
"

echo "Starting Singularity container..."
singularity exec \
    --nv \
    --writable \
    --bind $CACHE_DIR:$CACHE_DIR \
    --bind $TORCH_EXT_DIR:$TORCH_EXT_DIR \
    --bind $SANDBOX_DIR:$SANDBOX_DIR \
    --bind $SCRATCH_DIR:$SCRATCH_DIR \
    --bind /gpfs:/gpfs \
    --bind $MLC_DATA_DIR:/mnt/mirror1/datasets/MLC-SLM_Workshop_2026 \
    --pwd $SANDBOX_DIR \
    $SANDBOX_DIR \
    bash -c "$CMD"
