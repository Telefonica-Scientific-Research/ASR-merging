import json, re
from pathlib import Path

TARGET_TYPES = [
    "voxtral_mcq_mixed_bal_transcript_4gpu",
    "voxtral_mcq_mixed_bal_weighted_4gpu",
    "voxtral_mcq_mixed_transcript_4gpu",
    "voxtral_mcq_mixed_weighted_4gpu",
    "voxtral_mcq_128mixed_bal_transcript_4gpu",
    "voxtral_mcq_128mixed_bal_weighted_4gpu",
    "voxtral_mcq_128mixed_transcript_4gpu",
    "voxtral_mcq_128mixed_weighted_4gpu",
    "voxtral_mcq_128noen_bal_transcript_4gpu",
    "voxtral_mcq_128noen_bal_weighted_4gpu",
    "voxtral_mcq_128noen_transcript_4gpu",
    "voxtral_mcq_128noen_weighted_4gpu",
    "voxtral_mcq_noen_mixed_bal_transcript_4gpu",
    "voxtral_mcq_noen_mixed_bal_weighted_4gpu",
    "voxtral_mcq_noen_mixed_transcript_4gpu",
    "voxtral_mcq_noen_mixed_weighted_4gpu",
]

print(f"{'Experiment':<44} {'Steps':>6} {'eval_loss':>9} {'train_loss':>10} {'best_gap':>10} {'@step':>6}  {'#runs':>5}  timestamp")
print('-'*115)

for typ in TARGET_TYPES:
    runs = sorted(Path('experiments').glob(f'{typ}_202606*'))
    n_runs = len(runs)
    if not runs:
        name = typ.replace('voxtral_mcq_','').replace('_4gpu','')
        print(f"{name:<44}  NO RUNS")
        continue
    exp = runs[-1]
    log_file = exp / 'training_ckpt_log.jsonl'
    name = typ.replace('voxtral_mcq_','').replace('_4gpu','')
    ts_match = re.search(r'(\d{8}_\d{6})$', exp.name)
    ts = ts_match.group(1) if ts_match else '?'
    if not log_file.exists():
        print(f"{name:<44}  no log — {ts}")
        continue
    raw = [l for l in log_file.read_text().strip().split('\n') if l.strip()]
    if not raw:
        print(f"{name:<44}  empty log — {ts}")
        continue
    entries = [json.loads(l) for l in raw]
    last = entries[-1]
    train_key = 'avg_train_loss_since_last_eval' if 'avg_train_loss_since_last_eval' in last else 'train_loss'
    best = min(entries, key=lambda x: abs(x.get('generalization_gap', 999)))
    gap = best['generalization_gap']
    gap_step = best['global_step']
    print(f"{name:<44} {last['global_step']:>6} {last['eval_loss']:>9.4f} {last.get(train_key,0):>10.4f} {gap:>10.4f} {gap_step:>6}  {n_runs:>5}  {ts}")
