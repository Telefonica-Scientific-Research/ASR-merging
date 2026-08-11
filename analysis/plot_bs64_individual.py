import json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BS64_TYPES = [
    ("voxtral_mcq_bs64_128mixed_bal_mildw_4gpu",  "128mixed_bal_mildw"),
    ("voxtral_mcq_bs64_128mixed_bal_trx_4gpu",    "128mixed_bal_trx"),
    ("voxtral_mcq_bs64_128mixed_mildw_4gpu",      "128mixed_mildw"),
    ("voxtral_mcq_bs64_128mixed_trx_4gpu",        "128mixed_trx"),
    ("voxtral_mcq_bs64_128noen_bal_mildw_4gpu",   "128noen_bal_mildw"),
    ("voxtral_mcq_bs64_128noen_bal_trx_4gpu",     "128noen_bal_trx"),
    ("voxtral_mcq_bs64_128noen_mildw_4gpu",       "128noen_mildw"),
    ("voxtral_mcq_bs64_128noen_trx_4gpu",         "128noen_trx"),
    ("voxtral_mcq_bs64_mixed_bal_mildw_4gpu",     "mixed_bal_mildw"),
    ("voxtral_mcq_bs64_mixed_bal_trx_4gpu",       "mixed_bal_trx"),
    ("voxtral_mcq_bs64_mixed_mildw_4gpu",         "mixed_mildw"),
    ("voxtral_mcq_bs64_mixed_trx_4gpu",           "mixed_trx"),
    ("voxtral_mcq_bs64_noen_bal_mildw_4gpu",      "noen_bal_mildw"),
    ("voxtral_mcq_bs64_noen_bal_trx_4gpu",        "noen_bal_trx"),
    ("voxtral_mcq_bs64_noen_mildw_4gpu",          "noen_mildw"),
    ("voxtral_mcq_bs64_noen_trx_4gpu",            "noen_trx"),
]

SMOOTH_WIN = 5  # rolling average window for train loss

def smooth(values, w):
    if len(values) < w:
        return values
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='same')

fig, axes = plt.subplots(4, 4, figsize=(22, 18))
fig.suptitle("bs64 15-epoch — Train & Eval Loss (individual)", fontsize=14, y=1.01)

for ax, (typ, label) in zip(axes.flat, BS64_TYPES):
    # Find latest 20260624 run
    runs = sorted(Path('experiments').glob(f'{typ}_20260624*'))
    if not runs:
        runs = sorted(Path('experiments').glob(f'{typ}_202606*'))
    if not runs:
        ax.set_title(label, fontsize=8)
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        continue

    exp = runs[-1]
    log_file = exp / 'training_ckpt_log.jsonl'
    if not log_file.exists():
        ax.set_title(label, fontsize=8)
        ax.text(0.5, 0.5, 'no log', ha='center', va='center', transform=ax.transAxes)
        continue

    entries = [json.loads(l) for l in log_file.read_text().strip().split('\n') if l.strip()]
    if not entries:
        continue

    steps      = [e['global_step'] for e in entries]
    eval_loss  = [e['eval_loss'] for e in entries]
    train_key  = 'avg_train_loss_since_last_eval' if 'avg_train_loss_since_last_eval' in entries[0] else 'train_loss'
    train_loss = [e.get(train_key, float('nan')) for e in entries]
    gen_gaps   = [e.get('generalization_gap', float('nan')) for e in entries]

    # Best gen_gap index
    best_gap_idx = int(np.nanargmin(gen_gaps))
    # Best eval_loss index
    best_eval_idx = int(np.nanargmin(eval_loss))

    # Skip first point for y-axis range (spike removal)
    skip = 1 if len(entries) > 3 else 0
    eval_clipped  = eval_loss[skip:]
    train_clipped = [t for t in train_loss[skip:] if not np.isnan(t)]

    # Y-axis: clip to 98th percentile of non-first values
    all_vals = eval_clipped + train_clipped
    if all_vals:
        y_max = min(np.percentile(all_vals, 98) * 1.15, max(all_vals) * 1.05)
        y_min = max(0, min(all_vals) * 0.92)
    else:
        y_max, y_min = 1.0, 0.0

    # Smooth train loss
    train_smooth = smooth(np.array(train_loss, dtype=float), SMOOTH_WIN)

    # Plot raw train (faint) and smoothed train
    ax.plot(steps, train_loss, color='steelblue', alpha=0.2, linewidth=0.8)
    ax.plot(steps, train_smooth, color='steelblue', linewidth=1.5, label='train (smooth)')
    ax.plot(steps, eval_loss,   color='darkorange', linewidth=1.5, label='eval')

    # Mark best gen_gap
    ax.axvline(steps[best_gap_idx], color='green', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.scatter([steps[best_gap_idx]], [eval_loss[best_gap_idx]],
               color='green', s=50, zorder=5,
               label=f'best gap={gen_gaps[best_gap_idx]:.4f}@{steps[best_gap_idx]}')

    # Mark best eval_loss (only if different from best_gap)
    if best_eval_idx != best_gap_idx:
        ax.scatter([steps[best_eval_idx]], [eval_loss[best_eval_idx]],
                   color='red', marker='*', s=80, zorder=5,
                   label=f'best eval={eval_loss[best_eval_idx]:.4f}@{steps[best_eval_idx]}')

    ax.set_ylim(y_min, y_max)
    ax.set_title(label, fontsize=8, fontweight='bold')
    ax.set_xlabel('step', fontsize=7)
    ax.set_ylabel('loss', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path('analysis_outputs/bs64_15ep_individual.png')
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"Saved: {out}")
