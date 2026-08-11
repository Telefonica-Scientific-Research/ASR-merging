"""Plot train/eval loss for the 6 ICL-crop30 experiments from 2026-06-28."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("experiments")

EXPERIMENTS = [
    ("voxtral_mcq_nllb_icl_crop30_frzenc_4gpu_20260628_004742",
     "frzenc (5e-5)\nenc+conn frozen, LLM trainable"),
    ("voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_4gpu_20260628_004740",
     "frzenc_lr1e5 (1e-5)\nenc+conn frozen, LLM trainable"),
    ("voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_bal_4gpu_20260628_004749",
     "frzenc_lr1e5_bal (1e-5)\nenc+conn frozen, LLM trainable, balanced"),
    ("voxtral_mcq_nllb_icl_crop30_frzenc_only_4gpu_20260628_004842",
     "frzenc_only (5e-5)\nenc frozen, conn+LLM trainable"),
    ("voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_4gpu_20260628_005302",
     "frzenc_only_lr1e5 (1e-5)\nenc frozen, conn+LLM trainable"),
    ("voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_bal_4gpu_20260628_010215",
     "frzenc_only_lr1e5_bal (1e-5)\nenc frozen, conn+LLM trainable, balanced"),
]

SMOOTH_WIN = 5

def smooth(values, w):
    if len(values) < w:
        return np.array(values)
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='same')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("ICL crop30 — 6 ICL shots, random-crop 300s  |  Train & Eval Loss  (2026-06-28)",
             fontsize=13, y=1.02)

for ax, (exp_name, label) in zip(axes.flat, EXPERIMENTS):
    log_file = BASE / exp_name / "training_ckpt_log.jsonl"
    if not log_file.exists():
        ax.set_title(label, fontsize=8)
        ax.text(0.5, 0.5, 'no log', ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        continue

    entries = [json.loads(l) for l in log_file.read_text().strip().split('\n') if l.strip()]
    if not entries:
        ax.text(0.5, 0.5, 'empty log', ha='center', va='center', transform=ax.transAxes)
        continue

    steps      = [e['global_step'] for e in entries]
    train_loss = [e['train_loss'] for e in entries]
    eval_loss  = [e['eval_loss'] for e in entries]
    gen_gap    = [e.get('generalization_gap', 0) for e in entries]

    train_smooth = smooth(train_loss, SMOOTH_WIN)

    ax.plot(steps, train_smooth, color='steelblue', linewidth=1.5, label='train (smoothed)')
    ax.scatter(steps, train_loss, color='steelblue', alpha=0.3, s=12)
    ax.plot(steps, eval_loss, color='coral', linewidth=1.8, label='eval')
    ax.scatter(steps, eval_loss, color='coral', s=20, zorder=3)

    # mark best eval
    best_idx = int(np.argmin(eval_loss))
    ax.axvline(steps[best_idx], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.annotate(f"best eval={eval_loss[best_idx]:.4f}\n@step {steps[best_idx]}",
                xy=(steps[best_idx], eval_loss[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=7, color='coral',
                arrowprops=dict(arrowstyle='->', color='coral', lw=0.8))

    ax.set_title(label, fontsize=8.5)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("loss", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # summary text
    final_train = train_loss[-1]
    final_eval  = eval_loss[-1]
    ax.text(0.02, 0.05,
            f"final train={final_train:.4f}\nfinal eval={final_eval:.4f}\n"
            f"steps={steps[-1]}",
            transform=ax.transAxes, fontsize=7, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))

plt.tight_layout()
out = Path("analysis_outputs/icl_crop30_20260628_loss.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=140, bbox_inches='tight')
print(f"Saved → {out}")
