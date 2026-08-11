"""Class distribution comparison: fine-tuned variants vs voxtral-24B reference."""
import collections, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = "/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm/opt/ASR-merging/experiments"

FILES = {
    "24B\nicl6-mm\n(best LB)": f"{BASE}/voxtral_ootb_voxtral-small-24b-2507_phase2_eval/challenge_phase2_orig_notx_mnt128_icl6_mm_hyp.txt",
    "frzenc\n5e5\nen":         f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_4gpu_20260628_004742/challenge_phase2_orig_notx_icl6_mm_hyp.txt",
    "frzenc\n5e5\nnon-en":     f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_4gpu_20260628_004742/challenge_phase2_orig_notx_icl6_mm_nonen_hyp.txt",
    "frzenc\n1e5\nen":         f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_4gpu_20260628_004740/challenge_phase2_orig_notx_icl6_mm_hyp.txt",
    "frzenc\n1e5\nnon-en":     f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_4gpu_20260628_004740/challenge_phase2_orig_notx_icl6_mm_nonen_hyp.txt",
    "frzenc-only\n5e5\nen":    f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_only_4gpu_20260628_004842/challenge_phase2_orig_notx_icl6_mm_hyp.txt",
    "frzenc-only\n5e5\nnon-en":f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_only_4gpu_20260628_004842/challenge_phase2_orig_notx_icl6_mm_nonen_hyp.txt",
    "frzenc-only\n1e5\nen":    f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_4gpu_20260628_005302/challenge_phase2_orig_notx_icl6_mm_hyp.txt",
    "frzenc-only\n1e5\nnon-en":f"{BASE}/voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_4gpu_20260628_005302/challenge_phase2_orig_notx_icl6_mm_nonen_hyp.txt",
}

CLASSES = list("ABCD")
COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

data = {}
for label, path in FILES.items():
    if not os.path.exists(path):
        print(f"MISSING: {path}"); continue
    counts = collections.Counter()
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                counts[parts[-1]] += 1
    total = sum(counts.values())
    data[label] = {c: counts.get(c, 0) / total * 100 for c in CLASSES}

names = list(data.keys())
n = len(names)
x = np.arange(n)
width = 0.18

fig, ax = plt.subplots(figsize=(14, 5))

for i, (cls, color) in enumerate(zip(CLASSES, COLORS)):
    vals = [data[nm][cls] for nm in names]
    bars = ax.bar(x + (i - 1.5) * width, vals, width, label=f"Class {cls}", color=color, alpha=0.85)

# Reference uniform line at 25%
ax.axhline(25, color="gray", linestyle="--", linewidth=0.8, label="Uniform (25%)")

# Highlight 24B reference column
ax.axvspan(-0.5, 0.5, alpha=0.06, color="gold", zorder=0)
ax.text(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 65,
        "★ ref", ha="center", va="bottom", fontsize=8, color="goldenrod")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("% of predictions")
ax.set_title("Predicted Class Distribution — Phase 2 Challenge\nFine-tuned 3B variants vs. Voxtral-24B (best LB)", fontsize=11)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 72)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Annotate A% on top of each A bar
for i, nm in enumerate(names):
    a_pct = data[nm]["A"]
    ax.text(x[i] + (0 - 1.5) * width, a_pct + 0.8, f"{a_pct:.0f}%",
            ha="center", va="bottom", fontsize=6.5, color=COLORS[0], fontweight="bold")

plt.tight_layout()
out = "/gpfs/projects/ehpc628/jls/singularity_containers/flower_speech_llm/opt/ASR-merging/analysis_outputs/class_dist_phase2_20260628.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
