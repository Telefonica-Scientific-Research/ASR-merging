"""Plot train/eval loss for the two balanced 12-epoch resumed runs (2026-06-28).
Reads from the latest checkpoint's trainer_state.json."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("experiments")

RUNS = [
    (
        "voxtral_mcq_nllb_icl_crop30_frzenc_lr1e5_bal_12ep_4gpu_20260628_100526",
        "frzenc_lr1e5_bal 12ep\n(enc+conn frozen, LLM trainable, balanced, 1e-5)",
        "steelblue",
    ),
    (
        "voxtral_mcq_nllb_icl_crop30_frzenc_only_lr1e5_bal_12ep_4gpu_20260628_100501",
        "frzenc_only_lr1e5_bal 12ep\n(enc frozen, conn+LLM trainable, balanced, 1e-5)",
        "darkorange",
    ),
]

SMOOTH_WIN = 5


def smooth(values, w):
    if len(values) < w:
        return np.array(values)
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode="same")


def load_latest_state(exp_dir: Path):
    """Return log_history from the highest-step checkpoint-* (full history).
    final_model is a copy of best_gen_gap_checkpoint and has truncated history."""
    candidates = sorted(
        exp_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    if not candidates:
        return None, None
    chosen = candidates[-1]
    state = json.loads(chosen.read_text())
    # If final_model exists, training is done — report total steps from it
    final = exp_dir / "final_model" / "trainer_state.json"
    if final.exists():
        final_state = json.loads(final.read_text())
        total_steps = int(chosen.parent.name.split("-")[1])
    else:
        total_steps = state.get("global_step")
    return state["log_history"], total_steps


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "Balanced 12-epoch resumed runs — Train & Eval Loss  (2026-06-28)",
    fontsize=13,
    y=1.02,
)

for ax, (exp_name, label, color) in zip(axes, RUNS):
    log_history, global_step = load_latest_state(BASE / exp_name)

    if log_history is None:
        ax.set_title(label, fontsize=9)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        continue

    # Separate train and eval entries
    train_entries = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_entries  = [e for e in log_history if "eval_loss" in e]

    t_steps = [e["step"] for e in train_entries]
    t_loss  = [e["loss"] for e in train_entries]
    e_steps = [e["step"] for e in eval_entries]
    e_loss  = [e["eval_loss"] for e in eval_entries]

    t_smooth = smooth(t_loss, SMOOTH_WIN)

    ax.plot(t_steps, t_smooth, color=color, linewidth=1.5, label="train (smoothed)")
    ax.scatter(t_steps, t_loss, color=color, alpha=0.25, s=10)
    ax.plot(e_steps, e_loss, color="crimson", linewidth=2.0, label="eval", marker="o", markersize=5)

    # Mark best eval
    if e_loss:
        best_idx = int(np.argmin(e_loss))
        ax.axvline(e_steps[best_idx], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.annotate(
            f"best eval={e_loss[best_idx]:.4f}\n@step {e_steps[best_idx]}",
            xy=(e_steps[best_idx], e_loss[best_idx]),
            xytext=(14, 8),
            textcoords="offset points",
            fontsize=8,
            color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
        )

    # Epoch boundaries (50 steps/epoch in balanced runs)
    steps_per_epoch = 50
    max_step = t_steps[-1] if t_steps else (e_steps[-1] if e_steps else 0)
    for ep in range(1, 13):
        ep_step = ep * steps_per_epoch
        if ep_step <= max_step + steps_per_epoch:
            ax.axvline(ep_step, color="lightgray", linestyle=":", linewidth=0.6, alpha=0.8)
            ax.text(ep_step, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 0.8,
                    f"ep{ep}", fontsize=6, color="gray", ha="center", va="bottom")

    ax.set_title(label, fontsize=9)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("loss", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Progress annotation
    total_steps = 12 * steps_per_epoch  # = 600
    pct = 100 * (global_step or 0) / total_steps
    ax.text(
        0.02, 0.97,
        f"current step: {global_step}  ({pct:.1f}% of 600)\nbest eval: {min(e_loss):.4f} @step {e_steps[int(np.argmin(e_loss))]}",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

plt.tight_layout()
out = Path("analysis_outputs/bal12ep_20260628.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
