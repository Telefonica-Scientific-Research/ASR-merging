#!/usr/bin/env python3
"""Analyze cyclic-permutation debias detail files.

For each question the detail row has:
  - pred_choice: debiased argmax (original-option label of the winning content)
  - n_options:   number of options (2/3/4)
  - content_probs: {label -> aggregated prob across rotations} for the content
                   originally at that label position
  - per_rotation_logprobs: list (len n) of {slot_label -> logprob}; in rotation r
                   slot j holds content (j+r) % n  (mapping from the predictor).

Key diagnostics:
  * Raw (rotation-0) prediction = model's natural pick with original order
    (rotation 0 is identity: slot j == content j). This is the PRE-debias pred.
  * Debiased prediction = pred_choice (argmax content_probs).
  * Slot bias = how often / how strongly the model picks slot A regardless of
    the content rotated into it (averaging over rotations isolates pure position).
  * Permutation consistency = does the SAME content win across all rotations?
    Pure content reasoning -> consistent; pure slot bias -> picks a different
    content each rotation (whatever lands in slot A) -> 0% consistent.
"""
import json
import sys
from collections import Counter

LABELS = ["A", "B", "C", "D"]


def argmax_slot(rot: dict, n: int):
    """Return slot index (0..n-1) with max logprob among the first n labels."""
    best_i, best_v = 0, float("-inf")
    for i in range(n):
        v = rot.get(LABELS[i])
        if v is None:
            continue
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def analyze(path: str) -> dict:
    n_rows = 0
    opt_counts = Counter()
    raw_dist = Counter()        # rotation-0 argmax (pre-debias)
    deb_dist = Counter()        # debiased pred_choice (post-debias)
    slot_argmax = Counter()     # which slot wins, over all (q, rotation)
    slot_logprob_sum = [0.0, 0.0, 0.0, 0.0]
    slot_logprob_cnt = [0, 0, 0, 0]
    consistent = 0              # same content wins across all rotations
    changed = 0                 # debiased != raw rotation-0
    margin_sum = 0.0            # top1 - top2 of content_probs
    n_with_rot = 0
    empty = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_rows += 1
            cp = r.get("content_probs") or {}
            pr = r.get("per_rotation_logprobs") or []
            n = r.get("n_options") or (len(cp) if cp else 0)
            if not cp or not pr or n < 2:
                empty += 1
                # still count the written pred_choice for distribution
                if r.get("pred_choice"):
                    deb_dist[r["pred_choice"]] += 1
                    raw_dist[r["pred_choice"]] += 1
                continue
            n_with_rot += 1
            opt_counts[n] += 1

            # debiased prediction
            deb_dist[r["pred_choice"]] += 1
            # decisiveness margin
            probs = sorted(cp.values(), reverse=True)
            if len(probs) >= 2:
                margin_sum += probs[0] - probs[1]

            # per-rotation slot stats + content picked
            contents_picked = []
            for rot_idx, rot in enumerate(pr):
                j = argmax_slot(rot, n)
                slot_argmax[LABELS[j]] += 1
                contents_picked.append((j + rot_idx) % n)
                for i in range(n):
                    v = rot.get(LABELS[i])
                    if v is not None:
                        slot_logprob_sum[i] += v
                        slot_logprob_cnt[i] += 1

            # raw rotation-0 prediction (identity mapping: slot == content)
            raw_pred_idx = argmax_slot(pr[0], n)
            raw_dist[LABELS[raw_pred_idx]] += 1
            if LABELS[raw_pred_idx] != r["pred_choice"]:
                changed += 1

            # permutation consistency: same content across all rotations
            if len(set(contents_picked)) == 1:
                consistent += 1

    def pct(c: Counter, k):
        tot = sum(c.values())
        return 100.0 * c.get(k, 0) / tot if tot else 0.0

    slot_mean = [
        (slot_logprob_sum[i] / slot_logprob_cnt[i]) if slot_logprob_cnt[i] else float("nan")
        for i in range(4)
    ]

    return {
        "path": path,
        "n_rows": n_rows,
        "n_with_rot": n_with_rot,
        "empty": empty,
        "opt_counts": dict(opt_counts),
        "raw_dist": {k: round(pct(raw_dist, k), 1) for k in LABELS},
        "deb_dist": {k: round(pct(deb_dist, k), 1) for k in LABELS},
        "slot_argmax_pct": {k: round(pct(slot_argmax, k), 1) for k in LABELS},
        "slot_mean_logprob": [round(x, 3) for x in slot_mean],
        "consistency_pct": round(100.0 * consistent / n_with_rot, 1) if n_with_rot else 0.0,
        "changed_pct": round(100.0 * changed / n_with_rot, 1) if n_with_rot else 0.0,
        "mean_margin": round(margin_sum / n_with_rot, 4) if n_with_rot else 0.0,
    }


def main():
    paths = sys.argv[1:]
    results = [analyze(p) for p in paths]
    for res in results:
        print("=" * 70)
        print(res["path"].split("/")[-1])
        print("=" * 70)
        print(f"  rows={res['n_rows']}  with_rotations={res['n_with_rot']}  empty={res['empty']}")
        print(f"  option-count dist: {res['opt_counts']}")
        print(f"  RAW  (rotation-0, pre-debias)  A/B/C/D %: {res['raw_dist']}")
        print(f"  DEBIASED (content argmax)      A/B/C/D %: {res['deb_dist']}")
        print(f"  slot-argmax % (which SLOT wins, all rotations): {res['slot_argmax_pct']}")
        print(f"  mean logprob per SLOT [A,B,C,D]: {res['slot_mean_logprob']}")
        print(f"  permutation consistency (same content all rotations): {res['consistency_pct']}%")
        print(f"  predictions changed by debias (vs raw rot-0): {res['changed_pct']}%")
        print(f"  mean decisiveness margin (top1-top2 content_prob): {res['mean_margin']}")
        print()

    if len(results) >= 2:
        print("=" * 70)
        print("SIDE-BY-SIDE")
        print("=" * 70)
        hdr = "metric".ljust(32) + "".join(r["path"].split("/")[-1][:22].ljust(24) for r in results)
        print(hdr)
        rows = [
            ("RAW A%", lambda r: r["raw_dist"]["A"]),
            ("DEBIASED A%", lambda r: r["deb_dist"]["A"]),
            ("slot-A win %", lambda r: r["slot_argmax_pct"]["A"]),
            ("consistency %", lambda r: r["consistency_pct"]),
            ("changed by debias %", lambda r: r["changed_pct"]),
            ("mean margin", lambda r: r["mean_margin"]),
        ]
        for name, fn in rows:
            print(name.ljust(32) + "".join(str(fn(r)).ljust(24) for r in results))


if __name__ == "__main__":
    main()
