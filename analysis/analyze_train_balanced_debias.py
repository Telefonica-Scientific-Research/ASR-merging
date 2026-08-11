#!/usr/bin/env python3
"""Stage-1 (cyclic) vs Stage-2 (global marginal calibration) accuracy analysis.

Runs on the BALANCED full-train OOTB debias eval (job 42417089). Because that data
is exactly 25/25/25/25 A/B/C/D, the *true* answer marginal is uniform by construction,
so any residual post-cyclic tilt is pure model label-bias — and we can measure whether
removing it with a single GLOBAL vector g actually raises ACCURACY (not just reshapes
the histogram).

Reports, on the labeled balanced data:
  1. raw      = rotation-0 argmax (no debias)
  2. cyclic   = argmax of per-question debiased content_probs (current submission method)
  3. cyclic+g = cyclic, then global calibration g_L = -alpha*log(prior_L), swept over alpha
                (alpha=0 -> cyclic; alpha=1 -> divide-by-prior / contextual calibration)
  4. cyclic+g(uniform) = g iteratively fit so the aggregate predicted marginal == uniform

Usage:
  python analysis/analyze_train_balanced_debias.py \
      [EXP_DIR=experiments/voxtral_ootb_train_balanced_debias] \
      [BALANCED_JSONL=data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_balanced.jsonl]
"""
import json
import math
import os
import sys
import glob
from collections import Counter, defaultdict

L = ["A", "B", "C", "D"]
EPS = 1e-9


def _session_id_from_path(path: str) -> str:
    # mlc-slm-2nd-dev/English_Australian/1127_015_phone.wav -> English_Australian_1127_015_phone
    parts = path.replace("\\", "/").split("/")
    stem = os.path.splitext(parts[-1])[0]
    accent = parts[-2] if len(parts) >= 2 else ""
    return f"{accent}_{stem}" if accent else stem


def _load_gold_map(balanced_jsonl: str):
    gold = {}
    if not os.path.exists(balanced_jsonl):
        return gold
    for line in open(balanced_jsonl):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        sid = _session_id_from_path(r.get("path") or r.get("audio") or "")
        for q in (r.get("questions") or []):
            ca = q.get("correct_answer")
            qid = q.get("question_id")
            if ca and qid:
                gold[(sid, str(qid))] = str(ca).strip().upper()[:1]
    return gold


def _load_items(exp_dir: str, gold_map):
    pred_files = sorted(glob.glob(os.path.join(exp_dir, "train_balanced_*_shard_*", "forgetting_eval_predictions.jsonl")))
    items = []  # (content_probs, per_rotation, gold)
    n_files = 0
    for pf in pred_files:
        n_files += 1
        for line in open(pf):
            r = json.loads(line)
            cp = r.get("debias_content_probs")
            pr = r.get("debias_per_rotation")
            if not cp or len([k for k in L if k in cp]) != 4:
                continue
            g = r.get("gold_choice")
            if g not in L:
                sid = r.get("session_id")
                qid = r.get("question_id")
                g = gold_map.get((str(sid), str(qid)))
            if g not in L:
                continue
            items.append((cp, pr, g))
    return items, n_files, pred_files


def _marginal(preds):
    n = len(preds) or 1
    c = Counter(preds)
    return {k: c[k] / n for k in L}


def _per_class_recall(preds, golds):
    tot = Counter(golds)
    hit = Counter(p for p, g in zip(preds, golds) if p == g)
    return {k: (hit[k] / tot[k] if tot[k] else float("nan")) for k in L}


def _pred_raw(items):
    out = []
    for cp, pr, g in items:
        if pr and all(k in pr[0] for k in L):
            out.append(max(L, key=lambda k: pr[0][k]))
        else:  # no rotation info -> fall back to cyclic
            out.append(max(L, key=lambda k: cp[k]))
    return out


def _pred_cyclic(items):
    return [max(L, key=lambda k: cp[k]) for cp, pr, g in items]


def _pred_calibrated(items, g_vec):
    out = []
    for cp, pr, g in items:
        s = {k: math.log(max(cp[k], EPS)) + g_vec[k] for k in L}
        out.append(max(L, key=lambda k: s[k]))
    return out


def _acc(preds, golds):
    return sum(p == gg for p, gg in zip(preds, golds)) / (len(golds) or 1)


def _fit_g_to_uniform(items, golds, iters=800):
    """Iteratively shift g so the aggregate predicted marginal == uniform (0.25 each)."""
    g = {k: 0.0 for k in L}
    target = 0.25
    for t in range(iters):
        step = 0.6 * (1.0 - t / iters) + 0.05
        preds = _pred_calibrated(items, g)
        frac = _marginal(preds)
        for k in L:
            g[k] += step * (target - frac[k])
        m = sum(g.values()) / 4.0
        for k in L:
            g[k] -= m  # center (argmax is shift-invariant)
    return g


def main():
    exp_dir = sys.argv[1] if len(sys.argv) > 1 else "experiments/voxtral_ootb_train_balanced_debias"
    balanced_jsonl = sys.argv[2] if len(sys.argv) > 2 else "data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_balanced.jsonl"

    gold_map = _load_gold_map(balanced_jsonl)
    items, n_files, pred_files = _load_items(exp_dir, gold_map)
    if not items:
        print(f"No predictions found under {exp_dir} (job 42417089 may still be running).")
        print(f"Looked for: {exp_dir}/train_balanced_*_shard_*/forgetting_eval_predictions.jsonl")
        return

    golds = [g for cp, pr, g in items]
    print(f"Loaded {len(items)} four-option questions from {n_files} shard file(s).")
    print(f"Gold marginal (sanity, should be ~25% each): "
          + "  ".join(f"{k}:{100*Counter(golds)[k]/len(golds):.1f}%" for k in L))

    # model prior = mean content-prob mass per letter (contextual-calibration prior)
    prior = {k: sum(cp[k] for cp, pr, g in items) / len(items) for k in L}

    def report(tag, preds):
        a = _acc(preds, golds)
        marg = _marginal(preds)
        rec = _per_class_recall(preds, golds)
        print(f"\n[{tag}]  accuracy = {100*a:.2f}%")
        print("   pred marginal : " + "  ".join(f"{k}:{100*marg[k]:.1f}%" for k in L))
        print("   per-class recall: " + "  ".join(f"{k}:{100*rec[k]:.1f}%" for k in L))
        return a

    raw_preds = _pred_raw(items)
    cyc_preds = _pred_cyclic(items)
    a_raw = report("1. raw (rotation-0, no debias)", raw_preds)
    a_cyc = report("2. cyclic (current submission)", cyc_preds)

    # Stage-2: sweep alpha for g_L = -alpha*log(prior_L)
    print("\n[3. cyclic + global calibration g_L = -alpha*log(prior_L)]  (alpha=0 cyclic, alpha=1 divide-by-prior)")
    print(f"   model prior mass: " + "  ".join(f"{k}:{100*prior[k]:.1f}%" for k in L))
    best = (a_cyc, 0.0, cyc_preds)
    print(f"   {'alpha':>6}{'accuracy':>11}{'  pred marginal (A/B/C/D)':>30}")
    a = 0.0
    while a <= 2.0001:
        g_vec = {k: -a * math.log(max(prior[k], EPS)) for k in L}
        preds = _pred_calibrated(items, g_vec)
        acc = _acc(preds, golds)
        marg = _marginal(preds)
        flag = ""
        if acc > best[0]:
            best = (acc, a, preds); flag = "  <-- best"
        print(f"   {a:>6.2f}{100*acc:>10.2f}%   {100*marg['A']:.0f}/{100*marg['B']:.0f}/{100*marg['C']:.0f}/{100*marg['D']:.0f}{flag}")
        a += 0.25

    # Stage-2 alt: fit g so aggregate marginal == uniform exactly
    g_uni = _fit_g_to_uniform(items, golds)
    uni_preds = _pred_calibrated(items, g_uni)
    a_uni = report("4. cyclic + g(fit to uniform marginal)", uni_preds)
    print("   fitted g (centered): " + "  ".join(f"{k}:{g_uni[k]:+.2f}" for k in L))

    # Persist g so the (free) transfer to the challenge predictions is one command later.
    # g_uniform == g_model: on balanced data the true marginal is uniform, so fitting g to
    # uniform removes the model's intrinsic label bias (the transferable part).
    g_best = {k: -best[1] * math.log(max(prior[k], EPS)) for k in L}
    out_g = {
        "prior": prior,
        "g_model": g_uni,            # fit-to-uniform == removes model label bias (TRANSFER THIS)
        "best_alpha": best[1],
        "g_best_alpha": g_best,
        "accuracy": {"raw": a_raw, "cyclic": a_cyc, "best_calib": best[0], "uniform": a_uni},
        "n_items": len(items),
    }
    g_path = os.path.join(exp_dir, "stage2_calibration_g.json")
    with open(g_path, "w") as f:
        json.dump(out_g, f, indent=2)
    print(f"\n   saved calibration vectors -> {g_path}")

    print("\n================ SUMMARY ================")
    print(f"  raw (no debias)              : {100*a_raw:.2f}%")
    print(f"  cyclic (current)             : {100*a_cyc:.2f}%   (delta vs raw {100*(a_cyc-a_raw):+.2f})")
    print(f"  cyclic + best calib (a={best[1]:.2f}) : {100*best[0]:.2f}%   (delta vs cyclic {100*(best[0]-a_cyc):+.2f})")
    print(f"  cyclic + g(->uniform)        : {100*a_uni:.2f}%   (delta vs cyclic {100*(a_uni-a_cyc):+.2f})")
    print("  NOTE: the cyclic delta transfers to the LB (prior-agnostic). The calibration")
    print("        delta is measured with the TRUE marginal known (uniform); on the LB you")
    print("        must instead target the LB's own marginal (~45% A), not uniform.")


if __name__ == "__main__":
    main()
