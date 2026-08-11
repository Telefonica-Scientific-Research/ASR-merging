#!/usr/bin/env python3
"""Apply a global calibration vector g (estimated on the BALANCED training data) to the
already-computed CHALLENGE OOTB cyclic predictions, producing a new submission hyp.txt.

This is the TRANSFER step: g_model removes the model's intrinsic per-letter bias (estimated
where we know the labels). Applying it to the challenge does NOT force the challenge marginal
to uniform — it strips the model's letter tendency and lets the content signal decide. The LB
then tells us whether that estimate translates.

No GPU: reuses the challenge `*_detail.jsonl` (per-question cyclic content_probs + session/qid),
so the output is line-for-line aligned with the original submission.

Usage:
  python analysis/apply_calibration_to_challenge.py \
      [DETAIL=experiments/voxtral_ootb_challenge_eval/challenge_hyp_crop_debias_detail.jsonl] \
      [G_JSON=experiments/voxtral_ootb_train_balanced_debias/stage2_calibration_g.json] \
      [G_KEY=g_model] [SCALE=1.0]

Output: <detail_dir>/<detail_stem>__calib_<G_KEY>_s<SCALE>.hyp.txt
"""
import json
import math
import os
import sys
from collections import Counter

L = ["A", "B", "C", "D"]
EPS = 1e-9


def main():
    detail = sys.argv[1] if len(sys.argv) > 1 else "experiments/voxtral_ootb_challenge_eval/challenge_hyp_crop_debias_detail.jsonl"
    g_json = sys.argv[2] if len(sys.argv) > 2 else "experiments/voxtral_ootb_train_balanced_debias/stage2_calibration_g.json"
    g_key = sys.argv[3] if len(sys.argv) > 3 else "g_model"
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    if not os.path.exists(detail):
        print(f"ERROR: challenge detail not found: {detail}")
        return
    if not os.path.exists(g_json):
        print(f"ERROR: g vector not found: {g_json} (run analyze_train_balanced_debias.py first)")
        return

    gj = json.load(open(g_json))
    g_raw = gj.get(g_key)
    if g_raw is None:
        print(f"ERROR: key '{g_key}' not in {g_json}. Available: {list(gj.keys())}")
        return
    g = {k: scale * float(g_raw.get(k, 0.0)) for k in L}
    print(f"Applying g_key='{g_key}' scale={scale}:  " + "  ".join(f"{k}:{g[k]:+.3f}" for k in L))

    stem = os.path.splitext(os.path.basename(detail))[0]
    out_path = os.path.join(os.path.dirname(detail), f"{stem}__calib_{g_key}_s{scale:g}.hyp.txt")

    changed = 0
    n = 0
    marg_old = Counter()
    marg_new = Counter()
    with open(out_path, "w") as out:
        for line in open(detail):
            r = json.loads(line)
            cp = r.get("content_probs") or {}
            keys = [k for k in cp.keys()]
            if not keys:
                # no debias info -> keep original pred_choice
                pred = r.get("pred_choice")
                out.write(f"{r.get('session_id')} {r.get('question_id')} {pred}\n")
                n += 1
                continue
            old = max(keys, key=lambda k: cp[k])
            new = max(keys, key=lambda k: math.log(max(cp[k], EPS)) + g.get(k, 0.0))
            out.write(f"{r.get('session_id')} {r.get('question_id')} {new}\n")
            n += 1
            marg_old[old] += 1
            marg_new[new] += 1
            if new != old:
                changed += 1

    print(f"Wrote {n} predictions -> {out_path}")
    print(f"Changed vs cyclic: {changed}/{n} ({100*changed/max(n,1):.1f}%)")
    print("  cyclic marginal : " + "  ".join(f"{k}:{100*marg_old[k]/max(n,1):.1f}%" for k in L))
    print("  calib  marginal : " + "  ".join(f"{k}:{100*marg_new[k]/max(n,1):.1f}%" for k in L))
    print("\nThis is an LB candidate. Submit only if the balanced-train gate showed cyclic+g > cyclic.")


if __name__ == "__main__":
    main()
