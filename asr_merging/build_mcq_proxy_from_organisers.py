#!/usr/bin/env python3
"""Build a proxy MCQ JSONL from organisers with controlled label distribution.

This script reshapes correct-answer label priors (A/B/C/D) by creating
option-swap variants per question and sampling them to hit a target prior.

Input schema expected (organisers style):
  {"path": ..., "questions": [{"question_id", "question_stem", "options", "correct_answer"}, ...]}

Output schema is the same grouped-per-audio JSONL.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class QRef:
    row_idx: int
    q_idx: int
    path: str
    question: Dict


def _norm_label(x: str) -> str:
    return str(x or "").strip().upper()


def _find_correct_idx(options: List[Dict], correct_answer: str) -> int:
    ca = _norm_label(correct_answer)
    for i, opt in enumerate(options):
        if _norm_label(opt.get("label", "")) == ca:
            return i
    return -1


def _swap_option_text(a: Dict, b: Dict) -> Tuple[Dict, Dict]:
    # Keep labels fixed; swap only texts so the correct content moves to target label.
    a_new = copy.deepcopy(a)
    b_new = copy.deepcopy(b)
    a_new["text"], b_new["text"] = b_new.get("text"), a_new.get("text")
    return a_new, b_new


def _variant_with_target_label(question: Dict, target_label: str) -> Dict:
    q = copy.deepcopy(question)
    opts = q.get("options") or []
    ci = _find_correct_idx(opts, q.get("correct_answer"))
    if ci < 0:
        return q

    tlabel = _norm_label(target_label)
    if _norm_label(opts[ci].get("label", "")) == tlabel:
        q["correct_answer"] = tlabel
        return q

    ti = -1
    for i, opt in enumerate(opts):
        if _norm_label(opt.get("label", "")) == tlabel:
            ti = i
            break

    if ti < 0:
        return q

    c_new, t_new = _swap_option_text(opts[ci], opts[ti])
    opts[ci] = c_new
    opts[ti] = t_new
    q["options"] = opts
    q["correct_answer"] = tlabel
    return q


def _parse_target_prior(raw: str) -> Dict[str, float]:
    # Format: A:0.25,B:0.25,C:0.25,D:0.25
    out: Dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid target prior token: {part}")
        k, v = part.split(":", 1)
        out[_norm_label(k)] = float(v)
    if not out:
        raise ValueError("Empty target prior")
    s = sum(out.values())
    if s <= 0:
        raise ValueError("Target prior sum must be > 0")
    for k in list(out.keys()):
        out[k] /= s
    return out


def _compute_counts(total: int, prior: Dict[str, float], labels: List[str]) -> Dict[str, int]:
    # Largest remainder rounding.
    raw = {l: total * prior.get(l, 0.0) for l in labels}
    base = {l: int(raw[l]) for l in labels}
    rem = total - sum(base.values())
    order = sorted(labels, key=lambda l: (raw[l] - base[l]), reverse=True)
    for i in range(rem):
        base[order[i % len(order)]] += 1
    return base


def build_proxy(
    input_jsonl: Path,
    output_jsonl: Path,
    report_json: Path,
    target_prior: Dict[str, float],
    size_multiplier: float,
    seed: int,
    max_copies_per_question: int,
) -> None:
    rng = random.Random(seed)

    rows: List[Dict] = []
    qrefs: List[QRef] = []
    label_pool: Dict[int, List[str]] = {}

    with input_jsonl.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(row)
            qlist = row.get("questions") or []
            for q_idx, q in enumerate(qlist):
                if not isinstance(q, dict):
                    continue
                opts = q.get("options") or []
                labels = [_norm_label(o.get("label", "")) for o in opts if isinstance(o, dict)]
                labels = [x for x in labels if x]
                if len(labels) < 2:
                    continue
                qrefs.append(QRef(row_idx=row_idx, q_idx=q_idx, path=str(row.get("path", "")), question=q))
                label_pool[len(qrefs) - 1] = sorted(set(labels))

    if not qrefs:
        raise RuntimeError("No valid questions found in input JSONL")

    n_out = max(1, int(round(len(qrefs) * float(size_multiplier))))
    labels_all = sorted({l for labs in label_pool.values() for l in labs})
    target_counts = _compute_counts(n_out, target_prior, labels_all)

    # Index question ids by label availability.
    by_label: Dict[str, List[int]] = {l: [] for l in labels_all}
    for qid, labs in label_pool.items():
        for l in labs:
            by_label[l].append(qid)

    for l in labels_all:
        rng.shuffle(by_label[l])

    selected: List[Tuple[int, str]] = []
    copies = Counter()
    made = Counter()

    for _ in range(n_out):
        # pick label with largest remaining deficit
        deficits = {l: target_counts[l] - made[l] for l in labels_all}
        best_label = max(labels_all, key=lambda l: deficits[l])

        # candidate pool for that label with copy cap
        cand = [qid for qid in by_label[best_label] if copies[qid] < max_copies_per_question]

        if not cand:
            # fallback to any label/qid still available
            fallback = []
            for l in labels_all:
                fallback.extend((qid, l) for qid in by_label[l] if copies[qid] < max_copies_per_question)
            if not fallback:
                break
            qid, lbl = rng.choice(fallback)
        else:
            qid = rng.choice(cand)
            lbl = best_label

        selected.append((qid, lbl))
        copies[qid] += 1
        made[lbl] += 1

    # Build grouped output rows.
    out_grouped: Dict[str, List[Dict]] = defaultdict(list)
    out_dist = Counter()

    for i, (qid, lbl) in enumerate(selected):
        qr = qrefs[qid]
        qv = _variant_with_target_label(qr.question, lbl)
        qv = copy.deepcopy(qv)
        if "question_id" in qv:
            qv["question_id"] = f"{qv['question_id']}__proxy_{i}"
        out_grouped[qr.path].append(qv)
        out_dist[_norm_label(qv.get("correct_answer", ""))] += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for path in sorted(out_grouped.keys()):
            row = {"path": path, "questions": out_grouped[path]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "n_input_questions": len(qrefs),
        "n_output_questions": len(selected),
        "target_prior": target_prior,
        "target_counts": target_counts,
        "achieved_counts": dict(out_dist),
        "labels_all": labels_all,
        "size_multiplier": size_multiplier,
        "seed": seed,
        "max_copies_per_question": max_copies_per_question,
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote proxy dataset: {output_jsonl}")
    print(f"Wrote report: {report_json}")
    print(f"Input questions: {len(qrefs)}")
    print(f"Output questions: {len(selected)}")
    print(f"Achieved counts: {dict(out_dist)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build organisers-based proxy MCQ dataset with label-prior control")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--report-json", required=True)
    p.add_argument(
        "--target-prior",
        default="A:0.25,B:0.25,C:0.25,D:0.25",
        help="Comma-separated prior, e.g. A:0.3,B:0.3,C:0.25,D:0.15",
    )
    p.add_argument("--size-multiplier", type=float, default=1.0)
    p.add_argument("--max-copies-per-question", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_proxy(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
        target_prior=_parse_target_prior(args.target_prior),
        size_multiplier=float(args.size_multiplier),
        seed=int(args.seed),
        max_copies_per_question=int(args.max_copies_per_question),
    )


if __name__ == "__main__":
    main()
