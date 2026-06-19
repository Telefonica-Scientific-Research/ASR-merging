#!/usr/bin/env python3
"""Augment MCQ JSONL by swapping correct answer with each incorrect option.

For each question with N options and one correct label, this script creates
N-1 augmented variants by swapping:
- option label/text at the correct-answer position
- option label/text at one selected incorrect position
and sets correct_answer to the swapped-in label.

This preserves option-text consistency while changing the correct letter.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class Stats:
    n_rows_in: int = 0
    n_rows_out: int = 0
    n_questions_in: int = 0
    n_questions_out: int = 0
    n_augmented_questions_added: int = 0
    n_skipped_questions: int = 0


def _find_correct_index(options: List[Dict], correct_answer: str) -> Optional[int]:
    ca = str(correct_answer or "").strip().upper()
    for i, opt in enumerate(options):
        label = str(opt.get("label", "")).strip().upper()
        if label == ca:
            return i
    return None


def _swap_option_payload(a: Dict, b: Dict) -> Tuple[Dict, Dict]:
    # We swap both label and text for exact consistency requirement.
    a_new = copy.deepcopy(a)
    b_new = copy.deepcopy(b)
    a_new["label"], b_new["label"] = b_new.get("label"), a_new.get("label")
    a_new["text"], b_new["text"] = b_new.get("text"), a_new.get("text")
    return a_new, b_new


def _augment_question_variants(question: Dict) -> Tuple[List[Dict], int]:
    q = copy.deepcopy(question)
    options = q.get("options")
    if not isinstance(options, list) or len(options) < 2:
        return [], 1

    correct_idx = _find_correct_index(options, q.get("correct_answer"))
    if correct_idx is None:
        return [], 1

    variants: List[Dict] = []
    for j in range(len(options)):
        if j == correct_idx:
            continue

        v = copy.deepcopy(q)
        v_opts = copy.deepcopy(options)

        c_new, j_new = _swap_option_payload(v_opts[correct_idx], v_opts[j])
        v_opts[correct_idx] = c_new
        v_opts[j] = j_new

        v["options"] = v_opts
        # After swap, the correct content moved to index j, so correct label is now there.
        v["correct_answer"] = str(v_opts[j].get("label", "")).strip().upper()

        if "question_id" in v:
            v["question_id"] = f"{v['question_id']}__swap_{j}"

        variants.append(v)

    return variants, 0


def augment_jsonl(
    input_path: Path,
    output_path: Path,
    keep_original: bool,
    shuffle_rows: bool,
    seed: int,
) -> Stats:
    rng = random.Random(seed)
    stats = Stats()

    all_out_rows: List[Dict] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            stats.n_rows_in += 1

            questions = row.get("questions") or []
            if not isinstance(questions, list):
                questions = []

            stats.n_questions_in += len(questions)

            out_questions: List[Dict] = []
            if keep_original:
                out_questions.extend(copy.deepcopy(questions))

            for q in questions:
                variants, skipped = _augment_question_variants(q)
                if skipped:
                    stats.n_skipped_questions += 1
                else:
                    stats.n_augmented_questions_added += len(variants)
                out_questions.extend(variants)

            if out_questions:
                out_row = copy.deepcopy(row)
                out_row["questions"] = out_questions
                stats.n_questions_out += len(out_questions)
                all_out_rows.append(out_row)

    if shuffle_rows:
        rng.shuffle(all_out_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in all_out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.n_rows_out = len(all_out_rows)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment MCQ JSONL by correct-answer swaps")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--drop-original", action="store_true", help="Only output augmented variants")
    p.add_argument("--shuffle-rows", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats = augment_jsonl(
        input_path=Path(args.input_jsonl),
        output_path=Path(args.output_jsonl),
        keep_original=not bool(args.drop_original),
        shuffle_rows=bool(args.shuffle_rows),
        seed=int(args.seed),
    )

    growth = (stats.n_questions_out / stats.n_questions_in) if stats.n_questions_in else 0.0
    print(f"rows: {stats.n_rows_in} -> {stats.n_rows_out}")
    print(f"questions: {stats.n_questions_in} -> {stats.n_questions_out}")
    print(f"added_augmented_questions: {stats.n_augmented_questions_added}")
    print(f"skipped_questions: {stats.n_skipped_questions}")
    print(f"growth_factor: {growth:.6f}x")


if __name__ == "__main__":
    main()
