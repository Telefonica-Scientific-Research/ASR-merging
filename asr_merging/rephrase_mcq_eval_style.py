#!/usr/bin/env python3
"""Rephrase MCQ question stems into eval-style instructional language.

Input/Output schema:
  {"path": ..., "questions": [{"question_id", "question_stem", "options", "correct_answer"}, ...]}

The script keeps labels/options/answers unchanged and rewrites only question_stem.
A JSON report is produced with counts and template usage.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Timestamp-like atom: mm:ss(.x), hh:mm:ss(.x), or raw seconds (up to 4 digits).
_TS_ATOM = r"(?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?|\d{1,4}(?:\.\d+)?)"
_RANGE_PAT = re.compile(rf"(?P<a>{_TS_ATOM})\s*(?:-|to|~|–|—)\s*(?P<b>{_TS_ATOM})")
_SPACE_PAT = re.compile(r"\s+")


@dataclass
class Stats:
    n_rows_in: int = 0
    n_rows_out: int = 0
    n_questions_in: int = 0
    n_questions_changed: int = 0
    n_questions_unchanged: int = 0


def _norm_space(s: str) -> str:
    return _SPACE_PAT.sub(" ", (s or "").strip())


def _strip_question_prefix(q: str) -> str:
    s = _norm_space(q)
    # Remove common lead-ins so templates do not duplicate wording too much.
    prefixes = [
        "listen to",
        "compare",
        "based on",
        "according to",
        "in the segment",
        "in this segment",
    ]
    low = s.lower()
    for p in prefixes:
        if low.startswith(p + " "):
            s = s[len(p) :].strip(" ,.-")
            break
    return s.rstrip("?").strip()


def _strip_leading_range_fragment(s: str) -> str:
    # Example: "88.348-94.654, the speaker ..." -> "the speaker ..."
    out = re.sub(rf"^\s*{_TS_ATOM}\s*(?:-|to|~|–|—)\s*{_TS_ATOM}\s*[,.:;-]?\s*", "", s)
    return out.strip()


def _lower_initial_if_questionword(s: str) -> str:
    if not s:
        return s
    qwords = ("What ", "Which ", "How ", "Why ", "Who ", "Where ", "When ")
    if s.startswith(qwords):
        return s[0].lower() + s[1:]
    return s


def _extract_ranges(q: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in _RANGE_PAT.finditer(q or ""):
        a = m.group("a")
        b = m.group("b")
        out.append((a, b))
    return out


def _render_range(a: str, b: str) -> str:
    return f"[{a}, {b}]"


def _eval_style_rephrase(q: str) -> Tuple[str, str]:
    raw = _norm_space(q)
    if not raw:
        return raw, "empty"

    core = _strip_leading_range_fragment(_strip_question_prefix(raw))
    ranges = _extract_ranges(raw)

    if len(ranges) >= 2:
        left = _render_range(*ranges[0])
        right = _render_range(*ranges[1])
        text = f"Compare the audio evidence between {left} and {right}. {core}".strip()
        if not text.endswith("?"):
            text += "?"
        return text, "compare_multi_range"

    if len(ranges) == 1:
        seg = _render_range(*ranges[0])
        text = f"Listen to the segment {seg}. {core}".strip()
        if not text.endswith("?"):
            text += "?"
        return text, "listen_single_range"

    low = raw.lower()
    if low.startswith("according to"):
        text = f"According to the conversation, {_lower_initial_if_questionword(core)}".strip()
        if not text.endswith("?"):
            text += "?"
        return text, "according_no_range"

    if low.startswith("based on"):
        text = f"Based on the conversation, {_lower_initial_if_questionword(core)}".strip()
        if not text.endswith("?"):
            text += "?"
        return text, "based_no_range"

    text = f"Based on the conversation, {_lower_initial_if_questionword(core)}".strip()
    if not text.endswith("?"):
        text += "?"
    return text, "generic_eval_style"


def rephrase_jsonl(input_jsonl: Path, output_jsonl: Path, report_json: Path) -> Dict:
    stats = Stats()
    template_counts: Counter = Counter()
    out_rows: List[Dict] = []

    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats.n_rows_in += 1

            questions = row.get("questions") or []
            if not isinstance(questions, list):
                questions = []

            out_qs: List[Dict] = []
            for q in questions:
                if not isinstance(q, dict):
                    continue
                stats.n_questions_in += 1
                q2 = copy.deepcopy(q)
                old_stem = "" if q2.get("question_stem") is None else str(q2.get("question_stem"))
                new_stem, template_name = _eval_style_rephrase(old_stem)
                template_counts[template_name] += 1
                q2["question_stem"] = new_stem
                out_qs.append(q2)

                if _norm_space(old_stem) != _norm_space(new_stem):
                    stats.n_questions_changed += 1
                else:
                    stats.n_questions_unchanged += 1

            row2 = copy.deepcopy(row)
            row2["questions"] = out_qs
            out_rows.append(row2)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.n_rows_out = len(out_rows)

    report = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "n_rows_in": stats.n_rows_in,
        "n_rows_out": stats.n_rows_out,
        "n_questions_in": stats.n_questions_in,
        "n_questions_changed": stats.n_questions_changed,
        "n_questions_unchanged": stats.n_questions_unchanged,
        "change_rate": (stats.n_questions_changed / stats.n_questions_in) if stats.n_questions_in else 0.0,
        "template_counts": dict(template_counts),
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rephrase MCQ question stems into eval-style language")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--report-json", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = rephrase_jsonl(
        input_jsonl=Path(args.input_jsonl),
        output_jsonl=Path(args.output_jsonl),
        report_json=Path(args.report_json),
    )
    print(f"Wrote rephrased dataset: {report['output_jsonl']}")
    print(f"Wrote report: {args.report_json}")
    print(f"Rows: {report['n_rows_in']} -> {report['n_rows_out']}")
    print(f"Questions changed: {report['n_questions_changed']} / {report['n_questions_in']}")
    print("Template counts:", report["template_counts"])


if __name__ == "__main__":
    main()
