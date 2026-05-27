#!/usr/bin/env python3
"""Build cache-and-shard layout for Voxtral MCQ JSONL tasks.

Expected JSONL schema per line:
{
  "path": "relative/or/absolute/audio.wav",
  "questions": [
    {
      "question_id": "Q001",
      "question_stem": "...",
      "options": [{"label": "A", "text": "..."}, ...],
      "correct_answer": "A"
    }
  ]
}

Output cache layout:
  <output_root>/<cache_name>/
    train/processed/
    dev/processed/
    test/processed/
    clean_index_cache/
      train_clean_indices.json
      dev_clean_indices.json
      test_clean_indices.json
      train_subset/
        train_eval_subset_indices.json
        train_finetune_indices.json
    shards/
      <split>/
        shard_00000_of_00008/processed/
        ...
    manifest.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from datasets import Dataset


def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_choice_label(s: str) -> str:
    return (s or "").strip().upper()


def _build_dynamic_choice_map(options: Sequence[Dict]) -> Dict[str, str]:
    choice_map: Dict[str, str] = {}
    for opt in options:
        if not isinstance(opt, dict):
            continue
        label = _normalize_choice_label(str(opt.get("label", "")))
        if not label:
            continue
        text = "" if opt.get("text") is None else str(opt.get("text"))
        choice_map[label] = text
    return choice_map


def _resolve_correct_choice_dynamic(choice_map: Dict[str, str], answer_gt: str) -> Optional[str]:
    answer_raw = (answer_gt or "").strip()
    answer_label = _normalize_choice_label(answer_raw)
    if answer_label in choice_map:
        return answer_label

    answer_norm = _normalize_label(answer_raw)
    for key, value in choice_map.items():
        if _normalize_label(value) == answer_norm:
            return key
    return None


def _resolve_jsonl_audio_path(raw_path: str, audio_root: Optional[str]) -> str:
    p = Path(raw_path)

    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        if audio_root:
            root = Path(audio_root)
            candidates.append(root / p)

            parts = list(p.parts)
            if "mlc-slm-2nd-dev" in parts:
                idx = parts.index("mlc-slm-2nd-dev")
                if idx + 1 < len(parts) and parts[idx + 1] != "data":
                    with_data = parts[: idx + 1] + ["data"] + parts[idx + 1 :]
                    candidates.append(root / Path(*with_data))

        candidates.append(p)

    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())

    if p.is_absolute():
        return str(p)
    if audio_root:
        return str((Path(audio_root) / p).resolve())
    return str(p.resolve())


def _resolve_jsonl_audio_path_checked(raw_path: str, audio_root: Optional[str], check_exists: bool) -> str:
    resolved = _resolve_jsonl_audio_path(raw_path, audio_root)
    if check_exists and not Path(resolved).exists():
        raise FileNotFoundError(
            f"Audio file not found for path='{raw_path}' with audio_root='{audio_root}'. Resolved to: {resolved}"
        )
    return resolved


def _build_mcq_prompt(question: str, choice_map: Dict[str, str]) -> str:
    choice_lines = [f"{lbl}. {txt}" for lbl, txt in choice_map.items()]
    return (
        "Choose the most suitable answer from the options below. "
        f"You must respond with only one label from: {', '.join(choice_map.keys())}.\n\n"
        f"Question: {question}\n\n"
        + "\n".join(choice_lines)
    )


def _load_jsonl_rows(
    jsonl_path: str,
    audio_root: Optional[str],
    max_questions_per_audio: int,
    max_samples: int,
    seed: int,
    check_audio_exists: bool,
) -> List[Dict]:
    src = Path(jsonl_path)
    if not src.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    rows: List[Dict] = []

    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            raw_audio_path = str(row.get("path") or row.get("audio_path") or "").strip()
            if not raw_audio_path:
                continue

            audio_path = _resolve_jsonl_audio_path_checked(raw_audio_path, audio_root, check_audio_exists)

            questions = row.get("questions") or []
            if not isinstance(questions, list):
                continue
            if max_questions_per_audio > 0:
                questions = questions[:max_questions_per_audio]

            for qi, q in enumerate(questions):
                if not isinstance(q, dict):
                    continue

                question = "" if q.get("question_stem") is None else str(q.get("question_stem"))
                choice_map = _build_dynamic_choice_map(q.get("options") or [])
                if len(choice_map) < 2:
                    continue

                gold_raw = "" if q.get("correct_answer") is None else str(q.get("correct_answer"))
                gold_choice = _resolve_correct_choice_dynamic(choice_map, gold_raw)
                if gold_choice is None:
                    continue

                qid = q.get("question_id") or str(qi)
                sample_id = f"jsonl:{line_no}:{qid}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "audio_path": audio_path,
                        "question": question,
                        "prompt_text": _build_mcq_prompt(question, choice_map),
                        "gold_choice": gold_choice,
                        "choice_labels": list(choice_map.keys()),
                        "choices": choice_map,
                        "metadata": {
                            "jsonl_line": line_no,
                            "source_audio_path": raw_audio_path,
                            "question_id": qid,
                            "question": question,
                            "task_name": q.get("task_name") if q.get("task_name") is not None else row.get("task_name"),
                            "language": q.get("language") if q.get("language") is not None else row.get("language"),
                            "difficulty": q.get("difficulty") if q.get("difficulty") is not None else row.get("difficulty"),
                            "category": q.get("category") if q.get("category") is not None else row.get("category"),
                            "subtype": q.get("subtype") if q.get("subtype") is not None else row.get("subtype"),
                            "sub-category": q.get("sub-category") if q.get("sub-category") is not None else row.get("sub-category"),
                            "sub-sub-category": q.get("sub-sub-category") if q.get("sub-sub-category") is not None else row.get("sub-sub-category"),
                            "linguistics_sub_discipline": q.get("linguistics_sub_discipline")
                            if q.get("linguistics_sub_discipline") is not None
                            else row.get("linguistics_sub_discipline"),
                        },
                    }
                )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_samples > 0:
        rows = rows[: min(max_samples, len(rows))]

    if not rows:
        raise RuntimeError(f"No usable MCQ samples found in JSONL: {jsonl_path}")

    return rows


def _split_train_eval(rows: List[Dict], eval_fraction: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if eval_fraction <= 0.0:
        return rows, []

    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)

    n_eval = max(1, int(round(len(rows) * eval_fraction)))
    n_eval = min(n_eval, max(0, len(rows) - 1))

    eval_idx = set(idx[:n_eval])
    train_rows = [r for i, r in enumerate(rows) if i not in eval_idx]
    eval_rows = [r for i, r in enumerate(rows) if i in eval_idx]
    return train_rows, eval_rows


def _save_split(cache_dir: Path, split_name: str, rows: List[Dict]) -> None:
    ds = Dataset.from_list(rows)
    out = cache_dir / split_name / "processed"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))


def _audio_id_from_path(audio_path: str) -> str:
    return hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()


def _assign_audio_ids(split_map: Dict[str, List[Dict]]) -> Dict[str, str]:
    audio_map: Dict[str, str] = {}
    for rows in split_map.values():
        for r in rows:
            ap = str(r.get("audio_path") or "")
            if not ap:
                continue
            aid = _audio_id_from_path(ap)
            r["audio_id"] = aid
            audio_map[aid] = ap
    return dict(sorted(audio_map.items(), key=lambda kv: kv[0]))


def _write_audio_index(cache_dir: Path, split_map: Dict[str, List[Dict]], audio_map: Dict[str, str]) -> None:
    out_root = cache_dir / "audio_index"
    out_root.mkdir(parents=True, exist_ok=True)

    split_audio_ids: Dict[str, List[str]] = {}
    for split_name, rows in split_map.items():
        ids = sorted({str(r.get("audio_id") or "") for r in rows if r.get("audio_id")})
        split_audio_ids[split_name] = ids

    payload = {
        "n_unique_audio": len(audio_map),
        "audio": [{"audio_id": aid, "audio_path": ap} for aid, ap in audio_map.items()],
        "split_audio_ids": split_audio_ids,
    }
    (out_root / "audio_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_audio_shards(cache_dir: Path, split_name: str, rows: List[Dict], num_shards: int, seed: int) -> None:
    if num_shards <= 1:
        return

    split_root = cache_dir / "audio_shards" / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    audio_ids = sorted({str(r.get("audio_id") or "") for r in rows if r.get("audio_id")})
    rnd = random.Random(seed)
    rnd.shuffle(audio_ids)

    q_per_audio: Dict[str, int] = {}
    for r in rows:
        aid = str(r.get("audio_id") or "")
        if not aid:
            continue
        q_per_audio[aid] = q_per_audio.get(aid, 0) + 1

    for shard_index in range(num_shards):
        shard_tag = f"shard_{shard_index:05d}_of_{num_shards:05d}"
        shard_audio_ids = [aid for i, aid in enumerate(audio_ids) if (i % num_shards) == shard_index]
        n_samples = sum(q_per_audio.get(aid, 0) for aid in shard_audio_ids)

        out = {
            "split": split_name,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "n_audio": len(shard_audio_ids),
            "n_samples_estimate": int(n_samples),
            "audio_ids": shard_audio_ids,
        }
        (split_root / f"{shard_tag}_audio_ids.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def _write_clean_indices(cache_dir: Path, split_name: str, n: int) -> None:
    clean_root = cache_dir / "clean_index_cache"
    clean_root.mkdir(parents=True, exist_ok=True)
    payload = {"keep_indices": list(range(int(n)))}
    (clean_root / f"{split_name}_clean_indices.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_train_subset(cache_dir: Path, n_train: int, eval_fraction: float, seed: int) -> None:
    idx = list(range(int(n_train)))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    n_eval = max(1, int(round(n_train * eval_fraction)))
    n_eval = min(n_eval, n_train - 1) if n_train > 1 else 1

    eval_idx = sorted(idx[:n_eval])
    finetune_idx = sorted(idx[n_eval:])

    subset_root = cache_dir / "clean_index_cache" / "train_subset"
    subset_root.mkdir(parents=True, exist_ok=True)
    (subset_root / "train_eval_subset_indices.json").write_text(json.dumps({"indices": eval_idx}, indent=2), encoding="utf-8")
    (subset_root / "train_finetune_indices.json").write_text(
        json.dumps({"indices": finetune_idx}, indent=2),
        encoding="utf-8",
    )


def _iter_shard_rows(rows: List[Dict], num_shards: int, shard_index: int) -> List[Dict]:
    return [r for i, r in enumerate(rows) if (i % num_shards) == shard_index]


def _build_split_shards(cache_dir: Path, split_name: str, rows: List[Dict], num_shards: int) -> None:
    if num_shards <= 1:
        return

    split_root = cache_dir / "shards" / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    for shard_index in range(num_shards):
        shard_rows = _iter_shard_rows(rows, num_shards, shard_index)
        shard_tag = f"shard_{shard_index:05d}_of_{num_shards:05d}"
        shard_dir = split_root / shard_tag
        _save_split(shard_dir, "", shard_rows)
        # _save_split writes <dir>/processed when split_name is ""
        # so keep explicit metadata in a sidecar for easier discovery.
        meta = {
            "split": split_name,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "n_samples": len(shard_rows),
        }
        (shard_dir / "shard_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _resolve_cache_name(args: argparse.Namespace) -> str:
    if args.cache_name:
        return str(args.cache_name)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"mcq_jsonl_{stamp}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cache+shards for Voxtral MCQ JSONL training")

    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--eval-jsonl", default=None)
    p.add_argument("--audio-root", default=None)

    p.add_argument("--max-questions-per-audio", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--eval-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--output-root", default="data/cache/voxtral_mcq")
    p.add_argument("--cache-name", default=None)
    p.add_argument("--no-check-audio-exists", dest="check_audio_exists", action="store_false")
    p.set_defaults(check_audio_exists=True)

    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--shard-splits",
        default="train",
        help="Comma-separated splits to shard (train,dev,test). Default: train",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.output_root) / _resolve_cache_name(args)
    cache_dir.mkdir(parents=True, exist_ok=True)

    max_q = max(0, int(args.max_questions_per_audio))
    max_train = max(0, int(args.max_train_samples))
    max_eval = max(0, int(args.max_eval_samples))
    seed = int(args.seed)

    train_rows_all = _load_jsonl_rows(
        jsonl_path=args.train_jsonl,
        audio_root=args.audio_root,
        max_questions_per_audio=max_q,
        max_samples=max_train,
        seed=seed,
        check_audio_exists=bool(args.check_audio_exists),
    )

    if args.eval_jsonl:
        eval_rows = _load_jsonl_rows(
            jsonl_path=args.eval_jsonl,
            audio_root=args.audio_root,
            max_questions_per_audio=max_q,
            max_samples=max_eval,
            seed=seed,
            check_audio_exists=bool(args.check_audio_exists),
        )
        train_rows = train_rows_all
    else:
        train_rows, eval_rows = _split_train_eval(train_rows_all, float(args.eval_fraction), seed)
        if max_eval > 0 and eval_rows:
            eval_rows = eval_rows[: min(max_eval, len(eval_rows))]

    test_rows = list(eval_rows)

    split_map = {"train": train_rows, "dev": eval_rows, "test": test_rows}
    audio_map = _assign_audio_ids(split_map)
    _write_audio_index(cache_dir, split_map, audio_map)

    _save_split(cache_dir, "train", train_rows)
    _save_split(cache_dir, "dev", eval_rows)
    _save_split(cache_dir, "test", test_rows)

    _write_clean_indices(cache_dir, "train", len(train_rows))
    _write_clean_indices(cache_dir, "dev", len(eval_rows))
    _write_clean_indices(cache_dir, "test", len(test_rows))
    _write_train_subset(cache_dir, len(train_rows), eval_fraction=0.02, seed=seed)

    shard_splits = {x.strip() for x in str(args.shard_splits).split(",") if x.strip()}
    for split_name in sorted(shard_splits):
        if split_name not in split_map:
            raise ValueError(f"Unknown split in --shard-splits: {split_name}")
        _build_split_shards(cache_dir, split_name, split_map[split_name], int(args.num_shards))
        _build_audio_shards(cache_dir, split_name, split_map[split_name], int(args.num_shards), seed)

    manifest = {
        "cache_dir": str(cache_dir.resolve()),
        "created_at": dt.datetime.now().isoformat(),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "audio_root": str(args.audio_root) if args.audio_root else None,
        "max_questions_per_audio": max_q,
        "max_train_samples": max_train,
        "max_eval_samples": max_eval,
        "eval_fraction": float(args.eval_fraction),
        "seed": seed,
        "check_audio_exists": bool(args.check_audio_exists),
        "num_shards": int(args.num_shards),
        "shard_splits": sorted(shard_splits),
        "counts": {
            "train": len(train_rows),
            "dev": len(eval_rows),
            "test": len(test_rows),
        },
        "audio_counts": {
            "unique_total": len(audio_map),
            "train": len({r.get("audio_id") for r in train_rows if r.get("audio_id")}),
            "dev": len({r.get("audio_id") for r in eval_rows if r.get("audio_id")}),
            "test": len({r.get("audio_id") for r in test_rows if r.get("audio_id")}),
        },
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Built MCQ cache at: {cache_dir}")
    print(f"  train={len(train_rows)}")
    print(f"  dev={len(eval_rows)}")
    print(f"  test={len(test_rows)}")
    print(f"  unique_audio={len(audio_map)}")
    if int(args.num_shards) > 1:
        print(f"  row_shards={int(args.num_shards)} for splits={','.join(sorted(shard_splits))}")
        print(f"  audio_shards={int(args.num_shards)} for splits={','.join(sorted(shard_splits))}")


if __name__ == "__main__":
    main()
