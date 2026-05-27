#!/usr/bin/env python3
"""Build MLC cache layout expected by *train_router.py from raw WAV/TXT trees.

Expected raw layout (examples):
- training: <train_root>/<Language>/<Dialect>/<Speaker>/<utt>.wav + <utt>.txt
- development: <dev_root>/<Language>/<utt>.wav + <utt>.txt
- development: <dev_root>/<Language>/<Dialect>/<utt>.wav + <utt>.txt

The script writes a cache like:
  <output_root>/mlc_slm_<tag>/
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
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import Audio, Dataset


@dataclass
class Example:
    utt_id: str
    audio_path: str
    language: str
    text: str


def _normalize_token(s: str) -> str:
    return " ".join(s.replace("_", " ").split()).strip().lower()


def _build_label(language_dir: str, dialect_dir: str, mode: str) -> str:
    base = _normalize_token(language_dir)
    dialect = _normalize_token(dialect_dir)
    if mode == "base":
        return base
    if base == dialect:
        return base
    return f"{base} ({dialect})"


def _collect_examples(split_root: Path, label_mode: str) -> List[Example]:
    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    examples: List[Example] = []
    for wav_path in split_root.rglob("*.wav"):
        rel = wav_path.relative_to(split_root)
        if len(rel.parts) < 2:
            # Need at least language/file.wav
            continue

        language_dir = rel.parts[0]
        dialect_dir = rel.parts[1] if len(rel.parts) >= 3 else language_dir
        label = _build_label(language_dir, dialect_dir, label_mode)

        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        utt_id = wav_path.stem
        examples.append(
            Example(
                utt_id=utt_id,
                audio_path=str(wav_path.resolve()),
                language=label,
                text=text,
            )
        )

    if not examples:
        raise RuntimeError(f"No valid wav/txt examples found under: {split_root}")

    examples.sort(key=lambda x: x.utt_id)
    return examples


def _to_dataset(examples: Iterable[Example]) -> Dataset:
    rows = list(examples)
    ds = Dataset.from_dict(
        {
            "utt_id": [r.utt_id for r in rows],
            "audio": [r.audio_path for r in rows],
            "language": [r.language for r in rows],
            "text": [r.text for r in rows],
        }
    )
    # Keep paths in the dataset with consistent feature typing.
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def _save_split(cache_dir: Path, split_name: str, ds: Dataset) -> None:
    out = cache_dir / split_name / "processed"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))


def _write_clean_indices(cache_dir: Path, split_name: str, n: int) -> None:
    clean_root = cache_dir / "clean_index_cache"
    clean_root.mkdir(parents=True, exist_ok=True)
    payload = {"keep_indices": list(range(n))}
    (clean_root / f"{split_name}_clean_indices.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _write_train_subset(cache_dir: Path, n_train: int, eval_fraction: float, seed: int) -> None:
    import random

    idx = list(range(n_train))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    n_eval = max(1, int(round(n_train * eval_fraction)))
    n_eval = min(n_eval, n_train - 1) if n_train > 1 else 1

    eval_idx = sorted(idx[:n_eval])
    finetune_idx = sorted(idx[n_eval:])

    subset_root = cache_dir / "clean_index_cache" / "train_subset"
    subset_root.mkdir(parents=True, exist_ok=True)
    (subset_root / "train_eval_subset_indices.json").write_text(
        json.dumps({"indices": eval_idx}, indent=2), encoding="utf-8"
    )
    (subset_root / "train_finetune_indices.json").write_text(
        json.dumps({"indices": finetune_idx}, indent=2), encoding="utf-8"
    )


def _default_cache_name(train_root: Path, dev_root: Path, label_mode: str) -> str:
    langs = sorted({p.name for p in train_root.iterdir() if p.is_dir()} | {p.name for p in dev_root.iterdir() if p.is_dir()})
    lang_tag = "_".join(s.replace(" ", "") for s in langs)
    mode_tag = "dialect" if label_mode == "dialect" else "base"
    return f"mlc_slm_{lang_tag}_{mode_tag}"


def main() -> None:
    p = argparse.ArgumentParser(description="Build mlc_slm cache from raw MLC WAV/TXT folders")
    p.add_argument("--train-root", required=True, help="Raw training data root (contains language dirs)")
    p.add_argument("--dev-root", required=True, help="Raw dev data root (contains language dirs)")
    p.add_argument(
        "--test-root",
        default=None,
        help="Optional raw test root. If omitted, dev split is reused for test split.",
    )
    p.add_argument(
        "--output-root",
        default="data/cache/voxtral",
        help="Directory under which mlc_slm_* cache directory will be created",
    )
    p.add_argument(
        "--cache-name",
        default=None,
        help="Explicit cache directory name (default: auto-generated mlc_slm_<...>)",
    )
    p.add_argument(
        "--label-mode",
        choices=["base", "dialect"],
        default="base",
        help="Use base language labels or dialect-specific labels in 'language' column",
    )
    p.add_argument(
        "--train-eval-fraction",
        type=float,
        default=0.02,
        help="Fraction of train_clean used to build train_eval subset indices",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for train subset split")
    args = p.parse_args()

    train_root = Path(args.train_root)
    dev_root = Path(args.dev_root)
    test_root = Path(args.test_root) if args.test_root else dev_root

    output_root = Path(args.output_root)
    cache_name = args.cache_name or _default_cache_name(train_root, dev_root, args.label_mode)
    cache_dir = output_root / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building cache at: {cache_dir}")

    train_examples = _collect_examples(train_root, args.label_mode)
    dev_examples = _collect_examples(dev_root, args.label_mode)
    test_examples = _collect_examples(test_root, args.label_mode)

    print(f"Collected train={len(train_examples):,} dev={len(dev_examples):,} test={len(test_examples):,}")

    train_ds = _to_dataset(train_examples)
    dev_ds = _to_dataset(dev_examples)
    test_ds = _to_dataset(test_examples)

    _save_split(cache_dir, "train", train_ds)
    _save_split(cache_dir, "dev", dev_ds)
    _save_split(cache_dir, "test", test_ds)

    _write_clean_indices(cache_dir, "train", len(train_ds))
    _write_clean_indices(cache_dir, "dev", len(dev_ds))
    _write_clean_indices(cache_dir, "test", len(test_ds))
    _write_train_subset(cache_dir, len(train_ds), args.train_eval_fraction, args.seed)

    langs = sorted(set(train_ds["language"]) | set(dev_ds["language"]) | set(test_ds["language"]))
    print(f"Language labels ({len(langs)}): {', '.join(langs)}")
    print("Done.")


if __name__ == "__main__":
    main()
