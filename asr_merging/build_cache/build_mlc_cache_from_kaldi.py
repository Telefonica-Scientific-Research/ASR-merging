#!/usr/bin/env python3
"""Build router-compatible mlc_slm cache from Kaldi-style files.

Expected per-split files under <kaldi_root>/<split>/:
  wav.scp              (required)
  text                 (required)
  segments             (optional)
  utt2lang             (optional)
  reco2lang            (optional)

Outputs cache layout compatible with train/eval routers:
  <output_root>/<cache_name>/
    train/processed
    dev/processed
    test/processed
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
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from datasets import Audio, Dataset, Features, Value


@dataclass
class SegmentRef:
    utt_id: str
    rec_id: str
    start_sec: float
    end_sec: float
    text: str
    language: str


def _read_kaldi_map(path: Path, value_columns: int = 1) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split(maxsplit=value_columns)
        if len(parts) < 2:
            continue
        key = parts[0]
        value = parts[1] if len(parts) == 2 else parts[1]
        out[key] = value
    return out


def _read_kaldi_text(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split(maxsplit=1)
        if len(parts) != 2:
            continue
        out[parts[0]] = parts[1].strip()
    return out


def _read_segments(path: Path) -> List[Tuple[str, str, float, float]]:
    out: List[Tuple[str, str, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        utt_id, rec_id, start_s, end_s = parts[:4]
        start = float(start_s)
        end = float(end_s)
        if end <= start:
            continue
        out.append((utt_id, rec_id, start, end))
    return out


def _normalize_label(x: Optional[str]) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip().lower()
    return s if s else "unknown"


def _read_wav_as_float32(wav_path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sample_width == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width={sample_width} bytes for file: {wav_path}")

    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)

    return arr.astype(np.float32), int(sr)


def _build_segment_refs(split_dir: Path) -> List[SegmentRef]:
    wav_scp = split_dir / "wav.scp"
    text_file = split_dir / "text"
    segments_file = split_dir / "segments"
    utt2lang_file = split_dir / "utt2lang"
    reco2lang_file = split_dir / "reco2lang"

    if not wav_scp.exists():
        raise FileNotFoundError(f"Missing required file: {wav_scp}")
    if not text_file.exists():
        raise FileNotFoundError(f"Missing required file: {text_file}")

    texts = _read_kaldi_text(text_file)
    utt2lang = _read_kaldi_map(utt2lang_file) if utt2lang_file.exists() else {}
    reco2lang = _read_kaldi_map(reco2lang_file) if reco2lang_file.exists() else {}

    refs: List[SegmentRef] = []

    if segments_file.exists():
        for utt_id, rec_id, start, end in _read_segments(segments_file):
            if utt_id not in texts:
                continue
            lang = _normalize_label(utt2lang.get(utt_id, reco2lang.get(rec_id)))
            refs.append(
                SegmentRef(
                    utt_id=utt_id,
                    rec_id=rec_id,
                    start_sec=float(start),
                    end_sec=float(end),
                    text=texts[utt_id],
                    language=lang,
                )
            )
    else:
        # Fallback: treat each recording/text as a full-utterance segment.
        wav_map = _read_kaldi_map(wav_scp)
        for utt_id, txt in texts.items():
            rec_id = utt_id
            if rec_id not in wav_map:
                continue
            lang = _normalize_label(utt2lang.get(utt_id, reco2lang.get(rec_id)))
            refs.append(
                SegmentRef(
                    utt_id=utt_id,
                    rec_id=rec_id,
                    start_sec=0.0,
                    end_sec=-1.0,
                    text=txt,
                    language=lang,
                )
            )

    refs.sort(key=lambda r: r.utt_id)
    if not refs:
        raise RuntimeError(f"No usable segments found in split dir: {split_dir}")
    return refs


def _iter_examples_from_refs(
    refs: List[SegmentRef],
    wav_map: Dict[str, str],
    min_duration_sec: float,
) -> Iterator[Dict]:
    refs_by_rec: Dict[str, List[SegmentRef]] = {}
    for r in refs:
        refs_by_rec.setdefault(r.rec_id, []).append(r)

    for rec_id in sorted(refs_by_rec.keys()):
        wav_path_s = wav_map.get(rec_id)
        if not wav_path_s:
            continue
        wav_path = Path(wav_path_s)
        if not wav_path.exists():
            continue

        audio, sr = _read_wav_as_float32(wav_path)
        n = audio.shape[0]

        for r in refs_by_rec[rec_id]:
            if r.end_sec < 0:
                s0 = 0
                s1 = n
            else:
                s0 = max(0, min(n, int(round(r.start_sec * sr))))
                s1 = max(0, min(n, int(round(r.end_sec * sr))))
            if s1 <= s0:
                continue
            dur = float(s1 - s0) / float(sr)
            if dur < min_duration_sec:
                continue

            clip = np.asarray(audio[s0:s1], dtype=np.float32)
            if clip.size == 0:
                continue

            yield {
                "utt_id": r.utt_id,
                "audio": {"array": clip, "sampling_rate": sr},
                "language": r.language,
                "text": r.text,
            }


def _build_dataset_from_split(split_dir: Path, min_duration_sec: float) -> Dataset:
    wav_map = _read_kaldi_map(split_dir / "wav.scp")
    refs = _build_segment_refs(split_dir)

    features = Features(
        {
            "utt_id": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "language": Value("string"),
            "text": Value("string"),
        }
    )

    ds = Dataset.from_generator(
        _iter_examples_from_refs,
        features=features,
        gen_kwargs={"refs": refs, "wav_map": wav_map, "min_duration_sec": float(min_duration_sec)},
    )

    if len(ds) == 0:
        raise RuntimeError(f"No samples built for split dir: {split_dir}")
    return ds


def _empty_canonical_dataset() -> Dataset:
    ds = Dataset.from_dict(
        {
            "utt_id": [],
            "audio": [],
            "language": [],
            "text": [],
        }
    )
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def _placeholder_test_dataset() -> Dataset:
    """Work around datasets save_to_disk bug on empty Audio datasets.

    Some datasets versions raise ZeroDivisionError when saving an empty dataset
    with complex features. We store one harmless placeholder row and then force
    test_clean indices to empty, so evaluation can safely use test_clean (empty)
    while test from this cache should be ignored in mixed-cache setups.
    """
    ds = Dataset.from_dict(
        {
            "utt_id": ["__empty_test_placeholder__"],
            "audio": [{"array": np.zeros(1600, dtype=np.float32), "sampling_rate": 16000}],
            "language": ["__placeholder__"],
            "text": [""],
        }
    )
    return ds.cast_column("audio", Audio(sampling_rate=16000))


def _save_split(cache_dir: Path, split_name: str, ds: Dataset) -> None:
    out = cache_dir / split_name / "processed"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))


def _write_clean_indices(cache_dir: Path, split_name: str, n: int, keep_indices: Optional[List[int]] = None) -> None:
    clean_root = cache_dir / "clean_index_cache"
    clean_root.mkdir(parents=True, exist_ok=True)
    payload = {"keep_indices": list(range(n)) if keep_indices is None else [int(i) for i in keep_indices]}
    (clean_root / f"{split_name}_clean_indices.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_train_subset(cache_dir: Path, n_train: int, eval_fraction: float, seed: int) -> None:
    idx = list(range(n_train))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    if n_train <= 1:
        eval_idx = [0] if n_train == 1 else []
        finetune_idx = []
    else:
        n_eval = max(1, int(round(n_train * eval_fraction)))
        n_eval = min(n_eval, n_train - 1)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mlc_slm cache from Kaldi-style split files")
    p.add_argument("--kaldi-root", required=True, help="Root containing train/dev/test subdirs with Kaldi files")
    p.add_argument("--output-root", default="data/cache/voxtral")
    p.add_argument("--cache-name", required=True)
    p.add_argument("--train-eval-fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-duration-sec", type=float, default=0.10)
    p.add_argument(
        "--allow-empty-test",
        action="store_true",
        help="Allow an empty test split (useful when test comes from a different cache at runtime).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    kaldi_root = Path(args.kaldi_root)
    split_dirs = {
        "train": kaldi_root / "train",
        "dev": kaldi_root / "dev",
        "test": kaldi_root / "test",
    }
    for sn, sd in split_dirs.items():
        if not sd.exists():
            raise FileNotFoundError(f"Missing Kaldi split dir for {sn}: {sd}")

    cache_dir = Path(args.output_root) / args.cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Building cache at: {cache_dir}")

    ds_train = _build_dataset_from_split(split_dirs["train"], min_duration_sec=args.min_duration_sec)
    ds_dev = _build_dataset_from_split(split_dirs["dev"], min_duration_sec=args.min_duration_sec)

    test_placeholder_used = False
    try:
        ds_test = _build_dataset_from_split(split_dirs["test"], min_duration_sec=args.min_duration_sec)
    except Exception:
        if not args.allow_empty_test:
            raise
        print("Warning: test split produced no usable samples; creating empty canonical test split (--allow-empty-test enabled).")
        ds_test = _placeholder_test_dataset()
        test_placeholder_used = True

    _save_split(cache_dir, "train", ds_train)
    _save_split(cache_dir, "dev", ds_dev)
    _save_split(cache_dir, "test", ds_test)

    _write_clean_indices(cache_dir, "train", len(ds_train))
    _write_clean_indices(cache_dir, "dev", len(ds_dev))
    if test_placeholder_used:
        _write_clean_indices(cache_dir, "test", len(ds_test), keep_indices=[])
    else:
        _write_clean_indices(cache_dir, "test", len(ds_test))
    _write_train_subset(cache_dir, len(ds_train), args.train_eval_fraction, args.seed)

    langs = sorted(set(ds_train["language"]) | set(ds_dev["language"]) | set(ds_test["language"]))
    print(f"Built splits: train={len(ds_train):,} dev={len(ds_dev):,} test={len(ds_test):,}")
    print(f"Language labels ({len(langs)}): {', '.join(langs)}")
    print("Done.")


if __name__ == "__main__":
    main()
