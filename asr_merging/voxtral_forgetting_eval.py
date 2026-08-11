#!/usr/bin/env python3
"""Evaluate Voxtral on non-ASR audio understanding tasks.

This script is intended to measure potential task forgetting after ASR fine-tuning.
It downloads benchmark datasets from Hugging Face (if needed), runs Voxtral in
Audio Instruct mode, and computes task-level accuracy.

Supported tasks:
- mmsu: spoken language understanding and reasoning multiple-choice benchmark
- minds14_intent: spoken intent classification (PolyAI/minds14, en-US)
- esc50_event: environmental sound classification (ashraq/esc50)
- jsonl_audio_mc: local JSONL audio multiple-choice task (custom schema)

Example:
  python -m asr_merging.voxtral_forgetting_eval \
    --model-id mistralai/Voxtral-Mini-3B-2507 \
    --adapter-path experiments/mlc_fast_debug_20260415_233439/checkpoint-7000 \
        --tasks mmsu \
        --split train \
    --max-samples-per-task 200 \
    --timestamped-output
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime as dt
import difflib
import json
import os
import random
import re
import sys
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor
from asr_merging.voxtral_train_router import (
    _offline_aware_from_pretrained_kwargs,
    _resolve_pretrained_source,
)
from asr_merging.voxtral_train_MCQ import (
    _build_audio_source_from_question,
    _extract_time_ranges_from_text,
)

try:
    # Fix for environments where default cuBLAS BF16 GEMM path is unstable.
    torch.backends.cuda.preferred_blas_library("cublaslt")
except Exception:
    pass

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None


DEFAULT_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"


@dataclass
class EvalSample:
    sample_id: str
    audio_path: Optional[str]
    gold_label: str
    metadata: Dict
    audio_obj: Optional[object] = None
    prompt_text: Optional[str] = None
    choice_map: Optional[Dict[str, str]] = None
    gold_choice: Optional[str] = None


@dataclass
class TaskData:
    task_name: str
    split_name: str
    labels: List[str]
    samples: List[EvalSample]
    prompt_template: str
    evaluation_mode: str = "label_match"


SUPPORTED_TASKS = "mmsu, minds14_intent, esc50_event, jsonl_audio_mc"


def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_choice_label(s: str) -> str:
    return (s or "").strip().upper()


def _build_choice_map(row: Dict) -> Dict[str, str]:
    choice_map: Dict[str, str] = {}
    for key in ("A", "B", "C", "D"):
        value = row.get(f"choice_{key.lower()}", "")
        choice_map[key] = "" if value is None else str(value)
    return choice_map


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


def _resolve_correct_choice(choice_map: Dict[str, str], answer_gt: str) -> Optional[str]:
    answer_norm = _normalize_label(answer_gt)
    for key, value in choice_map.items():
        if _normalize_label(value) == answer_norm:
            return key
    return None


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

            # Many MLC26 paths in JSONL use:
            #   mlc-slm-2nd-dev/<lang>/<file>
            # while disk layout is:
            #   mlc-slm-2nd-dev/data/<lang>/<file>
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

    # Keep a deterministic fallback to support downstream error logging.
    if p.is_absolute():
        return str(p)
    if audio_root:
        return str((Path(audio_root) / p).resolve())
    return str(p.resolve())


def _normalize_submission_session_id(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("/", "_")
    s = s.replace("(", "_").replace(")", "_")
    s = re.sub(r"__+", "_", s)
    return s.strip("_")


def _session_id_from_audio_path(audio_path: str) -> str:
    p = Path(audio_path)
    stem = p.stem
    parts = list(p.parts)

    # Heuristic: use path segments after a "data" directory when present.
    rel = parts
    if "data" in parts:
        idx = len(parts) - 1 - parts[::-1].index("data")
        rel = parts[idx + 1 :]
    if len(rel) >= 2:
        rel_dirs = rel[:-1]
    else:
        rel_dirs = parts[:-1]

    # Common multilingual folder mappings.
    if len(rel_dirs) >= 2 and rel_dirs[-2] == "English":
        sub = rel_dirs[-1]
        if sub in {"American", "Australian", "British", "Filipino", "Indian"}:
            return f"English_{sub}_{stem}"

    if len(rel_dirs) >= 1:
        last = rel_dirs[-1]
        if last == "French(Canada)":
            return f"French_Canada_{stem}"
        if last == "Portuguese(Brazil)":
            return f"Portuguese_Brazil_{stem}"
        if last == "Spanish(Mexico)":
            return f"Spanish_Mexico_{stem}"

    # Generic fallback for single-language directories (e.g., Thai/xxx.wav).
    if len(rel_dirs) >= 1:
        return f"{_normalize_submission_session_id(rel_dirs[-1])}_{stem}"
    return stem


def _build_audio_index(audio_root: Optional[str]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    by_session: Dict[str, str] = {}
    by_stem: Dict[str, List[str]] = defaultdict(list)

    if not audio_root:
        return by_session, by_stem

    root = Path(audio_root)
    if not root.exists():
        return by_session, by_stem

    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    for fp in root.rglob("*"):
        if not fp.is_file() or fp.suffix.lower() not in exts:
            continue
        resolved = str(fp.resolve())
        sid = _normalize_submission_session_id(_session_id_from_audio_path(resolved))
        by_session.setdefault(sid, resolved)
        by_stem[fp.stem].append(resolved)
    return by_session, by_stem


def _relative_audio_paths_from_session_id(session_id: str) -> List[str]:
    sid = _normalize_submission_session_id(session_id)
    if not sid:
        return []
    parts = sid.split("_")
    if len(parts) < 2:
        return []

    # Session-id schema is language-prefix + audio stem, where language-prefix
    # can be one token (e.g., French) or two tokens (e.g., English_British).
    two_token_langs = {
        ("English", "American"),
        ("English", "Australian"),
        ("English", "British"),
        ("English", "Filipino"),
        ("English", "Indian"),
        ("French", "Canada"),
        ("Portuguese", "Brazil"),
        ("Spanish", "Mexico"),
    }

    prefix_len = 2 if len(parts) >= 2 and (parts[0], parts[1]) in two_token_langs else 1
    stem_parts = parts[prefix_len:]
    if not stem_parts:
        return []
    stem = "_".join(stem_parts)
    lang = parts[0]

    if sid.startswith("English_American_"):
        return [f"English/American/{stem}.wav"]
    if sid.startswith("English_Australian_"):
        return [f"English/Australian/{stem}.wav"]
    if sid.startswith("English_British_"):
        return [f"English/British/{stem}.wav"]
    if sid.startswith("English_Filipino_"):
        return [f"English/Filipino/{stem}.wav"]
    if sid.startswith("English_Indian_"):
        return [f"English/Indian/{stem}.wav"]
    if sid.startswith("French_Canada_"):
        return [
            f"French(Canada)/{stem}.wav",
            f"French_Canada/{stem}.wav",
        ]
    if sid.startswith("Portuguese_Brazil_"):
        return [
            f"Portuguese(Brazil)/{stem}.wav",
            f"Portuguese_Brazil/{stem}.wav",
        ]
    if sid.startswith("Spanish_Mexico_"):
        return [
            f"Spanish(Mexico)/{stem}.wav",
            f"Spanish_Mexico/{stem}.wav",
        ]

    return [f"{lang}/{stem}.wav"]


def _extract_audio_path_from_row(row: Dict) -> Optional[str]:
    audio = row.get("audio")

    # datasets<=2 often returns a dict with a concrete path.
    if isinstance(audio, dict):
        p = audio.get("path")
        if p:
            return str(p)

    # datasets with torchcodec may return an AudioDecoder-like object.
    meta = getattr(audio, "metadata", None)
    if meta is not None:
        p = getattr(meta, "path", None)
        if p:
            return str(p)

    # Dataset-level fallback columns (e.g., PolyAI/minds14 has `path`).
    for key in ("path", "audio_path", "file", "filename"):
        p = row.get(key)
        if isinstance(p, str) and p.strip():
            return p

    return None


def _is_existing_local_path(p: str) -> bool:
    if not p:
        return False
    if p.startswith("http://") or p.startswith("https://"):
        return True
    return Path(p).exists()


def _write_wav_from_samples(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim == 2:
        x = x[0]
    x = np.clip(x, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(x_i16.tobytes())


def _materialize_audio_path(sample: EvalSample, temp_audio_dir: Path) -> str:
    if sample.audio_path and _is_existing_local_path(sample.audio_path):
        return sample.audio_path

    audio = sample.audio_obj
    if isinstance(audio, dict) and audio.get("array") is not None:
        sr = int(audio.get("sampling_rate") or 16000)
        out = temp_audio_dir / f"{sample.sample_id.replace(':', '_')}.wav"
        _write_wav_from_samples(out, np.asarray(audio["array"]), sr)
        return str(out)

    if audio is not None and hasattr(audio, "get_all_samples"):
        decoded = audio.get_all_samples()
        data = getattr(decoded, "data", None)
        sr = int(getattr(decoded, "sample_rate", 16000))
        if data is not None:
            arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
            out = temp_audio_dir / f"{sample.sample_id.replace(':', '_')}.wav"
            _write_wav_from_samples(out, arr, sr)
            return str(out)

    raise RuntimeError(
        f"Could not resolve audio for sample {sample.sample_id}. "
        f"audio_path={sample.audio_path!r} audio_obj_type={type(sample.audio_obj)}"
    )


def _load_model(model_id: str, adapter_path: Optional[str], dtype: torch.dtype):
    # Single-device placement is more stable for this eval path under some
    # runtime stacks than multi-GPU auto-sharding.
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model = VoxtralForConditionalGeneration.from_pretrained(
        _resolve_pretrained_source(model_id),
        torch_dtype=dtype,
        device_map=device_map,
        **_offline_aware_from_pretrained_kwargs(),
    )
    model_mode = "baseline"
    resolved_adapter = None

    if adapter_path:
        if PeftModel is None:
            raise RuntimeError("peft is required for adapter evaluation but is not installed.")
        resolved = _resolve_adapter_path(adapter_path)
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve adapter path from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(resolved), is_trainable=False)
        model_mode = "adapter"
        resolved_adapter = str(resolved)

    return model, model_mode, resolved_adapter


def _resolve_adapter_path(path_str: str) -> Optional[Path]:
    cp = Path(path_str)
    if not cp.exists():
        return None

    if (cp / "adapter_config.json").exists():
        return cp

    final_model = cp / "final_model"
    if (final_model / "adapter_config.json").exists():
        return final_model

    checkpoints = sorted(
        [p for p in cp.glob("checkpoint-*") if p.is_dir() and (p / "adapter_config.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def _pick_split(ds: Dataset, desired: str) -> Dataset:
    # Handle datasets where only one split is available.
    return ds if isinstance(ds, Dataset) else ds[desired]


def _build_stratified_partitions(
    labels: List[int],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> Dict[str, List[int]]:
    buckets: Dict[int, List[int]] = {}
    for idx, y in enumerate(labels):
        buckets.setdefault(int(y), []).append(idx)

    rng = random.Random(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, idxs in buckets.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n <= 2:
            # Tiny class buckets go to test to avoid training leakage into eval partitions.
            test_idx.extend(idxs)
            continue

        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        while n_test + n_val >= n:
            if n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        test_idx.extend(idxs[:n_test])
        val_idx.extend(idxs[n_test : n_test + n_val])
        train_idx.extend(idxs[n_test + n_val :])

    return {
        "train": sorted(train_idx),
        "validation": sorted(val_idx),
        "test": sorted(test_idx),
    }


def _build_minds14_task(
    split: str,
    max_samples: int,
    seed: int,
    cache_dir: Optional[str],
) -> TaskData:
    dsd = load_dataset("PolyAI/minds14", "en-US", cache_dir=cache_dir)
    ds = dsd["train"]

    split_norm = str(split).strip().lower()
    if split_norm in {"train", "all", "all_folds"}:
        raise ValueError(
            "minds14_intent does not allow training-split evaluation. "
            "Use --split test (recommended) or --split validation."
        )
    if split_norm not in {"test", "validation"}:
        raise ValueError("minds14_intent supports only split=test or split=validation.")

    feat = ds.features["intent_class"]
    labels = [str(x) for x in feat.names]

    partitions = _build_stratified_partitions(
        [int(x) for x in ds["intent_class"]],
        seed=seed,
        val_fraction=0.15,
        test_fraction=0.15,
    )

    indices = list(partitions[split_norm])
    random.Random(seed).shuffle(indices)
    if max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]

    samples: List[EvalSample] = []
    for i in indices:
        row = ds[int(i)]
        audio_path = _extract_audio_path_from_row(row)
        audio_obj = row.get("audio")
        if not audio_path and audio_obj is None:
            continue
        label = labels[int(row["intent_class"])]
        samples.append(
            EvalSample(
                sample_id=f"minds14:{split_norm}:{i}",
                audio_path=(None if audio_path is None else str(audio_path)),
                gold_label=label,
                metadata={
                    "transcription": row.get("transcription"),
                    "english_transcription": row.get("english_transcription"),
                    "lang_id": row.get("lang_id"),
                },
                audio_obj=audio_obj,
            )
        )

    prompt = (
        "You are given one spoken utterance for banking support. "
        "Classify the user intent into exactly one label from this list: {labels}. "
        "Respond with only the label and nothing else."
    )

    if not samples:
        row0 = ds[0] if len(ds) > 0 else {}
        raise RuntimeError(
            "minds14_intent resolved zero usable samples. "
            f"Available columns={ds.column_names}; first-row keys={list(row0.keys()) if isinstance(row0, dict) else type(row0)}"
        )

    return TaskData(
        task_name="minds14_intent",
        split_name=f"stratified_{split_norm}",
        labels=labels,
        samples=samples,
        prompt_template=prompt,
    )


def _build_esc50_task(
    split: str,
    max_samples: int,
    seed: int,
    cache_dir: Optional[str],
) -> TaskData:
    # ESC-50 on HF provides a single train split with folds 1..5.
    ds = load_dataset("ashraq/esc50", split="train", cache_dir=cache_dir)

    if split == "test":
        ds = ds.filter(lambda x: int(x["fold"]) == 5)
        split_name = "fold5"
    elif split == "validation":
        ds = ds.filter(lambda x: int(x["fold"]) == 4)
        split_name = "fold4"
    elif split.startswith("fold") and split[4:].isdigit():
        fold_id = int(split[4:])
        if fold_id < 1 or fold_id > 5:
            raise ValueError("ESC-50 fold must be in fold1..fold5")
        ds = ds.filter(lambda x: int(x["fold"]) == fold_id)
        split_name = f"fold{fold_id}"
    else:
        split_name = "all_folds"

    labels = sorted({str(x) for x in ds["category"]})

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    if max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]

    samples: List[EvalSample] = []
    for i in indices:
        row = ds[int(i)]
        audio_path = _extract_audio_path_from_row(row)
        audio_obj = row.get("audio")
        if not audio_path and audio_obj is None:
            continue
        samples.append(
            EvalSample(
                sample_id=f"esc50:{i}",
                audio_path=(None if audio_path is None else str(audio_path)),
                gold_label=str(row["category"]),
                metadata={
                    "fold": row.get("fold"),
                    "esc10": row.get("esc10"),
                    "target": row.get("target"),
                    "filename": row.get("filename"),
                },
                audio_obj=audio_obj,
            )
        )

    prompt = (
        "Classify the environmental sound in this audio clip into exactly one label from this list: {labels}. "
        "Respond with only the label and nothing else."
    )

    if not samples:
        row0 = ds[0] if len(ds) > 0 else {}
        raise RuntimeError(
            "esc50_event resolved zero usable samples. "
            f"Available columns={ds.column_names}; first-row keys={list(row0.keys()) if isinstance(row0, dict) else type(row0)}"
        )

    return TaskData(
        task_name="esc50_event",
        split_name=split_name,
        labels=labels,
        samples=samples,
        prompt_template=prompt,
    )


def _build_mmsu_task(
    split: str,
    max_samples: int,
    seed: int,
    cache_dir: Optional[str],
) -> TaskData:
    ds = load_dataset("ddwang2000/MMSU", split=split, cache_dir=cache_dir)

    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    if max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]

    samples: List[EvalSample] = []
    for i in indices:
        row = ds[int(i)]
        audio_path = _extract_audio_path_from_row(row)
        audio_obj = row.get("audio")
        if not audio_path and audio_obj is None:
            continue

        choice_map = _build_choice_map(row)
        answer_gt = "" if row.get("answer_gt") is None else str(row.get("answer_gt"))
        gold_choice = _resolve_correct_choice(choice_map, answer_gt)
        if gold_choice is None:
            raise RuntimeError(
                f"Could not resolve correct MMSU choice for sample {row.get('id', i)}. "
                f"answer_gt={answer_gt!r} choices={choice_map}"
            )

        question = "" if row.get("question") is None else str(row.get("question"))
        prompt = (
            "Choose the most suitable answer from options A, B, C, and D. "
            "You must respond with only A, B, C, or D.\n\n"
            f"Question: {question}\n\n"
            f"A. {choice_map['A']}\n"
            f"B. {choice_map['B']}\n"
            f"C. {choice_map['C']}\n"
            f"D. {choice_map['D']}"
        )

        samples.append(
            EvalSample(
                sample_id=f"mmsu:{row.get('id', i)}",
                audio_path=(None if audio_path is None else str(audio_path)),
                gold_label=answer_gt,
                metadata={
                    "id": row.get("id", i),
                    "question": question,
                    "task_name": row.get("task_name"),
                    "category": row.get("category"),
                    "sub-category": row.get("sub-category"),
                    "sub-sub-category": row.get("sub-sub-category"),
                    "linguistics_sub_discipline": row.get("linguistics_sub_discipline"),
                },
                audio_obj=audio_obj,
                prompt_text=prompt,
                choice_map=choice_map,
                gold_choice=gold_choice,
            )
        )

    if not samples:
        row0 = ds[0] if len(ds) > 0 else {}
        raise RuntimeError(
            "mmsu resolved zero usable samples. "
            f"Available columns={ds.column_names}; first-row keys={list(row0.keys()) if isinstance(row0, dict) else type(row0)}"
        )

    return TaskData(
        task_name="mmsu",
        split_name=str(split),
        labels=["A", "B", "C", "D"],
        samples=samples,
        prompt_template="",
        evaluation_mode="multiple_choice",
    )


def _build_jsonl_audio_mc_task(
    split: str,
    max_samples: int,
    seed: int,
    cache_dir: Optional[str],
    jsonl_path: Optional[str],
    audio_root: Optional[str],
    max_questions_per_audio: int,
    prediction_only: bool,
    yesno_binary: bool = False,
) -> TaskData:
    del cache_dir  # Unused but kept for a common task-builder signature style.

    if not jsonl_path:
        raise ValueError("jsonl_audio_mc requires --jsonl-path")

    src = Path(jsonl_path)
    if not src.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    samples: List[EvalSample] = []
    all_labels = set()
    audio_by_session, audio_by_stem = _build_audio_index(audio_root)

    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            raw_audio_path = str(row.get("path") or row.get("audio_path") or "").strip()

            # Supported schema A (grouped): one audio row with "questions": [...]
            # Supported schema B (challenge): one question per row with session_id/question_stem/options.
            if isinstance(row.get("questions"), list):
                questions = row.get("questions") or []
                if not raw_audio_path:
                    continue
            else:
                q_single = {
                    "question_stem": row.get("question_stem"),
                    "options": row.get("options"),
                    "correct_answer": row.get("correct_answer"),
                    "question_id": row.get("question_id"),
                }
                questions = [q_single]

                sess_raw = _normalize_submission_session_id(str(row.get("session_id") or "").strip())
                if raw_audio_path:
                    pass
                elif sess_raw:
                    if sess_raw in audio_by_session:
                        raw_audio_path = audio_by_session[sess_raw]
                    else:
                        stem_guess = "_".join(sess_raw.split("_")[-3:])
                        cands = audio_by_stem.get(stem_guess, [])
                        if len(cands) == 1:
                            raw_audio_path = cands[0]
                        else:
                            rels = _relative_audio_paths_from_session_id(sess_raw)
                            for rel in rels:
                                cand = _resolve_jsonl_audio_path(rel, audio_root)
                                if Path(cand).exists():
                                    raw_audio_path = cand
                                    break
                            if not raw_audio_path and rels:
                                raw_audio_path = rels[0]

            if max_questions_per_audio > 0:
                questions = questions[:max_questions_per_audio]

            if not raw_audio_path:
                continue
            audio_path = _resolve_jsonl_audio_path(raw_audio_path, audio_root)
            fixed_session_id = _normalize_submission_session_id(
                _session_id_from_audio_path(audio_path)
                if audio_path
                else str(row.get("session_id") or "")
            )

            for qi, q in enumerate(questions):
                if not isinstance(q, dict):
                    continue

                question = "" if q.get("question_stem") is None else str(q.get("question_stem"))
                choice_map = _build_dynamic_choice_map(q.get("options") or [])
                if len(choice_map) < 2:
                    continue

                gold_raw = "" if q.get("correct_answer") is None else str(q.get("correct_answer"))
                gold_choice = _resolve_correct_choice_dynamic(choice_map, gold_raw)
                if not prediction_only and gold_choice is None:
                    continue

                # Neutral ordinal relabeling for 2-option questions.
                # Replaces A/B with 1/2 in the prompt and scoring to reduce the
                # MCQ-specific A-slot positional prior without introducing any
                # semantic assumption (unlike Yes/No which implies affirmative/negative).
                # The canonical A/B labels are kept in metadata for reverse mapping.
                yn_to_orig: Optional[Dict[str, str]] = None
                orig_choice_map = choice_map
                if yesno_binary and len(choice_map) == 2:
                    orig_labels = list(choice_map.keys())
                    yn_labels = ["1", "2"]
                    yn_to_orig = {yn: orig for yn, orig in zip(yn_labels, orig_labels)}
                    choice_map = {yn: choice_map[orig] for yn, orig in yn_to_orig.items()}
                    if gold_choice is not None:
                        # Remap gold_choice to the 1/2 label space
                        orig_to_yn = {v: k for k, v in yn_to_orig.items()}
                        gold_choice = orig_to_yn.get(gold_choice, gold_choice)

                choice_lines = [f"{lbl}. {txt}" for lbl, txt in choice_map.items()]
                prompt = (
                    _MCQ_AUDIO_FOCUS_PREFIX
                    + "Choose the most suitable answer from the options below. "
                    f"You must respond with only one label from: {', '.join(choice_map.keys())}.\n\n"
                    f"Question: {question}\n\n"
                    + "\n".join(choice_lines)
                )

                qid = q.get("question_id") or str(qi)
                sample_id = f"jsonl:{line_no}:{qid}"
                samples.append(
                    EvalSample(
                        sample_id=sample_id,
                        audio_path=audio_path,
                        gold_label=gold_raw,
                        metadata={
                            "jsonl_line": line_no,
                            "source_audio_path": raw_audio_path,
                            "question_id": qid,
                            "session_id": fixed_session_id,
                            "question": question,
                            "yn_to_orig": yn_to_orig,
                            "orig_choice_map": orig_choice_map if yn_to_orig else None,
                        },
                        prompt_text=prompt,
                        choice_map=choice_map,
                        gold_choice=gold_choice,
                    )
                )
                all_labels.update(choice_map.keys())

    if not prediction_only:
        rng = random.Random(seed)
        rng.shuffle(samples)
    if max_samples > 0:
        samples = samples[: min(max_samples, len(samples))]

    if not samples:
        raise RuntimeError(
            "jsonl_audio_mc resolved zero usable samples. "
            "Check JSONL schema, correct_answer labels, and audio paths."
        )

    return TaskData(
        task_name="jsonl_audio_mc",
        split_name=str(split),
        labels=sorted(all_labels),
        samples=samples,
        prompt_template="",
        evaluation_mode="multiple_choice",
    )


def _parse_tasks(raw: str) -> List[str]:
    out = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            out.append(x)
    if not out:
        raise ValueError(f"No tasks selected. Use --tasks {SUPPORTED_TASKS}")
    return out


def _select_predicted_label(text: str, labels: Sequence[str]) -> str:
    raw = (text or "").strip()
    nraw = _normalize_label(raw)

    norm_to_label = {_normalize_label(lbl): lbl for lbl in labels}

    # 1) exact normalized match
    if nraw in norm_to_label:
        return norm_to_label[nraw]

    # 2) substring match (prefer longer labels first)
    for nlabel, orig in sorted(norm_to_label.items(), key=lambda kv: len(kv[0]), reverse=True):
        if nlabel and nlabel in nraw:
            return orig

    # 3) fuzzy fallback
    cand = difflib.get_close_matches(nraw, list(norm_to_label.keys()), n=1, cutoff=0.0)
    if cand:
        return norm_to_label[cand[0]]

    return raw


def _select_multiple_choice_option(text: str, options: Sequence[str]) -> Optional[str]:
    raw = (text or "").strip()
    if not raw:
        return None

    allowed = [_normalize_choice_label(str(x)) for x in options]
    allowed_set = set(allowed)

    raw_up = _normalize_choice_label(raw)
    if raw_up in allowed_set:
        return raw_up

    raw_norm = _normalize_label(raw)
    for label in allowed:
        if _normalize_label(label) == raw_norm:
            return label

    for token in re.findall(r"[A-Za-z0-9_]+", raw_up):
        if token in allowed_set:
            return token

    for label in sorted(allowed, key=len):
        if len(label) == 1 and label in raw_up:
            return label

    return None


def _option_label_token_ids(processor, label: str) -> List[int]:
    """First-token ids the model would emit for an option label (e.g. "A").

    Handles both the bare letter and the space-prefixed sub-word variant that
    sentencepiece/tekken tokenizers produce at the start of a generated answer.
    Results are cached because the option labels (A/B/C/D) repeat across every
    sample within a run.
    """
    cache = getattr(_option_label_token_ids, "_cache", None)
    if cache is None:
        cache = {}
        _option_label_token_ids._cache = cache
    if label in cache:
        return cache[label]

    tok = getattr(processor, "tokenizer", processor)
    ids: set = set()
    for variant in (str(label), " " + str(label)):
        try:
            enc = tok.encode(variant, add_special_tokens=False)
        except TypeError:
            enc = tok.encode(variant)
        if enc:
            ids.add(int(enc[-1]))
    result = sorted(ids)
    cache[label] = result
    return result


# ---------------------------------------------------------------------------
# Audio-focus prefix prepended to ALL MCQ prompts (main questions and ICL shots).
# Trains/reminds the model to focus on paralinguistic audio cues, not just words.
# ---------------------------------------------------------------------------
_MCQ_AUDIO_FOCUS_PREFIX: str = (
    "Listen carefully to the audio \u2014 the answer depends on how the "
    "speaker sounds (tone, intonation, pace, voice quality), not just "
    "the words spoken. "
)

# ICL few-shot format examples (text-only, no audio).
# These are prepended as user/assistant turns before the actual audio question
# when --few-shot-count > 0, showing the model to output a bare letter only.
#
# Layout (indices):
#   [0], [1] — 2-option (A/B only) examples: used when --few-shot-count 2 to
#               demonstrate the model must pick from {A, B} and not hallucinate C/D.
#   [2], [3], [4] — 4-option (A/B/C/D) examples: used alone with --few-shot-count 2
#               if a different slice is selected, or combined via --few-shot-count 4/5.
# ---------------------------------------------------------------------------
_ICL_FORMAT_EXAMPLES: List[Tuple[str, str]] = [
    # --- 2-option examples (slots [0] and [1]) ---
    (
        "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B.\n\n"
        "Question: Are the two voices heard in this conversation the same person or two different speakers?\n\n"
        "A. The same speaker throughout.\nB. Two distinct speakers.",
        "B",
    ),
    (
        "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B.\n\n"
        "Question: Does the speaker sound hesitant or confident when delivering the statement?\n\n"
        "A. Confident and assertive.\nB. Hesitant and uncertain.",
        "B",
    ),
    # --- 4-option examples (slots [2], [3], [4]) ---
    (
        "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        "Question: Which term best describes speech delivered at a normal, comfortable speaking rate?\n\n"
        "A. Unusually rapid\nB. Slightly accelerated\nC. Normal pace\nD. Very slow",
        "C",
    ),
    (
        "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        "Question: What label fits a recording environment with minimal background noise?\n\n"
        "A. Quiet indoor setting\nB. Noisy outdoor space\nC. Crowded venue\nD. Windy exterior",
        "A",
    ),
    (
        "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        "Question: Which descriptor applies to speech that is clearly articulated and easy to understand?\n\n"
        "A. Muffled\nB. Intelligible\nC. Distorted\nD. Incoherent",
        "B",
    ),
]


# ---------------------------------------------------------------------------
# Multimodal ICL examples (audio-grounded, 30-second clips).
# Audio filenames are relative to _ICL_MM_AUDIO_DIR (resolved from this file).
# Layout:
#   [0-3] — 4-option examples, balanced labels A/B/C/D:
#            diverse dialects (Australian, British×2, Indian) and topics
#            (speaker ID, speaking rate, emotion, disfluency).
#   [4-5] — 2-option examples, balanced labels A/B:
#            (Australian voice comparison, Indian speaker tracking).
# When --icl-multimodal is set, 4-opt questions receive examples [0-3] and
# 2-opt questions receive examples [4-5]; few-shot-count is not used.
# ---------------------------------------------------------------------------
_ICL_MM_AUDIO_DIR: str = "data/mlc26_task2/icl_examples"
_ICL_MULTIMODAL_EXAMPLES: List[Tuple[str, str, str]] = [
    # fmt: off  (long string literals — keep on one line for readability)
    # --- 4-option examples (slots [0-3], answers A / B / C / D) ---
    (
        "icl_4opt_A_eng_aus.wav",
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        'Question: Who speaks the line "They\'re grooming us" at [518.87, 520.22]?\n\n'
        "A. The speaker who previously owned a cafe.\n"
        "B. The speaker who used to be a programmer.\n"
        "C. Both speakers say it at the same time.\n"
        "D. It is not clear who the speaker is.",
        "A",
    ),
    (
        "icl_4opt_B_eng_brit.wav",
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        "Question: At [43.88, 48.99], the speaker says \'Yes, that\'s okay with me right, what\'s your first\'. "
        "What does his delivery reveal about his attitude?\n\n"
        "A. He speaks very slowly and hesitantly, showing reluctance.\n"
        "B. His quick pace and eager tone after agreeing show he is keen to continue.\n"
        "C. He laughs, indicating he thinks the topic is funny.\n"
        "D. He sighs, showing he is bored with the topic already.",
        "B",
    ),
    (
        "icl_4opt_C_eng_brit.wav",
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        "Question: How does the speaker at [365.75, 372.42] sound when he says \'Really. Wow\' "
        "in reaction to something he just heard?\n\n"
        "A. Excited and curious.\n"
        "B. Calm and indifferent.\n"
        "C. Shocked and slightly disgusted.\n"
        "D. Amused and intrigued.",
        "C",
    ),
    (
        "icl_4opt_D_eng_ind.wav",
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        "Question: In the phrase \'Uh my just a co - co - college is open\' [641.94, 648.49], "
        "what do the sounds the speaker makes primarily indicate?\n\n"
        "A. He is angry.\n"
        "B. He is excited.\n"
        "C. He is emphasizing the word \'college\'.\n"
        "D. He is hesitating or stumbling over his words.",
        "D",
    ),
    # --- 2-option examples (slots [4-5], answers A / B) ---
    (
        "icl_2opt_A_eng_aus.wav",
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B.\n\n"
        "Question: In the exchange between [127.92, 130.56], two speakers both say "
        "\'it does the job\'. Which speaker proposed the phrase first?\n\n"
        "A. The speaker with the slightly deeper voice who proposed the phrase.\n"
        "B. The speaker with the slightly higher-pitched voice who agreed with the phrase.",
        "A",
    ),
    (
        "icl_2opt_B_eng_ind.wav",
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B.\n\n"
        "Question: Between [215.68, 222.71], one speaker asks about the developer of cricket "
        "and the other admits not knowing. Who asks the question?\n\n"
        "A. The speaker who advocates for PUBG.\n"
        "B. The speaker who advocates for cricket.",
        "B",
    ),
]  # fmt: on

# Non-English audio variant: identical to _ICL_MULTIMODAL_EXAMPLES except
# slot [3] (4-opt-D) uses a Vietnamese audio clip with an English-translated
# question, teaching the model to answer in English regardless of audio language.
_ICL_MULTIMODAL_EXAMPLES_NONEN: List[Tuple[str, str, str]] = [
    _ICL_MULTIMODAL_EXAMPLES[0],  # 4opt-A  English_Australian
    _ICL_MULTIMODAL_EXAMPLES[1],  # 4opt-B  English_British
    _ICL_MULTIMODAL_EXAMPLES[2],  # 4opt-C  English_British
    (                             # 4opt-D  Vietnamese audio + English question
        "icl_4opt_D_vie.wav",
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        "You must respond with only one label from: A, B, C, D.\n\n"
        "Question: What emotions are expressed in the word \"H\u1ea3?\" said at [383.54, 383.92]?\n\n"
        "A. Agreeing and understanding.\n"
        "B. Joy and excitement.\n"
        "C. Confusion and hearing difficulty.\n"
        "D. Shocked and disbelief.",
        "D",
    ),
    _ICL_MULTIMODAL_EXAMPLES[4],  # 2opt-A  English_Australian
    _ICL_MULTIMODAL_EXAMPLES[5],  # 2opt-B  English_Indian
]


def _render_mc_prompt(question: str, choice_map: Dict[str, str]) -> str:
    """Rebuild the jsonl_audio_mc core prompt for a given label->text mapping.

    Mirrors the template used in _build_jsonl_audio_mc_task so that
    option-permutation debiasing presents prompts identical to the canonical
    eval apart from which content sits at each label slot.
    """
    choice_lines = [f"{lbl}. {txt}" for lbl, txt in choice_map.items()]
    return (
        _MCQ_AUDIO_FOCUS_PREFIX
        + "Choose the most suitable answer from the options below. "
        f"You must respond with only one label from: {', '.join(choice_map.keys())}.\n\n"
        f"Question: {question}\n\n"
        + "\n".join(choice_lines)
    )


def _run_audio_instruction(
    model,
    processor,
    sample: EvalSample,
    audio_path: str,
    prompt: str,
    model_id: str,
    max_new_tokens: int,
    prompt_prefix: str = "",
    crop_from_question_refs: bool = False,
    crop_collar_seconds: float = 30.0,
    random_crop_seconds: float = 0.0,
    transcript_dir: Optional[str] = None,
    transcript_max_words: int = 0,
    score_labels: Optional[Sequence[str]] = None,
    few_shot_count: int = 0,
    icl_multimodal: bool = False,
    icl_multimodal_nonen: bool = False,
) -> Union[str, Dict[str, float]]:
    # Preferred audio-instruct path.
    effective_prompt = (prompt_prefix + prompt) if prompt_prefix else prompt

    # Optionally prepend the ASR transcript so the LLM can cross-reference text
    # with audio when answering the MCQ.
    transcript = _load_transcript(audio_path, transcript_dir, transcript_max_words)
    if transcript:
        effective_prompt = f"Transcript:\n{transcript}\n\n{effective_prompt}"

    # Eval-time audio cropping (opt-in via crop_from_question_refs=True).
    # Uses the question/prompt text for timestamp detection.
    # Questions without detected timestamps pass full audio (random_crop_seconds=0).
    if crop_from_question_refs:
        crop_audio_source, _crop_key, _allow_cache, _crop_windows = _build_audio_source_from_question(
            audio_path=audio_path,
            question_text=sample.prompt_text or prompt,
            sample_id=sample.sample_id,
            crop_from_question_refs=True,
            crop_collar_seconds=crop_collar_seconds,
            random_crop_seconds=random_crop_seconds,
        )
        if isinstance(crop_audio_source, np.ndarray):
            # processor.apply_chat_template requires a file path for audio;
            # save the cropped ndarray to a system temp file (NOT the audio
            # parent dir — that would pollute the dataset with tmp*_crop.wav files).
            import tempfile as _tempfile
            _crop_tmp_fd, _crop_tmp_str = _tempfile.mkstemp(suffix="_crop.wav")
            os.close(_crop_tmp_fd)
            _crop_tmp = Path(_crop_tmp_str)
            _write_wav_from_samples(_crop_tmp, crop_audio_source, 16000)
            audio_content: Dict = {"type": "audio", "path": str(_crop_tmp)}
        else:
            _crop_tmp = None
            audio_content = {"type": "audio", "path": str(crop_audio_source)}
    else:
        audio_content = {"type": "audio", "path": audio_path}

    conversation = []
    # Prepend ICL examples (text-only or multimodal) to teach the model to output a bare letter.
    if icl_multimodal and few_shot_count > 0:
        _icl_base = _ICL_MULTIMODAL_EXAMPLES_NONEN if icl_multimodal_nonen else _ICL_MULTIMODAL_EXAMPLES
        _n_opts = len(sample.choice_map or {})
        if few_shot_count >= len(_icl_base):
            # Type-adaptive: use 2-opt pool for 2-opt questions, 4-opt pool otherwise.
            _icl_pool = _icl_base[4:6] if _n_opts == 2 else _icl_base[0:4]
        else:
            # Non-adaptive: use first few_shot_count examples for all question types.
            _icl_pool = _icl_base[:few_shot_count]
        _icl_audio_base = Path(__file__).resolve().parent.parent / _ICL_MM_AUDIO_DIR
        for _icl_fname, _icl_q, _icl_a in _icl_pool:
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(_icl_audio_base / _icl_fname)},
                    {"type": "text", "text": _icl_q},
                ],
            })
            conversation.append({"role": "assistant", "content": _icl_a})
    else:
        # Text-only ICL examples (legacy).
        for _icl_q, _icl_a in _ICL_FORMAT_EXAMPLES[:few_shot_count]:
            conversation.append({"role": "user", "content": _icl_q})
            conversation.append({"role": "assistant", "content": _icl_a})
    conversation.append(
        {
            "role": "user",
            "content": [
                audio_content,
                {"type": "text", "text": effective_prompt},
            ],
        }
    )

    try:
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    except Exception as chat_err:  # noqa: BLE001
        # Fallback for runtimes where processor chat-template compilation fails
        # (e.g., "Can't compile non template nodes").
        if not getattr(_run_audio_instruction, "_warned_template_fallback", False):
            print(
                "Warning: processor.apply_chat_template failed; using tokenizer chat-template fallback. "
                f"error={type(chat_err).__name__}: {chat_err}"
            )
            _run_audio_instruction._warned_template_fallback = True

        encoded = processor.tokenizer.apply_chat_template(
            [conversation],
            return_tensors=None,
        )
        if isinstance(encoded, dict):
            chat_audio = encoded.pop("audio", None)
        else:
            chat_audio = encoded["audio"] if "audio" in encoded else None
            if "audio" in encoded:
                del encoded["audio"]

        if chat_audio is None:
            raise RuntimeError("Tokenizer chat template did not return audio content for audio-instruct fallback.")

        inputs = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "input_features": _build_voxtral_chat_input_features(processor, chat_audio),
        }
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    if score_labels is not None:
        # Logit-scoring path for option-permutation debiasing.  Instead of
        # generating free text, read the next-token distribution and return the
        # log-sum-exp logprob mass for each requested option label.
        with torch.no_grad():
            model_out = model(**inputs)
        next_token_logits = model_out.logits[0, -1, :].float()
        logprobs = torch.log_softmax(next_token_logits, dim=-1)
        scores: Dict[str, float] = {}
        for lbl in score_labels:
            tok_ids = _option_label_token_ids(processor, str(lbl))
            if tok_ids:
                cand = torch.stack([logprobs[t] for t in tok_ids])
                scores[str(lbl)] = float(torch.logsumexp(cand, dim=0))
            else:
                scores[str(lbl)] = float("-inf")

        # Clean up the temporary crop WAV written to the system temp dir.
        if crop_from_question_refs and "_crop_tmp" in dir() and _crop_tmp is not None:
            try:
                _crop_tmp.unlink(missing_ok=True)
            except Exception:
                pass
        return scores

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)

    # Clean up the temporary crop WAV written to the system temp dir.
    if crop_from_question_refs and "_crop_tmp" in dir() and _crop_tmp is not None:
        try:
            _crop_tmp.unlink(missing_ok=True)
        except Exception:
            pass

    return decoded[0] if decoded else ""


def _predict_choice_cyclic_debias(
    model,
    processor,
    sample: EvalSample,
    audio_path: str,
    model_id: str,
    prompt_prefix: str = "",
    crop_from_question_refs: bool = False,
    crop_collar_seconds: float = 30.0,
    random_crop_seconds: float = 0.0,
    transcript_dir: Optional[str] = None,
    transcript_max_words: int = 0,
    few_shot_count: int = 0,
    icl_multimodal: bool = False,
    icl_multimodal_nonen: bool = False,
) -> Tuple[Optional[str], Dict[str, float], List[Dict[str, float]]]:
    """Position-debiased MCQ prediction via cyclic option permutation.

    Each answer content is presented once at every label slot across N cyclic
    rotations (N = number of options).  For each rotation we read the model's
    next-token logprobs for the option labels, convert them to a normalized
    distribution over the slots, and accumulate each rotation's probability back
    onto the *content* currently occupying that slot.  Averaging across rotations
    cancels the per-slot prior (e.g. the model's tendency to answer "A"), and the
    content with the highest mean probability wins.

    Returns (predicted_canonical_label, content_probs_by_label, per_rotation_scores).
    """
    choice_map = sample.choice_map or {}
    labels = list(choice_map.keys())
    contents = list(choice_map.values())
    n = len(labels)
    if n < 2:
        return (labels[0] if labels else None), {}, []

    question = str((sample.metadata or {}).get("question", "") or "")
    content_prob = [0.0] * n
    per_rotation: List[Dict[str, float]] = []

    for r in range(n):
        # Rotation r: label slot j shows content[(j + r) % n].
        permuted = {labels[j]: contents[(j + r) % n] for j in range(n)}
        prompt_r = _render_mc_prompt(question, permuted)
        scores = _run_audio_instruction(
            model=model,
            processor=processor,
            sample=sample,
            audio_path=audio_path,
            prompt=prompt_r,
            model_id=model_id,
            max_new_tokens=1,
            prompt_prefix=prompt_prefix,
            crop_from_question_refs=crop_from_question_refs,
            crop_collar_seconds=crop_collar_seconds,
            random_crop_seconds=random_crop_seconds,
            transcript_dir=transcript_dir,
            transcript_max_words=transcript_max_words,
            score_labels=labels,
            few_shot_count=few_shot_count,
            icl_multimodal=icl_multimodal,
            icl_multimodal_nonen=icl_multimodal_nonen,
        )
        per_rotation.append({lbl: float(scores.get(lbl, float("-inf"))) for lbl in labels})

        logp = torch.tensor(
            [scores.get(lbl, float("-inf")) for lbl in labels],
            dtype=torch.float32,
        )
        prob = torch.softmax(logp, dim=0)
        for j in range(n):
            content_prob[(j + r) % n] += float(prob[j]) / n

    best_idx = max(range(n), key=lambda k: content_prob[k])
    content_probs_by_label = {labels[k]: content_prob[k] for k in range(n)}
    return labels[best_idx], content_probs_by_label, per_rotation


def _predict_choice_calibrated(
    model,
    processor,
    sample: EvalSample,
    audio_path: str,
    model_id: str,
    prompt_prefix: str = "",
    crop_from_question_refs: bool = False,
    crop_collar_seconds: float = 30.0,
    random_crop_seconds: float = 0.0,
    transcript_dir: Optional[str] = None,
    transcript_max_words: int = 0,
    few_shot_count: int = 0,
    icl_multimodal: bool = False,
    icl_multimodal_nonen: bool = False,
) -> Tuple[Optional[str], Dict[str, float], Dict[str, float]]:
    """Prior-calibrated MCQ prediction (Zhao et al., 2021 — "Calibrate Before Use").

    Scores each label on the *real* prompt and on a *null* prompt that keeps the
    question and audio unchanged but replaces every option's text with an empty
    string.  The null prompt reveals the model's label-position prior
    P(label | question, empty_options) independently of option semantics.

    Calibrated log-score = log P(label | real) − log P(label | null).

    The label with the highest calibrated score is returned.  Unlike cyclic
    debias, this requires exactly 2 forward passes regardless of N, but it only
    removes the *prior conditioned on the question* — if the prior varies across
    questions it is captured per-question.

    Returns (pred_label, calibrated_scores_by_label, null_logprobs_by_label).
    """
    choice_map = sample.choice_map or {}
    labels = list(choice_map.keys())
    n = len(labels)
    if n < 2:
        return (labels[0] if labels else None), {}, {}

    question = str((sample.metadata or {}).get("question", "") or "")

    # Null prompt: same MCQ format but option texts are empty.
    null_choice_map = {lbl: "" for lbl in labels}
    null_prompt = _render_mc_prompt(question, null_choice_map)

    # Real prompt: normal MCQ prompt with actual option texts.
    real_prompt = _render_mc_prompt(question, choice_map)

    shared_kwargs = dict(
        model=model,
        processor=processor,
        sample=sample,
        audio_path=audio_path,
        model_id=model_id,
        max_new_tokens=1,
        prompt_prefix=prompt_prefix,
        crop_from_question_refs=crop_from_question_refs,
        crop_collar_seconds=crop_collar_seconds,
        random_crop_seconds=random_crop_seconds,
        transcript_dir=transcript_dir,
        transcript_max_words=transcript_max_words,
        score_labels=labels,
        few_shot_count=few_shot_count,
        icl_multimodal=icl_multimodal,
        icl_multimodal_nonen=icl_multimodal_nonen,
    )

    null_scores: Dict[str, float] = _run_audio_instruction(prompt=null_prompt, **shared_kwargs)
    real_scores: Dict[str, float] = _run_audio_instruction(prompt=real_prompt, **shared_kwargs)

    # Calibrated score in log-space: log P(real) - log P(null).
    calibrated = {
        lbl: real_scores.get(lbl, float("-inf")) - null_scores.get(lbl, float("-inf"))
        for lbl in labels
    }

    pred_choice = max(labels, key=lambda l: calibrated[l])
    return pred_choice, calibrated, {lbl: null_scores.get(lbl, float("-inf")) for lbl in labels}


def _build_voxtral_chat_input_features(processor, chat_audio: List[np.ndarray]) -> torch.Tensor:
    target_frames = 3000
    pad_to_samples = 480000
    feature_tensors = []
    for audio_array in chat_audio:
        audio_inputs = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            padding=True,
            truncation=False,
            pad_to_multiple_of=pad_to_samples,
        )
        raw_features = None
        if isinstance(audio_inputs, dict):
            raw_features = audio_inputs.get("input_features")
        elif hasattr(audio_inputs, "input_features"):
            raw_features = audio_inputs.input_features
        elif isinstance(audio_inputs, (list, tuple)) and audio_inputs:
            raw_features = audio_inputs[0]

        if raw_features is None:
            raise RuntimeError("Could not extract input_features for Voxtral chat audio.")

        feats = np.asarray(raw_features, dtype=np.float32)
        if feats.ndim == 3 and feats.shape[0] == 1:
            feats = feats[0]
        if feats.ndim == 2:
            # Match VoxtralProcessor: split long mel sequence into chunks of 3000 frames.
            if feats.shape[1] % target_frames != 0:
                pad_size = target_frames - (feats.shape[1] % target_frames)
                feats = np.pad(feats, ((0, 0), (0, pad_size)), mode="constant")
            feats = feats.reshape(feats.shape[0], -1, target_frames).transpose(1, 0, 2)
        elif feats.ndim == 3:
            # Already chunked [num_chunks, mel_bins, frames].
            if feats.shape[2] != target_frames:
                raise RuntimeError(
                    f"Unexpected chunked feature shape from Voxtral feature extractor: {feats.shape}"
                )
        else:
            raise RuntimeError(f"Unexpected feature shape from Voxtral feature extractor: {feats.shape}")

        feature_tensors.append(torch.as_tensor(feats))

    return torch.cat(feature_tensors, dim=0)


def _task_builder(task_name: str):
    if task_name == "mmsu":
        return _build_mmsu_task
    if task_name == "minds14_intent":
        return _build_minds14_task
    if task_name == "esc50_event":
        return _build_esc50_task
    if task_name == "jsonl_audio_mc":
        return _build_jsonl_audio_mc_task
    raise ValueError(f"Unsupported task: {task_name}. Supported tasks: {SUPPORTED_TASKS}")


def _load_transcript(audio_path: str, transcript_dir: Optional[str],
                     max_words: int = 0) -> str:
    """Load the plain-text transcript for *audio_path* from *transcript_dir*.

    Returns an empty string only when the transcript file is not found or
    *transcript_dir* is None/empty.

    Hallucination-loop detection: if any 5-gram appears more than 5 times the
    transcript is treated as corrupted.  The clean tokens before the loop onset
    are kept and an error notice is appended so the model knows the transcript is
    unreliable::

        {pre-loop text}
        [Note: ASR transcription may contain errors]

    When *max_words* > 0 the (possibly truncated) transcript is further limited
    to that many words (applied before the error notice).
    """
    if not transcript_dir:
        return ""
    stem = Path(audio_path).stem
    txt_path = Path(transcript_dir) / f"{stem}.txt"
    if not txt_path.exists():
        return ""
    text = txt_path.read_text(encoding="utf-8").strip()
    words = text.split()
    has_errors = False
    # Hallucination-loop filter: sliding-window density check.
    # A real loop fills a 150-word window with many repetitions of one 5-gram.
    # A naturally repeated phrase (e.g. "how about you i am") appears at most
    # 1-2 times per window and is never flagged.
    if len(words) >= 6:
        from collections import Counter
        _NGRAM, _WIN, _MIN = 5, 150, 5
        _step = max(1, _WIN // 3)
        _loop_phrase, _loop_start = None, len(words)
        for _ws in range(0, max(1, len(words) - _WIN + 1), _step):
            _ww = words[_ws: _ws + _WIN]
            _ng = [" ".join(_ww[_i:_i+_NGRAM]) for _i in range(len(_ww)-_NGRAM)]
            _cnt = Counter(_ng)
            if not _cnt:
                continue
            _top, _c = _cnt.most_common(1)[0]
            if _c > _MIN:
                _tp = _top.split()
                _n = len(_tp)
                _seen = False
                for _i in range(len(words) - _n + 1):
                    if words[_i:_i+_n] == _tp:
                        if _seen:
                            _loop_start = _i
                            break
                        _seen = True
                else:
                    _loop_start = _ws
                _loop_phrase = _top
                break
        if _loop_phrase is not None:
            has_errors = True
            words = words[:_loop_start]
    if max_words > 0 and len(words) > max_words:
        words = words[:max_words]
    clean_text = " ".join(words)
    if has_errors:
        return (clean_text + "\n[Note: ASR transcription may contain errors]").strip()
    return clean_text if max_words > 0 else text


def _build_prompt_for_sample(task: TaskData, sample: EvalSample) -> str:
    if task.evaluation_mode == "multiple_choice":
        if not sample.prompt_text:
            raise RuntimeError(f"Missing prompt_text for sample {sample.sample_id}")
        return sample.prompt_text
    return task.prompt_template.format(labels=", ".join(task.labels))


def _compute_accuracy_breakdown(rows: List[Dict], field_name: str) -> Dict[str, Dict[str, float]]:
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        if row.get("error"):
            continue
        metadata = row.get("metadata") or {}
        key = metadata.get(field_name)
        if key in (None, ""):
            continue
        key_str = str(key)
        counts[key_str]["total"] += 1
        counts[key_str]["correct"] += int(bool(row.get("is_correct")))

    breakdown: Dict[str, Dict[str, float]] = {}
    for key, value in sorted(counts.items()):
        total = int(value["total"])
        correct = int(value["correct"])
        breakdown[key] = {
            "n_total": total,
            "n_correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }
    return breakdown


def _add_multiple_choice_summary_fields(summary: Dict, rows: List[Dict]) -> Dict:
    summary["task_name_breakdown"] = _compute_accuracy_breakdown(rows, "task_name")
    summary["category_breakdown"] = _compute_accuracy_breakdown(rows, "category")
    summary["sub_category_breakdown"] = _compute_accuracy_breakdown(rows, "sub-category")
    summary["sub_sub_category_breakdown"] = _compute_accuracy_breakdown(rows, "sub-sub-category")
    summary["linguistics_sub_discipline_breakdown"] = _compute_accuracy_breakdown(
        rows,
        "linguistics_sub_discipline",
    )
    return summary


def _evaluate_task(
    task: TaskData,
    model,
    processor,
    model_id: str,
    max_new_tokens: int,
    temp_audio_dir: Path,
    prediction_only: bool,
    prompt_prefix: str = "",
    crop_from_question_refs: bool = False,
    crop_collar_seconds: float = 30.0,
    random_crop_seconds: float = 0.0,
    transcript_dir: Optional[str] = None,
    transcript_max_words: int = 0,
    debias_cyclic: bool = False,
    calibrate_prior: bool = False,
    few_shot_count: int = 0,
    icl_multimodal: bool = False,
    icl_multimodal_nonen: bool = False,
) -> Tuple[Dict, List[Dict]]:
    rows: List[Dict] = []
    correct = 0
    skipped = 0
    error_counts: Counter[str] = Counter()
    max_error_examples = 8

    for idx, s in enumerate(task.samples):
        try:
            audio_path = _materialize_audio_path(s, temp_audio_dir)
            debias_probs: Optional[Dict[str, float]] = None
            debias_per_rotation: Optional[List[Dict[str, float]]] = None
            calib_null_logprobs: Optional[Dict[str, float]] = None
            if calibrate_prior and task.evaluation_mode == "multiple_choice":
                pred_choice, debias_probs, calib_null_logprobs = _predict_choice_calibrated(
                    model=model,
                    processor=processor,
                    sample=s,
                    audio_path=audio_path,
                    model_id=model_id,
                    prompt_prefix=prompt_prefix,
                    crop_from_question_refs=crop_from_question_refs,
                    crop_collar_seconds=crop_collar_seconds,
                    random_crop_seconds=random_crop_seconds,
                    transcript_dir=transcript_dir,
                    transcript_max_words=transcript_max_words,
                    few_shot_count=few_shot_count,
                    icl_multimodal=icl_multimodal,
                    icl_multimodal_nonen=icl_multimodal_nonen,
                )
                output = ""
            elif debias_cyclic and task.evaluation_mode == "multiple_choice":
                pred_choice, debias_probs, debias_per_rotation = _predict_choice_cyclic_debias(
                    model=model,
                    processor=processor,
                    sample=s,
                    audio_path=audio_path,
                    model_id=model_id,
                    prompt_prefix=prompt_prefix,
                    crop_from_question_refs=crop_from_question_refs,
                    crop_collar_seconds=crop_collar_seconds,
                    random_crop_seconds=random_crop_seconds,
                    transcript_dir=transcript_dir,
                    transcript_max_words=transcript_max_words,
                    few_shot_count=few_shot_count,
                    icl_multimodal=icl_multimodal,
                    icl_multimodal_nonen=icl_multimodal_nonen,
                )
                output = ""
            elif not calibrate_prior:
                prompt = _build_prompt_for_sample(task, s)
                output = _run_audio_instruction(
                    model=model,
                    processor=processor,
                    sample=s,
                    audio_path=audio_path,
                    prompt=prompt,
                    model_id=model_id,
                    max_new_tokens=max_new_tokens,
                    prompt_prefix=prompt_prefix,
                    crop_from_question_refs=crop_from_question_refs,
                    crop_collar_seconds=crop_collar_seconds,
                    random_crop_seconds=random_crop_seconds,
                    transcript_dir=transcript_dir,
                    transcript_max_words=transcript_max_words,
                    few_shot_count=few_shot_count,
                    icl_multimodal=icl_multimodal,
                    icl_multimodal_nonen=icl_multimodal_nonen,
                )
            if task.evaluation_mode == "multiple_choice":
                if not debias_cyclic and not calibrate_prior:
                    pred_choice = _select_multiple_choice_option(
                        output,
                        list((s.choice_map or {}).keys()),
                    )
                if prediction_only and pred_choice is None:
                    labels = list((s.choice_map or {}).keys())
                    pred_choice = labels[0] if labels else None
                pred = (s.choice_map or {}).get(pred_choice or "", "")
                # Remap Yes/No predictions back to canonical A/B labels.
                yn_to_orig = (s.metadata or {}).get("yn_to_orig")
                orig_choice_map = (s.metadata or {}).get("orig_choice_map") or s.choice_map or {}
                canonical_pred_choice = yn_to_orig.get(pred_choice, pred_choice) if yn_to_orig and pred_choice else pred_choice
                canonical_gold_choice = yn_to_orig.get(s.gold_choice, s.gold_choice) if yn_to_orig and s.gold_choice else s.gold_choice
                ok = (canonical_pred_choice == canonical_gold_choice) if (not prediction_only and canonical_gold_choice is not None) else None
                row = {
                    "task": task.task_name,
                    "sample_id": s.sample_id,
                    "audio_path": audio_path,
                    "gold_label": s.gold_label,
                    "gold_choice": canonical_gold_choice,
                    "pred_label": orig_choice_map.get(canonical_pred_choice or "", pred),
                    "pred_choice": canonical_pred_choice,
                    "yn_pred_raw": pred_choice if yn_to_orig else None,
                    "is_correct": ok,
                    "model_output": output,
                    "response": output,
                    "debias_content_probs": debias_probs,
                    "debias_per_rotation": debias_per_rotation,
                    "calib_null_logprobs": calib_null_logprobs,
                    "session_id": s.metadata.get("session_id"),
                    "question_id": s.metadata.get("question_id"),
                    "id": s.metadata.get("id"),
                    "question": s.metadata.get("question"),
                    "choice_a": orig_choice_map.get("A", ""),
                    "choice_b": orig_choice_map.get("B", ""),
                    "choice_c": orig_choice_map.get("C", ""),
                    "choice_d": orig_choice_map.get("D", ""),
                    "choices": orig_choice_map,
                    "answer_gt": s.gold_label,
                    "task_name": s.metadata.get("task_name"),
                    "category": s.metadata.get("category"),
                    "sub-category": s.metadata.get("sub-category"),
                    "sub-sub-category": s.metadata.get("sub-sub-category"),
                    "linguistics_sub_discipline": s.metadata.get("linguistics_sub_discipline"),
                    "metadata": s.metadata,
                }
            else:
                pred = _select_predicted_label(output, task.labels)
                ok = (pred == s.gold_label) if not prediction_only else None
                row = {
                    "task": task.task_name,
                    "sample_id": s.sample_id,
                    "audio_path": audio_path,
                    "gold_label": s.gold_label,
                    "pred_label": pred,
                    "is_correct": ok,
                    "model_output": output,
                    "metadata": s.metadata,
                }
            if ok is True:
                correct += 1
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            error_msg = f"{type(e).__name__}: {e}"
            error_counts[error_msg] += 1
            print(
                f"[{task.task_name}] SKIP sample_id={s.sample_id} reason={error_msg}",
                file=sys.stderr,
            )
            if isinstance(e, (RuntimeError, torch.cuda.OutOfMemoryError)):
                traceback.print_exc(file=sys.stderr)

            error_row = {
                "task": task.task_name,
                "sample_id": s.sample_id,
                "audio_path": s.audio_path,
                "gold_label": s.gold_label,
                "pred_label": None,
                "is_correct": False,
                "model_output": "",
                "error": error_msg,
                "metadata": s.metadata,
            }
            if task.evaluation_mode == "multiple_choice":
                error_row.update(
                    {
                        "gold_choice": s.gold_choice,
                        "pred_choice": None,
                        "response": "",
                        "session_id": s.metadata.get("session_id"),
                        "question_id": s.metadata.get("question_id"),
                        "id": s.metadata.get("id"),
                        "question": s.metadata.get("question"),
                        "choice_a": (s.choice_map or {}).get("A", ""),
                        "choice_b": (s.choice_map or {}).get("B", ""),
                        "choice_c": (s.choice_map or {}).get("C", ""),
                        "choice_d": (s.choice_map or {}).get("D", ""),
                        "choices": s.choice_map,
                        "answer_gt": s.gold_label,
                        "task_name": s.metadata.get("task_name"),
                        "category": s.metadata.get("category"),
                        "sub-category": s.metadata.get("sub-category"),
                        "sub-sub-category": s.metadata.get("sub-sub-category"),
                        "linguistics_sub_discipline": s.metadata.get("linguistics_sub_discipline"),
                    }
                )
            rows.append(error_row)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(task.samples):
            denom = max(1, idx + 1 - skipped)
            acc_so_far = (correct / denom) if not prediction_only else 0.0
            print(
                f"[{task.task_name}] {idx + 1}/{len(task.samples)} processed | "
                f"acc_so_far={acc_so_far:.4f} | skipped={skipped}"
            )

    eval_count = max(0, len(task.samples) - skipped)
    if prediction_only:
        eval_count = 0
    accuracy = (correct / eval_count) if eval_count > 0 else 0.0

    if skipped > 0:
        print(f"[{task.task_name}] skip summary (top reasons):", file=sys.stderr)
        for reason, count in error_counts.most_common():
            print(f"[{task.task_name}]   {count}x {reason}", file=sys.stderr)

    summary = {
        "task": task.task_name,
        "split": task.split_name,
        "n_total": len(task.samples),
        "n_eval": eval_count,
        "n_skipped": skipped,
        "n_correct": correct,
        "accuracy": accuracy,
        "labels": task.labels,
    }
    if task.evaluation_mode == "multiple_choice":
        summary = _add_multiple_choice_summary_fields(summary, rows)
    return summary, rows


def _choose_dtype(use_bf16: bool, use_fp16: bool) -> torch.dtype:
    if use_bf16:
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return torch.float32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Voxtral on non-ASR tasks to monitor forgetting.",
    )

    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--adapter-path", default=None)

    p.add_argument(
        "--tasks",
        default="mmsu",
        help=f"Comma-separated tasks. Supported: {SUPPORTED_TASKS}",
    )
    p.add_argument(
        "--split",
        default="train",
        help=(
            "Requested split. MMSU passes this directly to Hugging Face dataset loading and commonly uses split=train. "
            "minds14_intent supports only test/validation via stratified held-out partition. "
            "For esc50_event, split=test runs 5-fold aggregation."
        ),
    )
    p.add_argument("--max-samples-per-task", type=int, default=200)
    p.add_argument(
        "--jsonl-path",
        default=None,
        help="Path to JSONL file for task=jsonl_audio_mc.",
    )
    p.add_argument(
        "--audio-root",
        default=None,
        help="Root directory to prepend to relative audio paths in --jsonl-path.",
    )
    p.add_argument(
        "--jsonl-max-questions-per-audio",
        type=int,
        default=0,
        help="If > 0, keep at most this many questions per audio item in JSONL.",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument(
        "--few-shot-count",
        type=int,
        default=0,
        help="Number of text-only ICL examples to prepend as user/assistant turns before the "
             "audio question. Teaches OOTB models to output a bare letter (A/B/C/D). "
             f"Max {len(_ICL_FORMAT_EXAMPLES)}. Slots [0-1] are 2-option (A/B) examples; "
             "slots [2-4] are 4-option (A/B/C/D) examples. Default 0 (disabled).",
    )
    p.add_argument(
        "--icl-multimodal",
        action="store_true",
        default=False,
        help="Use multimodal (audio+text) ICL examples from _ICL_MULTIMODAL_EXAMPLES instead of "
             "text-only examples. When --few-shot-count equals the full pool size (6), selects "
             "4-opt shots for 4-option questions and 2-opt shots for 2-option questions. "
             "When --few-shot-count < 6, uses the first N examples for all question types.",
    )
    p.add_argument(
        "--icl-multimodal-nonen",
        action="store_true",
        default=False,
        help="When --icl-multimodal is set, replace the 4-opt-D English slot with a "
             "Vietnamese audio clip (English-translated question) to demonstrate that "
             "the model should answer in English regardless of audio language.",
    )
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--output-dir", default="experiments/voxtral_forgetting_eval")
    p.add_argument("--timestamped-output", action="store_true")

    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--no-use-bf16", dest="use_bf16", action="store_false")
    p.set_defaults(use_bf16=True)

    p.add_argument("--use-fp16", action="store_true")
    p.add_argument(
        "--prediction-only",
        action="store_true",
        help="Run inference without requiring gold labels; writes hyp.txt for jsonl_audio_mc.",
    )

    p.add_argument(
        "--prompt-language",
        default="en",
        help="Language tag used in the transcription hint prefix (e.g. 'en', 'fr').  "
             "Only relevant when --use-transcription-hint-format is set.",
    )
    p.add_argument(
        "--use-transcription-hint-format",
        action="store_true",
        help="Prepend 'lang:{lang}\\n[TRANSCRIBE]\\n' to every MCQ question prompt, "
             "embedding the ASR transcription cue inside the [INST] block.  "
             "Matches the training format produced by voxtral_train_MCQ.py "
             "--use-transcription-hint-format.",
    )
    p.add_argument(
        "--eval-crop-from-question-refs",
        action="store_true",
        help="Apply timestamp-based audio cropping during inference. "
             "Questions with detected time references are cropped to the relevant window "
             "(± --eval-crop-collar-seconds); questions without time references receive "
             "full audio when --eval-random-crop-seconds=0 (default).",
    )
    p.add_argument(
        "--eval-crop-collar-seconds",
        type=float,
        default=30.0,
        help="Collar in seconds around detected timestamps for eval crop. Default 30.0.",
    )
    p.add_argument(
        "--eval-random-crop-seconds",
        type=float,
        default=0.0,
        help="Fallback crop duration (seconds) for questions without timestamps. "
             "0 (default) passes full audio for those questions.",
    )
    p.add_argument(
        "--transcript-dir",
        default=None,
        metavar="DIR",
        help="Directory containing ASR transcript .txt files (one per session, named {stem}.txt). "
             "When set, the full transcript is prepended to every MCQ prompt as "
             "'Transcript:\\n{text}\\n\\n'. Off by default.",
    )
    p.add_argument(
        "--transcript-max-words",
        type=int,
        default=0,
        help="Truncate transcript to at most this many words before inserting into the prompt. "
             "0 (default) = no truncation (use full transcript).",
    )

    p.add_argument(
        "--debias-cyclic-permutation",
        action="store_true",
        help="Position-debias multiple-choice predictions via cyclic option permutation. "
             "Each option content is presented once at every label slot across N rotations "
             "(N = number of options); the model's next-token option-label logprobs are read, "
             "normalized per rotation, mapped back to each content, and averaged across rotations "
             "to cancel the per-slot prior (e.g. the tendency to answer 'A'). Costs N forward "
             "passes per question instead of one generate call. Only affects multiple_choice tasks.",
    )
    p.add_argument(
        "--calibrate-prior",
        action="store_true",
        help="Prior-calibrated prediction (Zhao et al., 2021 — 'Calibrate Before Use'). "
             "For each question, runs a second forward pass on a null prompt that keeps the "
             "question and audio unchanged but replaces every option's text with an empty string. "
             "The null pass reveals P(label | question, no_option_text) — the model's label-position "
             "prior independent of option semantics. "
             "Calibrated score = log P(label | real) - log P(label | null). "
             "Costs exactly 2 forward passes per question regardless of N options. "
             "Particularly effective for 2-option questions where cyclic debias only does 2 rotations. "
             "Only affects multiple_choice tasks.",
    )
    p.add_argument(
        "--yesno-binary",
        action="store_true",
        help="For 2-option questions, relabel options as '1'/'2' instead of 'A'/'B' in the "
             "prompt and scoring. '1' maps to the first option (A), '2' to the second (B). "
             "The output pred_choice is always remapped back to the canonical A/B label. "
             "Addresses the MCQ-specific A-slot positional prior without introducing semantic "
             "assumptions (unlike Yes/No). Only applies to jsonl_audio_mc tasks.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks = _parse_tasks(args.tasks)
    if "jsonl_audio_mc" in tasks and not args.jsonl_path:
        raise ValueError("--jsonl-path is required when --tasks includes jsonl_audio_mc")

    prompt_prefix = (
        f"lang:{args.prompt_language}\n[TRANSCRIBE]\n"
        if args.use_transcription_hint_format
        else ""
    )

    dtype = _choose_dtype(use_bf16=bool(args.use_bf16), use_fp16=bool(args.use_fp16))

    output_dir = Path(args.output_dir)
    if args.timestamped_output:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / ts
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_dir = output_dir / "_tmp_audio"

    print("Loading model and processor...")
    model, model_mode, resolved_adapter = _load_model(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        dtype=dtype,
    )
    processor = VoxtralProcessor.from_pretrained(_resolve_pretrained_source(args.model_id))

    task_summaries: List[Dict] = []
    detailed_task_summaries: List[Dict] = []
    all_rows: List[Dict] = []

    for task_name in tasks:
        if task_name == "esc50_event" and str(args.split).strip().lower() == "test":
            fold_summaries: List[Dict] = []
            fold_rows: List[Dict] = []
            for fold in range(1, 6):
                task_data = _build_esc50_task(
                    split=f"fold{fold}",
                    max_samples=max(0, int(args.max_samples_per_task)),
                    seed=int(args.seed) + fold,
                    cache_dir=args.cache_dir,
                )
                print(
                    f"Starting task={task_data.task_name} split={task_data.split_name} "
                    f"n={len(task_data.samples)}"
                )
                summary, rows = _evaluate_task(
                    task=task_data,
                    model=model,
                    processor=processor,
                    model_id=args.model_id,
                    max_new_tokens=int(args.max_new_tokens),
                    temp_audio_dir=temp_audio_dir,
                    prediction_only=bool(args.prediction_only),
                    prompt_prefix=prompt_prefix,
                    crop_from_question_refs=bool(args.eval_crop_from_question_refs),
                    crop_collar_seconds=float(args.eval_crop_collar_seconds),
                    random_crop_seconds=float(args.eval_random_crop_seconds),
                    transcript_dir=args.transcript_dir,
                    transcript_max_words=int(args.transcript_max_words),
                    few_shot_count=int(args.few_shot_count),
                )
                fold_summaries.append(summary)
                fold_rows.extend(rows)

            n_eval = sum(int(s["n_eval"]) for s in fold_summaries)
            n_total = sum(int(s["n_total"]) for s in fold_summaries)
            n_skipped = sum(int(s["n_skipped"]) for s in fold_summaries)
            n_correct = sum(int(s["n_correct"]) for s in fold_summaries)
            acc_macro = (
                sum(float(s["accuracy"]) for s in fold_summaries) / len(fold_summaries)
                if fold_summaries
                else 0.0
            )
            acc_micro = (n_correct / n_eval) if n_eval > 0 else 0.0

            agg = {
                "task": "esc50_event",
                "split": "folds_1_to_5",
                "n_total": n_total,
                "n_eval": n_eval,
                "n_skipped": n_skipped,
                "n_correct": n_correct,
                "accuracy": acc_micro,
                "accuracy_macro_folds": acc_macro,
            }
            task_summaries.append(agg)
            detailed_task_summaries.extend(fold_summaries)
            all_rows.extend(fold_rows)
            print(
                f"[esc50_event aggregate] acc_micro={acc_micro:.4f} "
                f"acc_macro_folds={acc_macro:.4f} n_eval={n_eval}"
            )
            continue

        if task_name == "jsonl_audio_mc":
            task_data = _build_jsonl_audio_mc_task(
                split=args.split,
                max_samples=max(0, int(args.max_samples_per_task)),
                seed=int(args.seed),
                cache_dir=args.cache_dir,
                jsonl_path=args.jsonl_path,
                audio_root=args.audio_root,
                max_questions_per_audio=max(0, int(args.jsonl_max_questions_per_audio)),
                prediction_only=bool(args.prediction_only),
                yesno_binary=bool(args.yesno_binary),
            )
            print(
                f"Starting task={task_data.task_name} split={task_data.split_name} "
                f"n={len(task_data.samples)}"
            )
            summary, rows = _evaluate_task(
                task=task_data,
                model=model,
                processor=processor,
                model_id=args.model_id,
                max_new_tokens=int(args.max_new_tokens),
                temp_audio_dir=temp_audio_dir,
                prediction_only=bool(args.prediction_only),
                prompt_prefix=prompt_prefix,
                crop_from_question_refs=bool(args.eval_crop_from_question_refs),
                crop_collar_seconds=float(args.eval_crop_collar_seconds),
                random_crop_seconds=float(args.eval_random_crop_seconds),
                transcript_dir=args.transcript_dir,
                transcript_max_words=int(args.transcript_max_words),
                debias_cyclic=bool(args.debias_cyclic_permutation),
                calibrate_prior=bool(args.calibrate_prior),
                few_shot_count=int(args.few_shot_count),
                icl_multimodal=bool(args.icl_multimodal),
                icl_multimodal_nonen=bool(args.icl_multimodal_nonen),
            )
            task_summaries.append(summary)
            detailed_task_summaries.append(summary)
            all_rows.extend(rows)
            continue

        builder = _task_builder(task_name)
        task_data = builder(
            split=args.split,
            max_samples=max(0, int(args.max_samples_per_task)),
            seed=int(args.seed),
            cache_dir=args.cache_dir,
        )
        print(
            f"Starting task={task_data.task_name} split={task_data.split_name} "
            f"n={len(task_data.samples)}"
        )
        summary, rows = _evaluate_task(
            task=task_data,
            model=model,
            processor=processor,
            model_id=args.model_id,
            max_new_tokens=int(args.max_new_tokens),
            temp_audio_dir=temp_audio_dir,
            prediction_only=bool(args.prediction_only),
            prompt_prefix=prompt_prefix,
            crop_from_question_refs=bool(args.eval_crop_from_question_refs),
            crop_collar_seconds=float(args.eval_crop_collar_seconds),
            random_crop_seconds=float(args.eval_random_crop_seconds),
            transcript_dir=args.transcript_dir,
            transcript_max_words=int(args.transcript_max_words),
            debias_cyclic=bool(args.debias_cyclic_permutation),
            calibrate_prior=bool(args.calibrate_prior),
            few_shot_count=int(args.few_shot_count),
            icl_multimodal=bool(args.icl_multimodal),
            icl_multimodal_nonen=bool(args.icl_multimodal_nonen),
        )
        task_summaries.append(summary)
        detailed_task_summaries.append(summary)
        all_rows.extend(rows)

    macro_acc = 0.0
    if task_summaries:
        macro_acc = sum(x["accuracy"] for x in task_summaries) / len(task_summaries)

    payload = {
        "timestamp": dt.datetime.now().isoformat(),
        "model_id": args.model_id,
        "model_mode": model_mode,
        "adapter_path_input": args.adapter_path,
        "adapter_path_resolved": resolved_adapter,
        "dtype": str(dtype),
        "tasks": tasks,
        "split": args.split,
        "max_samples_per_task": int(args.max_samples_per_task),
        "jsonl_path": args.jsonl_path,
        "audio_root": args.audio_root,
        "jsonl_max_questions_per_audio": int(args.jsonl_max_questions_per_audio),
        "max_new_tokens": int(args.max_new_tokens),
        "prediction_only": bool(args.prediction_only),
        "macro_accuracy": macro_acc,
        "task_summaries": task_summaries,
        "detailed_task_summaries": detailed_task_summaries,
    }

    metrics_path = output_dir / "forgetting_eval_metrics.json"
    rows_path = output_dir / "forgetting_eval_predictions.jsonl"

    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    hyp_path = output_dir / "hyp.txt"
    if bool(args.prediction_only) and "jsonl_audio_mc" in tasks:
        hyp_rows = [r for r in all_rows if r.get("task") == "jsonl_audio_mc"]
        with hyp_path.open("w", encoding="utf-8") as f:
            for r in hyp_rows:
                sid = _normalize_submission_session_id(str(r.get("session_id") or ""))
                qid = str(r.get("question_id") or "")
                ans = _normalize_choice_label(str(r.get("pred_choice") or ""))
                if not ans:
                    ans = "A"
                f.write(f"{sid} {qid} {ans}\n")

        # When cyclic debiasing is active, also emit an intermediate detail file
        # carrying the SAME predictions completed with the per-permutation option
        # logprobs (and the averaged content probabilities) for each question.
        if bool(args.debias_cyclic_permutation):
            detail_path = output_dir / "hyp_debias_detail.jsonl"
            with detail_path.open("w", encoding="utf-8") as f:
                for r in hyp_rows:
                    sid = _normalize_submission_session_id(str(r.get("session_id") or ""))
                    qid = str(r.get("question_id") or "")
                    ans = _normalize_choice_label(str(r.get("pred_choice") or "")) or "A"
                    content_probs = r.get("debias_content_probs") or {}
                    detail = {
                        "session_id": sid,
                        "question_id": qid,
                        "pred_choice": ans,
                        "n_options": len(content_probs) or None,
                        "content_probs": content_probs,
                        "per_rotation_logprobs": r.get("debias_per_rotation"),
                    }
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            print(f"Saved debias detail: {detail_path}")

    print("\nEvaluation complete.")
    print(f"Macro accuracy: {macro_acc:.4f}")
    for s in task_summaries:
        print(
            f"  - {s['task']} ({s['split']}): "
            f"acc={s['accuracy']:.4f} n_eval={s['n_eval']} skipped={s['n_skipped']}"
        )
        for bk_key, bk_label in [
            ("category_breakdown", "category"),
            ("sub_category_breakdown", "sub-category"),
            ("sub_sub_category_breakdown", "sub-sub-category"),
        ]:
            bk = s.get(bk_key)
            if not bk:
                continue
            print(f"\n    Accuracy by {bk_label}:")
            for name, vals in sorted(bk.items(), key=lambda x: -x[1]["accuracy"]):
                print(
                    f"      {name:40s}  acc={vals['accuracy']:.4f}  "
                    f"({vals['n_correct']}/{vals['n_total']})"
                )
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved predictions: {rows_path}")
    if bool(args.prediction_only) and "jsonl_audio_mc" in tasks:
        print(f"Saved hyp: {hyp_path}")


if __name__ == "__main__":
    main()
