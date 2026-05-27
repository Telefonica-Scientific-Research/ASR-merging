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
import random
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

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
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
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
) -> TaskData:
    del cache_dir  # Unused but kept for a common task-builder signature style.

    if not jsonl_path:
        raise ValueError("jsonl_audio_mc requires --jsonl-path")

    src = Path(jsonl_path)
    if not src.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    samples: List[EvalSample] = []
    all_labels = set()

    with src.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            raw_audio_path = str(row.get("path") or row.get("audio_path") or "").strip()
            if not raw_audio_path:
                continue
            audio_path = _resolve_jsonl_audio_path(raw_audio_path, audio_root)

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

                choice_lines = [f"{lbl}. {txt}" for lbl, txt in choice_map.items()]
                prompt = (
                    "Choose the most suitable answer from the options below. "
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
                            "question": question,
                        },
                        prompt_text=prompt,
                        choice_map=choice_map,
                        gold_choice=gold_choice,
                    )
                )
                all_labels.update(choice_map.keys())

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


def _run_audio_instruction(
    model,
    processor,
    sample: EvalSample,
    audio_path: str,
    prompt: str,
    model_id: str,
    max_new_tokens: int,
) -> str:
    # Preferred audio-instruct path.
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": audio_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

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

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return decoded[0] if decoded else ""


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
) -> Tuple[Dict, List[Dict]]:
    rows: List[Dict] = []
    correct = 0
    skipped = 0
    error_counts: Counter[str] = Counter()
    max_error_examples = 8

    for idx, s in enumerate(task.samples):
        try:
            audio_path = _materialize_audio_path(s, temp_audio_dir)
            prompt = _build_prompt_for_sample(task, s)
            output = _run_audio_instruction(
                model=model,
                processor=processor,
                sample=s,
                audio_path=audio_path,
                prompt=prompt,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
            )
            if task.evaluation_mode == "multiple_choice":
                pred_choice = _select_multiple_choice_option(
                    output,
                    list((s.choice_map or {}).keys()),
                )
                pred = (s.choice_map or {}).get(pred_choice or "", "")
                ok = pred_choice == s.gold_choice
                row = {
                    "task": task.task_name,
                    "sample_id": s.sample_id,
                    "audio_path": audio_path,
                    "gold_label": s.gold_label,
                    "gold_choice": s.gold_choice,
                    "pred_label": pred,
                    "pred_choice": pred_choice,
                    "is_correct": ok,
                    "model_output": output,
                    "response": output,
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
                    "metadata": s.metadata,
                }
            else:
                pred = _select_predicted_label(output, task.labels)
                ok = pred == s.gold_label
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
            correct += int(ok)
            rows.append(row)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            error_msg = f"{type(e).__name__}: {e}"
            error_counts[error_msg] += 1

            if sum(error_counts.values()) <= max_error_examples:
                print(
                    f"[{task.task_name}] skip sample_id={s.sample_id} "
                    f"reason={error_msg}"
                )

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
            print(
                f"[{task.task_name}] {idx + 1}/{len(task.samples)} processed | "
                f"acc_so_far={(correct / max(1, idx + 1 - skipped)):.4f} | skipped={skipped}"
            )

    eval_count = max(0, len(task.samples) - skipped)
    accuracy = (correct / eval_count) if eval_count > 0 else 0.0

    if skipped > 0:
        print(f"[{task.task_name}] skip summary (top reasons):")
        for reason, count in error_counts.most_common(8):
            print(f"[{task.task_name}]   {count}x {reason}")

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
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--output-dir", default="experiments/voxtral_forgetting_eval")
    p.add_argument("--timestamped-output", action="store_true")

    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--no-use-bf16", dest="use_bf16", action="store_false")
    p.set_defaults(use_bf16=True)

    p.add_argument("--use-fp16", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    tasks = _parse_tasks(args.tasks)
    if "jsonl_audio_mc" in tasks and not args.jsonl_path:
        raise ValueError("--jsonl-path is required when --tasks includes jsonl_audio_mc")

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
    processor = VoxtralProcessor.from_pretrained(args.model_id)

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


if __name__ == "__main__":
    main()
