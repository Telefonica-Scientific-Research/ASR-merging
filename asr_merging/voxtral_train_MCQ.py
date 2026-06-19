#!/usr/bin/env python3
"""Train/evaluate Voxtral on JSONL multiple-choice audio understanding tasks.

This script is separate from ASR-oriented training in `voxtral_train_router.py` and
is focused on acoustic/semantic MCQ supervision.

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
"""

from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter, OrderedDict, defaultdict
import hashlib
import inspect
import json
import math
import random
import re
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import Trainer, TrainerCallback, TrainingArguments, VoxtralForConditionalGeneration, VoxtralProcessor

# Match voxtral_train_router.py runtime behavior:
# on some CUDA/PyTorch stacks, cuBLAS StridedBatched can fail while cuBLASLt works.
try:
    torch.backends.cuda.preferred_blas_library("cublaslt")
except Exception:
    pass


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


@dataclass
class MCQSample:
    sample_id: str
    audio_path: str
    question: str
    prompt_text: str
    choice_map: Dict[str, str]
    gold_choice: str
    metadata: Dict


@dataclass
class MCQTaskData:
    name: str
    samples: List[MCQSample]


def _build_mcq_prompt(question: str, choice_map: Dict[str, str]) -> str:
    choice_lines = [f"{lbl}. {txt}" for lbl, txt in choice_map.items()]
    return (
        "Choose the most suitable answer from the options below. "
        f"You must respond with only one label from: {', '.join(choice_map.keys())}.\n\n"
        f"Question: {question}\n\n"
        + "\n".join(choice_lines)
    )


def load_jsonl_audio_mcq(
    jsonl_path: str,
    audio_root: Optional[str],
    max_questions_per_audio: int,
    max_samples: int,
    seed: int,
) -> MCQTaskData:
    src = Path(jsonl_path)
    if not src.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    samples: List[MCQSample] = []

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

                qid = q.get("question_id") or str(qi)
                sample_id = f"jsonl:{line_no}:{qid}"
                samples.append(
                    MCQSample(
                        sample_id=sample_id,
                        audio_path=audio_path,
                        question=question,
                        prompt_text=_build_mcq_prompt(question, choice_map),
                        choice_map=choice_map,
                        gold_choice=gold_choice,
                        metadata={
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
                    )
                )

    rng = random.Random(seed)
    rng.shuffle(samples)
    if max_samples > 0:
        samples = samples[: min(max_samples, len(samples))]

    if not samples:
        raise RuntimeError("No usable MCQ samples found in JSONL.")

    return MCQTaskData(name="jsonl_audio_mc", samples=samples)


def _samples_to_hf_dataset(samples: List[MCQSample]) -> Dataset:
    rows = []
    for s in samples:
        rows.append(
            {
                "sample_id": s.sample_id,
                "audio_path": s.audio_path,
                "question": s.question,
                "prompt_text": s.prompt_text,
                "gold_choice": s.gold_choice,
                "choice_labels": list(s.choice_map.keys()),
                "choices": s.choice_map,
                "metadata": s.metadata,
            }
        )
    return Dataset.from_list(rows)


def _dataset_to_samples(ds: Dataset) -> List[MCQSample]:
    out: List[MCQSample] = []
    for row in ds:
        metadata = row.get("metadata") or {}
        choice_map = row.get("choices") or {}
        out.append(
            MCQSample(
                sample_id=str(row.get("sample_id") or ""),
                audio_path=str(row.get("audio_path") or ""),
                question=str(row.get("question") or metadata.get("question") or ""),
                prompt_text=str(row.get("prompt_text") or ""),
                choice_map={str(k): str(v) for k, v in choice_map.items()},
                gold_choice=str(row.get("gold_choice") or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return out


def _resolve_processed_split_dir(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / split_name / "processed"


def _resolve_sharded_split_dir(cache_dir: Path, split_name: str, num_shards: int, shard_index: int) -> Path:
    shard_tag = f"shard_{int(shard_index):05d}_of_{int(num_shards):05d}"
    return cache_dir / "shards" / split_name / shard_tag / "processed"


def _load_cache_split_dataset(
    cache_dir: Path,
    split_name: str,
    num_shards: int,
    shard_index: int,
) -> Dataset:
    shard_dir = _resolve_sharded_split_dir(cache_dir, split_name, num_shards, shard_index)
    if num_shards > 1 and shard_dir.exists():
        return load_from_disk(str(shard_dir))

    split_dir = _resolve_processed_split_dir(cache_dir, split_name)
    if not split_dir.exists():
        raise FileNotFoundError(f"Cached split not found: {split_dir}")

    ds = load_from_disk(str(split_dir))
    if num_shards > 1:
        indices = [i for i in range(len(ds)) if (i % num_shards) == shard_index]
        ds = ds.select(indices)
    return ds


def _split_train_eval(samples: List[MCQSample], eval_fraction: float, seed: int) -> Tuple[List[MCQSample], List[MCQSample]]:
    if eval_fraction <= 0.0:
        return samples, []
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_eval = max(1, int(round(len(samples) * eval_fraction)))
    n_eval = min(n_eval, max(0, len(samples) - 1))
    eval_idx = set(idx[:n_eval])
    train_samples = [s for i, s in enumerate(samples) if i not in eval_idx]
    eval_samples = [s for i, s in enumerate(samples) if i in eval_idx]
    return train_samples, eval_samples


def _apply_shard(samples: List[MCQSample], num_shards: int, shard_index: int) -> List[MCQSample]:
    if num_shards <= 1:
        return samples
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    return [s for i, s in enumerate(samples) if (i % num_shards) == shard_index]


def _load_audio_ids_for_shard(cache_dir: Path, split_name: str, num_shards: int, shard_index: int) -> Set[str]:
    shard_tag = f"shard_{int(shard_index):05d}_of_{int(num_shards):05d}"
    p = cache_dir / "audio_shards" / split_name / f"{shard_tag}_audio_ids.json"
    if not p.exists():
        raise FileNotFoundError(f"Audio shard mapping not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    ids = data.get("audio_ids") or []
    return {str(x) for x in ids}


def _audio_id_from_path(audio_path: str) -> str:
    # Keep this aligned with build_mcq_cache_from_jsonl.py.
    import hashlib

    return hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()


def _apply_audio_shard(
    samples: List[MCQSample],
    audio_shard_cache_dir: str,
    split_name: str,
    num_shards: int,
    shard_index: int,
) -> List[MCQSample]:
    if num_shards <= 1:
        return samples
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")

    cache_dir = Path(audio_shard_cache_dir)
    shard_audio_ids = _load_audio_ids_for_shard(cache_dir, split_name, num_shards, shard_index)
    return [s for s in samples if _audio_id_from_path(s.audio_path) in shard_audio_ids]


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


def _choose_dtype(use_bf16: bool, use_fp16: bool) -> torch.dtype:
    if use_bf16:
        return torch.bfloat16
    if use_fp16:
        return torch.float16
    return torch.float32


def _load_model(
    model_id: str,
    model_mode: str,
    adapter_path: Optional[str],
    dtype: torch.dtype,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )

    if model_mode == "adapter":
        if not adapter_path:
            raise ValueError("--adapter-path is required when --model-mode adapter")
        resolved = _resolve_adapter_path(adapter_path)
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve adapter path from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(resolved), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_cfg)

    return model


def _print_trainable_params(model) -> None:
    try:
        if hasattr(model, "get_nb_trainable_parameters"):
            trainable, total = model.get_nb_trainable_parameters()
        else:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pct = (100.0 * float(trainable) / float(total)) if total else 0.0
        print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] Could not compute trainable parameter summary: {type(e).__name__}: {e}")


def _is_encoder_connector_param_name(name: str) -> bool:
    n = (name or "").lower()
    encoder_or_connector_tokens = [
        "audio_tower",
        "audio_encoder",
        "speech_encoder",
        "feature_extractor",
        "multi_modal_projector",
        "modality_projector",
        "audio_projector",
        "projector",
        "connector",
        "bridge",
    ]
    return any(tok in n for tok in encoder_or_connector_tokens)


def _split_dual_lr_named_groups(model) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    no_decay_terms = ["bias", "layernorm.weight", "layer_norm.weight", "norm.weight"]
    grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
        "enc_decay": [],
        "enc_nodecay": [],
        "llm_decay": [],
        "llm_nodecay": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_enc = _is_encoder_connector_param_name(name)
        is_nodecay = any(t in name.lower() for t in no_decay_terms)
        if is_enc and is_nodecay:
            grouped["enc_nodecay"].append((name, param))
        elif is_enc:
            grouped["enc_decay"].append((name, param))
        elif is_nodecay:
            grouped["llm_nodecay"].append((name, param))
        else:
            grouped["llm_decay"].append((name, param))
    return grouped


def _print_dual_lr_group_preview(model, topk: int) -> None:
    groups = _split_dual_lr_named_groups(model)
    order = ["enc_decay", "enc_nodecay", "llm_decay", "llm_nodecay"]
    topk = max(1, int(topk))
    print("Dual-LR dry-run: parameter group preview")
    for key in order:
        items = groups[key]
        n_tensors = len(items)
        n_params = sum(int(p.numel()) for _, p in items)
        print(f"  - {key}: tensors={n_tensors:,}, params={n_params:,}")
        top_items = sorted(items, key=lambda x: int(x[1].numel()), reverse=True)[:topk]
        for rank, (name, p) in enumerate(top_items, start=1):
            print(f"      {rank:02d}. {name} [{int(p.numel()):,}]")


class DualLRTrainer(Trainer):
    """Trainer with two LR groups: encoder+multi_modal_projector vs LLM."""

    def __init__(
        self,
        *args,
        encoder_connector_learning_rate: float,
        llm_learning_rate: float,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder_connector_learning_rate = float(encoder_connector_learning_rate)
        self.llm_learning_rate = float(llm_learning_rate)

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        named_groups = _split_dual_lr_named_groups(opt_model)
        grouped = {k: [p for _, p in v] for k, v in named_groups.items()}

        total_trainable = 0
        for _, param in opt_model.named_parameters():
            if param.requires_grad:
                total_trainable += int(param.numel())

        optimizer_grouped_parameters = []
        if grouped["enc_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["enc_decay"],
                    "lr": float(self.encoder_connector_learning_rate),
                    "weight_decay": float(self.args.weight_decay),
                }
            )
        if grouped["enc_nodecay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["enc_nodecay"],
                    "lr": float(self.encoder_connector_learning_rate),
                    "weight_decay": 0.0,
                }
            )
        if grouped["llm_decay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["llm_decay"],
                    "lr": float(self.llm_learning_rate),
                    "weight_decay": float(self.args.weight_decay),
                }
            )
        if grouped["llm_nodecay"]:
            optimizer_grouped_parameters.append(
                {
                    "params": grouped["llm_nodecay"],
                    "lr": float(self.llm_learning_rate),
                    "weight_decay": 0.0,
                }
            )

        enc_params = sum(int(p.numel()) for p in grouped["enc_decay"] + grouped["enc_nodecay"])
        llm_params = sum(int(p.numel()) for p in grouped["llm_decay"] + grouped["llm_nodecay"])
        print(
            "Dual-LR optimizer enabled: "
            f"encoder+multi_modal_projector lr={self.encoder_connector_learning_rate:g} ({enc_params:,} params), "
            f"llm lr={self.llm_learning_rate:g} ({llm_params:,} params), "
            f"total trainable={total_trainable:,}."
        )

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(float(self.args.adam_beta1), float(self.args.adam_beta2)),
            eps=float(self.args.adam_epsilon),
        )
        return self.optimizer


class CheckpointLossLoggerCallback(TrainerCallback):
    """Save latest train/eval losses each time a checkpoint is written."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / "training_ckpt_log.jsonl"
        self._latest_train_loss: Optional[float] = None
        self._latest_eval_loss: Optional[float] = None

    @staticmethod
    def _as_float_or_none(v) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except Exception:
            return None

    @staticmethod
    def _latest_from_history(state, key: str) -> Optional[float]:
        hist = getattr(state, "log_history", None) or []
        for row in reversed(hist):
            if isinstance(row, dict) and key in row:
                try:
                    return float(row[key])
                except Exception:
                    continue
        return None

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        if not logs:
            return
        if "loss" in logs and "eval_loss" not in logs:
            self._latest_train_loss = self._as_float_or_none(logs.get("loss"))
        if "eval_loss" in logs:
            self._latest_eval_loss = self._as_float_or_none(logs.get("eval_loss"))

    def on_save(self, args, state, control, **kwargs):  # type: ignore[override]
        if not getattr(state, "is_world_process_zero", True):
            return

        step = int(getattr(state, "global_step", 0) or 0)
        epoch = self._as_float_or_none(getattr(state, "epoch", None))
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"

        train_loss = self._latest_train_loss
        eval_loss = self._latest_eval_loss
        if train_loss is None:
            train_loss = self._latest_from_history(state, "loss")
        if eval_loss is None:
            eval_loss = self._latest_from_history(state, "eval_loss")

        payload = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "global_step": step,
            "epoch": epoch,
            "checkpoint_dir": str(ckpt_dir),
            "train_loss": train_loss,
            "eval_loss": eval_loss,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
            if feats.shape[1] % target_frames != 0:
                pad_size = target_frames - (feats.shape[1] % target_frames)
                feats = np.pad(feats, ((0, 0), (0, pad_size)), mode="constant")
            feats = feats.reshape(feats.shape[0], -1, target_frames).transpose(1, 0, 2)
        elif feats.ndim == 3:
            if feats.shape[2] != target_frames:
                raise RuntimeError(f"Unexpected chunked feature shape: {feats.shape}")
        else:
            raise RuntimeError(f"Unexpected feature shape: {feats.shape}")

        feature_tensors.append(torch.as_tensor(feats))

    return torch.cat(feature_tensors, dim=0)


def _to_1d_long(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    t = t.to(dtype=torch.long)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    elif t.ndim >= 2:
        t = t[0]
    return t.reshape(-1)


def _parse_time_value(token: str) -> Optional[float]:
    s = (token or "").strip().lower()
    s = s.replace(",", ".")
    if not s:
        return None

    if ":" in s:
        parts = s.split(":")
        try:
            vals = [float(x) for x in parts]
        except Exception:
            return None
        if len(vals) == 2:
            mm, ss = vals
            if ss < 0:
                return None
            return max(0.0, mm * 60.0 + ss)
        if len(vals) == 3:
            hh, mm, ss = vals
            if mm < 0 or ss < 0:
                return None
            return max(0.0, hh * 3600.0 + mm * 60.0 + ss)
        return None

    m = re.match(
        r"^([0-9]+(?:[\.,][0-9]+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)?$",
        s,
    )
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or "s"
    if unit.startswith("h"):
        return max(0.0, val * 3600.0)
    if unit.startswith("m"):
        return max(0.0, val * 60.0)
    return max(0.0, val)


def _extract_time_ranges_from_text(text: str) -> List[Tuple[float, float]]:
    q = str(text or "")
    if not q:
        return []

    ranges: List[Tuple[float, float]] = []
    used_spans: List[Tuple[int, int]] = []

    time_atom = (
        r"(?:"
        r"\d+(?::\d{1,2}){1,2}"
        r"|"
        r"\d+(?:[\.,]\d+)?(?:\s*(?:h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds))?"
        r")"
    )
    range_re = re.compile(rf"(?P<a>{time_atom})\s*(?:-|–|—|to)\s*(?P<b>{time_atom})", flags=re.IGNORECASE)

    for m in range_re.finditer(q):
        a = _parse_time_value(m.group("a"))
        b = _parse_time_value(m.group("b"))
        if a is None or b is None:
            continue
        lo, hi = (a, b) if a <= b else (b, a)
        ranges.append((lo, hi))
        used_spans.append((m.start(), m.end()))

    point_kw_re = re.compile(
        rf"(?:at|around|near|timestamp|time)\s*(?P<t>{time_atom})",
        flags=re.IGNORECASE,
    )
    for m in point_kw_re.finditer(q):
        t = _parse_time_value(m.group("t"))
        if t is None:
            continue
        ranges.append((t, t))
        used_spans.append((m.start(), m.end()))

    def _overlaps_used(start: int, end: int) -> bool:
        for s0, e0 in used_spans:
            if not (end <= s0 or start >= e0):
                return True
        return False

    colon_time_re = re.compile(r"(?<![\d:])\d+(?::\d{1,2}){1,2}(?![\d:])")
    for m in colon_time_re.finditer(q):
        if _overlaps_used(m.start(), m.end()):
            continue
        t = _parse_time_value(m.group(0))
        if t is None:
            continue
        ranges.append((t, t))

    # Keep deterministic order and remove near-duplicates.
    ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
    deduped: List[Tuple[float, float]] = []
    for lo, hi in ranges:
        if deduped and abs(deduped[-1][0] - lo) < 1e-3 and abs(deduped[-1][1] - hi) < 1e-3:
            continue
        deduped.append((lo, hi))
    return deduped


def _format_seconds_like_token(seconds: float, token: str) -> str:
    t = str(token or "").strip().lower().replace(",", ".")
    v = max(0.0, float(seconds))

    if ":" in t:
        parts = t.split(":")
        sec_rounded = int(round(v))
        if len(parts) == 3 or v >= 3600:
            hh = sec_rounded // 3600
            mm = (sec_rounded % 3600) // 60
            ss = sec_rounded % 60
            return f"{hh:02d}:{mm:02d}:{ss:02d}"
        mm = sec_rounded // 60
        ss = sec_rounded % 60
        return f"{mm:02d}:{ss:02d}"

    m = re.match(
        r"^([0-9]+(?:[\.,][0-9]+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds)?$",
        t,
    )
    if not m:
        return f"{int(round(v))}s"

    unit = m.group(2) or "s"
    has_decimal = "." in (m.group(1) or "")

    if unit.startswith("h"):
        val = v / 3600.0
    elif unit.startswith("m"):
        val = v / 60.0
    else:
        val = v

    if has_decimal:
        return f"{val:.2f}{unit}"
    return f"{int(round(val))}{unit}"


def _map_time_to_concat_local(time_s: float, windows: Sequence[Tuple[float, float]]) -> Optional[float]:
    t = float(time_s)
    offset = 0.0
    for start_s, end_s in windows:
        a = float(start_s)
        b = float(end_s)
        if b <= a:
            continue
        if a <= t <= b:
            return offset + (t - a)
        offset += (b - a)
    return None


def _rewrite_timestamps_to_cropped_local_time(
    text: str,
    windows: Sequence[Tuple[float, float]],
) -> str:
    if not text:
        return text

    valid_windows = sorted(
        [
            (float(a), float(b))
            for a, b in windows
            if b is not None and a is not None and float(b) > float(a)
        ],
        key=lambda x: (x[0], x[1]),
    )
    if not valid_windows:
        return text

    q = str(text)
    time_atom = (
        r"(?:"
        r"\d+(?::\d{1,2}){1,2}"
        r"|"
        r"\d+(?:[\.,]\d+)?(?:\s*(?:h|hr|hrs|hour|hours|m|min|mins|minute|minutes|s|sec|secs|second|seconds))?"
        r")"
    )
    range_re = re.compile(
        rf"(?P<a>{time_atom})(?P<lws>\s*)(?P<sep>-|–|—|to)(?P<rws>\s*)(?P<b>{time_atom})",
        flags=re.IGNORECASE,
    )
    point_kw_re = re.compile(
        rf"(?P<prefix>(?:at|around|near|timestamp|time)\s*)(?P<t>{time_atom})",
        flags=re.IGNORECASE,
    )
    colon_time_re = re.compile(r"(?<![\d:])\d+(?::\d{1,2}){1,2}(?![\d:])")

    def _range_repl(m: re.Match) -> str:
        a_tok = m.group("a")
        b_tok = m.group("b")
        a = _parse_time_value(a_tok)
        b = _parse_time_value(b_tok)
        if a is None or b is None:
            return m.group(0)
        a_new = _map_time_to_concat_local(a, valid_windows)
        b_new = _map_time_to_concat_local(b, valid_windows)
        if a_new is None or b_new is None:
            return m.group(0)
        return (
            f"{_format_seconds_like_token(a_new, a_tok)}"
            f"{m.group('lws')}{m.group('sep')}{m.group('rws')}"
            f"{_format_seconds_like_token(b_new, b_tok)}"
        )

    q = range_re.sub(_range_repl, q)

    def _point_repl(m: re.Match) -> str:
        tok = m.group("t")
        t = _parse_time_value(tok)
        if t is None:
            return m.group(0)
        t_new = _map_time_to_concat_local(t, valid_windows)
        if t_new is None:
            return m.group(0)
        return f"{m.group('prefix')}{_format_seconds_like_token(t_new, tok)}"

    q = point_kw_re.sub(_point_repl, q)

    used_spans: List[Tuple[int, int]] = []
    for m in range_re.finditer(q):
        used_spans.append((m.start(), m.end()))
    for m in point_kw_re.finditer(q):
        used_spans.append((m.start(), m.end()))

    def _overlaps_used(start: int, end: int) -> bool:
        for s0, e0 in used_spans:
            if not (end <= s0 or start >= e0):
                return True
        return False

    pieces: List[str] = []
    pos = 0
    for m in colon_time_re.finditer(q):
        if _overlaps_used(m.start(), m.end()):
            continue
        tok = m.group(0)
        t = _parse_time_value(tok)
        if t is None:
            continue
        t_new = _map_time_to_concat_local(t, valid_windows)
        if t_new is None:
            continue
        pieces.append(q[pos : m.start()])
        pieces.append(_format_seconds_like_token(t_new, tok))
        pos = m.end()
    if pos == 0:
        return q
    pieces.append(q[pos:])
    return "".join(pieces)


def _load_audio_for_cropping(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    arr, sr = sf.read(str(audio_path), dtype="float32")
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    arr = np.asarray(arr, dtype=np.float32)

    if int(sr) != int(target_sr):
        try:
            import librosa

            arr = librosa.resample(arr, orig_sr=int(sr), target_sr=int(target_sr))
            sr = int(target_sr)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Audio resample failed for {audio_path}: {e}")

    return arr, int(sr)


def _build_audio_source_from_question(
    audio_path: str,
    question_text: str,
    sample_id: str,
    crop_from_question_refs: bool,
    crop_collar_seconds: float,
    random_crop_seconds: float,
) -> Tuple[object, str, bool, Optional[List[Tuple[float, float]]]]:
    # Returns (audio_source, audio_cache_key, allow_cache, crop_windows).
    # audio_source is either original path (str) or cropped waveform (np.ndarray).
    if not bool(crop_from_question_refs):
        return str(audio_path), str(audio_path), True, None

    waveform, sr = _load_audio_for_cropping(str(audio_path), target_sr=16000)
    total_samples = int(waveform.shape[0])
    if total_samples <= 0:
        return str(audio_path), str(audio_path), True, None

    duration_s = float(total_samples) / float(sr)
    refs = _extract_time_ranges_from_text(str(question_text))
    collar = max(0.0, float(crop_collar_seconds))

    if refs:
        segments: List[np.ndarray] = []
        windows: List[Tuple[float, float]] = []
        for lo, hi in refs:
            start_s = max(0.0, float(lo) - collar)
            end_s = min(duration_s, float(hi) + collar)
            if end_s <= start_s:
                continue
            s0 = int(start_s * sr)
            s1 = int(end_s * sr)
            if s1 <= s0:
                continue
            segments.append(waveform[s0:s1])
            windows.append((start_s, end_s))

        if segments:
            merged = np.concatenate(segments, axis=0).astype(np.float32, copy=False)
            key_payload = {
                "audio_path": str(audio_path),
                "mode": "refs",
                "windows": [[round(a, 3), round(b, 3)] for a, b in windows],
            }
            key = "crop:" + hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
            return merged, key, True, windows

    # No explicit references: random crop on-the-fly.
    crop_s = max(0.0, float(random_crop_seconds))
    crop_n = int(crop_s * sr)
    if crop_n > 0 and crop_n < total_samples:
        max_start = total_samples - crop_n
        start = random.randint(0, max_start)
        end = start + crop_n
        chunk = waveform[start:end].astype(np.float32, copy=False)
        key_payload = {
            "audio_path": str(audio_path),
            "mode": "random",
            "sample_id": str(sample_id),
            "start": int(start),
            "end": int(end),
        }
        key = "crop:" + hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
        return chunk, key, False, None

    return waveform.astype(np.float32, copy=False), str(audio_path), True, None


def _encode_chat_sample(
    processor: VoxtralProcessor,
    model_id: str,
    prompt_language: str,
    audio_source: object,
    prompt_text: str,
    assistant_text: str,
    max_prompt_tokens: int,
    audio_prompt_cache: Optional[OrderedDict] = None,
    audio_prompt_cache_size: int = 0,
    audio_cache_key: Optional[str] = None,
    allow_audio_cache: bool = True,
) -> Dict[str, torch.Tensor]:
    conversation = [
        {
            "role": "user",
            "content": [
                ({"type": "audio", "path": str(audio_source)} if isinstance(audio_source, str) else {"type": "audio", "audio": audio_source}),
                {"type": "text", "text": prompt_text},
            ],
        },
        {
            "role": "assistant",
            "content": assistant_text,
        },
    ]

    try:
        cache_key = str(audio_cache_key) if audio_cache_key else (str(audio_source) if isinstance(audio_source, str) else None)
        cache_entry = None
        use_audio_prompt_cache = (
            bool(allow_audio_cache)
            and cache_key is not None
            and audio_prompt_cache is not None
            and int(audio_prompt_cache_size) > 0
        )
        if use_audio_prompt_cache:
            cache_entry = audio_prompt_cache.get(cache_key)
            if cache_entry is not None:
                audio_prompt_cache.move_to_end(cache_key)

        if cache_entry is None:
            # Prefer the robust transcription-request API used in voxtral_train_router.py.
            if isinstance(audio_source, str):
                prompt = processor.apply_transcription_request(
                    language=str(prompt_language),
                    model_id=str(model_id),
                    audio=[str(audio_source)],
                    format=["WAV"],
                    sampling_rate=16000,
                    return_tensors="pt",
                )
            else:
                prompt = processor.apply_transcription_request(
                    language=str(prompt_language),
                    model_id=str(model_id),
                    audio=[audio_source],
                    sampling_rate=16000,
                    return_tensors="pt",
                )
            prompt_ids = _to_1d_long(prompt["input_ids"])
            prompt_mask = _to_1d_long(prompt["attention_mask"])
            input_features = prompt["input_features"]

            if use_audio_prompt_cache:
                input_features_cached = input_features
                if isinstance(input_features_cached, torch.Tensor):
                    input_features_cached = input_features_cached.detach().cpu()
                audio_prompt_cache[cache_key] = {
                    "prompt_ids": prompt_ids.detach().cpu(),
                    "prompt_mask": prompt_mask.detach().cpu(),
                    "input_features": input_features_cached,
                }
                audio_prompt_cache.move_to_end(cache_key)
                while len(audio_prompt_cache) > int(audio_prompt_cache_size):
                    audio_prompt_cache.popitem(last=False)
        else:
            prompt_ids = cache_entry["prompt_ids"].clone()
            prompt_mask = cache_entry["prompt_mask"].clone()
            input_features = cache_entry["input_features"]
            if isinstance(input_features, torch.Tensor):
                input_features = input_features.clone()

        prompt_text_ids = processor.tokenizer(
            str(prompt_text),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to(dtype=torch.long)
        if prompt_text_ids.ndim == 0:
            prompt_text_ids = prompt_text_ids.unsqueeze(0)
        prompt_text_ids = prompt_text_ids.reshape(-1)

        if prompt_text_ids.shape[0] > 0:
            prompt_ids = torch.cat([prompt_ids, prompt_text_ids], dim=0)
            prompt_mask = torch.cat(
                [prompt_mask, torch.ones((prompt_text_ids.shape[0],), dtype=torch.long)],
                dim=0,
            )

        if int(max_prompt_tokens) > 0 and prompt_ids.shape[0] > int(max_prompt_tokens):
            prompt_ids = prompt_ids[-int(max_prompt_tokens) :]
            prompt_mask = prompt_mask[-int(max_prompt_tokens) :]

        target_ids = processor.tokenizer(
            str(assistant_text),
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0].to(dtype=torch.long)
        eos_id = processor.tokenizer.eos_token_id
        if eos_id is not None:
            target_ids = torch.cat([target_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)

        input_ids = torch.cat([prompt_ids, target_ids], dim=0)
        attention_mask = torch.cat(
            [prompt_mask, torch.ones((target_ids.shape[0],), dtype=torch.long)],
            dim=0,
        )
        assistant_mask = torch.zeros((input_ids.shape[0],), dtype=torch.bool)
        assistant_mask[prompt_ids.shape[0] :] = True
    except Exception:
        try:
            # Older API: same processor path but no assistant mask support.
            encoded = processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = _to_1d_long(encoded["input_ids"])
            attention_mask = _to_1d_long(encoded["attention_mask"])
            input_features = encoded["input_features"]

            prompt_text_ids = processor.tokenizer(
                str(prompt_text),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(dtype=torch.long)
            if prompt_text_ids.ndim == 0:
                prompt_text_ids = prompt_text_ids.unsqueeze(0)
            prompt_text_ids = prompt_text_ids.reshape(-1)
            if prompt_text_ids.shape[0] > 0:
                input_ids = torch.cat([input_ids, prompt_text_ids], dim=0)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((prompt_text_ids.shape[0],), dtype=torch.long)],
                    dim=0,
                )

            assistant_mask = None
        except Exception:
            # Tokenizer fallback for versions where processor template path is unavailable.
            tmp_audio_path = None
            if isinstance(audio_source, str):
                fallback_audio_path = str(audio_source)
            else:
                fallback_audio_arr = np.asarray(audio_source, dtype=np.float32)
                try:
                    import soundfile as sf

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tmp_audio_path = tf.name
                    sf.write(tmp_audio_path, fallback_audio_arr, 16000)
                    fallback_audio_path = tmp_audio_path
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Failed to materialize cropped audio for tokenizer fallback: {e}")

            user_message = {
                "role": "user",
                "content": [
                    {"type": "audio", "path": fallback_audio_path},
                    {"type": "text", "text": str(prompt_text)},
                ],
            }

            try:
                try:
                    encoded = processor.tokenizer.apply_chat_template(
                        [user_message],
                        return_tensors=None,
                        return_assistant_tokens_mask=True,
                    )
                except Exception:
                    encoded = processor.tokenizer.apply_chat_template(
                        [user_message],
                        return_tensors=None,
                    )
            finally:
                if tmp_audio_path is not None:
                    try:
                        Path(tmp_audio_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            chat_audio = encoded.pop("audio", None)
            if chat_audio is None:
                raise RuntimeError("Tokenizer chat template did not return audio for fallback.")

            # Build teacher-forcing targets manually from assistant text to avoid
            # strict validator errors for chats ending with assistant in older APIs.
            prompt_ids = _to_1d_long(encoded["input_ids"])
            prompt_mask = _to_1d_long(encoded["attention_mask"])

            prompt_text_ids = processor.tokenizer(
                str(prompt_text),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(dtype=torch.long)
            if prompt_text_ids.ndim == 0:
                prompt_text_ids = prompt_text_ids.unsqueeze(0)
            prompt_text_ids = prompt_text_ids.reshape(-1)
            if prompt_text_ids.shape[0] > 0:
                prompt_ids = torch.cat([prompt_ids, prompt_text_ids], dim=0)
                prompt_mask = torch.cat(
                    [prompt_mask, torch.ones((prompt_text_ids.shape[0],), dtype=torch.long)],
                    dim=0,
                )

            if int(max_prompt_tokens) > 0 and prompt_ids.shape[0] > int(max_prompt_tokens):
                prompt_ids = prompt_ids[-int(max_prompt_tokens) :]
                prompt_mask = prompt_mask[-int(max_prompt_tokens) :]

            target_ids = processor.tokenizer(
                str(assistant_text),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0].to(dtype=torch.long)
            eos_id = processor.tokenizer.eos_token_id
            if eos_id is not None:
                target_ids = torch.cat([target_ids, torch.tensor([eos_id], dtype=torch.long)], dim=0)

            input_ids = torch.cat([prompt_ids, target_ids], dim=0)
            attention_mask = torch.cat(
                [prompt_mask, torch.ones((target_ids.shape[0],), dtype=torch.long)],
                dim=0,
            )
            input_features = _build_voxtral_chat_input_features(processor, chat_audio)

            assistant_mask = torch.zeros((input_ids.shape[0],), dtype=torch.bool)
            assistant_mask[prompt_ids.shape[0] :] = True

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if assistant_mask is not None and assistant_mask.shape[0] == labels.shape[0]:
        labels[~assistant_mask] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_features": input_features,
    }


class VoxtralMCQCollator:
    def __init__(
        self,
        processor: VoxtralProcessor,
        model_id: str,
        prompt_language: str,
        max_prompt_tokens: int,
        audio_prompt_cache_size: int,
        crop_from_question_refs: bool,
        remap_timestamps_after_crop: bool,
        crop_collar_seconds: float,
        random_crop_seconds: float,
    ):
        self.processor = processor
        self.model_id = model_id
        self.prompt_language = prompt_language
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.audio_prompt_cache_size = max(0, int(audio_prompt_cache_size))
        self.audio_prompt_cache: OrderedDict = OrderedDict()
        self.crop_from_question_refs = bool(crop_from_question_refs)
        self.remap_timestamps_after_crop = bool(remap_timestamps_after_crop)
        self.crop_collar_seconds = float(crop_collar_seconds)
        self.random_crop_seconds = float(random_crop_seconds)
        tok = self.processor.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        encoded = []
        for f in features:
            audio_source, audio_cache_key, allow_audio_cache, crop_windows = _build_audio_source_from_question(
                audio_path=str(f["audio_path"]),
                question_text=str(f.get("question") or f.get("prompt_text") or ""),
                sample_id=str(f.get("sample_id") or ""),
                crop_from_question_refs=bool(self.crop_from_question_refs),
                crop_collar_seconds=float(self.crop_collar_seconds),
                random_crop_seconds=float(self.random_crop_seconds),
            )
            prompt_text = str(f["prompt_text"])
            if bool(self.remap_timestamps_after_crop) and crop_windows:
                prompt_text = _rewrite_timestamps_to_cropped_local_time(
                    text=prompt_text,
                    windows=crop_windows,
                )
            encoded.append(
                _encode_chat_sample(
                    processor=self.processor,
                    model_id=self.model_id,
                    prompt_language=self.prompt_language,
                    audio_source=audio_source,
                    prompt_text=prompt_text,
                    assistant_text=str(f["gold_choice"]),
                    max_prompt_tokens=self.max_prompt_tokens,
                    audio_prompt_cache=self.audio_prompt_cache,
                    audio_prompt_cache_size=self.audio_prompt_cache_size,
                    audio_cache_key=audio_cache_key,
                    allow_audio_cache=allow_audio_cache,
                )
            )

        max_len = max(x["input_ids"].shape[0] for x in encoded)

        def pad_1d(x: torch.Tensor, fill: int) -> torch.Tensor:
            if x.shape[0] == max_len:
                return x
            pad = torch.full((max_len - x.shape[0],), fill, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        input_ids = torch.stack([pad_1d(x["input_ids"], self.pad_id) for x in encoded], dim=0)
        attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in encoded], dim=0)
        labels = torch.stack([pad_1d(x["labels"], -100) for x in encoded], dim=0)
        input_features = torch.cat([x["input_features"] for x in encoded], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "input_features": input_features,
        }


def _run_audio_instruction(model, processor, audio_path: str, prompt: str, max_new_tokens: int) -> str:
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
            raise RuntimeError("Tokenizer chat template did not return audio for fallback.")

        inputs = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "input_features": _build_voxtral_chat_input_features(processor, chat_audio),
        }
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return decoded[0] if decoded else ""


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

    out: Dict[str, Dict[str, float]] = {}
    for key, v in sorted(counts.items()):
        total = int(v["total"])
        correct = int(v["correct"])
        out[key] = {
            "n_total": total,
            "n_correct": correct,
            "accuracy": (correct / total) if total > 0 else 0.0,
        }
    return out


def _add_mcq_breakdowns(summary: Dict, rows: List[Dict]) -> Dict:
    summary["language_breakdown"] = _compute_accuracy_breakdown(rows, "language")
    summary["difficulty_breakdown"] = _compute_accuracy_breakdown(rows, "difficulty")
    summary["category_breakdown"] = _compute_accuracy_breakdown(rows, "category")
    summary["subtype_breakdown"] = _compute_accuracy_breakdown(rows, "subtype")
    summary["task_name_breakdown"] = _compute_accuracy_breakdown(rows, "task_name")
    summary["sub_category_breakdown"] = _compute_accuracy_breakdown(rows, "sub-category")
    summary["sub_sub_category_breakdown"] = _compute_accuracy_breakdown(rows, "sub-sub-category")
    summary["linguistics_sub_discipline_breakdown"] = _compute_accuracy_breakdown(
        rows,
        "linguistics_sub_discipline",
    )
    return summary


def evaluate_mcq(
    model,
    processor,
    samples: List[MCQSample],
    max_new_tokens: int,
    output_dir: Path,
) -> Dict:
    rows = []
    correct = 0
    error_counts: Counter[str] = Counter()

    for i, s in enumerate(samples, start=1):
        try:
            out = _run_audio_instruction(
                model=model,
                processor=processor,
                audio_path=s.audio_path,
                prompt=s.prompt_text,
                max_new_tokens=max_new_tokens,
            )
            pred_choice = _select_multiple_choice_option(out, list(s.choice_map.keys()))
            ok = pred_choice == s.gold_choice
            correct += int(ok)
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "audio_path": s.audio_path,
                    "question": s.question,
                    "gold_choice": s.gold_choice,
                    "pred_choice": pred_choice,
                    "is_correct": bool(ok),
                    "response": out,
                    "choices": s.choice_map,
                    "metadata": s.metadata,
                }
            )
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            error_counts[err] += 1
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "audio_path": s.audio_path,
                    "question": s.question,
                    "gold_choice": s.gold_choice,
                    "pred_choice": None,
                    "is_correct": False,
                    "response": "",
                    "error": err,
                    "choices": s.choice_map,
                    "metadata": s.metadata,
                }
            )

        if i % 20 == 0 or i == len(samples):
            done = max(1, i)
            print(f"[eval] {i}/{len(samples)} processed | acc_so_far={correct/done:.4f}")

    n = len(samples)
    acc = (correct / n) if n > 0 else 0.0
    summary = {
        "n_total": n,
        "n_correct": correct,
        "accuracy": acc,
        "n_error": int(sum(error_counts.values())),
        "error_counts": dict(sorted(error_counts.items(), key=lambda kv: kv[0])),
    }
    summary = _add_mcq_breakdowns(summary, rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mcq_eval_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "mcq_eval_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Voxtral on JSONL audio MCQ")

    p.add_argument("--model-id", default="mistralai/Voxtral-Mini-3B-2507")
    p.add_argument("--model-mode", choices=["baseline", "adapter"], default="baseline")
    p.add_argument("--adapter-path", default=None)

    p.add_argument("--train-jsonl", required=False)
    p.add_argument("--eval-jsonl", default=None)
    p.add_argument("--audio-root", default=None)
    p.add_argument("--prompt-language", default="en")
    p.add_argument("--mcq-cache-dir", default=None)
    p.add_argument("--cache-train-split", default="train")
    p.add_argument("--cache-eval-split", default="dev")
    p.add_argument("--audio-shard-cache-dir", default=None)
    p.add_argument("--audio-shard-train-split", default="train")

    p.add_argument("--do-train", action="store_true")
    p.add_argument("--do-eval", action="store_true")
    p.add_argument("--eval-fraction", type=float, default=0.0)

    p.add_argument("--max-questions-per-audio", type=int, default=0)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)

    p.add_argument("--num-epochs", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument(
        "--encoder-connector-learning-rate",
        type=float,
        default=None,
        help="Optional LR for encoder+multi_modal_projector parameter groups (requires --llm-learning-rate).",
    )
    p.add_argument(
        "--llm-learning-rate",
        type=float,
        default=None,
        help="Optional LR for LLM parameter groups (requires --encoder-connector-learning-rate).",
    )
    p.add_argument(
        "--dual-lr-dry-run",
        action="store_true",
        help="Print dual-LR parameter groups and exit before training.",
    )
    p.add_argument(
        "--dual-lr-dry-run-topk",
        type=int,
        default=20,
        help="Number of top parameter names to print per dual-LR group in dry-run mode.",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument(
        "--audio-prompt-cache-size",
        type=int,
        default=256,
        help="Collator-side LRU cache size (audio-path keyed) for reusable audio prompt encoding; set 0 to disable.",
    )
    p.add_argument(
        "--crop-from-question-refs",
        action="store_true",
        help="Enable collator-side audio cropping based on timestamp/range mentions in the question text.",
    )
    p.add_argument(
        "--remap-timestamps-after-crop",
        action="store_true",
        help=(
            "When using --crop-from-question-refs and explicit timestamp ranges, rewrite question/prompt "
            "timestamps to concatenated-local crop time."
        ),
    )
    p.add_argument(
        "--crop-collar-seconds",
        type=float,
        default=30.0,
        help="Extra seconds added to both sides of each detected timestamp/range.",
    )
    p.add_argument(
        "--random-crop-seconds",
        type=float,
        default=300.0,
        help="When no timestamp/range is detected, use an on-the-fly random crop of this duration (seconds).",
    )

    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--output-root", default="experiments")
    p.add_argument("--experiment-name", default="voxtral_mcq")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--timestamped-exp-dir", action="store_true")

    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--no-use-bf16", dest="use_bf16", action="store_false")
    p.set_defaults(use_bf16=True)
    p.add_argument("--use-fp16", action="store_true")

    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    base = Path(args.output_root)
    exp = str(args.experiment_name)
    if args.timestamped_exp_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        return base / f"{exp}_{stamp}"
    return base / exp


def _save_run_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    args_payload = {
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "args": vars(args),
    }
    (output_dir / "run_args.json").write_text(
        json.dumps(args_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = "python " + " ".join(shlex.quote(str(x)) for x in sys.argv)
    cmd_sh_path = output_dir / "run_command.sh"
    cmd_sh_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + cmd + "\n",
        encoding="utf-8",
    )
    try:
        cmd_sh_path.chmod(0o755)
    except Exception:
        pass

    (output_dir / "run_command.txt").write_text(cmd + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.do_train and not args.do_eval:
        args.do_eval = True

    # Runtime safety: many GPUs (e.g., pre-Ampere) cannot run BF16 matmuls.
    # In that case fallback to FP16 automatically to avoid cublasGemmEx invalid-value crashes.
    if bool(args.use_bf16) and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print(
            "[warn] CUDA BF16 is not supported on this GPU/runtime. "
            "Falling back to FP16 to avoid CUDA cublas errors."
        )
        args.use_bf16 = False
        if not bool(args.use_fp16):
            args.use_fp16 = True

    if bool(args.use_bf16) and bool(args.use_fp16):
        # Keep settings deterministic; BF16 takes precedence when supported.
        args.use_fp16 = False

    if bool(args.remap_timestamps_after_crop) and not bool(args.crop_from_question_refs):
        print(
            "[warn] --remap-timestamps-after-crop has no effect unless --crop-from-question-refs is enabled."
        )

    if args.mcq_cache_dir:
        cache_dir = Path(args.mcq_cache_dir)
        train_ds = _load_cache_split_dataset(
            cache_dir=cache_dir,
            split_name=str(args.cache_train_split),
            num_shards=int(args.num_shards),
            shard_index=int(args.shard_index),
        )
        if args.max_train_samples > 0:
            train_ds = train_ds.select(range(min(int(args.max_train_samples), len(train_ds))))

        eval_samples: List[MCQSample] = []
        eval_ds: Optional[Dataset] = None
        if args.do_eval or args.do_train:
            eval_split = str(args.cache_eval_split).strip()
            if eval_split:
                eval_ds = _load_cache_split_dataset(
                    cache_dir=cache_dir,
                    split_name=eval_split,
                    num_shards=1,
                    shard_index=0,
                )
                if args.max_eval_samples > 0:
                    eval_ds = eval_ds.select(range(min(int(args.max_eval_samples), len(eval_ds))))
                eval_samples = _dataset_to_samples(eval_ds)

        train_samples = _dataset_to_samples(train_ds)
    else:
        if not args.train_jsonl:
            raise ValueError("Either --train-jsonl or --mcq-cache-dir is required.")

        train_task = load_jsonl_audio_mcq(
            jsonl_path=args.train_jsonl,
            audio_root=args.audio_root,
            max_questions_per_audio=max(0, int(args.max_questions_per_audio)),
            max_samples=max(0, int(args.max_train_samples)),
            seed=int(args.seed),
        )

        if args.eval_jsonl:
            eval_task = load_jsonl_audio_mcq(
                jsonl_path=args.eval_jsonl,
                audio_root=args.audio_root,
                max_questions_per_audio=max(0, int(args.max_questions_per_audio)),
                max_samples=max(0, int(args.max_eval_samples)),
                seed=int(args.seed),
            )
            train_samples = train_task.samples
            eval_samples = eval_task.samples
        else:
            train_samples, eval_samples = _split_train_eval(
                train_task.samples,
                eval_fraction=float(args.eval_fraction),
                seed=int(args.seed),
            )
            if args.max_eval_samples > 0 and eval_samples:
                eval_samples = eval_samples[: min(int(args.max_eval_samples), len(eval_samples))]

        if args.audio_shard_cache_dir and int(args.num_shards) > 1:
            train_samples = _apply_audio_shard(
                samples=train_samples,
                audio_shard_cache_dir=str(args.audio_shard_cache_dir),
                split_name=str(args.audio_shard_train_split),
                num_shards=int(args.num_shards),
                shard_index=int(args.shard_index),
            )
        else:
            train_samples = _apply_shard(train_samples, int(args.num_shards), int(args.shard_index))
        train_ds = _samples_to_hf_dataset(train_samples)
        eval_ds = _samples_to_hf_dataset(eval_samples) if eval_samples else None

    print(f"Train samples: {len(train_samples)}")
    print(f"Eval samples: {len(eval_samples)}")

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_run_metadata(output_dir=output_dir, args=args)

    processor = VoxtralProcessor.from_pretrained(args.model_id)
    dtype = _choose_dtype(use_bf16=bool(args.use_bf16), use_fp16=bool(args.use_fp16))
    model = _load_model(
        model_id=args.model_id,
        model_mode=args.model_mode,
        adapter_path=args.adapter_path,
        dtype=dtype,
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    _print_trainable_params(model)

    if args.do_train:
        dual_lr_enabled = args.encoder_connector_learning_rate is not None or args.llm_learning_rate is not None
        if dual_lr_enabled and (args.encoder_connector_learning_rate is None or args.llm_learning_rate is None):
            raise ValueError(
                "Both --encoder-connector-learning-rate and --llm-learning-rate must be set together."
            )
        if bool(args.dual_lr_dry_run) and not dual_lr_enabled:
            raise ValueError(
                "--dual-lr-dry-run requires both --encoder-connector-learning-rate and --llm-learning-rate."
            )
        if bool(args.dual_lr_dry_run):
            _print_dual_lr_group_preview(model, topk=int(args.dual_lr_dry_run_topk))
            print("Dual-LR dry-run complete. Exiting before trainer initialization.")
            return

        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

        collator = VoxtralMCQCollator(
            processor=processor,
            model_id=str(args.model_id),
            prompt_language=str(args.prompt_language),
            max_prompt_tokens=int(args.max_prompt_tokens),
            audio_prompt_cache_size=int(args.audio_prompt_cache_size),
            crop_from_question_refs=bool(args.crop_from_question_refs),
            remap_timestamps_after_crop=bool(args.remap_timestamps_after_crop),
            crop_collar_seconds=float(args.crop_collar_seconds),
            random_crop_seconds=float(args.random_crop_seconds),
        )

        train_args_kwargs = {
            "output_dir": str(output_dir),
            "per_device_train_batch_size": int(args.train_batch_size),
            "per_device_eval_batch_size": int(args.eval_batch_size),
            "gradient_accumulation_steps": int(args.grad_accum_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "warmup_ratio": float(args.warmup_ratio),
            "logging_steps": int(args.logging_steps),
            "save_steps": int(args.save_steps),
            "eval_steps": int(args.eval_steps),
            "num_train_epochs": int(args.num_epochs),
            "bf16": bool(args.use_bf16),
            "fp16": bool(args.use_fp16),
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "save_strategy": "steps",
            "load_best_model_at_end": False,
            "report_to": [],
        }

        eval_mode = "steps" if eval_ds is not None else "no"
        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if "evaluation_strategy" in ta_params:
            train_args_kwargs["evaluation_strategy"] = eval_mode
        elif "eval_strategy" in ta_params:
            train_args_kwargs["eval_strategy"] = eval_mode

        train_args = TrainingArguments(**train_args_kwargs)

        trainer_cls = DualLRTrainer if dual_lr_enabled else Trainer
        trainer_kwargs = {
            "model": model,
            "args": train_args,
            "data_collator": collator,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "callbacks": [CheckpointLossLoggerCallback(output_dir=output_dir)],
        }
        if dual_lr_enabled:
            trainer_kwargs["encoder_connector_learning_rate"] = float(args.encoder_connector_learning_rate)
            trainer_kwargs["llm_learning_rate"] = float(args.llm_learning_rate)

        trainer = trainer_cls(**trainer_kwargs)

        trainer.train()

        final_dir = output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        processor.save_pretrained(str(final_dir))
        print(f"Training complete. Saved model/artifacts to: {final_dir}")

    if args.do_eval:
        eval_model = model
        summary = evaluate_mcq(
            model=eval_model,
            processor=processor,
            samples=eval_samples,
            max_new_tokens=int(args.max_new_tokens),
            output_dir=output_dir,
        )
        print("\nEvaluation complete:")
        print(f"  Accuracy: {summary['accuracy']:.4f}")
        print(f"  n_total: {summary['n_total']}")
        print(f"  n_correct: {summary['n_correct']}")
        print(f"  Metrics saved to: {output_dir / 'mcq_eval_metrics.json'}")


if __name__ == "__main__":
    main()
