#!/usr/bin/env python3
"""Voxtral training/evaluation router for VC, OpenSLR, and MLC datasets.

This script mirrors the notebook's configuration style and canonical Voxtral path:
- Canonical schema: samples must contain `audio` and `text`.
- VAD settings/config are exposed in one place (Config dataclass).
- Dataset recovery is cache-first for VC/OpenSLR/MLC.
- Train/eval routing is controlled by CLI arguments.

Examples:
  python -m asr_merging.voxtral_train_router \
        --source vc --language sr --do-train --do-eval \
        --model-mode baseline --train-set train --valid-set valid --evaluation-set test

  python -m asr_merging.voxtral_train_router \
        --source mlc --do-eval --model-mode adapter \
    --adapter-path experiments/my_mlc_run/checkpoint-1000 \
        --evaluation-set test_clean
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import re
import shutil
import sys
import tempfile
import unicodedata
import wave
from datetime import datetime
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import jiwer
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import Audio, Dataset, concatenate_datasets, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    VoxtralForConditionalGeneration,
    VoxtralProcessor,
)
# FIX for PyTorch 2.10 + CUDA 12.8 + L40S: cuBLAS StridedBatched is broken, use cuBLASLt instead
torch.backends.cuda.preferred_blas_library("cublaslt")

MLC_LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "french": "fr",
    "french(canada)": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "portuguese(brazil)": "pt",
    "russian": "ru",
    "spanish": "es",
    "spanish(mexico)": "es",
    "tagalog": "tl",
    "thai": "th",
    "turkish": "tr",
    "urdu": "ur",
    "vietnamese": "vi",
}

LANGUAGE_FIELD_CANDIDATES_DEFAULT = ("language", "lang", "lang_code", "language_code", "locale")

try:
    from num2words import num2words
except Exception:
    num2words = None

# Languages where spacing between characters improves jiwer tokenization.
WER_CHAR_SPACING_LANGS = {"ja", "ko", "th", "zh", "zh-cn", "zh-tw", "yue"}

# Map common ASR tags to num2words language codes.
NUM2WORDS_LANG_MAP = {
    "ar": "ar",
    "cs": "cs",
    "da": "da",
    "de": "de",
    "en": "en",
    "es": "es",
    "fi": "fi",
    "fr": "fr",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "nl": "nl",
    "no": "no",
    "pl": "pl",
    "pt": "pt",
    "pt-br": "pt_BR",
    "ru": "ru",
    "sv": "sv",
    "tr": "tr",
    "uk": "uk",
    "zh": "zh",
}

WER_POLICY = {
    "apply_to_refs": False,
    "apply_to_preds": True,
    "replace_zero_between_words": False,
    "convert_numbers_to_words": True,
    "add_char_spacing_for_cjkt": True,
    "remove_punctuation": True,
    "lowercase": True,
}


def _map_language_for_model(language: Optional[str], *, use_name_mapping: bool) -> Optional[str]:
    if language is None:
        return None
    s = str(language).strip().lower()
    if not s:
        return None
    if use_name_mapping:
        return MLC_LANGUAGE_NAME_TO_CODE.get(s, s)
    return s

@dataclass
class Config:
    model_id: str = "mistralai/Voxtral-Mini-3B-2507"
    sample_rate: int = 16000

    # Optimization / training knobs (aligned to notebook style)
    num_epochs: int = 1
    train_batch_size: int = 16
    eval_batch_size: int = 16
    grad_accum_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.0
    warmup_steps: int = 0
    logging_steps: int = 20
    save_steps: int = 200
    eval_steps: int = 200
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    early_stopping_patience: Optional[int] = None
    step_decay_gamma: float = 0.7
    step_decay_epochs: int = 3

    # Runtime
    use_quantization: bool = False
    use_grad_checkpoint: bool = False
    use_bf16: bool = True
    use_fp16: bool = False
    dataloader_num_workers: int = 0

    # VAD knobs (kept for parity with notebook configuration)
    use_vad_filtering: bool = True
    vad_backend: str = "silero"
    vad_threshold: float = 0.25
    vad_min_speech_ms: int = 250
    vad_speech_pad_ms: int = 250
    vad_min_retained_ratio: float = 0.35
    vad_max_length: int = 30

    # Paths (canonical Voxtral cache roots)
    processed_cv_cache_dir: Path = Path("data/cache/processed_cv_datasets/voxtral_prompt_aligned")
    openslr_cache_dir: Path = Path("data/cache/processed_openslr/voxtral_prompt_aligned")
    mlc_cache_dir: Path = Path("data/cache/voxtral")
    mlc_cache_name: Optional[str] = None
    mlc_train_dev_cache_names: List[str] = field(default_factory=list)
    mlc_test_cache_name: Optional[str] = None
    adaptive_sampling_enabled: bool = False
    adaptive_sampling_languages: List[str] = field(default_factory=list)
    adaptive_sampling_multiplier: float = 1.5
    adaptive_sampling_stage1_fraction: float = 0.25
    adaptive_sampling_seed: int = 42

    # Prompt/eval language for Voxtral path
    prompt_language: str = "en"
    enable_per_sample_language_prompt: bool = False
    language_dropout_fraction: float = 0.0
    language_dropout_seed: int = 42
    enable_mlc_language_name_mapping: bool = True
    language_field_candidates: List[str] = field(default_factory=lambda: list(LANGUAGE_FIELD_CANDIDATES_DEFAULT))
    eval_decoding_language_mode: str = "fixed"
    eval_scoring_mode: str = "normalization"


def _default_cli_options() -> Dict:
    return {
        "source": None,
        "language": None,
        "model_mode": "baseline",
        "adapter_path": None,
        "do_train": False,
        "do_eval": False,
        "do_test": False,
        "train_set": None,
        "valid_set": None,
        "evaluation_set": "test",
        "test_set": None,
        "output_dir": None,
        "resume_from_checkpoint": None,
        "output_root": "experiments",
        "experiment_name": None,
        "timestamped_exp_dir": False,
        "tf_tracking": False,
        "max_new_tokens": 256,
        "num_epochs": 1,
        "train_batch_size": 2,
        "eval_batch_size": 4,
        "grad_accum_steps": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.0,
        "warmup_steps": 0,
        "prompt_language": "en",
        "enable_per_sample_language_prompt": False,
        "language_dropout_fraction": 0.0,
        "language_dropout_seed": 42,
        "enable_mlc_language_name_mapping": True,
        "eval_decoding_language_mode": "fixed",
        "eval_scoring_mode": "normalization",
        "save_predictions_file": False,
        "predictions_file_path": None,
        "early_stopping_patience": None,
        "validation_split_ratio": 0.1,
        "validation_split_seed": 42,
        "use_vad": True,
        "mlc_cache_name": None,
        "mlc_train_dev_cache_names": [],
        "mlc_test_cache_name": None,
        "adaptive_sampling_enabled": False,
        "adaptive_sampling_languages": [],
        "adaptive_sampling_multiplier": 1.5,
        "adaptive_sampling_stage1_fraction": 0.25,
        "adaptive_sampling_seed": 42,
        "config_json": None,
    }


def _convert_value_like_default(value, default):
    if default is None or value is None:
        return value
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "yes", "y", "on"}:
                return True
            if v in {"0", "false", "no", "n", "off"}:
                return False
        raise ValueError(f"Expected boolean value, got: {value!r}")
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    if isinstance(default, list):
        if isinstance(value, list):
            return [str(v) for v in value]
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        raise ValueError(f"Expected list value, got: {value!r}")
    return value


def _coerce_config_payload(raw_payload: Dict) -> Dict:
    defaults = _default_cli_options()
    known_keys = set(defaults.keys()) | {f.name for f in fields(Config)}

    payload = dict(raw_payload)
    if "datasets" in payload and isinstance(payload["datasets"], dict):
        payload = {**payload, **payload["datasets"]}
    if "args" in payload and isinstance(payload["args"], dict):
        payload = {**payload, **payload["args"]}
    if "config" in payload and isinstance(payload["config"], dict):
        payload = {**payload, **payload["config"]}

    unsupported = sorted(k for k in payload.keys() if k not in known_keys and k not in {"datasets", "args", "config"})
    if unsupported:
        raise ValueError(f"Unsupported keys in config json: {unsupported}")

    coerced = {}
    for k, v in payload.items():
        if k in defaults:
            coerced[k] = _convert_value_like_default(v, defaults[k])
        elif k in {f.name for f in fields(Config)}:
            coerced[k] = v
    return coerced


def _load_json_config(config_path: Optional[str]) -> Tuple[Dict, Dict, Optional[Path]]:
    if not config_path:
        return {}, {}, None
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config json file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config json must be a JSON object.")
    coerced = _coerce_config_payload(raw)
    return raw, coerced, path


def _merge_cli_with_config(cli_ns: argparse.Namespace) -> Tuple[Dict, Dict, Dict, Optional[Path]]:
    defaults = _default_cli_options()
    provided_cli = vars(cli_ns)
    raw_cfg, cfg_values, cfg_path = _load_json_config(provided_cli.get("config_json"))

    merged = copy.deepcopy(defaults)
    for k, v in cfg_values.items():
        if k in merged:
            merged[k] = v
    # CLI has highest priority.
    for k, v in provided_cli.items():
        merged[k] = v

    if merged.get("source") is None:
        raise ValueError("Missing required argument 'source'. Provide --source or set 'source' in config json.")

    return merged, raw_cfg, cfg_values, cfg_path


def _build_config_from_inputs(merged: Dict, cfg_values: Dict) -> Config:
    config_overrides = {}
    for f in fields(Config):
        if f.name in cfg_values:
            config_overrides[f.name] = cfg_values[f.name]

    config_overrides.update(
        {
            "num_epochs": merged["num_epochs"],
            "train_batch_size": merged["train_batch_size"],
            "eval_batch_size": merged["eval_batch_size"],
            "grad_accum_steps": merged["grad_accum_steps"],
            "learning_rate": merged["learning_rate"],
            "weight_decay": merged["weight_decay"],
            "warmup_ratio": merged["warmup_ratio"],
            "warmup_steps": merged["warmup_steps"],
            "prompt_language": merged["prompt_language"],
            "enable_per_sample_language_prompt": merged["enable_per_sample_language_prompt"],
            "language_dropout_fraction": merged["language_dropout_fraction"],
            "language_dropout_seed": merged["language_dropout_seed"],
            "enable_mlc_language_name_mapping": merged["enable_mlc_language_name_mapping"],
            "eval_decoding_language_mode": merged["eval_decoding_language_mode"],
            "eval_scoring_mode": merged["eval_scoring_mode"],
            "early_stopping_patience": merged["early_stopping_patience"],
            "use_vad_filtering": merged["use_vad"],
            "mlc_cache_name": merged["mlc_cache_name"],
            "mlc_train_dev_cache_names": list(merged.get("mlc_train_dev_cache_names") or []),
            "mlc_test_cache_name": merged["mlc_test_cache_name"],
            "adaptive_sampling_enabled": merged["adaptive_sampling_enabled"],
            "adaptive_sampling_languages": list(merged.get("adaptive_sampling_languages") or []),
            "adaptive_sampling_multiplier": merged["adaptive_sampling_multiplier"],
            "adaptive_sampling_stage1_fraction": merged["adaptive_sampling_stage1_fraction"],
            "adaptive_sampling_seed": merged["adaptive_sampling_seed"],
        }
    )

    path_fields = {"processed_cv_cache_dir", "openslr_cache_dir", "mlc_cache_dir"}
    for k in path_fields:
        if k in config_overrides and config_overrides[k] is not None:
            config_overrides[k] = Path(config_overrides[k])

    return Config(**config_overrides)


def _print_experiment_summary(
    *,
    merged: Dict,
    config: Config,
    output_dir: Path,
    run_train: bool,
    run_eval: bool,
    train_set_name: Optional[str],
    valid_set_name: Optional[str],
    eval_set_name: Optional[str],
    train_ds: Optional[Dataset],
    valid_ds: Optional[Dataset],
    eval_ds: Optional[Dataset],
) -> None:
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()}

    print("\n=== Experiment Configuration ===")
    print(f"  source: {merged['source']}")
    print(f"  language: {merged.get('language')}")
    print(f"  model_mode: {merged['model_mode']}")
    print(f"  adapter_path: {merged.get('adapter_path')}")
    print(f"  run_train: {run_train}")
    print(f"  run_eval: {run_eval}")
    print(f"  output_dir: {output_dir}")
    print(f"  config_json: {merged.get('config_json')}")
    print("  resolved_config:")
    for key in sorted(cfg.keys()):
        print(f"    - {key}: {cfg[key]}")

    print("\n=== Data Splits Used ===")
    if run_train:
        print(f"  train_set: {train_set_name} (n={len(train_ds):,})" if train_ds is not None else f"  train_set: {train_set_name} (missing)")
        print(f"  valid_set: {valid_set_name} (n={len(valid_ds):,})" if valid_ds is not None else f"  valid_set: {valid_set_name} (missing)")
    else:
        print("  train_set: not used")
        print("  valid_set: not used")

    if run_eval:
        print(f"  evaluation_set: {eval_set_name} (n={len(eval_ds):,})" if eval_ds is not None else f"  evaluation_set: {eval_set_name} (missing)")
    else:
        print("  evaluation_set: not used")
    print("===============================\n")


class VoxtralCanonicalCollator:
    """Canonical collator: builds prompt + transcription labels for Voxtral."""

    def __init__(
        self,
        processor: VoxtralProcessor,
        model_id: str,
        language: str = "en",
        text_key: str = "text",
        enable_per_sample_language_prompt: bool = False,
        language_dropout_fraction: float = 0.0,
        language_dropout_seed: int = 42,
        use_name_mapping: bool = True,
        language_field_candidates: Optional[List[str]] = None,
    ):
        self.processor = processor
        self.model_id = model_id
        self.language = language
        self.text_key = text_key
        self.enable_per_sample_language_prompt = bool(enable_per_sample_language_prompt)
        self.language_dropout_fraction = float(language_dropout_fraction)
        self.use_name_mapping = bool(use_name_mapping)
        self.language_field_candidates = tuple(language_field_candidates or LANGUAGE_FIELD_CANDIDATES_DEFAULT)
        self._rng = np.random.default_rng(int(language_dropout_seed))

        if not (0.0 <= self.language_dropout_fraction <= 1.0):
            raise ValueError("language_dropout_fraction must be in [0, 1].")

    def _extract_feature_language(self, feature: dict) -> Optional[str]:
        for key in self.language_field_candidates:
            if key in feature and feature[key] is not None:
                return _map_language_for_model(feature[key], use_name_mapping=self.use_name_mapping)
        return None

    @staticmethod
    def _pad_and_cat_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Pad non-batch dimensions to max size and concatenate on batch dim.

        `apply_transcription_request` can return tensors with variable prompt lengths
        (e.g., language-conditioned prefixes), so raw torch.cat can fail.
        """
        if not tensors:
            raise ValueError("Expected at least one tensor to concatenate.")
        if len(tensors) == 1:
            return tensors[0]

        rank = tensors[0].dim()
        if any(t.dim() != rank for t in tensors):
            raise ValueError("All tensors must have the same rank for padded concatenation.")

        max_sizes = [0] * rank
        for d in range(rank):
            max_sizes[d] = max(t.size(d) for t in tensors)

        padded = []
        for t in tensors:
            pad = []
            # torch.nn.functional.pad consumes pad sizes from last dim to first dim.
            for d in range(rank - 1, 0, -1):
                diff = max_sizes[d] - t.size(d)
                pad.extend([0, diff])
            if any(pad):
                t = F.pad(t, pad, value=0)
            padded.append(t)

        return torch.cat(padded, dim=0)

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [f[self.text_key] for f in features]
        audios = [f["audio"]["array"] for f in features]

        if self.enable_per_sample_language_prompt:
            prompts = []
            for f, audio in zip(features, audios):
                sample_lang = self._extract_feature_language(f)
                if sample_lang is None:
                    sample_lang = _map_language_for_model(self.language, use_name_mapping=self.use_name_mapping)

                if self.language_dropout_fraction > 0.0 and self._rng.random() < self.language_dropout_fraction:
                    sample_lang = None

                req_kwargs = {
                    "model_id": self.model_id,
                    "audio": [audio],
                    "format": ["WAV"],
                    "return_tensors": "pt",
                }
                if sample_lang is not None:
                    req_kwargs["language"] = sample_lang
                prompts.append(self.processor.apply_transcription_request(**req_kwargs))

            prompt_ids = self._pad_and_cat_tensors([p["input_ids"] for p in prompts])
            prompt_attn = self._pad_and_cat_tensors([p["attention_mask"] for p in prompts])
            passthrough = {
                k: self._pad_and_cat_tensors([p[k] for p in prompts])
                for k in prompts[0].keys()
                if k not in ("input_ids", "attention_mask")
            }
        else:
            prompt = self.processor.apply_transcription_request(
                language=_map_language_for_model(self.language, use_name_mapping=self.use_name_mapping),
                model_id=self.model_id,
                audio=audios,
                format=["WAV"] * len(audios),
                return_tensors="pt",
            )
            prompt_ids = prompt["input_ids"]
            prompt_attn = prompt["attention_mask"]
            passthrough = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}

        tok = self.processor.tokenizer
        text_tok = tok(
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=256,
            return_tensors=None,
        )

        input_ids, attention_mask, labels = [], [], []
        for i, t_ids in enumerate(text_tok["input_ids"]):
            p_ids = prompt_ids[i].tolist()
            p_att = prompt_attn[i].tolist()

            ids = p_ids + t_ids + [tok.eos_token_id]
            attn = p_att + [1] * (len(t_ids) + 1)
            lab = [-100] * len(p_ids) + t_ids + [tok.eos_token_id]

            input_ids.append(ids)
            attention_mask.append(attn)
            labels.append(lab)

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        max_len = max(len(x) for x in input_ids)

        def pad_to(x, fill):
            return x + [fill] * (max_len - len(x))

        batch = {
            "input_ids": torch.tensor([pad_to(x, pad_id) for x in input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([pad_to(x, 0) for x in attention_mask], dtype=torch.long),
            "labels": torch.tensor([pad_to(x, -100) for x in labels], dtype=torch.long),
        }
        for k, v in passthrough.items():
            batch[k] = v
        return batch


def create_vad_filtering_preprocessor(
    processor: VoxtralProcessor,
    use_vad: bool,
    vad_threshold: float,
    min_speech_duration_ms: int,
    speech_pad_ms: int,
    min_retained_ratio: float,
    max_length: int,
    sampling_rate: int,
) -> Callable[[dict], dict]:
    """Notebook-style VAD wrapper; for this router it returns canonical audio/text.

    If VAD tooling is unavailable, it gracefully falls back to keeping the full sample.
    """

    vad_fn = None
    if use_vad:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            model = load_silero_vad()

            def _silero(audio_np: np.ndarray) -> Optional[np.ndarray]:
                ts = get_speech_timestamps(
                    audio_np,
                    model,
                    threshold=vad_threshold,
                    min_speech_duration_ms=min_speech_duration_ms,
                    speech_pad_ms=speech_pad_ms,
                    return_seconds=False,
                    sampling_rate=sampling_rate,
                )
                if not ts:
                    return None
                chunks = [audio_np[t["start"] : t["end"]] for t in ts if t["end"] > t["start"]]
                if not chunks:
                    return None
                cat = np.concatenate(chunks)
                if len(cat) < int(len(audio_np) * min_retained_ratio):
                    return None
                return cat

            vad_fn = _silero
        except Exception:
            vad_fn = None

    def _preprocess(batch: dict) -> dict:
        out_audio: List[dict] = []
        out_text: List[str] = []
        out_language: List[Optional[str]] = []

        audios = batch["audio"]
        texts = batch.get("sentence", batch.get("text", []))
        langs = batch.get("language", None)
        if not isinstance(texts, list):
            texts = [texts] * len(audios)
        if langs is not None and not isinstance(langs, list):
            langs = [langs] * len(audios)

        for idx, (a, t) in enumerate(zip(audios, texts)):
            arr = np.asarray(a["array"], dtype=np.float32)
            if use_vad and vad_fn is not None:
                kept = vad_fn(arr)
                if kept is not None:
                    arr = kept
            if max_length and len(arr) > max_length * sampling_rate:
                arr = arr[: max_length * sampling_rate]
            out_audio.append({"array": arr, "sampling_rate": sampling_rate})
            out_text.append(str(t))
            if langs is not None:
                out_language.append(None if langs[idx] is None else str(langs[idx]))

        out = {"audio": out_audio, "text": out_text}
        if langs is not None:
            out["language"] = out_language
        return out

    return _preprocess


def build_voxtral_preprocessor(config: Config, processor: VoxtralProcessor) -> Callable[[dict], dict]:
    return create_vad_filtering_preprocessor(
        processor=processor,
        use_vad=config.use_vad_filtering,
        vad_threshold=config.vad_threshold,
        min_speech_duration_ms=config.vad_min_speech_ms,
        speech_pad_ms=config.vad_speech_pad_ms,
        min_retained_ratio=config.vad_min_retained_ratio,
        max_length=config.vad_max_length,
        sampling_rate=config.sample_rate,
    )


def load_voxtral_base_model(config: Config) -> VoxtralForConditionalGeneration:
    dtype = torch.bfloat16 if config.use_bf16 else (torch.float16 if config.use_fp16 else torch.float32)
    # device_map='auto' breaks distributed training launched via torchrun/accelerate.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    distributed = world_size > 1 or local_rank >= 0

    load_kwargs = {"torch_dtype": dtype}
    if not distributed:
        load_kwargs["device_map"] = "auto"

    model = VoxtralForConditionalGeneration.from_pretrained(config.model_id, **load_kwargs)
    if config.use_grad_checkpoint:
        model.gradient_checkpointing_enable()
    return model


def _normalize_indices(indices: List[int], size: int) -> List[int]:
    if size <= 0:
        return []
    idxs = [int(i) for i in indices]
    if idxs and min(idxs) >= 1 and max(idxs) == size and 0 not in idxs:
        idxs = [i - 1 for i in idxs]
    seen = set()
    out = []
    for i in idxs:
        if 0 <= i < size and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _ensure_canonical(ds: Dataset, split_name: str) -> Dataset:
    cols = set(ds.column_names)
    if not {"audio", "text"}.issubset(cols):
        raise RuntimeError(f"Split '{split_name}' is not canonical. Required columns: ['audio','text'], got {ds.column_names}")
    return ds


def recover_vc_splits(config: Config, language: str) -> Dict[str, Tuple[Dataset, List[str]]]:
    root = config.processed_cv_cache_dir / language
    if not root.exists():
        raise FileNotFoundError(f"VC canonical cache not found: {root}")

    out = {}
    split_map = {"train": "train", "valid": "valid", "test": "test"}
    for key, split_dir in split_map.items():
        ds = load_from_disk(str(root / split_dir))
        ds = _ensure_canonical(ds, key)
        refs_path = root / f"{split_dir}_refs.json"
        refs = json.loads(refs_path.read_text(encoding="utf-8")) if refs_path.exists() else list(ds["text"])
        out[key] = (ds, refs)
    return out


def recover_openslr_splits(config: Config) -> Dict[str, Tuple[Dataset, List[str]]]:
    root = config.openslr_cache_dir
    if not root.exists():
        raise FileNotFoundError(f"OpenSLR canonical cache root not found: {root}")

    candidates = sorted([p for p in root.glob("pipe=voxtral__*") if p.is_dir()], reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No canonical Voxtral OpenSLR cache found under: {root}")

    ds = load_from_disk(str(candidates[0]))
    ds = _ensure_canonical(ds, "openslr_test")
    return {"test": (ds, list(ds["text"]))}


def recover_mlc_splits(config: Config) -> Dict[str, Tuple[Dataset, List[str]]]:
    root = config.mlc_cache_dir
    if not root.exists():
        raise FileNotFoundError(f"MLC cache root not found: {root}")

    dirs = sorted([p for p in root.glob("mlc_slm_*") if p.is_dir()], reverse=True)
    if not dirs:
        raise FileNotFoundError(f"No mlc_slm_* cache found under: {root}")

    def _resolve_cache_dir(name: str) -> Path:
        p = root / name
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Requested MLC cache dir not found under {root}: {name}")
        return p

    if config.mlc_train_dev_cache_names:
        train_dev_dirs = [_resolve_cache_dir(n) for n in config.mlc_train_dev_cache_names]
    elif config.mlc_cache_name:
        train_dev_dirs = [_resolve_cache_dir(config.mlc_cache_name)]
    else:
        train_dev_dirs = [dirs[0]]

    if config.mlc_test_cache_name:
        test_dir = _resolve_cache_dir(config.mlc_test_cache_name)
    elif config.mlc_cache_name:
        test_dir = _resolve_cache_dir(config.mlc_cache_name)
    else:
        test_dir = train_dev_dirs[0]

    def _load_one_cache(mlc: Path) -> Dict[str, Tuple[Dataset, List[str]]]:
        split_proc = {sn: _ensure_canonical(load_from_disk(str(mlc / sn / "processed")), sn) for sn in ("train", "dev", "test")}
        refs = {sn: list(split_proc[sn]["text"]) for sn in split_proc}

        one = {
            "train": (split_proc["train"], refs["train"]),
            "dev": (split_proc["dev"], refs["dev"]),
            "test": (split_proc["test"], refs["test"]),
        }

        clean_root = mlc / "clean_index_cache"
        if clean_root.exists():
            clean = {}
            for sn in ("train", "dev", "test"):
                jp = clean_root / f"{sn}_clean_indices.json"
                if jp.exists():
                    meta = json.loads(jp.read_text(encoding="utf-8"))
                    keep = _normalize_indices(meta.get("keep_indices", []), len(split_proc[sn]))
                else:
                    keep = list(range(len(split_proc[sn])))
                dsc = split_proc[sn].select(keep)
                rsc = [refs[sn][i] for i in keep]
                clean[sn] = (dsc, rsc)

            one["train_clean"] = clean["train"]
            one["dev_clean"] = clean["dev"]
            one["test_clean"] = clean["test"]

            subset_root = clean_root / "train_subset"
            eval_j = subset_root / "train_eval_subset_indices.json"
            finetune_j = subset_root / "train_finetune_indices.json"
            if eval_j.exists() and finetune_j.exists():
                train_clean_ds, train_clean_refs = clean["train"]
                eval_idx = _normalize_indices(json.loads(eval_j.read_text(encoding="utf-8")).get("indices", []), len(train_clean_ds))
                finetune_idx = _normalize_indices(
                    json.loads(finetune_j.read_text(encoding="utf-8")).get("indices", []),
                    len(train_clean_ds),
                )
                one["train_eval"] = (train_clean_ds.select(eval_idx), [train_clean_refs[i] for i in eval_idx])
                one["train_finetune"] = (train_clean_ds.select(finetune_idx), [train_clean_refs[i] for i in finetune_idx])

        return one

    train_dev_loaded = [_load_one_cache(d) for d in train_dev_dirs]
    test_loaded = _load_one_cache(test_dir)

    def _concat_entries(entries: List[Tuple[Dataset, List[str]]], split_name: str) -> Tuple[Dataset, List[str]]:
        if not entries:
            raise RuntimeError(f"No entries available to build split: {split_name}")
        if len(entries) == 1:
            return entries[0]
        dsets = [e[0] for e in entries]
        refs = []
        for _, r in entries:
            refs.extend(r)
        return _ensure_canonical(concatenate_datasets(dsets), split_name), refs

    out = {
        "train": _concat_entries([x["train"] for x in train_dev_loaded], "train"),
        "dev": _concat_entries([x["dev"] for x in train_dev_loaded], "dev"),
        "test": test_loaded["test"],
    }

    train_clean_entries = [x.get("train_clean", x["train"]) for x in train_dev_loaded]
    dev_clean_entries = [x.get("dev_clean", x["dev"]) for x in train_dev_loaded]
    test_clean_entry = test_loaded.get("test_clean", test_loaded["test"])
    out["train_clean"] = _concat_entries(train_clean_entries, "train_clean")
    out["dev_clean"] = _concat_entries(dev_clean_entries, "dev_clean")
    out["test_clean"] = test_clean_entry

    train_eval_entries = [x["train_eval"] for x in train_dev_loaded if "train_eval" in x]
    if train_eval_entries:
        out["train_eval"] = _concat_entries(train_eval_entries, "train_eval")
    train_finetune_entries = [x["train_finetune"] for x in train_dev_loaded if "train_finetune" in x]
    if train_finetune_entries:
        out["train_finetune"] = _concat_entries(train_finetune_entries, "train_finetune")

    return out


def recover_dataset_pool(config: Config, source: str, language: Optional[str]) -> Dict[str, Tuple[Dataset, List[str]]]:
    source = source.lower()
    if source == "vc":
        if not language:
            raise ValueError("--language is required for source=vc (e.g. sr, mt, da)")
        return recover_vc_splits(config, language=language)
    if source == "openslr":
        return recover_openslr_splits(config)
    if source == "mlc":
        return recover_mlc_splits(config)
    raise ValueError(f"Unsupported source: {source}")


def _lang_to_code_for_policy(language: Optional[str]) -> str:
    if language is None:
        return ""
    s = str(language).strip().lower()
    return MLC_LANGUAGE_NAME_TO_CODE.get(s, s)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_for_wer(
    text,
    *,
    language: Optional[str] = None,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    keep_accents: bool = True,
    replace_zero_between_words: bool = False,
    convert_numbers_to_words: bool = True,
    add_char_spacing_for_cjkt: bool = True,
    num2words_lang: Optional[str] = None,
) -> str:
    t = "" if text is None else str(text)
    lang = _lang_to_code_for_policy(language)

    if lowercase:
        t = t.lower()

    if not keep_accents:
        t = _strip_accents(t)

    if replace_zero_between_words:
        t = re.sub(r"(?<=\b\w)0(?=\w\b)", "o", t)

    if remove_punctuation:
        t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)

    if add_char_spacing_for_cjkt and lang in WER_CHAR_SPACING_LANGS:
        t = " ".join(list(re.sub(r"\s+", "", t)))

    if convert_numbers_to_words and num2words is not None:
        n2w_lang = num2words_lang or NUM2WORDS_LANG_MAP.get(lang)

        def _num_to_words(m: re.Match) -> str:
            tok = m.group(0)
            try:
                if n2w_lang:
                    return num2words(int(tok), lang=n2w_lang)
                return num2words(int(tok))
            except Exception:
                return tok

        t = re.sub(r"\b\d+\b", _num_to_words, t)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def _normalize_list_for_policy(texts: List[str], language: Optional[str], apply_norm: bool) -> List[str]:
    if not apply_norm:
        return ["" if t is None else str(t) for t in texts]
    return [
        normalize_for_wer(
            t,
            language=language,
            lowercase=WER_POLICY["lowercase"],
            remove_punctuation=WER_POLICY["remove_punctuation"],
            keep_accents=True,
            replace_zero_between_words=WER_POLICY["replace_zero_between_words"],
            convert_numbers_to_words=WER_POLICY["convert_numbers_to_words"],
            add_char_spacing_for_cjkt=WER_POLICY["add_char_spacing_for_cjkt"],
        )
        for t in texts
    ]


def score_predictions_with_policy(
    refs: List[str],
    preds: List[str],
    *,
    language: Optional[str] = None,
    sample_languages: Optional[List[Optional[str]]] = None,
    apply_to_refs: Optional[bool] = None,
    apply_to_preds: Optional[bool] = None,
) -> Dict[str, float]:
    if len(refs) != len(preds):
        raise ValueError(f"Prediction/reference mismatch: {len(preds)} vs {len(refs)}")
    if sample_languages is not None and len(sample_languages) != len(refs):
        raise ValueError(f"Language/reference mismatch: {len(sample_languages)} vs {len(refs)}")

    refs_norm = WER_POLICY["apply_to_refs"] if apply_to_refs is None else bool(apply_to_refs)
    preds_norm = WER_POLICY["apply_to_preds"] if apply_to_preds is None else bool(apply_to_preds)

    if sample_languages is None:
        refs_n = _normalize_list_for_policy(refs, language=language, apply_norm=refs_norm)
        preds_n = _normalize_list_for_policy(preds, language=language, apply_norm=preds_norm)
    else:
        refs_n = []
        preds_n = []
        for r, p, lang in zip(refs, preds, sample_languages):
            effective_lang = lang or language
            refs_n.extend(_normalize_list_for_policy([r], language=effective_lang, apply_norm=refs_norm))
            preds_n.extend(_normalize_list_for_policy([p], language=effective_lang, apply_norm=preds_norm))

    return {"wer": float(jiwer.wer(refs_n, preds_n)), "cer": float(jiwer.cer(refs_n, preds_n))}


def _normalize_language_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    return s or None


def _extract_language_labels_from_dataset(
    ds_chunk: Dataset,
    fallback_language: Optional[str],
    language_field_candidates: List[str],
) -> Tuple[List[Optional[str]], bool]:
    for col in language_field_candidates:
        if col in ds_chunk.column_names:
            return [_normalize_language_label(x) for x in ds_chunk[col]], True
    return [_normalize_language_label(fallback_language)] * len(ds_chunk), False


def _compute_per_language_metrics(
    refs: List[str],
    preds: List[str],
    sample_languages_display: List[Optional[str]],
    *,
    use_name_mapping: bool,
    default_language: Optional[str],
    scoring_mode: str,
) -> Dict[str, Dict[str, float]]:
    grouped = {}
    for r, p, lang in zip(refs, preds, sample_languages_display):
        if lang is None:
            continue
        if lang not in grouped:
            grouped[lang] = {"refs": [], "preds": []}
        grouped[lang]["refs"].append(r)
        grouped[lang]["preds"].append(p)

    out: Dict[str, Dict[str, float]] = {}
    for lang in sorted(grouped.keys()):
        refs_l = grouped[lang]["refs"]
        preds_l = grouped[lang]["preds"]
        if not refs_l:
            continue
        lang_code = _map_language_for_model(lang, use_name_mapping=use_name_mapping) or default_language
        if scoring_mode == "legacy":
            metrics = {"wer": float(jiwer.wer(refs_l, preds_l)), "cer": float(jiwer.cer(refs_l, preds_l))}
        elif scoring_mode == "normalization_both_sides":
            metrics = score_predictions_with_policy(
                refs_l,
                preds_l,
                language=lang_code,
                apply_to_refs=True,
                apply_to_preds=True,
            )
        else:
            metrics = score_predictions_with_policy(refs_l, preds_l, language=lang_code)
        out[lang] = {"n": len(refs_l), "wer": metrics["wer"], "cer": metrics["cer"]}
    return out


def _save_predictions_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def _materialize_audio_path(audio_obj, temp_audio_dir: Path, sample_id: str) -> str:
    meta = getattr(audio_obj, "metadata", None)
    sample_rate = int(getattr(meta, "sample_rate", 16000) or 16000)

    if hasattr(audio_obj, "get_all_samples"):
        decoded = audio_obj.get_all_samples()
        data = getattr(decoded, "data", None)
        sample_rate = int(getattr(decoded, "sample_rate", sample_rate) or sample_rate)
        if data is not None:
            arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
            out = temp_audio_dir / f"{sample_id}.wav"
            _write_wav_from_samples(out, arr, sample_rate)
            return str(out)

    if isinstance(audio_obj, dict) and audio_obj.get("array") is not None:
        arr = np.asarray(audio_obj["array"])
        sample_rate = int(audio_obj.get("sampling_rate") or sample_rate)
        out = temp_audio_dir / f"{sample_id}.wav"
        _write_wav_from_samples(out, arr, sample_rate)
        return str(out)

    raise RuntimeError(f"Could not materialize audio for sample_id={sample_id}")


def _build_voxtral_chat_input_features(processor: VoxtralProcessor, chat_audio: List[np.ndarray]) -> torch.Tensor:
    """Build input features for Voxtral from chat audio without using _retrieve_input_features."""
    feature_tensors = []
    for audio_array in chat_audio:
        # Use feature extractor to get raw features
        audio_inputs = processor.feature_extractor(
            audio_array,
            sampling_rate=16000,
            padding=False,
            truncation=False,
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
        if feats.ndim != 2:
            raise RuntimeError(f"Unexpected feature shape from Voxtral feature extractor: {feats.shape}")

        # Convert to tensor without dangerous reshape
        feature_tensors.append(torch.as_tensor(feats).unsqueeze(0))

    # Pad all to same length and stack
    max_len = max(f.shape[2] for f in feature_tensors)
    padded = []
    for feat in feature_tensors:
        if feat.shape[2] < max_len:
            pad_size = max_len - feat.shape[2]
            feat = torch.nn.functional.pad(feat, (0, pad_size))
        padded.append(feat)

    return torch.cat(padded, dim=0)


def canonical_generate_predictions(
    model,
    dataset: Dataset,
    processor: VoxtralProcessor,
    model_id: str,
    language: Optional[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    decoding_language_mode: str = "fixed",
    use_name_mapping: bool = True,
    sample_languages: Optional[List[Optional[str]]] = None,
) -> List[str]:
    model.eval()
    preds: List[str] = []
    warned_autodetect_fallback = False
    temp_audio_root = Path(tempfile.mkdtemp(prefix="voxtral_train_autodetect_")) if decoding_language_mode == "autodetect" else None
    try:
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            batch = dataset.select(range(start, end))
            audio_objs = list(batch["audio"])
            audios = [x["array"] for x in audio_objs]

            def _decode_subset(sub_audios: List, lang_value: Optional[str]) -> List[str]:
                nonlocal warned_autodetect_fallback
                req_language = lang_value
                if req_language is None:
                    req_language = language or "en"
                    if decoding_language_mode == "autodetect" and not warned_autodetect_fallback:
                        print(
                            "Warning: multilingual autodetect fallback is active. "
                            f"Detected transformers=={transformers.__version__} where "
                            "VoxtralProcessor.apply_transcription_request requires an explicit "
                            "`language` argument in this runtime path. "
                            f"To avoid a runtime failure, decoding is forced with language='{req_language}'. "
                            "This can bias multilingual inference and should not be treated as true "
                            "language autodetection. For fair multilingual analysis, prefer "
                            "decoding_language_mode='oracle' (dataset language labels) or use the "
                            "chat-template autodetect path instead."
                        )
                        warned_autodetect_fallback = True

                req_kwargs = {
                    "model_id": model_id,
                    "language": req_language,
                    "audio": sub_audios,
                    "format": ["WAV"] * len(sub_audios),
                    "return_tensors": "pt",
                }
                req = processor.apply_transcription_request(**req_kwargs)
                req = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in req.items()}
                with torch.no_grad():
                    out = model.generate(**req, max_new_tokens=max_new_tokens)
                return processor.batch_decode(out, skip_special_tokens=True)

            def _decode_autodetect_subset(sub_audio_objs: List, sample_offset: int) -> List[str]:
                if temp_audio_root is None:
                    raise RuntimeError("Internal error: autodetect temp directory was not created.")
                conversations = []
                for idx_local, audio_obj in enumerate(sub_audio_objs):
                    sample_id = f"eval_{sample_offset + idx_local:08d}"
                    audio_path = _materialize_audio_path(audio_obj, temp_audio_root, sample_id)
                    conversations.append(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "audio", "path": audio_path},
                                    {"type": "text", "text": "Transcribe the audio exactly."},
                                ],
                            }
                        ]
                    )

                encoded = processor.tokenizer.apply_chat_template(
                    conversations,
                    return_tensors=None,
                )
                chat_audio = encoded.pop("audio", None)
                if chat_audio is None:
                    raise RuntimeError("Tokenizer chat template did not return audio content for autodetect decoding.")
                inputs = {
                    "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                    "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
                    "input_features": _build_voxtral_chat_input_features(processor, chat_audio),
                }
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                prompt_len = inputs["input_ids"].shape[1]
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
                decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
                return [str(x) for x in decoded]

            if decoding_language_mode == "fixed":
                preds.extend(_decode_subset(audios, _map_language_for_model(language, use_name_mapping=use_name_mapping)))
            elif decoding_language_mode == "autodetect":
                preds.extend(_decode_autodetect_subset(audio_objs, start))
            elif decoding_language_mode == "oracle":
                if sample_languages is None:
                    raise ValueError("decoding_language_mode='oracle' requires explicit sample language labels.")
                local_langs = sample_languages[start:end]
                grouped = {}
                for idx_local, raw_lang in enumerate(local_langs):
                    lang_code = _map_language_for_model(raw_lang, use_name_mapping=use_name_mapping)
                    if lang_code is None:
                        lang_code = _map_language_for_model(language, use_name_mapping=use_name_mapping)
                    grouped.setdefault(lang_code, []).append(idx_local)

                preds_local: List[Optional[str]] = [None] * len(audios)
                for lang_code, idxs in grouped.items():
                    sub_audios = [audios[i] for i in idxs]
                    sub_preds = _decode_subset(sub_audios, lang_code)
                    for i_local, p in zip(idxs, sub_preds):
                        preds_local[i_local] = p

                if any(p is None for p in preds_local):
                    raise RuntimeError("Internal decode error in oracle mode: missing predictions.")
                preds.extend([str(p) for p in preds_local])
            else:
                raise ValueError("decoding_language_mode must be one of: fixed, oracle, autodetect")
    finally:
        if temp_audio_root is not None:
            shutil.rmtree(temp_audio_root, ignore_errors=True)
    return preds


def score_predictions(refs: List[str], preds: List[str]) -> Dict[str, float]:
    if len(refs) != len(preds):
        raise ValueError(f"Prediction/reference mismatch: {len(preds)} vs {len(refs)}")
    return {"wer": float(jiwer.wer(refs, preds)), "cer": float(jiwer.cer(refs, preds))}


def _build_training_args(args_dict: Dict) -> TrainingArguments:
    """Build TrainingArguments across transformers versions.

    Handles key differences like `eval_strategy` vs `evaluation_strategy`
    and drops unsupported kwargs (e.g., `overwrite_output_dir` on older versions).
    """

    sig = inspect.signature(TrainingArguments.__init__)
    valid = set(sig.parameters.keys())
    d = dict(args_dict)

    if "eval_strategy" in d and "eval_strategy" not in valid and "evaluation_strategy" in valid:
        d["evaluation_strategy"] = d.pop("eval_strategy")
    elif "evaluation_strategy" in d and "evaluation_strategy" not in valid and "eval_strategy" in valid:
        d["eval_strategy"] = d.pop("evaluation_strategy")

    filtered = {k: v for k, v in d.items() if k in valid and v is not None}
    return TrainingArguments(**filtered)


def _make_training_args(
    *,
    config: Config,
    output_dir: Path,
    valid_dataset: Optional[Dataset],
    use_tf_tracking: bool,
    max_steps: Optional[int] = None,
) -> TrainingArguments:
    if use_tf_tracking:
        # Newer transformers deprecate TrainingArguments.logging_dir in favor of env var.
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "tensorboard")

    args_dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.num_epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.grad_accum_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio if config.warmup_ratio > 0 else None,
        "warmup_steps": config.warmup_steps if config.warmup_steps > 0 else None,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps if valid_dataset is not None else None,
        "eval_strategy": "steps" if valid_dataset is not None else "no",
        "save_strategy": "steps",
        "load_best_model_at_end": True if valid_dataset is not None else None,
        "metric_for_best_model": "eval_loss" if valid_dataset is not None else None,
        "greater_is_better": False if valid_dataset is not None else None,
        "report_to": "tensorboard" if use_tf_tracking else "none",
        "remove_unused_columns": False,
        "bf16": config.use_bf16,
        "fp16": config.use_fp16,
        "dataloader_num_workers": config.dataloader_num_workers,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": ("cosine" if config.lr_scheduler.lower() != "step" else "constant"),
        "overwrite_output_dir": True,
    }
    if max_steps is not None and max_steps > 0:
        args_dict["max_steps"] = int(max_steps)
    return _build_training_args(args_dict)


def _build_canonical_collator(config: Config, processor: VoxtralProcessor) -> VoxtralCanonicalCollator:
    return VoxtralCanonicalCollator(
        processor=processor,
        model_id=config.model_id,
        language=config.prompt_language,
        text_key="text",
        enable_per_sample_language_prompt=config.enable_per_sample_language_prompt,
        language_dropout_fraction=config.language_dropout_fraction,
        language_dropout_seed=config.language_dropout_seed,
        use_name_mapping=config.enable_mlc_language_name_mapping,
        language_field_candidates=config.language_field_candidates,
    )


def _fit_with_trainer(
    *,
    config: Config,
    processor: VoxtralProcessor,
    model,
    train_dataset: Dataset,
    valid_dataset: Optional[Dataset],
    output_dir: Path,
    use_tf_tracking: bool,
    max_steps: Optional[int] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> Trainer:
    collator = _build_canonical_collator(config=config, processor=processor)
    args = _make_training_args(
        config=config,
        output_dir=output_dir,
        valid_dataset=valid_dataset,
        use_tf_tracking=use_tf_tracking,
        max_steps=max_steps,
    )

    callbacks = []
    if valid_dataset is not None and config.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(config.early_stopping_patience)))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        callbacks=callbacks if callbacks else None,
    )
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    return trainer


def _normalize_label_for_sampling(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    return s or None


def _build_stage1_upsampled_train_dataset(train_dataset: Dataset, config: Config) -> Tuple[Dataset, Dict[str, object]]:
    info: Dict[str, object] = {
        "applied": False,
        "language_column": None,
        "target_count": 0,
        "original_target_count": 0,
        "extra_samples": 0,
    }

    if not config.adaptive_sampling_enabled:
        return train_dataset, info

    targets = {_normalize_label_for_sampling(x) for x in config.adaptive_sampling_languages}
    targets.discard(None)
    if not targets:
        return train_dataset, info

    if config.adaptive_sampling_multiplier <= 1.0:
        return train_dataset, info

    lang_col = None
    for col in config.language_field_candidates:
        if col in train_dataset.column_names:
            lang_col = col
            break
    if lang_col is None:
        return train_dataset, info

    matching_indices: List[int] = []
    for idx, raw_lang in enumerate(train_dataset[lang_col]):
        if _normalize_label_for_sampling(raw_lang) in targets:
            matching_indices.append(idx)

    if not matching_indices:
        info["language_column"] = lang_col
        return train_dataset, info

    original_target_count = len(matching_indices)
    target_count = int(math.ceil(original_target_count * config.adaptive_sampling_multiplier))
    extra_needed = max(0, target_count - original_target_count)
    if extra_needed == 0:
        info["language_column"] = lang_col
        return train_dataset, info

    rng = np.random.default_rng(int(config.adaptive_sampling_seed))
    extra_indices = rng.choice(np.asarray(matching_indices, dtype=np.int64), size=extra_needed, replace=True).tolist()

    all_indices = list(range(len(train_dataset))) + [int(i) for i in extra_indices]
    rng.shuffle(all_indices)

    boosted = train_dataset.select(all_indices)
    info.update(
        {
            "applied": True,
            "language_column": lang_col,
            "target_count": target_count,
            "original_target_count": original_target_count,
            "extra_samples": extra_needed,
        }
    )
    return boosted, info


def _estimate_total_optimizer_steps(dataset_size: int, config: Config) -> int:
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    micro_batch = max(1, int(config.train_batch_size) * world_size)
    batches_per_epoch = max(1, math.ceil(dataset_size / micro_batch))
    optimizer_steps_per_epoch = max(1, math.ceil(batches_per_epoch / max(1, int(config.grad_accum_steps))))
    return max(1, int(math.ceil(float(config.num_epochs) * optimizer_steps_per_epoch)))


def _compute_two_stage_steps(train_dataset: Dataset, config: Config) -> Tuple[int, int]:
    total_steps = _estimate_total_optimizer_steps(len(train_dataset), config)
    frac = float(config.adaptive_sampling_stage1_fraction)
    stage1_steps = int(round(total_steps * frac))
    if total_steps > 1:
        stage1_steps = max(1, min(stage1_steps, total_steps - 1))
    else:
        stage1_steps = 1
    stage2_steps = max(0, total_steps - stage1_steps)
    return stage1_steps, stage2_steps


def _is_distributed_ready() -> bool:
    return bool(torch.distributed.is_available() and torch.distributed.is_initialized())


def _is_main_process() -> bool:
    if _is_distributed_ready():
        return torch.distributed.get_rank() == 0
    return True


def _safe_barrier() -> None:
    if _is_distributed_ready():
        torch.distributed.barrier()


def train_voxtral_lora_canonical(
    config: Config,
    processor: VoxtralProcessor,
    train_dataset: Dataset,
    valid_dataset: Optional[Dataset],
    output_dir: Path,
    lora_cfg: LoraConfig,
    use_tf_tracking: bool = False,
):
    model = load_voxtral_base_model(config)
    if hasattr(model, "audio_tower"):
        for p in model.audio_tower.parameters():
            p.requires_grad = False
    model = get_peft_model(model, lora_cfg)

    trainer = _fit_with_trainer(
        config=config,
        processor=processor,
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        output_dir=output_dir,
        use_tf_tracking=use_tf_tracking,
    )
    return trainer, model


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Voxtral dataset router (VC/MLC/OpenSLR)", argument_default=argparse.SUPPRESS)

    p.add_argument("--config-json", default=argparse.SUPPRESS, help="Path to a JSON config file. CLI args override json values.")

    p.add_argument("--source", choices=["vc", "mlc", "openslr"])
    p.add_argument("--language", help="VC language code (sr|mt|da).")

    p.add_argument("--model-mode", choices=["baseline", "adapter"])
    p.add_argument("--adapter-path", help="Path to existing LoRA adapter for adapter mode inference/finetune.")

    p.add_argument("--do-train", action="store_true")
    p.add_argument("--do-eval", action="store_true", help="Run WER/CER evaluation on --evaluation-set.")
    p.add_argument("--do-test", action="store_true")

    p.add_argument("--train-set", help="Training split name. Required when --do-train is set.")
    p.add_argument("--valid-set", help="Validation split name. Required when --do-train is set.")
    p.add_argument("--evaluation-set", help="Split used for WER/CER evaluation.")
    p.add_argument("--test-set", help="Deprecated alias of --evaluation-set.")

    p.add_argument("--output-dir", help="If set, use this exact output directory.")
    p.add_argument(
        "--resume-from-checkpoint",
        help=(
            "Path to a Trainer checkpoint dir (checkpoint-*). "
            "For two-stage adaptive runs, checkpoints under stage1_adaptive/ or stage2_full_mix/ are stage-aware."
        ),
    )
    p.add_argument("--output-root", help="Base folder for timestamped experiments.")
    p.add_argument("--experiment-name", help="Optional experiment name prefix.")
    p.add_argument("--timestamped-exp-dir", action="store_true", help="Create output dir with timestamp under --output-root.")
    p.add_argument("--tf-tracking", action="store_true", help="Enable TensorBoard tracking (report_to=tensorboard).")
    p.add_argument("--max-new-tokens", type=int)

    p.add_argument("--num-epochs", type=int)
    p.add_argument("--train-batch-size", type=int)
    p.add_argument("--eval-batch-size", type=int)
    p.add_argument("--grad-accum-steps", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--warmup-ratio", type=float, help="Warmup ratio in [0,1]. Ignored when --warmup-steps > 0.")
    p.add_argument("--warmup-steps", type=int, help="Absolute warmup steps. Overrides warmup ratio when > 0.")
    p.add_argument("--prompt-language")
    p.add_argument("--enable-per-sample-language-prompt", action="store_true")
    p.add_argument("--no-enable-per-sample-language-prompt", dest="enable_per_sample_language_prompt", action="store_false")
    p.add_argument("--language-dropout-fraction", type=float)
    p.add_argument("--language-dropout-seed", type=int)
    p.add_argument("--enable-mlc-language-name-mapping", action="store_true")
    p.add_argument("--no-enable-mlc-language-name-mapping", dest="enable_mlc_language_name_mapping", action="store_false")
    p.add_argument("--eval-decoding-language-mode", choices=["fixed", "oracle", "autodetect"])
    p.add_argument("--eval-scoring-mode", choices=["legacy", "normalization", "normalization_both_sides", "notebook_policy"])
    p.add_argument(
        "--save-predictions-file",
        action="store_true",
        help="Save per-sample reference/hypothesis rows to JSONL during do_eval.",
    )
    p.add_argument(
        "--predictions-file-path",
        help="Optional explicit output path for do_eval predictions JSONL.",
    )
    p.add_argument("--early-stopping-patience", type=int, help="Enable early stopping with this patience in eval steps.")
    p.add_argument(
        "--validation-split-ratio",
        type=float,
        help="If --valid-set is not provided, split train set with this ratio as validation (default 0.1).",
    )
    p.add_argument(
        "--validation-split-seed",
        type=int,
        help="Seed for auto train/validation split when --valid-set is omitted (default 42).",
    )

    p.add_argument("--use-vad", action="store_true")
    p.add_argument("--no-use-vad", dest="use_vad", action="store_false")
    p.add_argument("--mlc-cache-name", help="Explicit mlc_slm_* cache folder name under --mlc-cache-dir.")
    p.add_argument(
        "--mlc-train-dev-cache-names",
        nargs="+",
        help="Optional list of mlc_slm_* cache folder names to concatenate for train/dev.",
    )
    p.add_argument(
        "--mlc-test-cache-name",
        help="Optional mlc_slm_* cache folder name to use only for test/test_clean.",
    )
    p.add_argument("--adaptive-sampling-enabled", action="store_true", help="Enable two-stage language upsampling for early training steps.")
    p.add_argument(
        "--adaptive-sampling-languages",
        nargs="+",
        help="Language labels to upsample during stage-1 (match against dataset language column).",
    )
    p.add_argument(
        "--adaptive-sampling-multiplier",
        type=float,
        help="Upsampling multiplier for selected languages during stage-1 (e.g., 1.5).",
    )
    p.add_argument(
        "--adaptive-sampling-stage1-fraction",
        type=float,
        help="Fraction of total optimizer steps allocated to stage-1 upsampled training.",
    )
    p.add_argument("--adaptive-sampling-seed", type=int, help="Random seed for adaptive stage-1 upsampling.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merged, raw_cfg, cfg_values, cfg_path = _merge_cli_with_config(args)

    eval_set_name = merged["evaluation_set"] or merged["test_set"] or "test"
    run_train = bool(merged["do_train"])
    run_eval = bool(merged["do_eval"] or merged["do_test"] or not run_train)

    if run_train and merged["train_set"] is None:
        raise ValueError("--do-train requires --train-set.")

    config = _build_config_from_inputs(merged=merged, cfg_values=cfg_values)

    pool = recover_dataset_pool(config=config, source=merged["source"], language=merged["language"])

    print("Recovered dataset splits:")
    for name, (ds, _) in sorted(pool.items()):
        print(f"  - {name}: n={len(ds):,} cols={ds.column_names}")

    auto_valid_name = None
    auto_valid_generated = False

    train_ds = pool[merged["train_set"]][0] if run_train else None
    valid_ds = None

    if run_train:
        if merged["train_set"] not in pool:
            raise ValueError(f"Train split not found: {merged['train_set']}")

        if merged["valid_set"] is not None:
            if merged["valid_set"] not in pool:
                raise ValueError(f"Validation split not found: {merged['valid_set']}")
            valid_ds = pool[merged["valid_set"]][0]
        else:
            ratio = float(merged["validation_split_ratio"])
            if not (0.0 < ratio < 1.0):
                raise ValueError(f"--validation-split-ratio must be in (0,1), got {ratio}")
            split = train_ds.train_test_split(test_size=ratio, seed=int(merged["validation_split_seed"]), shuffle=True)
            train_ds = split["train"]
            valid_ds = split["test"]
            pct = int(round(100.0 * ratio))
            auto_valid_name = f"{merged['train_set']}__auto_val_{pct}pct"
            auto_valid_generated = True

            print("Auto-created validation split from training dataset:")
            print(f"  - source train split: {merged['train_set']}")
            print(f"  - validation ratio: {ratio:.3f}")
            print(f"  - split seed: {int(merged['validation_split_seed'])}")
            print(f"  - train size: {len(train_ds):,}")
            print(f"  - val size: {len(valid_ds):,}")

    eval_ds, eval_refs = pool[eval_set_name] if run_eval else (None, None)

    effective_valid_set_name = merged["valid_set"] if merged["valid_set"] is not None else auto_valid_name

    if merged["output_dir"]:
        output_dir = Path(merged["output_dir"])
    elif merged["timestamped_exp_dir"]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_prefix = merged["experiment_name"] or f"voxtral_router_{merged['source']}"
        output_dir = Path(merged["output_root"]) / f"{exp_prefix}_{ts}"
    else:
        output_dir = Path(merged["output_root"]) / "voxtral_router_run"

    output_dir.mkdir(parents=True, exist_ok=True)

    _print_experiment_summary(
        merged=merged,
        config=config,
        output_dir=output_dir,
        run_train=run_train,
        run_eval=run_eval,
        train_set_name=merged["train_set"],
        valid_set_name=effective_valid_set_name,
        eval_set_name=eval_set_name,
        train_ds=train_ds,
        valid_ds=valid_ds,
        eval_ds=eval_ds,
    )

    experiment_config = {
        "timestamp": datetime.now().isoformat(),
        "cli_args": vars(args),
        "config_json_path": str(cfg_path) if cfg_path else None,
        "config_json_payload": raw_cfg if raw_cfg else None,
        "merged_cli_options": merged,
        "resolved": {
            "run_train": run_train,
            "run_eval": run_eval,
            "source": merged["source"],
            "language": merged["language"],
            "train_set": merged["train_set"],
            "valid_set": effective_valid_set_name,
            "auto_validation_generated": auto_valid_generated,
            "validation_split_ratio": float(merged["validation_split_ratio"]),
            "validation_split_seed": int(merged["validation_split_seed"]),
            "evaluation_set": eval_set_name,
            "output_dir": str(output_dir),
            "tf_tracking": bool(merged["tf_tracking"]),
            "resume_from_checkpoint": merged.get("resume_from_checkpoint"),
        },
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "recovered_splits": {
            name: {"n": len(ds), "columns": list(ds.column_names)} for name, (ds, _) in sorted(pool.items())
        },
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2),
        encoding="utf-8",
    )
    if cfg_path is not None:
        tracked_cfg = output_dir / "input_config.json"
        tracked_cfg.write_text(json.dumps(raw_cfg, indent=2), encoding="utf-8")

    processor = VoxtralProcessor.from_pretrained(config.model_id)

    active_model = None
    trainer = None

    if run_train:
        if train_ds is None:
            raise ValueError(f"Train split not found: {merged['train_set']}")

        resume_ckpt_path: Optional[Path] = None
        resume_stage_hint = "none"
        if merged.get("resume_from_checkpoint"):
            candidate = Path(str(merged["resume_from_checkpoint"]))
            if not candidate.exists() or not candidate.is_dir():
                raise FileNotFoundError(f"resume_from_checkpoint path is not a directory: {candidate}")
            resume_ckpt_path = candidate.resolve()
            if "stage2_full_mix" in resume_ckpt_path.parts:
                resume_stage_hint = "stage2"
            elif "stage1_adaptive" in resume_ckpt_path.parts:
                resume_stage_hint = "stage1"
            else:
                resume_stage_hint = "generic"
            print(f"Resuming from checkpoint: {resume_ckpt_path} (hint={resume_stage_hint})")

        if merged["model_mode"] == "adapter" and merged["adapter_path"]:
            base = load_voxtral_base_model(config)
            active_model = PeftModel.from_pretrained(base, merged["adapter_path"], is_trainable=True)
        else:
            base = load_voxtral_base_model(config)
            if hasattr(base, "audio_tower"):
                for p_ in base.audio_tower.parameters():
                    p_.requires_grad = False
            active_model = get_peft_model(base, build_lora_config())

        if config.warmup_ratio < 0 or config.warmup_ratio > 1:
            raise ValueError("warmup_ratio must be in [0, 1].")
        if config.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0.")

        if config.adaptive_sampling_enabled:
            if not (0.0 < config.adaptive_sampling_stage1_fraction < 1.0):
                raise ValueError("adaptive_sampling_stage1_fraction must be in (0, 1).")
            if config.adaptive_sampling_multiplier <= 1.0:
                raise ValueError("adaptive_sampling_multiplier must be > 1.0 when adaptive sampling is enabled.")

        stage1_ds, stage1_info = _build_stage1_upsampled_train_dataset(train_ds, config)
        use_two_stage = bool(config.adaptive_sampling_enabled and stage1_info.get("applied"))

        if use_two_stage:
            stage1_steps, stage2_steps = _compute_two_stage_steps(train_ds, config)
            print("Adaptive sampling stage-1 enabled:")
            print(f"  - language column: {stage1_info['language_column']}")
            print(f"  - selected language labels: {config.adaptive_sampling_languages}")
            print(f"  - multiplier: {config.adaptive_sampling_multiplier}")
            print(
                "  - target samples: "
                f"{stage1_info['original_target_count']:,} -> {stage1_info['target_count']:,} "
                f"(extra={stage1_info['extra_samples']:,})"
            )
            print(f"  - stage1 train size: {len(stage1_ds):,}")
            print(f"  - stage plan (optimizer steps): stage1={stage1_steps:,}, stage2={stage2_steps:,}")

            run_stage1 = True
            stage1_resume = None
            stage2_resume = None
            if resume_ckpt_path is not None:
                if resume_stage_hint == "stage2":
                    run_stage1 = False
                    stage2_resume = str(resume_ckpt_path)
                    print("Skipping stage-1 because resume checkpoint belongs to stage2_full_mix.")
                else:
                    stage1_resume = str(resume_ckpt_path)

            stage1_dir = output_dir / "stage1_adaptive"
            stage1_dir.mkdir(parents=True, exist_ok=True)
            if run_stage1:
                trainer = _fit_with_trainer(
                    config=config,
                    processor=processor,
                    model=active_model,
                    train_dataset=stage1_ds,
                    valid_dataset=valid_ds,
                    output_dir=stage1_dir,
                    use_tf_tracking=merged["tf_tracking"],
                    max_steps=stage1_steps,
                    resume_from_checkpoint=stage1_resume,
                )

            if stage2_steps > 0:
                stage2_dir = output_dir / "stage2_full_mix"
                stage2_dir.mkdir(parents=True, exist_ok=True)
                trainer = _fit_with_trainer(
                    config=config,
                    processor=processor,
                    model=active_model,
                    train_dataset=train_ds,
                    valid_dataset=valid_ds,
                    output_dir=stage2_dir,
                    use_tf_tracking=merged["tf_tracking"],
                    max_steps=stage2_steps,
                    resume_from_checkpoint=stage2_resume,
                )
        else:
            if config.adaptive_sampling_enabled:
                print(
                    "Adaptive sampling was requested but could not be applied "
                    "(missing language column, no matching labels, or no effective upsampling). "
                    "Falling back to regular single-stage training."
                )
            trainer = _fit_with_trainer(
                config=config,
                processor=processor,
                model=active_model,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                output_dir=output_dir,
                use_tf_tracking=merged["tf_tracking"],
                resume_from_checkpoint=str(resume_ckpt_path) if resume_ckpt_path is not None else None,
            )

        final_model_dir = output_dir / "final_model"
        if _is_main_process():
            if trainer is not None:
                trainer.save_model(str(final_model_dir))
            elif active_model is not None:
                active_model.save_pretrained(str(final_model_dir))

            # Ensure adapter artifacts are present even when trainer-side save hooks are incomplete.
            if active_model is not None and hasattr(active_model, "peft_config"):
                adapter_cfg = final_model_dir / "adapter_config.json"
                adapter_weights = final_model_dir / "adapter_model.safetensors"
                if not adapter_cfg.exists() or not adapter_weights.exists():
                    active_model.save_pretrained(str(final_model_dir))

            processor.save_pretrained(str(final_model_dir))
            print(f"Training complete. Saved model/artifacts to: {final_model_dir}")

        # Synchronize all ranks so eval (or process teardown) only starts after final artifacts are fully written.
        _safe_barrier()

    if run_eval:
        if eval_ds is None or eval_refs is None:
            raise ValueError(f"Evaluation split not found: {eval_set_name}")

        if active_model is None:
            base = load_voxtral_base_model(config)
            if merged["model_mode"] == "adapter":
                if not merged["adapter_path"]:
                    raise ValueError("--adapter-path is required for --model-mode adapter when evaluating without training")
                active_model = PeftModel.from_pretrained(base, merged["adapter_path"], is_trainable=False)
            else:
                active_model = base

        eval_scoring_mode = str(config.eval_scoring_mode).strip().lower()
        if eval_scoring_mode == "notebook_policy":
            eval_scoring_mode = "normalization"
        if eval_scoring_mode not in {"legacy", "normalization", "normalization_both_sides"}:
            raise ValueError("eval_scoring_mode must be one of: legacy, normalization, normalization_both_sides")

        sample_languages_display, has_explicit_language_labels = _extract_language_labels_from_dataset(
            eval_ds,
            merged.get("language") or config.prompt_language,
            config.language_field_candidates,
        )
        sample_languages_code = [
            _map_language_for_model(x, use_name_mapping=config.enable_mlc_language_name_mapping)
            for x in sample_languages_display
        ]

        if config.eval_decoding_language_mode == "oracle" and not has_explicit_language_labels:
            raise ValueError(
                "eval_decoding_language_mode='oracle' requires explicit per-sample language labels in the evaluation dataset. "
                "No supported language column was found."
            )

        preds = canonical_generate_predictions(
            model=active_model,
            dataset=eval_ds,
            processor=processor,
            model_id=config.model_id,
            language=config.prompt_language,
            decoding_language_mode=config.eval_decoding_language_mode,
            use_name_mapping=config.enable_mlc_language_name_mapping,
            batch_size=config.eval_batch_size,
            max_new_tokens=merged["max_new_tokens"],
            sample_languages=sample_languages_display,
        )

        if eval_scoring_mode == "legacy":
            metrics = score_predictions(eval_refs, preds)
        elif eval_scoring_mode == "normalization_both_sides":
            metrics = score_predictions_with_policy(
                eval_refs,
                preds,
                language=_map_language_for_model(
                    merged.get("language") or config.prompt_language,
                    use_name_mapping=config.enable_mlc_language_name_mapping,
                ),
                sample_languages=sample_languages_code,
                apply_to_refs=True,
                apply_to_preds=True,
            )
        else:
            metrics = score_predictions_with_policy(
                eval_refs,
                preds,
                language=_map_language_for_model(
                    merged.get("language") or config.prompt_language,
                    use_name_mapping=config.enable_mlc_language_name_mapping,
                ),
                sample_languages=sample_languages_code,
            )

        by_language = (
            _compute_per_language_metrics(
                eval_refs,
                preds,
                sample_languages_display,
                use_name_mapping=config.enable_mlc_language_name_mapping,
                default_language=_map_language_for_model(
                    merged.get("language") or config.prompt_language,
                    use_name_mapping=config.enable_mlc_language_name_mapping,
                ),
                scoring_mode=eval_scoring_mode,
            )
            if has_explicit_language_labels
            else {}
        )

        result = {
            "source": merged["source"],
            "language": merged["language"],
            "model_mode": merged["model_mode"],
            "evaluation_set": eval_set_name,
            "eval_decoding_language_mode": config.eval_decoding_language_mode,
            "eval_scoring_mode": eval_scoring_mode,
            "enable_mlc_language_name_mapping": config.enable_mlc_language_name_mapping,
            "n": len(eval_refs),
            "wer": metrics["wer"],
            "cer": metrics["cer"],
        }
        if by_language:
            result["by_language"] = by_language
        result["wer_policy"] = WER_POLICY
        out_path = output_dir / "eval_metrics.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        if merged.get("save_predictions_file", False):
            rows: List[Dict] = []
            for ref, hyp, lang_disp, lang_code in zip(
                eval_refs,
                preds,
                sample_languages_display,
                sample_languages_code,
            ):
                rows.append(
                    {
                        "split": eval_set_name,
                        "reference": "" if ref is None else str(ref),
                        "hypothesis": "" if hyp is None else str(hyp),
                        "language": lang_disp,
                        "language_code": lang_code,
                    }
                )
            preds_path = Path(merged["predictions_file_path"]) if merged.get("predictions_file_path") else (output_dir / "predictions.jsonl")
            _save_predictions_jsonl(preds_path, rows)
            result["predictions_file"] = str(preds_path)
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"  Predictions saved to: {preds_path}")

        print("Evaluation complete:")
        print(f"  WER: {metrics['wer']:.2%}")
        print(f"  CER: {metrics['cer']:.2%}")
        if by_language:
            print("  Per-language metrics:")
            for lang, lang_metrics in sorted(by_language.items()):
                print(
                    f"    - {lang}: n={lang_metrics['n']:,} "
                    f"WER={lang_metrics['wer']:.2%} CER={lang_metrics['cer']:.2%}"
                )
        print(f"  Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
