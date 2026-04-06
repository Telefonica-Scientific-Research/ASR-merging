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
import sys
from datetime import datetime
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import jiwer
import numpy as np
import torch
from datasets import Audio, Dataset, load_from_disk
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

    # Prompt/eval language for Voxtral path
    prompt_language: str = "en"


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
        "early_stopping_patience": None,
        "validation_split_ratio": 0.1,
        "validation_split_seed": 42,
        "use_vad": True,
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
            "early_stopping_patience": merged["early_stopping_patience"],
            "use_vad_filtering": merged["use_vad"],
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

    def __init__(self, processor: VoxtralProcessor, model_id: str, language: str = "en", text_key: str = "text"):
        self.processor = processor
        self.model_id = model_id
        self.language = language
        self.text_key = text_key

    def __call__(self, features: List[dict]) -> Dict[str, torch.Tensor]:
        texts = [f[self.text_key] for f in features]
        audios = [f["audio"]["array"] for f in features]

        prompt = self.processor.apply_transcription_request(
            language=self.language,
            model_id=self.model_id,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        passthrough = {k: v for k, v in prompt.items() if k not in ("input_ids", "attention_mask")}

        tok = self.processor.tokenizer
        prompt_ids = prompt["input_ids"]
        prompt_attn = prompt["attention_mask"]
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

        audios = batch["audio"]
        texts = batch.get("sentence", batch.get("text", []))
        if not isinstance(texts, list):
            texts = [texts] * len(audios)

        for a, t in zip(audios, texts):
            arr = np.asarray(a["array"], dtype=np.float32)
            if use_vad and vad_fn is not None:
                kept = vad_fn(arr)
                if kept is not None:
                    arr = kept
            if max_length and len(arr) > max_length * sampling_rate:
                arr = arr[: max_length * sampling_rate]
            out_audio.append({"array": arr, "sampling_rate": sampling_rate})
            out_text.append(str(t))

        return {"audio": out_audio, "text": out_text}

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
    load_kwargs = {"device_map": "auto", "torch_dtype": dtype}
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

    mlc = dirs[0]
    split_proc = {sn: _ensure_canonical(load_from_disk(str(mlc / sn / "processed")), sn) for sn in ("train", "dev", "test")}
    refs = {sn: list(split_proc[sn]["text"]) for sn in split_proc}

    clean_root = mlc / "clean_index_cache"
    out = {
        "train": (split_proc["train"], refs["train"]),
        "dev": (split_proc["dev"], refs["dev"]),
        "test": (split_proc["test"], refs["test"]),
    }

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

        out["train_clean"] = clean["train"]
        out["dev_clean"] = clean["dev"]
        out["test_clean"] = clean["test"]

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
            out["train_eval"] = (train_clean_ds.select(eval_idx), [train_clean_refs[i] for i in eval_idx])
            out["train_finetune"] = (train_clean_ds.select(finetune_idx), [train_clean_refs[i] for i in finetune_idx])

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


def canonical_generate_predictions(
    model,
    dataset: Dataset,
    processor: VoxtralProcessor,
    model_id: str,
    language: str,
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    model.eval()
    preds: List[str] = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        audios = [x["array"] for x in batch["audio"]]
        req = processor.apply_transcription_request(
            language=language,
            model_id=model_id,
            audio=audios,
            format=["WAV"] * len(audios),
            return_tensors="pt",
        )
        req = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in req.items()}
        with torch.no_grad():
            out = model.generate(**req, max_new_tokens=max_new_tokens)
        preds.extend(processor.batch_decode(out, skip_special_tokens=True))
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

    collator = VoxtralCanonicalCollator(
        processor=processor,
        model_id=config.model_id,
        language=config.prompt_language,
        text_key="text",
    )

    args = _build_training_args(
        {
            "output_dir": str(output_dir),
            "num_train_epochs": config.num_epochs,
            "per_device_train_batch_size": config.train_batch_size,
            "per_device_eval_batch_size": config.eval_batch_size,
            "gradient_accumulation_steps": config.grad_accum_steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "logging_steps": config.logging_steps,
            "save_steps": config.save_steps,
            "eval_steps": config.eval_steps if valid_dataset is not None else None,
            "eval_strategy": "steps" if valid_dataset is not None else "no",
            "save_strategy": "steps",
            "load_best_model_at_end": True if valid_dataset is not None else None,
            "metric_for_best_model": "eval_loss" if valid_dataset is not None else None,
            "greater_is_better": False if valid_dataset is not None else None,
            "report_to": "tensorboard" if use_tf_tracking else "none",
            "logging_dir": str(output_dir / "tensorboard") if use_tf_tracking else None,
            "remove_unused_columns": False,
            "bf16": config.use_bf16,
            "fp16": config.use_fp16,
            "dataloader_num_workers": config.dataloader_num_workers,
            "max_grad_norm": config.max_grad_norm,
            "lr_scheduler_type": ("cosine" if config.lr_scheduler.lower() != "step" else "constant"),
            "overwrite_output_dir": True,
        }
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
    trainer.train()
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

        if merged["model_mode"] == "adapter" and merged["adapter_path"]:
            base = load_voxtral_base_model(config)
            active_model = PeftModel.from_pretrained(base, merged["adapter_path"], is_trainable=True)
            # Continue training with Trainer API using canonical collator.
            collator = VoxtralCanonicalCollator(processor=processor, model_id=config.model_id, language=config.prompt_language)
            targs = _build_training_args(
                {
                    "output_dir": str(output_dir),
                    "num_train_epochs": config.num_epochs,
                    "per_device_train_batch_size": config.train_batch_size,
                    "per_device_eval_batch_size": config.eval_batch_size,
                    "gradient_accumulation_steps": config.grad_accum_steps,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "logging_steps": config.logging_steps,
                    "save_steps": config.save_steps,
                    "eval_steps": config.eval_steps if valid_ds is not None else None,
                    "eval_strategy": "steps" if valid_ds is not None else "no",
                    "save_strategy": "steps",
                    "load_best_model_at_end": True if valid_ds is not None else None,
                    "metric_for_best_model": "eval_loss" if valid_ds is not None else None,
                    "greater_is_better": False if valid_ds is not None else None,
                    "report_to": "tensorboard" if merged["tf_tracking"] else "none",
                    "logging_dir": str(output_dir / "tensorboard") if merged["tf_tracking"] else None,
                    "remove_unused_columns": False,
                    "bf16": config.use_bf16,
                    "fp16": config.use_fp16,
                    "dataloader_num_workers": config.dataloader_num_workers,
                    "max_grad_norm": config.max_grad_norm,
                    "overwrite_output_dir": True,
                }
            )
            trainer = Trainer(
                model=active_model,
                args=targs,
                train_dataset=train_ds,
                eval_dataset=valid_ds,
                data_collator=collator,
            )
            trainer.train()
        else:
            lora_cfg = build_lora_config()
            trainer, active_model = train_voxtral_lora_canonical(
                config=config,
                processor=processor,
                train_dataset=train_ds,
                valid_dataset=valid_ds,
                output_dir=output_dir,
                lora_cfg=lora_cfg,
                use_tf_tracking=merged["tf_tracking"],
            )

        if trainer is not None:
            trainer.save_model(str(output_dir / "final_model"))
        elif active_model is not None:
            active_model.save_pretrained(str(output_dir / "final_model"))
        processor.save_pretrained(str(output_dir / "final_model"))
        print(f"Training complete. Saved model/artifacts to: {output_dir / 'final_model'}")

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

        preds = canonical_generate_predictions(
            model=active_model,
            dataset=eval_ds,
            processor=processor,
            model_id=config.model_id,
            language=config.prompt_language,
            batch_size=config.eval_batch_size,
            max_new_tokens=merged["max_new_tokens"],
        )
        metrics = score_predictions(eval_refs, preds)

        result = {
            "source": merged["source"],
            "language": merged["language"],
            "model_mode": merged["model_mode"],
            "evaluation_set": eval_set_name,
            "n": len(eval_refs),
            "wer": metrics["wer"],
            "cer": metrics["cer"],
        }
        out_path = output_dir / "eval_metrics.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print("Evaluation complete:")
        print(f"  WER: {metrics['wer']:.2%}")
        print(f"  CER: {metrics['cer']:.2%}")
        print(f"  Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
