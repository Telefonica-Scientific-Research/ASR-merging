#!/usr/bin/env python3
"""Evaluate Voxtral baseline or adapter checkpoints on recovered dataset splits.

Examples:
  python -m asr_merging.voxtral_eval_router \
    --source mlc \
    --splits dev test

  python -m asr_merging.voxtral_eval_router \
    --source mlc \
    --splits dev test \
    --checkpoint-path experiments/mlc_train_eval_29k_20260406_224234

  python -m asr_merging.voxtral_eval_router \
    --source mlc \
    --splits dev test \
    --checkpoint-path experiments/mlc_train_eval_29k_20260406_224234/checkpoint-1000
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import re
import shutil
import soundfile as sf
import tempfile
import time
import unicodedata
import wave
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
import numpy as np
from peft import PeftModel
import torch
from tqdm.auto import tqdm
import transformers
from transformers import VoxtralProcessor

from .voxtral_train_router import (
    Config,
    load_voxtral_base_model,
    recover_dataset_pool,
    _resolve_pretrained_source,
    _force_audio_decode_false,
)

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

LANGUAGE_COLUMN_CANDIDATES = (
    "language",
    "lang",
    "lang_code",
    "language_code",
    "locale",
)


def _lang_to_code_for_policy(language: Optional[str]) -> str:
    if language is None:
        return ""
    s = str(language).strip().lower()
    return MLC_LANGUAGE_NAME_TO_CODE.get(s, s)


def _map_language_for_model(language: Optional[str], *, use_name_mapping: bool) -> Optional[str]:
    if language is None:
        return None
    s = str(language).strip().lower()
    if not s:
        return None
    if use_name_mapping:
        return MLC_LANGUAGE_NAME_TO_CODE.get(s, s)
    return s


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


def _extract_language_labels(ds_chunk, fallback_language: Optional[str]) -> Tuple[List[Optional[str]], bool]:
    for col in LANGUAGE_COLUMN_CANDIDATES:
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
    grouped = collections.defaultdict(lambda: {"refs": [], "preds": []})
    for r, p, lang in zip(refs, preds, sample_languages_display):
        if lang is None:
            continue
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


def _default_cli_options() -> Dict:
    return {
        "config_json": None,
        "source": None,
        "language": None,
        "splits": ["test"],
        "checkpoint_path": None,
        "model_id": "mistralai/Voxtral-Mini-3B-2507",
        "prompt_language": "en",
        "decoding_language_mode": "fixed",
        "scoring_mode": "normalization",
        "enable_mlc_language_name_mapping": True,
        "eval_batch_size": 4,
        "max_new_tokens": 256,
        "max_samples_per_split": None,
        "prepare_cache": False,
        "prepare_cache_only": False,
        "cache_dir": None,
        "cache_shard_size": 50000,
        "force_recache": False,
        "use_prepared_cache": True,
        "use_bf16": True,
        "use_fp16": False,
        "dataloader_num_workers": 0,
        "save_predictions_file": False,
        "predictions_file_path": None,
        "output_dir": "experiments/voxtral_eval",
        "timestamped_output": False,
        "processed_cv_cache_dir": None,
        "openslr_cache_dir": None,
        "mlc_cache_dir": None,
        "mlc_cache_name": None,
        "mlc_train_dev_cache_names": [],
        "mlc_test_cache_name": None,
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
    payload = dict(raw_payload)

    unsupported = sorted(k for k in payload.keys() if k not in defaults)
    if unsupported:
        raise ValueError(f"Unsupported keys in config json: {unsupported}")

    coerced = {}
    for k, v in payload.items():
        coerced[k] = _convert_value_like_default(v, defaults[k])
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

    merged = dict(defaults)
    for k, v in cfg_values.items():
        merged[k] = v
    for k, v in provided_cli.items():
        merged[k] = v

    if merged.get("source") is None:
        raise ValueError("Missing required argument 'source'. Provide --source or set 'source' in config json.")

    return merged, raw_cfg, cfg_values, cfg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Voxtral baseline or adapter checkpoint on one or more splits.",
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument("--config-json", help="Path to JSON config. CLI args override JSON values.")

    parser.add_argument("--source", choices=["vc", "mlc", "openslr"])
    parser.add_argument("--language", help="VC language code when --source vc.")
    parser.add_argument("--splits", nargs="+", help="Split names to evaluate.")

    parser.add_argument(
        "--checkpoint-path",
        help=(
            "Adapter/checkpoint path. You can pass: "
            "(1) an adapter dir with adapter_config.json, "
            "(2) a trainer checkpoint dir, or "
            "(3) an experiment dir containing final_model/ or checkpoint-*/"
        ),
    )

    parser.add_argument("--model-id")
    parser.add_argument("--prompt-language")
    parser.add_argument("--decoding-language-mode", choices=["fixed", "oracle", "autodetect"])
    parser.add_argument("--scoring-mode", choices=["legacy", "normalization", "normalization_both_sides", "notebook_policy"])
    parser.add_argument("--enable-mlc-language-name-mapping", action="store_true")
    parser.add_argument("--no-enable-mlc-language-name-mapping", dest="enable_mlc_language_name_mapping", action="store_false")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        help="Optional cap per split for faster sanity-check runs.",
    )
    parser.add_argument("--prepare-cache", action="store_true", help="Prepare sharded eval cache before evaluation.")
    parser.add_argument("--prepare-cache-only", action="store_true", help="Prepare sharded eval cache and exit.")
    parser.add_argument("--cache-dir", help="Directory for prepared sharded eval cache.")
    parser.add_argument("--cache-shard-size", type=int, help="Samples per saved shard (default 50000).")
    parser.add_argument("--force-recache", action="store_true", help="Delete existing prepared split cache and rebuild.")
    parser.add_argument("--no-use-prepared-cache", dest="use_prepared_cache", action="store_false")

    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--no-use-bf16", dest="use_bf16", action="store_false")
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int)

    parser.add_argument(
        "--save-predictions-file",
        action="store_true",
        help="Save per-sample reference/hypothesis rows to JSONL for inspection.",
    )
    parser.add_argument(
        "--predictions-file-path",
        help="Optional explicit output path for the predictions JSONL file.",
    )

    parser.add_argument("--output-dir")
    parser.add_argument("--timestamped-output", action="store_true")

    parser.add_argument("--processed-cv-cache-dir")
    parser.add_argument("--openslr-cache-dir")
    parser.add_argument("--mlc-cache-dir")
    parser.add_argument("--mlc-cache-name", help="Explicit mlc_slm_* cache folder name under --mlc-cache-dir.")
    parser.add_argument(
        "--mlc-train-dev-cache-names",
        nargs="+",
        help="Optional list of mlc_slm_* cache folder names to concatenate for train/dev.",
    )
    parser.add_argument(
        "--mlc-test-cache-name",
        help="Optional mlc_slm_* cache folder name to use only for test/test_clean.",
    )

    return parser.parse_args()


def _resolve_adapter_path(checkpoint_path: Optional[str]) -> Optional[Path]:
    if not checkpoint_path:
        return None

    cp = Path(checkpoint_path)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {cp}")

    # Direct adapter/checkpoint directory.
    if (cp / "adapter_config.json").exists():
        return cp

    # Experiment root with final_model/.
    final_model = cp / "final_model"
    if (final_model / "adapter_config.json").exists():
        return final_model

    # Experiment root with checkpoint-*/ directories.
    checkpoints = sorted(
        [p for p in cp.glob("checkpoint-*") if p.is_dir() and (p / "adapter_config.json").exists()],
        key=lambda p: p.name,
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        "Could not resolve an adapter directory from checkpoint path. "
        "Expected adapter_config.json in the provided path, final_model/, or checkpoint-*/."
    )


def _build_config(merged: Dict) -> Config:
    cfg = Config(
        model_id=merged["model_id"],
        eval_batch_size=merged["eval_batch_size"],
        prompt_language=merged["prompt_language"],
        use_bf16=merged["use_bf16"],
        use_fp16=merged["use_fp16"],
        dataloader_num_workers=merged["dataloader_num_workers"],
    )

    if merged["processed_cv_cache_dir"]:
        cfg.processed_cv_cache_dir = Path(merged["processed_cv_cache_dir"])
    if merged["openslr_cache_dir"]:
        cfg.openslr_cache_dir = Path(merged["openslr_cache_dir"])
    if merged["mlc_cache_dir"]:
        cfg.mlc_cache_dir = Path(merged["mlc_cache_dir"])
    cfg.mlc_cache_name = merged.get("mlc_cache_name")
    cfg.mlc_train_dev_cache_names = list(merged.get("mlc_train_dev_cache_names") or [])
    cfg.mlc_test_cache_name = merged.get("mlc_test_cache_name")

    return cfg


def _resolve_output_dir(base_dir: str, timestamped: bool) -> Path:
    out = Path(base_dir)
    if timestamped:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = out / f"voxtral_eval_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_predictions_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_eval_cache_dir(merged: Dict) -> Path:
    if merged.get("cache_dir"):
        return Path(merged["cache_dir"])

    source = merged["source"]
    lang_tag = merged.get("language") or "all"
    return Path("data/cache/voxtral_eval_shards") / source / lang_tag


def _prepared_split_dir(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / split_name


def _prepared_manifest_path(cache_dir: Path, split_name: str) -> Path:
    return _prepared_split_dir(cache_dir, split_name) / "manifest.json"


def _has_prepared_split(cache_dir: Path, split_name: str) -> bool:
    return _prepared_manifest_path(cache_dir, split_name).exists()


def _load_prepared_chunks(cache_dir: Path, split_name: str):
    split_dir = _prepared_split_dir(cache_dir, split_name)
    manifest = json.loads((split_dir / "manifest.json").read_text(encoding="utf-8"))

    from datasets import load_from_disk

    chunks = []
    for shard in manifest["shards"]:
        ds = load_from_disk(str(split_dir / shard["dataset_relpath"]))
        refs = json.loads((split_dir / shard["refs_relpath"]).read_text(encoding="utf-8"))
        chunks.append((ds, refs, shard["name"]))
    return chunks, manifest


def _prepare_sharded_split_cache(
    *,
    cache_dir: Path,
    split_name: str,
    ds,
    refs: List[str],
    source: str,
    language: Optional[str],
    shard_size: int,
    force_recache: bool,
) -> None:
    if shard_size <= 0:
        raise ValueError("cache_shard_size must be > 0")

    split_dir = _prepared_split_dir(cache_dir, split_name)
    manifest_path = split_dir / "manifest.json"

    if split_dir.exists() and force_recache:
        shutil.rmtree(split_dir)

    if manifest_path.exists() and not force_recache:
        print(f"Prepared cache already exists for split={split_name}: {manifest_path}")
        return

    split_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = split_dir / "shards"
    refs_dir = split_dir / "refs"
    shards_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    total = len(ds)
    num_shards = (total + shard_size - 1) // shard_size
    shard_entries = []

    print(f"Preparing cache split={split_name} n={total:,} shard_size={shard_size:,} num_shards={num_shards}")
    for idx, start in enumerate(range(0, total, shard_size)):
        end = min(start + shard_size, total)
        shard_name = f"shard_{idx:05d}"
        shard_ds = ds.select(range(start, end))
        shard_refs = refs[start:end]

        shard_rel = Path("shards") / shard_name
        refs_rel = Path("refs") / f"{shard_name}.json"

        shard_ds.save_to_disk(str(split_dir / shard_rel))
        (split_dir / refs_rel).write_text(json.dumps(shard_refs), encoding="utf-8")

        shard_entries.append(
            {
                "name": shard_name,
                "start": start,
                "end": end,
                "n": end - start,
                "dataset_relpath": shard_rel.as_posix(),
                "refs_relpath": refs_rel.as_posix(),
            }
        )

    manifest = {
        "source": source,
        "language": language,
        "split": split_name,
        "n": total,
        "shard_size": shard_size,
        "num_shards": num_shards,
        "created_at": datetime.now().isoformat(),
        "shards": shard_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved prepared cache manifest: {manifest_path}")


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


def _generate_predictions_with_progress(
    *,
    model,
    dataset,
    processor: VoxtralProcessor,
    model_id: str,
    language: Optional[str],
    decoding_language_mode: str,
    sample_languages: Optional[List[Optional[str]]],
    use_name_mapping: bool,
    batch_size: int,
    max_new_tokens: int,
    split_name: str,
) -> List[str]:
    model.eval()
    preds: List[str] = []
    warned_autodetect_fallback = False
    temp_audio_root = Path(tempfile.mkdtemp(prefix="voxtral_eval_autodetect_")) if decoding_language_mode == "autodetect" else None

    total = len(dataset)
    if total == 0:
        return preds

    t0 = time.time()
    pbar = tqdm(range(0, total, batch_size), desc=f"eval:{split_name}", unit="batch")
    try:
        for start in pbar:
            end = min(start + batch_size, total)
            batch = _force_audio_decode_false(dataset.select(range(start, end)), 16000)
            audio_objs = list(batch["audio"])
            audios = []
            for x in audio_objs:
                if x.get("array") is not None:
                    audios.append(np.asarray(x["array"], dtype=np.float32))
                elif x.get("bytes") is not None:
                    arr, _ = sf.read(io.BytesIO(x["bytes"]), dtype="float32", always_2d=False)
                    audios.append(arr)
                elif x.get("path"):
                    arr, _ = sf.read(x["path"], dtype="float32", always_2d=False)
                    audios.append(arr)
                else:
                    raise ValueError(f"Cannot decode audio entry: {list(x.keys()) if hasattr(x, 'keys') else type(x)}")

            batch_preds: List[Optional[str]] = [None] * len(audios)

            def _run_decode(audio_subset: List, lang_value: Optional[str]) -> List[str]:
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
                    "audio": audio_subset,
                    "format": ["WAV"] * len(audio_subset),
                    "return_tensors": "pt",
                }

                req = processor.apply_transcription_request(**req_kwargs)
                req = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in req.items()}

                with torch.no_grad():
                    out = model.generate(**req, max_new_tokens=max_new_tokens)
                return processor.batch_decode(out, skip_special_tokens=True)

            def _run_autodetect_decode(audio_subset_objs: List, sample_offset: int) -> List[str]:
                if temp_audio_root is None:
                    raise RuntimeError("Internal error: autodetect temp directory was not created.")
                conversations = []
                for idx_local, audio_obj in enumerate(audio_subset_objs):
                    sample_id = f"{split_name.replace(':', '_')}_{sample_offset + idx_local:08d}"
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
                if isinstance(encoded, dict):
                    chat_audio = encoded.pop("audio", None)
                else:
                    chat_audio = encoded["audio"] if "audio" in encoded else None
                    if "audio" in encoded:
                        del encoded["audio"]
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
                fixed_lang = _map_language_for_model(language, use_name_mapping=use_name_mapping)
                batch_preds = _run_decode(audios, fixed_lang)
            elif decoding_language_mode == "autodetect":
                try:
                    batch_preds = _run_autodetect_decode(audio_objs, start)
                except Exception as autodetect_err:  # noqa: BLE001
                    if not warned_autodetect_fallback:
                        print(
                            "Warning: chat-template autodetect path failed and is falling back "
                            f"to forced-language decoding. error={type(autodetect_err).__name__}: {autodetect_err}"
                        )
                        warned_autodetect_fallback = True
                    batch_preds = _run_decode(audios, None)
            elif decoding_language_mode == "oracle":
                if sample_languages is None:
                    raise ValueError("decoding_language_mode='oracle' requires explicit sample language labels.")

                local_langs = sample_languages[start:end]
                grouped = collections.defaultdict(list)
                for i_local, lang_raw in enumerate(local_langs):
                    lang_code = _map_language_for_model(lang_raw, use_name_mapping=use_name_mapping)
                    if lang_code is None:
                        lang_code = _map_language_for_model(language, use_name_mapping=use_name_mapping)
                    grouped[lang_code].append(i_local)

                for lang_code, idxs in grouped.items():
                    sub_audios = [audios[i] for i in idxs]
                    sub_preds = _run_decode(sub_audios, lang_code)
                    for i_local, p in zip(idxs, sub_preds):
                        batch_preds[i_local] = p

                if any(p is None for p in batch_preds):
                    raise RuntimeError("Internal decode error: missing predictions in oracle mode.")
            else:
                raise ValueError(f"Unsupported decoding_language_mode: {decoding_language_mode}")

            preds.extend([str(p) for p in batch_preds])

            elapsed = max(1e-6, time.time() - t0)
            done = end
            pbar.set_postfix(samples=done, speed=f"{done / elapsed:.2f} samp/s")
    finally:
        if temp_audio_root is not None:
            shutil.rmtree(temp_audio_root, ignore_errors=True)

    return preds


def main() -> None:
    args = parse_args()
    merged, raw_cfg, _, cfg_path = _merge_cli_with_config(args)

    cfg = _build_config(merged)
    adapter_path = _resolve_adapter_path(merged["checkpoint_path"])
    output_dir = _resolve_output_dir(merged["output_dir"], merged["timestamped_output"])
    cache_dir = _resolve_eval_cache_dir(merged)

    splits = merged["splits"]

    pool = None
    all_cached = merged.get("use_prepared_cache", True) and all(_has_prepared_split(cache_dir, s) for s in splits)
    if all_cached and not merged.get("prepare_cache") and not merged.get("prepare_cache_only"):
        print(f"Using prepared eval cache from: {cache_dir}")
    else:
        pool = recover_dataset_pool(config=cfg, source=merged["source"], language=merged["language"], run_train=False)
        print("Recovered dataset splits:")
        for split_name, (ds, _) in sorted(pool.items()):
            print(f"  - {split_name}: n={len(ds):,} cols={ds.column_names}")

    if merged.get("prepare_cache") or merged.get("prepare_cache_only"):
        if pool is None:
            pool = recover_dataset_pool(config=cfg, source=merged["source"], language=merged["language"], run_train=False)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for split_name in splits:
            if split_name not in pool:
                raise ValueError(
                    f"Requested split '{split_name}' not found for source={merged['source']}. "
                    f"Available: {sorted(pool.keys())}"
                )
            ds, refs = pool[split_name]
            _prepare_sharded_split_cache(
                cache_dir=cache_dir,
                split_name=split_name,
                ds=ds,
                refs=refs,
                source=merged["source"],
                language=merged["language"],
                shard_size=int(merged["cache_shard_size"]),
                force_recache=bool(merged.get("force_recache", False)),
            )
        if merged.get("prepare_cache_only"):
            print(f"Prepared cache only mode complete. Cache root: {cache_dir}")
            return

    model = load_voxtral_base_model(cfg)
    model_mode = "baseline"
    resolved_checkpoint = None
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model_mode = "adapter"
        resolved_checkpoint = str(adapter_path)

    processor = VoxtralProcessor.from_pretrained(_resolve_pretrained_source(cfg.model_id))

    decoding_mode = str(merged.get("decoding_language_mode", "fixed")).strip().lower()
    scoring_mode = str(merged.get("scoring_mode", "normalization")).strip().lower()
    use_name_mapping = bool(merged.get("enable_mlc_language_name_mapping", True))

    # Backward compatibility for older configs/CLI that used notebook_policy.
    if scoring_mode == "notebook_policy":
        scoring_mode = "normalization"

    if decoding_mode not in {"fixed", "oracle", "autodetect"}:
        raise ValueError("decoding_language_mode must be one of: fixed, oracle, autodetect")
    if scoring_mode not in {"legacy", "normalization", "normalization_both_sides"}:
        raise ValueError("scoring_mode must be one of: legacy, normalization, normalization_both_sides")

    split_metrics: Dict[str, Dict[str, float]] = {}
    prediction_rows: List[Dict] = []
    for split_name in splits:
        chunks = []
        if merged.get("use_prepared_cache", True) and _has_prepared_split(cache_dir, split_name):
            cached_chunks, manifest = _load_prepared_chunks(cache_dir, split_name)
            print(
                f"Starting evaluation split={split_name} from prepared cache "
                f"n={manifest['n']:,} shards={manifest['num_shards']} "
                f"batch_size={cfg.eval_batch_size} max_new_tokens={merged['max_new_tokens']}"
            )
            chunks = cached_chunks
        else:
            if pool is None:
                pool = recover_dataset_pool(config=cfg, source=merged["source"], language=merged["language"], run_train=False)
            if split_name not in pool:
                raise ValueError(
                    f"Requested split '{split_name}' not found for source={merged['source']}. "
                    f"Available: {sorted(pool.keys())}"
                )
            ds, refs = pool[split_name]
            chunks = [(ds, refs, split_name)]
            print(
                f"Starting evaluation split={split_name} n={len(refs):,} "
                f"batch_size={cfg.eval_batch_size} max_new_tokens={merged['max_new_tokens']}"
            )

        if merged.get("max_samples_per_split") is not None:
            cap = int(merged["max_samples_per_split"])
            if cap <= 0:
                raise ValueError("--max-samples-per-split must be > 0")

            capped_chunks = []
            kept = 0
            for ds_chunk, refs_chunk, label in chunks:
                if kept >= cap:
                    break
                remain = cap - kept
                take = min(remain, len(ds_chunk))
                if take <= 0:
                    continue
                if take < len(ds_chunk):
                    ds_chunk = ds_chunk.select(range(take))
                    refs_chunk = refs_chunk[:take]
                capped_chunks.append((ds_chunk, refs_chunk, label))
                kept += take
            chunks = capped_chunks
            print(f"Applying max sample cap for {split_name}: {kept:,}")

        all_refs: List[str] = []
        all_preds: List[str] = []
        all_sample_languages_display: List[Optional[str]] = []
        all_sample_languages_code: List[Optional[str]] = []
        has_explicit_language_labels = False
        for ds_chunk, refs_chunk, label in chunks:
            chunk_languages, explicit_langs = _extract_language_labels(ds_chunk, merged.get("language"))
            has_explicit_language_labels = has_explicit_language_labels or explicit_langs

            chunk_languages_code = [_map_language_for_model(x, use_name_mapping=use_name_mapping) for x in chunk_languages]

            if decoding_mode == "oracle" and not explicit_langs:
                raise ValueError(
                    "decoding_language_mode='oracle' requires explicit per-sample language labels in the dataset. "
                    "No language column was found in this chunk."
                )

            preds_chunk = _generate_predictions_with_progress(
                model=model,
                dataset=ds_chunk,
                processor=processor,
                model_id=cfg.model_id,
                language=merged.get("language") or cfg.prompt_language,
                decoding_language_mode=decoding_mode,
                sample_languages=chunk_languages,
                use_name_mapping=use_name_mapping,
                batch_size=cfg.eval_batch_size,
                max_new_tokens=merged["max_new_tokens"],
                split_name=f"{split_name}:{label}",
            )
            all_refs.extend(refs_chunk)
            all_preds.extend(preds_chunk)
            all_sample_languages_display.extend(chunk_languages)
            all_sample_languages_code.extend(chunk_languages_code)

            if merged.get("save_predictions_file", False):
                for r, p, lang_disp, lang_code in zip(refs_chunk, preds_chunk, chunk_languages, chunk_languages_code):
                    prediction_rows.append(
                        {
                            "split": split_name,
                            "reference": "" if r is None else str(r),
                            "hypothesis": "" if p is None else str(p),
                            "language": lang_disp,
                            "language_code": lang_code,
                        }
                    )

        if scoring_mode == "legacy":
            metrics = {"wer": float(jiwer.wer(all_refs, all_preds)), "cer": float(jiwer.cer(all_refs, all_preds))}
        elif scoring_mode == "normalization_both_sides":
            metrics = score_predictions_with_policy(
                all_refs,
                all_preds,
                language=_map_language_for_model(merged.get("language") or cfg.prompt_language, use_name_mapping=use_name_mapping),
                sample_languages=all_sample_languages_code,
                apply_to_refs=True,
                apply_to_preds=True,
            )
        else:
            metrics = score_predictions_with_policy(
                all_refs,
                all_preds,
                language=_map_language_for_model(merged.get("language") or cfg.prompt_language, use_name_mapping=use_name_mapping),
                sample_languages=all_sample_languages_code,
            )

        by_language = (
            _compute_per_language_metrics(
                all_refs,
                all_preds,
                all_sample_languages_display,
                use_name_mapping=use_name_mapping,
                default_language=_map_language_for_model(merged.get("language") or cfg.prompt_language, use_name_mapping=use_name_mapping),
                scoring_mode=scoring_mode,
            )
            if has_explicit_language_labels
            else {}
        )

        split_metrics[split_name] = {
            "n": len(all_refs),
            "wer": metrics["wer"],
            "cer": metrics["cer"],
        }
        if by_language:
            split_metrics[split_name]["by_language"] = by_language

        print(f"\n[{split_name}] n={len(all_refs):,}")
        print(f"  WER: {metrics['wer']:.2%}")
        print(f"  CER: {metrics['cer']:.2%}")
        if by_language:
            print("  Per-language metrics:")
            for lang, lang_metrics in sorted(by_language.items()):
                print(
                    f"    - {lang}: n={lang_metrics['n']:,} "
                    f"WER={lang_metrics['wer']:.2%} CER={lang_metrics['cer']:.2%}"
                )

    payload = {
        "timestamp": datetime.now().isoformat(),
        "source": merged["source"],
        "language": merged["language"],
        "model_id": cfg.model_id,
        "model_mode": model_mode,
        "checkpoint_path_input": merged["checkpoint_path"],
        "checkpoint_path_resolved": resolved_checkpoint,
        "prompt_language": cfg.prompt_language,
        "decoding_language_mode": decoding_mode,
        "scoring_mode": scoring_mode,
        "enable_mlc_language_name_mapping": use_name_mapping,
        "eval_batch_size": cfg.eval_batch_size,
        "max_new_tokens": merged["max_new_tokens"],
        "max_samples_per_split": merged.get("max_samples_per_split"),
        "cache_dir": str(cache_dir),
        "use_prepared_cache": bool(merged.get("use_prepared_cache", True)),
        "prepare_cache": bool(merged.get("prepare_cache", False)),
        "cache_shard_size": int(merged["cache_shard_size"]),
        "wer_policy": WER_POLICY,
        "config_json_path": str(cfg_path) if cfg_path else None,
        "config_json_payload": raw_cfg if raw_cfg else None,
        "merged_cli_options": merged,
        "splits": split_metrics,
    }

    out_json = output_dir / "eval_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved metrics: {out_json}")

    if merged.get("save_predictions_file", False):
        preds_path = Path(merged["predictions_file_path"]) if merged.get("predictions_file_path") else (output_dir / "predictions.jsonl")
        _save_predictions_jsonl(preds_path, prediction_rows)
        print(f"Saved prediction rows: {preds_path} (n={len(prediction_rows):,})")


if __name__ == "__main__":
    main()
