#!/usr/bin/env python3
"""
Audio ablation: run MCQ inference with zeroed input_features on a fine-tuned
Voxtral model.  Measures how much of the model's accuracy comes from audio vs.
question-text / answer-option patterns.

Usage (single GPU):
    python /opt/ASR-merging/asr_merging/eval_zero_audio.py \
        --adapter-path /opt/ASR-merging/experiments/<exp>/final_model \
        --eval-jsonl  data/mlc26_task2/eval_26files_challenge_repr.jsonl \
        --audio-root  data/mlc26_task2 \
        --output-dir  experiments/ablation_zero_audio

The output directory will contain the usual mcq_eval_metrics.json and
mcq_eval_predictions.jsonl files.
"""

import argparse
import sys
import os
from pathlib import Path

# Make the package importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from peft import PeftModel
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor

from asr_merging.voxtral_train_MCQ import (
    evaluate_mcq,
    load_jsonl_audio_mcq,
    _offline_aware_from_pretrained_kwargs,
)
from asr_merging.voxtral_train_router import _resolve_pretrained_source


# ---------------------------------------------------------------------------
# Zero-audio wrapper
# ---------------------------------------------------------------------------

class _ZeroAudioModel:
    """Thin wrapper that zeroes out `input_features` in every generate() call."""

    def __init__(self, model):
        self._model = model

    def generate(self, **kwargs):
        if "input_features" in kwargs and kwargs["input_features"] is not None:
            kwargs["input_features"] = torch.zeros_like(kwargs["input_features"])
        return self._model.generate(**kwargs)

    def __getattr__(self, name):
        # Delegate everything else (e.g. .device, .config, .eval()) to the real model
        return getattr(self._model, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="MCQ eval with zeroed audio features")
    p.add_argument("--model-id", default="mistralai/Voxtral-Mini-3B-2507")
    p.add_argument("--adapter-path", required=True,
                   help="Path to the fine-tuned adapter / final_model directory")
    p.add_argument("--eval-jsonl", required=True)
    p.add_argument("--audio-root", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--audio-cache-size", type=int, default=0,
                   help="Set to 0: no point caching since features will be zeroed anyway")
    p.add_argument("--max-eval-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load processor ──────────────────────────────────────────────────────
    pretrained_source = _resolve_pretrained_source(args.model_id)
    print(f"[zero_audio] Loading processor from {pretrained_source}")
    processor = VoxtralProcessor.from_pretrained(pretrained_source)

    # ── Load base model ─────────────────────────────────────────────────────
    print(f"[zero_audio] Loading base model from {pretrained_source}")
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    model_load_kwargs = _offline_aware_from_pretrained_kwargs(
        {"torch_dtype": torch.float16, "device_map": device_map}
    )
    base_model = VoxtralForConditionalGeneration.from_pretrained(
        pretrained_source, **model_load_kwargs
    )

    # ── Load fine-tuned adapter ─────────────────────────────────────────────
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    print(f"[zero_audio] Loading adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    model.eval()

    # ── Wrap with zero-audio patch ──────────────────────────────────────────
    zeroed_model = _ZeroAudioModel(model)
    print("[zero_audio] Audio features will be ZEROED for all inference calls.")

    # ── Load eval samples ───────────────────────────────────────────────────
    print(f"[zero_audio] Loading eval JSONL: {args.eval_jsonl}")
    task_data = load_jsonl_audio_mcq(
        jsonl_path=args.eval_jsonl,
        audio_root=args.audio_root,
        max_questions_per_audio=0,
        max_samples=args.max_eval_samples,
        seed=args.seed,
    )
    print(f"[zero_audio] {len(task_data.samples)} eval samples loaded.")

    # ── Run eval ────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_mcq(
        model=zeroed_model,
        processor=processor,
        samples=task_data.samples,
        max_new_tokens=args.max_new_tokens,
        output_dir=output_dir,
        audio_cache_size=args.audio_cache_size,
    )

    acc = summary.get("accuracy", 0.0)
    n = summary.get("n_total", 0)
    print(f"\n[zero_audio] RESULT: accuracy={acc:.4f} ({acc*100:.1f}%)  n={n}")
    print(f"[zero_audio] Metrics:     {output_dir / 'mcq_eval_metrics.json'}")
    print(f"[zero_audio] Predictions: {output_dir / 'mcq_eval_predictions.jsonl'}")
    print("\nInterpretation guide:")
    print(f"  ~25%  → model is guessing (audio was the only signal)")
    print(f"  ~50%  → model uses question/option text cues only")
    print(f"  ~90%  → model was largely ignoring audio (same as normal eval)")
    normal_acc = 0.923  # v2_mixed normal accuracy
    print(f"  Δ vs normal ({normal_acc*100:.1f}%): {(acc - normal_acc)*100:+.1f}%")


if __name__ == "__main__":
    main()
