#!/usr/bin/env python3
"""Transcribe long-form audio sessions using a fine-tuned Voxtral ASR model.

Voxtral handles up to 30 min per inference call (transcription mode).
Sessions longer than --max-chunk-minutes are split automatically and the
segment transcripts are concatenated.

Usage — single file:
    python -m asr_merging.transcribe_sessions \
        --audio-path data/mlc26_task2/mlc-slm-2nd-dev/Spanish/5094_002_phone.wav \
        --checkpoint-path experiments/mlc25_mlc26_continue_4gpu_fixed_bothsides_20260530_120641/final_model \
        --output-dir transcripts/test_spanish \
        --language es

Usage — full directory (all 150 sessions):
    python -m asr_merging.transcribe_sessions \
        --audio-dir data/mlc26_task2/mlc-slm-2nd-dev \
        --checkpoint-path experiments/mlc25_mlc26_continue_4gpu_fixed_bothsides_20260530_120641/final_model \
        --output-dir transcripts/mlc26_dev \
        --skip-existing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import VoxtralProcessor

from asr_merging.voxtral_train_router import (
    Config,
    _build_voxtral_chat_input_features,
    _resolve_pretrained_source,
    load_voxtral_base_model,
)


SAMPLE_RATE = 16000
# Voxtral's published transcription limit is 30 min; use 29 min as safe margin.
DEFAULT_MAX_CHUNK_SEC = 29 * 60


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio_mono_16k(path: str) -> Tuple[np.ndarray, float]:
    """Load audio as mono float32 at 16 kHz. Returns (array, duration_seconds)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            import scipy.signal as sig
            n_out = int(round(len(audio) * SAMPLE_RATE / sr))
            audio = sig.resample(audio, n_out).astype(np.float32)
    duration = len(audio) / SAMPLE_RATE
    return audio, duration


def _split_into_chunks(audio: np.ndarray, max_sec: float) -> List[np.ndarray]:
    """Split a 1-D float32 audio array into non-overlapping ≤max_sec segments."""
    max_samples = int(max_sec * SAMPLE_RATE)
    return [audio[i: i + max_samples] for i in range(0, len(audio), max_samples)]


def _split_into_chunks_overlapping(
    audio: np.ndarray, max_sec: float, overlap_sec: float
) -> List[Tuple[np.ndarray, float]]:
    """Split audio into overlapping windows.

    Returns a list of (chunk_array, start_sec) tuples.  Consecutive chunks
    share *overlap_sec* seconds of audio at their boundary so the model has
    audio context that prevents hallucination loops at the start of each chunk.
    """
    max_samples = int(max_sec * SAMPLE_RATE)
    overlap_samples = int(overlap_sec * SAMPLE_RATE)
    step = max_samples - overlap_samples
    if step <= 0:
        raise ValueError(
            f"overlap_sec ({overlap_sec}s) must be less than max_sec ({max_sec}s)"
        )
    chunks: List[Tuple[np.ndarray, float]] = []
    start = 0
    while start < len(audio):
        end = min(start + max_samples, len(audio))
        chunks.append((audio[start:end], start / SAMPLE_RATE))
        if end >= len(audio):
            break
        start += step
    return chunks


def _merge_overlapping_transcripts(texts: List[str], overlap_sec: float) -> str:
    """Concatenate chunk transcripts, deduplicating word-level overlaps.

    For each consecutive chunk pair we search for the longest suffix of the
    previous chunk's transcript that matches a prefix of the next chunk's
    transcript (up to a window sized by *overlap_sec*).  The matched prefix is
    dropped from the next chunk before appending.

    At ~2 words/second a window of ``ceil(overlap_sec * 3)`` words gives
    enough slack to tolerate minor transcription variability at boundaries.
    """
    import math

    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0].strip()

    window = max(4, math.ceil(overlap_sec * 3))  # words to search

    result_words = texts[0].split()
    for i in range(1, len(texts)):
        curr_words = texts[i].split()
        # Find longest suffix of result_words that equals a prefix of curr_words
        best = 0
        for n in range(min(window, len(result_words), len(curr_words)), 0, -1):
            if result_words[-n:] == curr_words[:n]:
                best = n
                break
        result_words.extend(curr_words[best:])
    return " ".join(result_words)


def _truncate_chunk_if_looped(text: str) -> str:
    """Truncate *text* at the onset of a hallucination loop inside a single chunk.

    Uses a sliding-window density check: if any 5-gram appears more than 5 times
    within any 150-word window the chunk is truncated just before the loop onset
    (second occurrence of the offending phrase in the full text).

    A naturally repeated phrase (e.g. "how about you") appears at most once or
    twice per window and is never flagged.  A real loop fills the window.

    Returns the truncated text (may be empty string if the loop starts at word 0).
    """
    words = text.split()
    if len(words) < 6:
        return text
    from collections import Counter as _Counter
    _NGRAM, _WIN, _MIN = 5, 150, 5
    _step = max(1, _WIN // 3)
    for _ws in range(0, max(1, len(words) - _WIN + 1), _step):
        _ww = words[_ws: _ws + _WIN]
        _ng = [" ".join(_ww[_i:_i+_NGRAM]) for _i in range(len(_ww) - _NGRAM)]
        _cnt = _Counter(_ng)
        if not _cnt:
            continue
        _top, _c = _cnt.most_common(1)[0]
        if _c > _MIN:
            # Locate the second occurrence in the full word list (= loop onset).
            _tp = _top.split()
            _n = len(_tp)
            _seen = False
            _loop_start = _ws  # fallback to window start
            for _i in range(len(words) - _n + 1):
                if words[_i:_i+_n] == _tp:
                    if _seen:
                        _loop_start = _i
                        break
                    _seen = True
            return " ".join(words[:_loop_start])
    return text


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _transcribe_chunk(
    chunk: np.ndarray,
    model,
    processor: VoxtralProcessor,
    model_id: str,
    language: Optional[str],
    max_new_tokens: int,
) -> str:
    """Transcribe a single ≤30 min audio chunk using apply_transcription_request.

    Matches the exact inference path used in ASR training/eval (voxtral_train_router.py)
    so the prompt format is consistent with the fine-tuning distribution.
    language='en' is recommended for all audio (verified better WER on multilingual).
    """
    req = processor.apply_transcription_request(
        language=language or "en",
        model_id=model_id,
        audio=[chunk],
        format=["WAV"],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    req = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in req.items()}

    with torch.no_grad():
        out = model.generate(**req, max_new_tokens=max_new_tokens)

    # The prompt tokens are all special tokens (audio + transcript markers),
    # so skip_special_tokens=True strips them, leaving only the generated text.
    texts = processor.batch_decode(out, skip_special_tokens=True)
    return (texts[0] if texts else "").strip()


def _transcribe_chunk_chat(
    chunk: np.ndarray,
    model,
    processor: VoxtralProcessor,
    max_new_tokens: int,
) -> str:
    """Transcribe using the chat-template path — produces punctuated output.

    Saves the chunk to a temp WAV file and passes the path to apply_chat_template,
    exactly as decoding_language_mode='autodetect' does in voxtral_train_router.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="voxtral_transcribe_"))
    try:
        tmp_wav = tmp_dir / "chunk.wav"
        sf.write(str(tmp_wav), chunk, SAMPLE_RATE)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": str(tmp_wav)},
                    {"type": "text", "text": "Transcribe the audio exactly."},
                ],
            }
        ]
        encoded = processor.tokenizer.apply_chat_template(
            [conversation],
            return_tensors=None,
        )
        chat_audio = encoded.pop("audio", None)
        if chat_audio is None:
            raise RuntimeError("apply_chat_template returned no audio content.")
        inputs = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "input_features": _build_voxtral_chat_input_features(processor, chat_audio, SAMPLE_RATE),
        }
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        return (decoded[0] if decoded else "").strip()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def transcribe_file(
    audio_path: str,
    model,
    processor: VoxtralProcessor,
    model_id: str,
    language: Optional[str],
    max_new_tokens: int,
    max_chunk_sec: float = DEFAULT_MAX_CHUNK_SEC,
    overlap_sec: float = 0.0,
    mode: str = "transcription",
) -> dict:
    """Transcribe one audio file; split if duration > max_chunk_sec.

    When *overlap_sec* > 0 consecutive chunks share that many seconds of audio
    at their boundary (reduces hallucination loops on difficult audio) and the
    overlapping text is deduplicated during merging.
    """
    t0 = time.time()
    audio, duration_sec = _load_audio_mono_16k(audio_path)
    basename = os.path.basename(audio_path)

    use_overlap = overlap_sec > 0.0 and duration_sec > max_chunk_sec

    if use_overlap:
        chunk_tuples = _split_into_chunks_overlapping(audio, max_chunk_sec, overlap_sec)
    else:
        raw_chunks = _split_into_chunks(audio, max_chunk_sec) if duration_sec > max_chunk_sec else [audio]
        chunk_tuples = [(c, i * max_chunk_sec) for i, c in enumerate(raw_chunks)]

    n_chunks = len(chunk_tuples)

    if n_chunks > 1:
        print(
            f"  [{basename}] {duration_sec/60:.1f} min → {n_chunks} segments"
            + (f" (overlap={overlap_sec}s)" if use_overlap else "")
        )

    texts: List[str] = []
    for idx, (chunk, start_sec) in enumerate(chunk_tuples):
        chunk_dur = len(chunk) / SAMPLE_RATE
        label = f"chunk {idx+1}/{n_chunks} ({start_sec/60:.1f}-{(start_sec+chunk_dur)/60:.1f} min)"
        print(f"    {label} ...", flush=True)
        if mode == "chat":
            text = _transcribe_chunk_chat(chunk, model, processor, max_new_tokens)
        else:
            text = _transcribe_chunk(chunk, model, processor, model_id, language, max_new_tokens)
        # Per-chunk loop detection: truncate hallucination loops before merging.
        # This prevents loopy garbage (e.g. "mga mga mga..." or "&amp;&amp;...")
        # from corrupting the merged transcript via the overlap-deduplication step.
        truncated = _truncate_chunk_if_looped(text)
        if len(truncated.split()) < len(text.split()):
            print(
                f"    {label} ⚠ loop truncated: {len(text.split())}→{len(truncated.split())} words"
            )
            text = truncated
        texts.append(text)
        print(f"    {label} → {len(text.split())} words: {text[:120]!r}")

    if use_overlap and len(texts) > 1:
        transcript = _merge_overlapping_transcripts(texts, overlap_sec)
    else:
        transcript = " ".join(t for t in texts if t)

    elapsed = time.time() - t0
    rt_factor = duration_sec / elapsed if elapsed > 0 else 0.0

    return {
        "audio_path": audio_path,
        "duration_sec": round(duration_sec, 1),
        "n_chunks": n_chunks,
        "language": language,
        "transcript": transcript,
        "word_count": len(transcript.split()),
        "elapsed_sec": round(elapsed, 1),
        "rt_factor": round(rt_factor, 2),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Transcribe long-form audio with a fine-tuned Voxtral ASR model"
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio-path", metavar="FILE",
                     help="Single audio file to transcribe")
    src.add_argument("--audio-dir", metavar="DIR",
                     help="Recursively transcribe all .wav/.flac files under this directory")
    src.add_argument("--file-list", metavar="TXT",
                     help="Text file with one audio path per line (blank/# lines ignored)")

    p.add_argument("--checkpoint-path", required=True, metavar="DIR",
                   help="LoRA adapter directory (contains adapter_config.json)")
    p.add_argument("--output-dir", required=True, metavar="DIR",
                   help="Write <stem>.txt and <stem>.json here")
    p.add_argument("--language", default=None,
                   help="ISO 639-1 language code, e.g. 'es', 'fr'. "
                        "Omit for automatic language detection.")
    p.add_argument("--model-id", default="mistralai/Voxtral-Mini-3B-2507")
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="Max generated transcript tokens per chunk "
                        "(default 8192 covers ~37 min at 150 wpm)")
    p.add_argument("--max-chunk-minutes", type=float, default=29.0,
                   help="Split sessions longer than this into segments (default 29 min)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip audio files whose .txt already exists in output-dir")
    p.add_argument("--no-bf16", action="store_true",
                   help="Disable bfloat16 (use float32)")
    p.add_argument("--shard-index", type=int, default=0,
                   help="0-based index of this GPU's shard (default 0)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of parallel shards / GPUs (default 1)")
    p.add_argument("--mode", choices=["transcription", "chat"], default="transcription",
                   help="transcription=apply_transcription_request (no punctuation, default); "
                        "chat=chat-template autodetect (punctuated output)")
    p.add_argument("--chunk-overlap-seconds", type=float, default=0.0,
                   help="Overlap between consecutive chunks in seconds (default 0 = no overlap). "
                        "Use e.g. 2 with --max-chunk-minutes 1 to reduce hallucination loops: "
                        "the model gets audio context from the previous chunk's tail.")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- collect audio files -----------------------------------------------
    if args.audio_path:
        audio_files = [args.audio_path]
    elif hasattr(args, 'file_list') and args.file_list:
        audio_files = [
            l.strip() for l in open(args.file_list)
            if l.strip() and not l.startswith("#")
        ]
    else:
        exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        audio_files = sorted(
            str(p) for p in Path(args.audio_dir).rglob("*")
            if p.suffix.lower() in exts
        )

    # ---- shard for multi-GPU parallel processing ---------------------------
    if args.num_shards > 1:
        audio_files = [f for i, f in enumerate(audio_files)
                       if i % args.num_shards == args.shard_index]
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(audio_files)} files")

    print(f"Audio files to transcribe: {len(audio_files)}")
    if not audio_files:
        print("Nothing to do.")
        return

    # ---- load model --------------------------------------------------------
    use_bf16 = not args.no_bf16
    print(f"\nLoading base model {args.model_id} (bf16={use_bf16}) ...")
    cfg = Config(model_id=args.model_id, use_bf16=use_bf16, use_fp16=False)
    pretrained_source = _resolve_pretrained_source(args.model_id)
    model = load_voxtral_base_model(cfg, model_source=pretrained_source)

    print(f"Applying LoRA adapter from {args.checkpoint_path} ...")
    model = PeftModel.from_pretrained(model, args.checkpoint_path, is_trainable=False)
    model.eval()

    processor = VoxtralProcessor.from_pretrained(pretrained_source)
    device = next(model.parameters()).device
    print(f"Model on {device}. Ready.\n")

    max_chunk_sec = args.max_chunk_minutes * 60
    results: List[dict] = []

    # ---- transcribe --------------------------------------------------------
    for audio_path in tqdm(audio_files, desc="Transcribing", unit="file"):
        stem = Path(audio_path).stem
        txt_out = output_dir / f"{stem}.txt"
        json_out = output_dir / f"{stem}.json"

        if args.skip_existing and txt_out.exists():
            print(f"  skip (exists): {stem}")
            continue

        print(f"\n[{stem}]", flush=True)
        try:
            meta = transcribe_file(
                audio_path=audio_path,
                model=model,
                processor=processor,
                model_id=args.model_id,
                language=args.language,
                max_new_tokens=args.max_new_tokens,
                max_chunk_sec=max_chunk_sec,
                overlap_sec=args.chunk_overlap_seconds,
                mode=args.mode,
            )
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            meta = {"audio_path": audio_path, "error": str(exc)}

        results.append(meta)

        if "transcript" in meta:
            txt_out.write_text(meta["transcript"] + "\n", encoding="utf-8")
            json_out.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"  {stem}: {meta['duration_sec']/60:.1f} min "
                f"→ {meta['word_count']} words  "
                f"({meta['rt_factor']:.1f}× RT)  →  {txt_out}"
            )

    # ---- summary -----------------------------------------------------------
    ok = [r for r in results if "transcript" in r]
    err = [r for r in results if "error" in r]
    print(f"\n{'='*60}")
    print(f"Done: {len(ok)} transcribed, {len(err)} errors")
    if ok:
        total_dur = sum(r["duration_sec"] for r in ok)
        total_words = sum(r["word_count"] for r in ok)
        avg_rt = sum(r["rt_factor"] for r in ok) / len(ok)
        avg_wpm = (total_words / (total_dur / 60)) if total_dur > 0 else 0
        print(f"Total audio : {total_dur/60:.0f} min")
        print(f"Total words : {total_words:,}  ({avg_wpm:.0f} wpm avg)")
        print(f"Avg RT factor: {avg_rt:.1f}×")
    if err:
        print("Errors:")
        for r in err:
            print(f"  {r['audio_path']}: {r['error']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
