#!/usr/bin/env python3
"""Translate English questions in a training JSONL to a target language using NLLB-200.

The input JSONL is assumed to be already in English (e.g. the en_translated file).
All question_stem and option texts are translated from English to the specified
target language. Quoted spans (verbatim audio speech inside quotes) are preserved
verbatim. Original English text is stored in *_original fields.

Usage:
    python -m asr_merging.translate_from_english_nllb \
        --input   data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl \
        --output  data/mlc26_task2/mlcslm_2nd_dev_qa_en_to_por.jsonl \
        --target-lang por_Latn \
        --model   /gpfs/scratch/ehpc628/models/models--facebook--nllb-200-distilled-1.3B/snapshots/7be3e24664b38ce1cac29b8aeed6911aa0cf0576 \
        --batch-size 64

Supported target-lang values (NLLB BCP-47 codes):
    por_Latn  fra_Latn  spa_Latn  rus_Cyrl  vie_Latn
    tur_Latn  tgl_Latn  deu_Latn  ita_Latn
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Matches quoted verbatim audio spans — preserved unchanged across all languages
QUOTE_PATTERN = re.compile(
    r'\u201e[^\u201d\u0022]{0,300}[\u201d\u0022]'   # „..."
    r'|\u201c[^\u201d]{0,300}\u201d'                 # "..."
    r'|\u2018[^\u2019]{0,300}\u2019'                 # '...'
    r'|\u00ab[^\u00bb]{0,300}\u00bb'                 # «...»
    r'|\u300c[^\u300d]{0,300}\u300d'                 # 「...」
    r'|\u300e[^\u300f]{0,300}\u300f'                 # 『...』
    r'|"[^"\n]{2,200}"',                             # "straight"
)

# Timestamp patterns — masked before translation to prevent numeric corruption
TIMESTAMP_PATTERN = re.compile(
    r'[\[\(]?\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?[\]\)]?'
    r'|\d+(?:[.,]\d+)+\w{0,2}'
)

MIN_ALPHA_CHARS = 4
SOURCE_NLLB = "eng_Latn"


def has_translatable_content(text: str) -> bool:
    return sum(1 for c in text if c.isalpha()) >= MIN_ALPHA_CHARS


def mask_timestamps(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    result = text
    for i, m in enumerate(TIMESTAMP_PATTERN.finditer(text)):
        token = f"TS{i}TS"
        placeholders[token] = m.group(0)
    for token, original in placeholders.items():
        result = result.replace(original, token, 1)
    return result, placeholders


def restore_timestamps(text: str, placeholders: dict[str, str]) -> str:
    for token, original in placeholders.items():
        text = text.replace(token, original)
    return text


def split_into_segments(text: str) -> list[tuple[str, bool]]:
    """Split text into (segment, is_quoted) pairs at quote boundaries."""
    segments = []
    last_end = 0
    for m in QUOTE_PATTERN.finditer(text):
        if m.start() > last_end:
            segments.append((text[last_end: m.start()], False))
        segments.append((m.group(0), True))
        last_end = m.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))
    return segments if segments else [(text, False)]


def translate_batch(
    texts: list[str],
    target_nllb: str,
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    max_new_tokens: int = 256,
) -> list[str]:
    tokenizer.src_lang = SOURCE_NLLB
    forced_bos = tokenizer.convert_tokens_to_ids(target_nllb)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos,
                max_new_tokens=max_new_tokens,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        results.extend(decoded)
    return results


def translate_texts_preserving_quotes(
    texts: list[str],
    target_nllb: str,
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    max_new_tokens: int = 256,
) -> list[str]:
    """Translate a list of English texts to target_nllb while keeping quoted spans verbatim."""
    all_segments = [split_into_segments(t) for t in texts]

    to_translate: list[str] = []
    positions: list[tuple[int, int]] = []
    ts_maps: dict[tuple[int, int], dict[str, str]] = {}

    for ti, segments in enumerate(all_segments):
        for si, (seg, is_quoted) in enumerate(segments):
            if not is_quoted and seg.strip() and has_translatable_content(seg):
                masked, placeholders = mask_timestamps(seg)
                to_translate.append(masked)
                positions.append((ti, si))
                if placeholders:
                    ts_maps[(ti, si)] = placeholders

    translated_map: dict[tuple[int, int], str] = {}
    if to_translate:
        translated_flat = translate_batch(
            to_translate, target_nllb, tokenizer, model, device, batch_size, max_new_tokens
        )
        for (ti, si), trans in zip(positions, translated_flat):
            if (ti, si) in ts_maps:
                trans = restore_timestamps(trans, ts_maps[(ti, si)])
            translated_map[(ti, si)] = trans

    results: list[str] = []
    for ti, segments in enumerate(all_segments):
        parts = []
        for si, (seg, is_quoted) in enumerate(segments):
            if is_quoted:
                parts.append(seg)
            elif (ti, si) in translated_map:
                parts.append(translated_map[(ti, si)])
            else:
                parts.append(seg)
        results.append("".join(parts))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Translate English training JSONL to a target language via NLLB-200"
    )
    parser.add_argument("--input", required=True, help="Source JSONL (nested format, English questions)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--target-lang", required=True,
        help="NLLB BCP-47 target language code, e.g. por_Latn, fra_Latn, spa_Latn"
    )
    parser.add_argument("--model", required=True, help="Path to NLLB model snapshot")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NLLB model from {args.model} on {device} ...")
    print(f"Translation direction: {SOURCE_NLLB} → {args.target_lang}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()
    print("Model loaded.")

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} audio records.")

    # Collect ALL question stems and option texts (source is always English)
    # items: list of (ri, qi, field, text)
    items = []
    for ri, rec in enumerate(records):
        for qi, q in enumerate(rec["questions"]):
            stem = q.get("question_stem", "")
            if stem.strip():
                items.append((ri, qi, "stem", stem))
            for opt in q.get("options", []):
                text = opt.get("text", "")
                if text.strip():
                    items.append((ri, qi, f"opt_{opt['label']}", text))

    print(f"Translating {len(items)} text pieces (EN → {args.target_lang}) ...")

    texts = [item[3] for item in items]
    translated = translate_texts_preserving_quotes(
        texts, args.target_lang, tokenizer, model, device,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
    )

    translations: dict[tuple, str] = {}
    for (ri, qi, field, _), trans in zip(items, translated):
        translations[(ri, qi, field)] = trans

    print("Writing output ...")
    with open(args.output, "w") as fout:
        for ri, rec in enumerate(records):
            new_rec = dict(rec)
            new_questions = []
            for qi, q in enumerate(rec["questions"]):
                new_q = dict(q)
                stem_key = (ri, qi, "stem")
                if stem_key in translations:
                    new_q["question_stem_original"] = q["question_stem"]
                    new_q["question_stem"] = translations[stem_key]
                new_options = []
                for opt in q.get("options", []):
                    opt_key = (ri, qi, f"opt_{opt['label']}")
                    new_opt = dict(opt)
                    if opt_key in translations:
                        new_opt["text_original"] = opt["text"]
                        new_opt["text"] = translations[opt_key]
                    new_options.append(new_opt)
                new_q["options"] = new_options
                new_questions.append(new_q)
            new_rec["questions"] = new_questions
            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    print(
        f"Done. {len(items)} texts translated EN → {args.target_lang}.\n"
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
