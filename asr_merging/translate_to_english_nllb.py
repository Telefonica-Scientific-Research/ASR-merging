#!/usr/bin/env python3
"""Translate non-English questions in a training JSONL to English using NLLB-200.

Translates both question_stem and option texts. English questions are kept as-is.
Quoted spans (verbatim audio speech inside the questions) are preserved in the
original language — they are masked before translation and restored afterwards.
Original text is preserved in *_original fields.

Usage:
    python -m asr_merging.translate_to_english_nllb \
        --input  data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource.jsonl \
        --output data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl \
        --model  /gpfs/scratch/ehpc628/models/models--facebook--nllb-200-distilled-1.3B/snapshots/7be3e24664b38ce1cac29b8aeed6911aa0cf0576 \
        --batch-size 64
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

# Matches quoted verbatim audio spans across all relevant languages:
#   „..."  German (U+201E open, U+201D or " close)
#   "..."  curly double (U+201C open, U+201D close)
#   '...'  curly single (U+2018 open, U+2019 close)
#   «...»  guillemets
#   「...」 Japanese corner brackets
#   『...』 Japanese white corner brackets
#   "..."  straight double quotes (min 2 chars to avoid false positives)
QUOTE_PATTERN = re.compile(
    r'\u201e[^\u201d\u0022]{0,300}[\u201d\u0022]'   # „..."
    r'|\u201c[^\u201d]{0,300}\u201d'                 # "..."
    r'|\u2018[^\u2019]{0,300}\u2019'                 # '...'
    r'|\u00ab[^\u00bb]{0,300}\u00bb'                 # «...»
    r'|\u300c[^\u300d]{0,300}\u300d'                 # 「...」
    r'|\u300e[^\u300f]{0,300}\u300f'                 # 『...』
    r'|"[^"\n]{2,200}"',                             # "straight"
)


# Matches timestamp patterns like 22.477, [22.1-25.3], (979.26-984.02), 51.836с
# We mask these before sending to NLLB to prevent clock-time misinterpretation
TIMESTAMP_PATTERN = re.compile(
    r'[\[\(]?\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?[\]\)]?'  # 22.1-25.3 / (979.26-984.02)
    r'|\d+(?:[.,]\d+)+\w{0,2}'                                   # 51.836 / 20.85с
)

# Segments with fewer than this many alphabetic characters are not worth
# translating — keep verbatim (handles ", " between quotes, "(timestamp) ?" etc.)
MIN_ALPHA_CHARS = 4


def has_translatable_content(text: str) -> bool:
    """True if text contains enough alphabetic characters to be worth translating."""
    return sum(1 for c in text if c.isalpha()) >= MIN_ALPHA_CHARS


def mask_timestamps(text: str) -> tuple[str, dict[str, str]]:
    """Replace timestamp spans with placeholders. Returns (masked_text, placeholder_map)."""
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
            segments.append((text[last_end : m.start()], False))
        segments.append((m.group(0), True))
        last_end = m.end()
    if last_end < len(text):
        segments.append((text[last_end:], False))
    return segments if segments else [(text, False)]


# langdetect ISO 639-1 -> NLLB BCP-47
LANG_TO_NLLB = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "pt": "por_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "tl": "tgl_Latn",
    "vi": "vie_Latn",
    "ur": "urd_Arab",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "af": "afr_Latn",
}
TARGET_NLLB = "eng_Latn"


def detect_lang(text: str) -> str:
    try:
        return detect_langs(text)[0].lang
    except LangDetectException:
        return "en"


def translate_batch(
    texts: list[str],
    src_nllb: str,
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    max_new_tokens: int = 256,
) -> list[str]:
    tokenizer.src_lang = src_nllb
    forced_bos = tokenizer.convert_tokens_to_ids(TARGET_NLLB)
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
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
    src_nllb: str,
    tokenizer,
    model,
    device: str,
    batch_size: int = 64,
    max_new_tokens: int = 256,
) -> list[str]:
    """Translate a list of texts while keeping quoted audio spans verbatim.

    Strategy:
      1. Split each text into (segment, is_quoted) pairs at quote boundaries.
      2. Collect all non-quoted, non-empty segments into a flat list for batch translation.
      3. Reassemble each text by joining translated non-quoted parts with the
         original quoted parts unchanged.
    """
    # Step 1: split
    all_segments = [split_into_segments(t) for t in texts]

    # Step 2: gather non-quoted segments to translate
    # Skip segments with no real words (timestamps/punctuation only) — sending
    # those to NLLB in isolation causes hallucinated conversational replies.
    # Also mask timestamps within translatable segments to prevent clock-time
    # misinterpretation (e.g. Urdu "20.85" → "8:85").
    to_translate: list[str] = []
    positions: list[tuple[int, int]] = []  # (text_idx, seg_idx)
    ts_maps: dict[tuple[int, int], dict[str, str]] = {}  # timestamp placeholder maps

    for ti, segments in enumerate(all_segments):
        for si, (seg, is_quoted) in enumerate(segments):
            if not is_quoted and seg.strip() and has_translatable_content(seg):
                masked, placeholders = mask_timestamps(seg)
                to_translate.append(masked)
                positions.append((ti, si))
                if placeholders:
                    ts_maps[(ti, si)] = placeholders

    # Step 3: batch translate
    translated_map: dict[tuple[int, int], str] = {}
    if to_translate:
        translated_flat = translate_batch(
            to_translate, src_nllb, tokenizer, model, device, batch_size, max_new_tokens
        )
        for (ti, si), trans in zip(positions, translated_flat):
            # Restore any masked timestamps
            if (ti, si) in ts_maps:
                trans = restore_timestamps(trans, ts_maps[(ti, si)])
            translated_map[(ti, si)] = trans

    # Step 4: reassemble
    results: list[str] = []
    for ti, segments in enumerate(all_segments):
        parts = []
        for si, (seg, is_quoted) in enumerate(segments):
            if is_quoted:
                parts.append(seg)  # keep verbatim
            elif (ti, si) in translated_map:
                parts.append(translated_map[(ti, si)])
            else:
                parts.append(seg)  # empty / whitespace only
        results.append("".join(parts))
    return results


def main():
    parser = argparse.ArgumentParser(description="Translate training JSONL to English via NLLB-200")
    parser.add_argument("--input", required=True, help="Source JSONL (nested format)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", required=True, help="Path to NLLB model snapshot")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NLLB model from {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
    ).to(device)
    model.eval()
    print("Model loaded.")

    # ---- Read records ----
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} audio records.")

    # ---- Detect language for every question stem ----
    print("Detecting languages ...")
    q_langs: dict[tuple, str] = {}
    for ri, rec in enumerate(records):
        for qi, q in enumerate(rec["questions"]):
            q_langs[(ri, qi)] = detect_lang(q["question_stem"])

    # ---- Collect texts to translate, grouped by NLLB source language ----
    # key: (ri, qi, field)  value: text
    # field is 'stem' or 'opt_<label>'
    lang_items: dict[str, list[tuple]] = defaultdict(list)

    for ri, rec in enumerate(records):
        for qi, q in enumerate(rec["questions"]):
            lang = q_langs[(ri, qi)]
            if lang == "en":
                continue
            nllb_src = LANG_TO_NLLB.get(lang)
            if nllb_src is None or nllb_src == TARGET_NLLB:
                continue
            lang_items[nllb_src].append((ri, qi, "stem", q["question_stem"]))
            for opt in q.get("options", []):
                lang_items[nllb_src].append((ri, qi, f"opt_{opt['label']}", opt["text"]))

    total = sum(len(v) for v in lang_items.values())
    print(f"Translating {total} text pieces across {len(lang_items)} source languages ...")

    # ---- Translate ----
    translations: dict[tuple, str] = {}
    for nllb_src, items in lang_items.items():
        texts = [item[3] for item in items]
        fields = [item[2] for item in items]
        print(f"  [{nllb_src}]  {len(texts)} texts (quote-preserving) ...")
        translated = translate_texts_preserving_quotes(
            texts, nllb_src, tokenizer, model, device,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
        )
        for (ri, qi, field, _), trans in zip(items, translated):
            translations[(ri, qi, field)] = trans

    # ---- Apply translations and write ----
    print("Writing output ...")
    translated_q = skipped_q = 0
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
                    translated_q += 1
                else:
                    skipped_q += 1
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
        f"Done. {translated_q} questions translated, {skipped_q} kept as-is (English).\n"
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
