#!/usr/bin/env python3
"""Translate non-English questions in a training JSONL to English using Qwen3-4B-Instruct.

Uses the LLM with an explicit instruction to preserve quoted audio spans verbatim.
Processes questions individually (no segment splitting needed — the LLM understands
the instruction naturally).
Original text is preserved in *_original fields.

Usage:
    python -m asr_merging.translate_to_english_qwen3 \
        --input  data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource.jsonl \
        --output data/mlc26_task2/mlcslm_2nd_dev_qa_successed_opensource_en_translated_qwen3.jsonl \
        --model  /gpfs/scratch/ehpc628/models/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554 \
        --batch-size 16
"""

import argparse
import json
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langdetect import detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

# Matches quoted verbatim audio spans across all relevant languages.
# Same pattern as translate_to_english_nllb.py — must stay in sync.
QUOTE_PATTERN = re.compile(
    r'\u201e[^\u201d\u0022]{0,300}[\u201d\u0022]'   # „..."
    r'|\u201c[^\u201d]{0,300}\u201d'                 # "..."
    r'|\u2018[^\u2019]{0,300}\u2019'                 # '...'
    r'|\u00ab[^\u00bb]{0,300}\u00bb'                 # «...»
    r'|\u300c[^\u300d]{0,300}\u300d'                 # 「...」
    r'|\u300e[^\u300f]{0,300}\u300f'                 # 『...』
    r'|"[^"\n]{2,200}"',                             # "straight"
)


def mask_quotes(text: str) -> tuple[str, list[str]]:
    """Replace quoted spans with [Q0], [Q1], ... placeholders.
    Returns (masked_text, list_of_original_quoted_spans).
    Replacements go right-to-left to preserve string positions."""
    matches = list(QUOTE_PATTERN.finditer(text))
    originals = [m.group(0) for m in matches]
    result = text
    for i, m in reversed(list(enumerate(matches))):
        result = result[: m.start()] + f"[Q{i}]" + result[m.end() :]
    return result, originals


def restore_quotes(text: str, originals: list[str]) -> str:
    """Restore [Q0], [Q1], ... placeholders with original quoted spans."""
    for i, original in enumerate(originals):
        text = text.replace(f"[Q{i}]", original, 1)
    return text


# langdetect ISO 639-1 -> human-readable name for the prompt
LANG_TO_NAME = {
    "es": "Spanish",
    "fr": "French",
    "pt": "Portuguese",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "tr": "Turkish",
    "tl": "Tagalog",
    "vi": "Vietnamese",
    "ur": "Urdu",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "af": "Afrikaans",
}

SYSTEM_PROMPT = (
    "You are a precise translator. Translate the given text to English. "
    "Rules:\n"
    "1. Tokens of the form [Q0], [Q1], [Q2], etc. are quote placeholders standing for "
    "verbatim quoted audio speech. Keep them EXACTLY as-is \u2014 do not translate, "
    "remove, or modify them in any way.\n"
    "2. Keep all timestamps (e.g. [22.1-25.3], (979.26-984.02)) exactly as-is.\n"
    "3. Output ONLY the translated text \u2014 no explanation, no preamble, no markdown.\n"
    "/no_think"
)


def detect_lang(text: str) -> str:
    try:
        return detect_langs(text)[0].lang
    except LangDetectException:
        return "en"


def build_prompt(text: str, src_lang_name: str, tokenizer) -> str:
    user_msg = f"Translate the following {src_lang_name} text to English:\n{text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def translate_batch(
    texts: list[str],
    src_lang_name: str,
    tokenizer,
    model,
    device: str,
    batch_size: int = 16,
    max_new_tokens: int = 512,
) -> list[str]:
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        prompts = [build_prompt(t, src_lang_name, tokenizer) for t in batch_texts]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        input_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens (strip the prompt)
        new_tokens = generated[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        # Strip any residual thinking tags that /no_think didn't suppress
        cleaned = [
            d.split("</think>")[-1].strip() if "</think>" in d else d.strip()
            for d in decoded
        ]
        results.extend(cleaned)
        print(f"    batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} done")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Translate training JSONL to English via Qwen3-4B-Instruct"
    )
    parser.add_argument("--input", required=True, help="Source JSONL (nested format)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--model", required=True, help="Path to Qwen3 model snapshot")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen3-4B from {args.model} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
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

    # ---- Collect texts to translate, grouped by source language ----
    # Quoted spans are masked to [Q0], [Q1], ... before translation so Qwen3
    # cannot accidentally translate verbatim audio content.
    lang_items: dict[str, list[tuple]] = defaultdict(list)
    quote_maps: dict[tuple, list[str]] = {}  # (ri, qi, field) -> original quotes
    for ri, rec in enumerate(records):
        for qi, q in enumerate(rec["questions"]):
            lang = q_langs[(ri, qi)]
            if lang == "en":
                continue
            lang_name = LANG_TO_NAME.get(lang, lang)
            masked_stem, stem_quotes = mask_quotes(q["question_stem"])
            stem_key = (ri, qi, "stem")
            lang_items[lang_name].append((ri, qi, "stem", masked_stem))
            if stem_quotes:
                quote_maps[stem_key] = stem_quotes
            for opt in q.get("options", []):
                masked_opt, opt_quotes = mask_quotes(opt["text"])
                opt_key = (ri, qi, f"opt_{opt['label']}")
                lang_items[lang_name].append((ri, qi, f"opt_{opt['label']}", masked_opt))
                if opt_quotes:
                    quote_maps[opt_key] = opt_quotes

    total = sum(len(v) for v in lang_items.values())
    print(
        f"Translating {total} text pieces across "
        f"{len(lang_items)} source languages (Qwen3-4B) ..."
    )

    # ---- Translate ----
    translations: dict[tuple, str] = {}
    for lang_name, items in lang_items.items():
        texts = [item[3] for item in items]
        print(f"  [{lang_name}]  {len(texts)} texts ...")
        translated = translate_batch(
            texts,
            lang_name,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        for (ri, qi, field, _), trans in zip(items, translated):
            key = (ri, qi, field)
            if key in quote_maps:
                trans = restore_quotes(trans, quote_maps[key])
            translations[key] = trans

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
