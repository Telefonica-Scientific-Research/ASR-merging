#!/usr/bin/env python3
"""Translate non-English questions in a challenge JSONL (flat format) to English using NLLB-200.

The challenge JSONL has one record per question (flat), with fields:
    session_id, question_id, question_stem, options

Language is detected via langdetect on question_stem (same strategy as training data).
Translated fields are written in-place; originals are preserved in *_original fields.

Usage:
    python -m asr_merging.translate_challenge_nllb \
        --input  data/mlc26_task2/task2_phase1_questions_options.jsonl \
        --output data/mlc26_task2/task2_phase1_questions_options_nllb_en.jsonl \
        --model  /gpfs/scratch/ehpc628/models/models--facebook--nllb-200-distilled-1.3B/snapshots/7be3e24664b38ce1cac29b8aeed6911aa0cf0576 \
        --batch-size 64
"""

import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import DetectorFactory

DetectorFactory.seed = 0

# Import shared utilities from the training translation module
from asr_merging.translate_to_english_nllb import (
    LANG_TO_NLLB,
    TARGET_NLLB,
    detect_lang,
    translate_texts_preserving_quotes,
)


def main():
    parser = argparse.ArgumentParser(
        description="Translate challenge JSONL (flat, one record per question) to English via NLLB-200"
    )
    parser.add_argument("--input", required=True, help="Source JSONL (flat format: one record per question)")
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

    # ---- Read flat records (one per question) ----
    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} question records.")

    # ---- Detect language for every question stem ----
    print("Detecting languages ...")
    langs = [detect_lang(rec["question_stem"]) for rec in records]
    lang_counts = defaultdict(int)
    for l in langs:
        lang_counts[l] += 1
    print(f"Language distribution: {dict(sorted(lang_counts.items()))}")

    # ---- Collect texts to translate, grouped by NLLB source language ----
    # Each item: (record_index, field_key, text)
    # field_key: "stem" or "opt_<label>"
    lang_items: dict[str, list[tuple]] = defaultdict(list)

    for ri, (rec, lang) in enumerate(zip(records, langs)):
        if lang == "en":
            continue
        nllb_src = LANG_TO_NLLB.get(lang)
        if nllb_src is None or nllb_src == TARGET_NLLB:
            continue
        lang_items[nllb_src].append((ri, "stem", rec["question_stem"]))
        for opt in rec.get("options", []):
            lang_items[nllb_src].append((ri, f"opt_{opt['label']}", opt["text"]))

    total_texts = sum(len(v) for v in lang_items.values())
    need_translation = sum(1 for l in langs if l != "en" and LANG_TO_NLLB.get(l, TARGET_NLLB) != TARGET_NLLB)
    print(f"Questions needing translation: {need_translation} / {len(records)}")
    print(f"Translating {total_texts} text pieces across {len(lang_items)} source languages ...")

    # ---- Translate per language group ----
    translations: dict[tuple, str] = {}
    for nllb_src, items in lang_items.items():
        texts = [item[2] for item in items]
        print(f"  [{nllb_src}]  {len(texts)} texts ...")
        translated = translate_texts_preserving_quotes(
            texts,
            nllb_src,
            tokenizer,
            model,
            device,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        for (ri, field, _), trans in zip(items, translated):
            translations[(ri, field)] = trans

    # ---- Apply translations and write output ----
    print("Writing output ...")
    n_translated = n_skipped = 0
    with open(args.output, "w") as fout:
        for ri, rec in enumerate(records):
            new_rec = dict(rec)

            stem_key = (ri, "stem")
            if stem_key in translations:
                new_rec["question_stem_original"] = rec["question_stem"]
                new_rec["question_stem"] = translations[stem_key]
                n_translated += 1
            else:
                n_skipped += 1

            new_options = []
            for opt in rec.get("options", []):
                new_opt = dict(opt)
                opt_key = (ri, f"opt_{opt['label']}")
                if opt_key in translations:
                    new_opt["text_original"] = opt["text"]
                    new_opt["text"] = translations[opt_key]
                new_options.append(new_opt)
            new_rec["options"] = new_options

            fout.write(json.dumps(new_rec, ensure_ascii=False) + "\n")

    print(f"Done. Translated: {n_translated}  kept as-is: {n_skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
