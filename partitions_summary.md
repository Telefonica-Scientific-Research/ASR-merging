# Data Partitions and Translation Strategy

## 1. Motivation: Training–Challenge Distribution Mismatch

The MLC-SLM 2nd Dev challenge (Phase 1) exposes a critical language-pairing mismatch between the training data and the actual evaluation distribution.

**Challenge Phase 1 distribution** (4,985 questions across 150 audio files):

| Pair type | Definition | Share |
|---|---|---|
| EN+EN | English audio, English question | 25.1% |
| Cross-lang | Non-English audio, English question | **52.0%** |
| Same-lang | Non-English audio, native-language question | 22.9% |

**Original organiser training data** (150 audio files, 4,500 questions):  
The questions provided by the organisers are almost exclusively in the same language as the audio (~99% same-language pairs). Training directly on this data therefore exposes the model to virtually no cross-language (non-EN audio → EN question) examples, while the challenge requires the model to handle them in the majority (52%) of cases.

### 1.1 Challenge Dataset Language Distribution (langdetect)

Per-question language distribution estimated via `langdetect` on `question_stem`, for both challenge phases. The majority (~75%) of questions are already in English (EN-audio sessions). Non-EN questions are the target of NLLB translation at inference time.

| Language (NLLB code) | Phase 1 questions | Phase 1 % | Phase 2 questions | Phase 2 % |
|---|---|---|---|---|
| English (`en`) | 3,841 | **77.1%** | 7,123 | **75.2%** |
| Portuguese (`por_Latn`) | 228 | 4.6% | 444 | 4.7% |
| Japanese (`jpn_Jpan`) | 180 | 3.6% | 360 | 3.8% |
| Thai (`tha_Thai`) | 165 | 3.3% | 245 | 2.6% |
| Russian (`rus_Cyrl`) | 153 | 3.1% | 243 | 2.6% |
| Spanish (`spa_Latn`) | 139 | 2.8% | 201 | 2.1% |
| Korean (`kor_Hang`) | 45 | 0.9% | 183 | 1.9% |
| French (`fra_Latn`) | 1 | 0.0% | 167 | 1.8% |
| Turkish (`tur_Latn`) | 30 | 0.6% | 151 | 1.6% |
| German (`deu_Latn`) | — | — | 115 | 1.2% |
| Vietnamese (`vie_Latn`) | 101 | 2.0% | 101 | 1.1% |
| Urdu (`urd_Arab`) | 69 | 1.4% | 69 | 0.7% |
| Tagalog (`tgl_Latn`) | 30 | 0.6% | 31 | 0.3% |
| Italian (`ita_Latn`) | — | — | 30 | 0.3% |
| **Total** | **4,985** | | **9,470** | |
| **Non-EN (to translate)** | **1,144** | **22.9%** | **2,347** | **24.8%** |

> Phase 2 gains French, German, and Italian sessions absent from Phase 1. Per-session majority-vote gives 119/151 (78.8%) EN sessions in Phase 1 and 113/151 (74.8%) in Phase 2.

---

## 2. Mismatch Detection via `langdetect`

To quantify the training–challenge gap precisely, we used the `langdetect` library (a Python port of Google's language-detect, v1.2.9) to classify every question in the dataset by its detected language.

### 2.1 Methodology

For each training record:
1. Extract the audio language from the file path (e.g., `mlc-slm-2nd-dev/French/file.wav` → `French`).
2. Classify the audio as **English-variety** (`English_American`, `English_Australian`, `English_British`, `English_Filipino`, `English_Indian`) or **non-English**.
3. For non-English audio, apply `langdetect.detect_langs(question_stem)[0].lang` with `DetectorFactory.seed = 0` (deterministic mode) to each question stem.
4. Classify the question as:
   - **EN+EN**: English-variety audio (question language is always English by construction).
   - **Cross-lang**: non-English audio + detected question language = `en`.
   - **Same-lang**: non-English audio + detected question language ≠ `en`.

> **Limitation**: `langdetect` can misidentify short questions or languages that share Latin script (e.g., Tagalog, Vietnamese, Malay) as another Latin-script language rather than English. However, for the purpose of detecting the overall distribution shift, the error rate is negligible across a corpus of 3,720 questions.

### 2.2 Per-Language Breakdown of Original Training Questions

The table below shows, for each audio-language group in the 124-file training set, how many questions were detected as cross-language (non-EN audio + EN question) vs. same-language (non-EN audio + native question):

| Audio language | Type | Files | EN+EN | Cross-lang | Same-lang |
|---|---|---|---|---|---|
| English_American | EN | 7 | 210 | — | — |
| English_Australian | EN | 4 | 120 | — | — |
| English_British | EN | 7 | 210 | — | — |
| English_Filipino | EN | 5 | 150 | — | — |
| English_Indian | EN | 4 | 120 | — | — |
| French | non-EN | 7 | — | 0 | 210 |
| French_Canada | non-EN | 8 | — | 0 | 240 |
| German | non-EN | 6 | — | 0 | 180 |
| Italian | non-EN | 5 | — | 0 | 150 |
| Japanese | non-EN | 10 | — | 0 | 300 |
| Korean | non-EN | 7 | — | 0 | 210 |
| Portuguese | non-EN | 9 | — | 0 | 270 |
| Portuguese_Brazil | non-EN | 6 | — | 0 | 180 |
| Russian | non-EN | 4 | — | 0 | 120 |
| Spanish | non-EN | 4 | — | 0 | 120 |
| Spanish_Mexico | non-EN | 4 | — | 0 | 120 |
| Tagalog | non-EN | 6 | — | 0 | 180 |
| Thai | non-EN | 4 | — | 0 | 120 |
| Turkish | non-EN | 6 | — | 0 | 180 |
| Urdu | non-EN | 4 | — | 0 | 120 |
| Vietnamese | non-EN | 7 | — | 0 | 210 |
| **TOTAL** | | **124** | **810** | **0** | **2,910** |
| **SHARE** | | | **21.8%** | **0.0%** | **78.2%** |

**Finding**: The original training data contains **zero cross-language questions** for non-English audio. Every non-English audio file is paired exclusively with questions written in the same language as the audio. This creates a severe mismatch against the challenge's 52% cross-language requirement.

### 2.3 Distribution Comparison Across Training Variants

After applying automatic translation, the question-language distribution shifts substantially. The table below compares all five training variants against the challenge target, with distributions computed via `langdetect` on the question stems:

| Training variant | Total Q | EN+EN | Cross-lang | Same-lang | EN+EN% | Cross% | Same% |
|---|---|---|---|---|---|---|---|
| **Original** | 3,720 | 810 | 0 | 2,910 | 21.8% | **0.0%** | 78.2% |
| **NLLB-200** | 3,720 | 810 | 2,778 | 132 | 21.8% | **74.7%** | 3.5% |
| **Qwen3-4B** | 3,720 | 810 | 2,796 | 114 | 21.8% | **75.2%** | 3.1% |
| **Mixed** (orig + NLLB) | 6,630 | 810 | 2,778 | 3,042 | 12.2% | **41.9%** | 45.9% |
| **Triple-Mixed** (orig + NLLB + Qwen3) | 9,540 | 810 | 5,574 | 3,156 | 8.5% | **58.4%** | 33.1% |
| **Challenge target** | — | — | — | — | **25.1%** | **52.0%** | **22.9%** |

**Key observations**:

- **Original**: Zero cross-lang coverage. Would require the model to generalise to a question type it never saw during training.
- **NLLB-200 / Qwen3-4B** (translation-only): The original 78.2% same-lang (non-EN audio, non-EN question) collapses to ~3.5% because we replace each non-EN question with its NLLB/Qwen3 English translation — those pairs switch from same-lang to cross-lang. The residual 3.5% (132 questions) are questions that could not be translated: either too short (< 4 alphabetic chars, skipped by `has_translatable_content`) or where langdetect misclassified the output. Total Q count stays at 3,720 (one-to-one replacement, no duplication). This overcorrects — cross-lang rises to ~75%, but same-lang collapses to ~3%. The challenge requires ~23% same-lang coverage, which these variants almost completely drop.
- **Mixed** (orig + NLLB): Cross-lang rises to 41.9% (below challenge target of 52%) but same-lang remains high at 45.9%. The model sees balanced exposure at the cost of a higher effective same-lang share.
- **Triple-Mixed** (orig + NLLB + Qwen3): Closest to the challenge distribution. Cross-lang at 58.4% (slightly above target 52%), same-lang at 33.1% (above target 22.9%), EN+EN at 8.5% (below target 25.1%). The EN+EN under-representation is an artefact of the augmentation strategy (non-EN records are duplicated, diluting the EN share), but the cross-lang/same-lang trade-off is the most balanced.

> **Note on residual non-English questions after translation** (NLLB: 132, Qwen3: 114): These correspond to cases where (a) the question was very short (< 4 translatable characters, skipped by the translation pipeline's `has_translatable_content` filter) or (b) `langdetect` misclassified the translated output as non-English. Both represent < 4% of non-EN audio questions.

---

## 3. Dataset Split: Training vs. Held-Out Evaluation

The 150 available audio files (21 language groups: 5 English varieties + 16 non-English languages) were split into:

- **Training set**: 124 audio files
- **Held-out evaluation set**: 26 audio files (0 audio-file overlap with training)

### 3.1 Language Coverage

| Language group | Total files | Train | Eval |
|---|---|---|---|
| English varieties (5) | 27 | 17 | 10 (2 per variety) |
| Non-EN languages — high-resource (11) | 77 | 66 | 11 (1 per language) |
| Non-EN languages — low-resource (5) | 20 | 15 | 5 (1 per language) |

> **High-resource non-EN languages** (≥5 files available): French, French_Canada, German, Italian, Japanese, Korean, Portuguese, Portuguese_Brazil, Tagalog, Turkish, Vietnamese.  
> **Low-resource non-EN languages** (4 files available): Russian, Spanish, Spanish_Mexico, Thai, Urdu.

The held-out 26-file split was designed to leave at least 3–4 audio files per language group in the training set to maintain training coverage.

---

## 4. Translation to English Using Automatic MT

To bridge the training–challenge distribution gap, the native-language questions for non-English audio files were automatically translated to English using two independent translation models. Quoted spans—verbatim audio content cited inside questions (e.g., *"What does the speaker mean by \u201cI\u2019ll take care of it\u201d?"*)—were explicitly preserved in their original form throughout translation.

### 4.1 NLLB-200 (facebook/nllb-200-distilled-1.3B)

- **Model**: Facebook NLLB-200 distilled 1.3B, a dedicated multilingual neural machine translation model supporting 200 languages ([Costajussà et al., 2022](https://arxiv.org/abs/2207.04672)).
- **Checkpoint**: `facebook/nllb-200-distilled-1.3B` (safetensors format).
- **Quote preservation**: A regex pattern (`QUOTE_PATTERN`) detects verbatim quoted spans using all common quote styles (curly doubles `""`, curly singles `''`, German `„"`, guillemets `«»`, Japanese `「」` `『』`, straight doubles `"`). Quoted spans are replaced by indexed placeholder tokens `[Q0]`, `[Q1]`, … before translation and restored afterwards.
- **Timestamp preservation**: Numeric timestamps and timestamp ranges (e.g., `22.477`, `[22.1–25.3]`) are similarly masked before translation to prevent the model from misinterpreting them as clock times (a known issue with seq2seq models).
- **Source language detection**: `langdetect` with seeded determinism; questions with fewer than 4 translatable alphabetic characters are passed through unchanged.
- **Already-English pass-through**: English questions (and all English-audio records) are left unchanged.
- **Batch size**: 64 sequences per forward pass (H100 80 GB).

#### 4.1.1 NLLB-200 Translation Quality (Flores-200 Benchmark, chrF++)

Official chrF++ scores from the NLLB-200 metrics CSV for both model sizes, evaluated on the Flores-200 benchmark (general-domain Wikipedia sentences). **Our use case is lang→EN** (translating non-EN challenge questions to English at inference time). EN→lang scores are shown for reference only — they are poor for Asian languages and are not used.

| Language | lang→EN 1.3B | lang→EN 3.3B | Δ | EN→lang 1.3B | EN→lang 3.3B |
|---|---|---|---|---|---|
| Portuguese | 70.6 | 71.3 | +0.7 | 67.9 | 69.4 |
| Tagalog | 67.2 | 68.2 | +1.0 | 59.0 | 60.6 |
| French | 67.2 | 68.1 | +0.9 | 68.8 | 69.6 |
| German | 66.7 | 67.4 | +0.7 | 61.2 | 62.8 |
| Turkish | 62.0 | 63.9 | +1.9 | 56.5 | 57.8 |
| Urdu | 60.9 | 61.7 | +0.8 | 47.9 | 48.3 |
| Russian | 60.6 | 61.3 | +0.7 | 54.6 | 56.1 |
| Italian | 60.6 | 61.2 | +0.6 | 56.2 | 57.1 |
| Vietnamese | 60.4 | 61.5 | +1.1 | 58.4 | 59.3 |
| Spanish | 58.3 | 59.1 | +0.8 | 53.6 | 54.2 |
| Korean | 55.0 | 56.1 | +1.1 | 34.0 | 34.3 |
| Thai | 54.9 | 56.8 | +1.9 | 38.6 | 40.5 |
| **Japanese** | **54.2** | **55.1** | +0.9 | **25.4** | **25.2** |

**Key findings**: (1) 3.3B offers only marginal gains (+0.6–+1.9 chrF++) over distilled-1.3B in the lang→EN direction — not worth the 2.5× compute cost. (2) EN→lang quality for Japanese (25.4) and Korean (34.0) is extremely poor — NLLB cannot reliably translate English questions into those scripts. This confirms that the **lang→EN inference strategy** (translate challenge questions to English) is far more reliable than the reverse. (3) These scores are on general-domain text; domain-specific acoustic/speech questions will score somewhat lower.

### 4.2 Qwen3-4B-Instruct (Qwen/Qwen3-4B-Instruct-2507)

- **Model**: Alibaba Qwen3-4B-Instruct, a 4B-parameter instruction-tuned LLM with strong multilingual understanding.
- **Checkpoint**: `Qwen/Qwen3-4B-Instruct-2507`.
- **Approach**: Zero-shot prompting. The system prompt instructs the model to translate from any source language to English while (a) keeping placeholder tokens `[Q0]`, `[Q1]`, … exactly as-is, and (b) not adding explanations.
- **Quote preservation**: Identical regex masking to NLLB (`QUOTE_PATTERN`) is applied before the LLM call and restored after. This was added after discovering that the LLM tended to paraphrase or re-translate verbatim audio quotes without masking (~85% preservation without masking → ~99.2% with masking). The masking result was validated against 1,599 questions containing at least one quoted span.
- **Batch size**: 16 sequences per call (generation with `max_new_tokens=512`, temperature=0 for determinism).

---

## 5. Training Data Variants

Six training data files were produced from the 124-file training split. All files share the same 124 unique audio files; the difference is exclusively in question language.

| File | Records | Questions | Description |
|---|---|---|---|
| `train_124files_original.jsonl` | 124 | 3,720 | Organiser questions as-is (native language) |
| `train_124files_nllb.jsonl` | 124 | 3,720 | All non-EN questions translated to EN by NLLB-200 |
| `train_124files_qwen3.jsonl` | 124 | 3,720 | All non-EN questions translated to EN by Qwen3-4B |
| `train_124files_mixed.jsonl` | 221 | 6,630 | Original + NLLB-translated copies for non-EN audio |
| `train_124files_triple_mixed.jsonl` | 318 | 9,540 | Original + NLLB + Qwen3 copies for non-EN audio |

**Notes on mixed files**:
- In `mixed` and `triple_mixed`, the 97 non-EN audio files appear 2× or 3× respectively (once per question-language variant), while the 27 English-audio files appear only once (already EN+EN).
- The `mixed` variant is the primary hypothesis: it exposes the model to both same-language and cross-language question styles, covering all three challenge pair types.

### 5.2 Round 5 Data Split (129/21 files — English-only strategy)

A new split was created for the English-only training strategy (Round 5). All files use exclusively English questions — either original (EN audio) or NLLB/Qwen3-translated (non-EN audio).

| File | Records | Questions | Description |
|---|---|---|---|
| `train_129files_nllb.jsonl` | 129 | 3,870 | 129-file train split, all questions NLLB→EN |
| `train_129files_qwen3.jsonl` | 129 | 3,870 | 129-file train split, all questions Qwen3→EN |
| `train_129files_nllb_qwen3.jsonl` | 258 | 7,740 | Both translations combined (each file appears twice) |
| `eval_21files_en_only.jsonl` | 21 | 630 | Held-out: 1 file/language group, English questions only |

> The 150-file EN-translated dataset (`mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl`, 150 records, 4,500q) is used for Round 1 and Round 6 training (all 150 files, organisers' eval for early stopping).

### 5.3 Curriculum Learning Augmentation — EN→Target Language (submitted 2026-06-21)

To prepare multilingual augmentation data for a curriculum learning phase, the full 150-file EN-translated dataset was translated from English into 8 target languages using NLLB-200 distilled-1.3B.

**Motivation**: Rather than mixing in the noisy original non-EN questions, translate the clean English questions into target languages. This gives:
- A controlled multilingual version of each audio file (no lang→EN→lang double-translation noise beyond what's already in the EN source)
- Flexibility to mix any subset at curriculum-learning time
- Direct coverage of the highest-frequency non-EN challenge languages

**Language selection criteria**: (1) present in training data (≥150 questions), (2) absent from JA/KO exclusion list (NLLB EN→lang chrF++ < 40), (3) present in challenge Phase 2 (≥100 questions). German added despite Phase 1 absence due to top-3 EN→DE quality (61.2 chrF++).

| SLURM job | Language | NLLB code | EN→lang chrF++ | Train data | Ph1 challenge | Ph2 challenge | Output file |
|---|---|---|---|---|---|---|
| 42201754 | French | `fra_Latn` | **68.8** | 11.3% (510q) | <1% | 1.8% | `mlcslm_2nd_dev_qa_en_to_fra.jsonl` |
| 42201753 | Portuguese | `por_Latn` | **67.9** | 11.3% (510q) | 4.6% | 4.7% | `mlcslm_2nd_dev_qa_en_to_por.jsonl` |
| 42201760 | German | `deu_Latn` | **61.2** | 4.7% (210q) | — | 1.2% | `mlcslm_2nd_dev_qa_en_to_deu.jsonl` |
| 42201759 | Tagalog | `tgl_Latn` | 59.0 | 4.7% (210q) | 0.6% | 0.3% | `mlcslm_2nd_dev_qa_en_to_tgl.jsonl` |
| 42201757 | Vietnamese | `vie_Latn` | 58.4 | 5.3% (240q) | 2.0% | 1.1% | `mlcslm_2nd_dev_qa_en_to_vie.jsonl` |
| 42201758 | Turkish | `tur_Latn` | 56.5 | 4.7% (210q) | 0.6% | 1.6% | `mlcslm_2nd_dev_qa_en_to_tur.jsonl` |
| 42201756 | Russian | `rus_Cyrl` | 54.6 | 3.3% (150q) | 3.1% | 2.6% | `mlcslm_2nd_dev_qa_en_to_rus.jsonl` |
| 42201755 | Spanish | `spa_Latn` | 53.6 | 6.7% (300q) | 2.8% | 2.1% | `mlcslm_2nd_dev_qa_en_to_spa.jsonl` |

> Italian excluded: only 30 Phase 2 challenge questions (0.3%), negligible impact. Japanese (25.4) and Korean (34.0) excluded: EN→lang chrF++ too low.

Each output file has 150 records / 4,500 questions (full 150-file dataset), with original English preserved in `question_stem_original` / `text_original` fields. Script: `asr_merging/translate_from_english_nllb.py`, SLURM template: `deploy/run_translate_from_eng_nllb_1gpu.sh`.

---

## 6. Held-Out Evaluation Partition

### 6.1 Design Principles

The evaluation set was constructed to be **representative of the challenge distribution** and to have **zero audio-file overlap** with training. The target challenge distribution (EN+EN 25.1%, cross-lang 52%, same-lang 22.9%) was approximated as closely as possible within the constraint of 26 audio files.

### 6.2 Evaluation Set Structure

| Audio group | Files | Records | Questions | Pair type |
|---|---|---|---|---|
| English varieties (5) × 2 files each | 10 | 10 | 300 | EN+EN (original questions) |
| High-resource non-EN (11) × 1 file each | 11 | 22 | 660 | Cross-lang (Qwen3-translated) **+** Same-lang (original) |
| Low-resource non-EN (5) × 1 file each | 5 | 5 | 150 | Cross-lang only (Qwen3-translated) |
| **Total** | **26** | **37** | **1,110** | |

> The 37-record count (for 26 audio files) reflects that high-resource non-EN files contribute 2 records each: one with Qwen3-translated (EN) questions and one with the original (native) questions.  
> Low-resource non-EN files contribute cross-lang records only; same-language records were excluded to preserve training data quality for these 5 low-resource groups (4 files/language total; using 1 for eval leaves only 3 for training).

### 6.3 Resulting Eval Distribution

| Pair type | Questions | Share | Challenge target |
|---|---|---|---|
| EN+EN | 300 | 27.0% | 25.1% |
| Cross-lang | 480 | 43.2% | 52.0% |
| Same-lang | 330 | 29.7% | 22.9% |

The evaluation set approximates the challenge distribution reasonably well given the file-count constraint (26 files, 30 questions each). The cross-lang share (43%) is lower than the challenge target (52%) due to the inclusion of same-language questions for high-resource non-EN files, which is considered a useful training signal to retain.

### 6.4 Question Source for Eval

- **EN audio records**: original organiser questions (already in English).
- **Non-EN audio cross-lang records**: Qwen3-4B-Instruct translations (preferred over NLLB for eval due to higher fluency and better quote preservation).
- **Non-EN audio same-lang records**: original organiser questions (native language).

**Rationale for Qwen3 on eval**: NLLB-200 translations are used as a training data augmentation signal; using NLLB-translated questions for eval would overfit the metric to NLLB output style. Qwen3 translations serve as a higher-quality, independently produced proxy for the challenge's cross-lang question distribution.

---

## 7. Training Experiments

Two rounds of experiments were conducted: an initial round (Round 1) using the full 150-file dataset with the organisers' evaluation sets, and a second round (Round 2, v2) using the new 124/26 split with the held-out evaluation partition.

**Shared training configuration**:
- Model: `mistralai/Voxtral-Mini-3B-2507`, LoRA (r=16, α=32, `target_modules="all-linear"`)
- 4×H100 (80 GB), fp16, lr=5×10⁻⁵, 3 epochs max, early-stopping patience=3
- Eval/save every 25 gradient-update steps, logging every 3 steps

---

### 7.1 Round 1 — Full 150-file Dataset, Organisers' Eval Sets

**Effective batch size sweep**: bs16 (grad_accum=4) vs bs32 (grad_accum=8), each on three data variants.

#### 7.1.1 Experiment summary table

| Experiment | Train data | Train Q | BS_eff | Total steps | Done | Status |
|---|---|---|---|---|---|---|
| `nllb_bs16` | NLLB-translated | 4,500 | 16 | 846 | 575 (68%) | ✅ Finished (early stopped) |
| `nllb_bs32` | NLLB-translated | 4,500 | 32 | 423 | 423 (100%) | ✅ Finished (all epochs) |
| `mixed_bs16` | orig + NLLB | 7,950 | 16 | 1,482 | 625 (42%) | ✅ Finished (early stopped) |
| `mixed_bs32` | orig + NLLB | 7,950 | 32 | 741 | 741 (100%) | ✅ Finished (all epochs) |
| `triple_bs16` | orig + NLLB + Qwen3 | 11,340 | 16 | 2,127 | 475 (22%) | ✅ Finished (early stopped) |
| `triple_bs32` | orig + NLLB + Qwen3 | 11,340 | 32 | 1,063 | ~1,063 | ✅ Finished |

#### 7.1.2 Training and eval loss metrics (final)

All eval losses are computed on the organisers' balanced eval sets (NLLB- or mixed-translated). Losses are cross-entropy over answer tokens (lower = better).

| Experiment | Init eval loss | Best eval loss | Last eval loss | Avg last-5 train | Gap† | Test acc‡ | Challenge |
|---|---|---|---|---|---|---|---|
| `nllb_bs16` | 0.3175 | 0.0961 | 0.1009 | 0.0763 | −0.025 | 90.7% (n=300) | **0.7382** 🥇 |
| `nllb_bs32` | 0.3483 | 0.0749 | 0.0901 | 0.0561 | −0.034 | 91.7% (n=300) | hyp pending |
| `mixed_bs16` | 0.3706 | 0.0926 | 0.0974 | 0.1060 | +0.009 | 93.5% (n=526) | hyp ready |
| `mixed_bs32` | 0.3955 | **0.0584** | 0.0666 | 0.0294 | −0.037 | **94.7% (n=526)** | 0.7225 ⚠️ |
| `triple_bs16` | 0.3332 | 0.1231 | 0.1305 | 0.1181 | −0.012 | 91.0% (n=752) | hyp ready |
| `triple_bs32` | 0.3171 | 0.0822 | 0.0866 | 0.0486 | −0.038 | **93.9% (n=752)** | **0.7370** 🥈 |

> † **Gap** = avg last-5 train loss − last eval loss. Negative = eval > train (model has capacity to improve further); positive = train > eval (eval set easier than training distribution at current step).  
> ‡ **Test acc** on organisers' test set (`organisers_*.jsonl`), **not** the held-out partition. Different `n` values: NLLB eval=300q (EN-only), mixed eval=526q (EN+native), triple eval=752q (EN+native+Qwen3). Direct cross-variant comparison is approximate only.

#### 7.1.3 Discussion and Conclusions

**1. Batch size: bs32 dominates bs16 uniformly**

Across all three dataset variants, bs32 reaches lower best eval loss despite having exactly half the gradient steps:

| Dataset | Best eval loss bs16 | Best eval loss bs32 | Relative gain | Test acc bs16 | Test acc bs32 |
|---|---|---|---|---|---|
| NLLB-translated | 0.0961 | 0.0749 | −22% | 90.7% | 91.7% |
| Mixed (orig + NLLB) | 0.0926 | 0.0584 | −37% | 93.5% | **94.7%** |
| Triple-mixed | 0.1231 | 0.0822 | −33% | 91.0% | **93.9%** |

Fewer but higher-quality gradient updates with larger batch size consistently outperform noisier updates at bs16. **Conclusion: bs32 is the preferred setting for all future experiments.**

**2. Data augmentation: mixed is the sweet spot**

Within each batch-size group, the ranking by both eval loss and test accuracy is:

| Rank | Variant (bs32) | Best eval loss | Test acc | Challenge dist. fit |
|---|---|---|---|---|
| 🥇 1 | **mixed_bs32** | **0.0584** | **94.7%** | Cross 41.9% / Same 45.9% |
| 🥈 2 | **triple_bs32** | 0.0822 | **93.9%** | Cross 58.4% / Same 33.1% |
| 🥉 3 | nllb_bs32 | 0.0749 | 91.7% | Cross 74.7% / Same 3.5% |

> Note: `triple_bs32` trained in `123213`; NCCL teardown crash in `123214` was benign — model weights were already saved before the crash.

`mixed_bs32` wins on both loss and accuracy, even though `nllb_bs32` has a lower best eval loss (0.0749 < 0.0584 is wrong — mixed_bs32 has 0.0584 which IS lower). The eval loss advantage of `mixed` reflects the combined same-lang + cross-lang coverage: the model learns both question styles without losing either. `triple_bs32` performs second: adding the Qwen3 copy introduces more diversity but also more noise/redundancy versus the NLLB copy (both translate to English), diluting training signal per step.

`nllb_bs32`, despite lower eval loss (0.0749) than `triple_bs32` (0.0822), ranks last on test accuracy (91.7%) because its test set only covers EN-translated questions (n=300), ignoring same-lang performance where it collapses to ~3%.

**3. Early stopping on the organisers' eval is unreliable as a convergence signal**

- `nllb_bs16` stopped at 575/846 steps (68%): the EN-only organisers eval is too easy to discriminate further learning.
- `mixed_bs16` stopped at 625/1,482 steps (42%): only one third of the training budget was consumed. The model may have significantly more to learn.
- `triple_bs16` stopped at 475/2,127 steps (22%): with 11,340 training questions, early stopping after <500 steps is almost certainly premature. The held-out eval (v2 experiments) using the more challenging 26-file partition should provide a better stopping signal.

**4. Train/eval gap interpretation**

`mixed_bs32` has the largest negative gap (−0.037): last train loss ≈ 0.029, last eval loss ≈ 0.067. This 2.3× ratio between eval and train loss suggests the model has near-memorised the mixed training set but generalises less perfectly to the eval set. This is typical of LoRA fine-tuning on a small dataset (<10k questions) and does not necessarily indicate harmful overfitting — the challenge score will be the true judge.

`mixed_bs16`'s slightly positive gap (+0.009) is the opposite anomaly: training loss is marginally above eval loss at the checkpoint, which simply means the model was evaluated mid-convergence on a relatively easy eval partition.

**5. Challenge submission priority (5 submissions/day)**

| Priority | Model | Hyp file | Rationale |
|---|---|---|---|
| ✅ Already submitted | `nllb_bs16` | Ready | Baseline → **0.74** |
| 📌 **#1** | **`mixed_bs32`** | Need to generate | Best loss + best test acc; highest confidence |
| 📌 **#2** | **`triple_bs32`** | Need to generate | Best challenge-distribution fit (58% cross-lang) |
| 📌 **#3** | `mixed_bs16` | Ready (4,985 lines) | Quick win — hyp already generated |
| 📌 **#4** | `triple_bs16` | Ready (4,985 lines) | Quick win — hyp already generated |
| ⏳ Hold | `nllb_bs32` | Need to generate | Weaker same-lang coverage; skip unless slots remain |

**Recommended action**: run `run_voxtral_challenge_eval_4gpu.sh` for `mixed_bs32` and `triple_bs32` to generate their hyp files (two SLURM jobs), then submit all four in today's remaining slots. Use the 5th slot for the best Round 2 (v2) model when it finishes.

---

### 7.2 Round 2 (v2) — 124-file training split, proper held-out eval

All five variants trained with effective bs=32 and `--eval-jsonl` = `--test-jsonl` = `eval_26files_challenge_repr.jsonl` (the held-out 26-file partition, 1,110q, zero audio overlap with training).

| SLURM job | Script variant | Train Q | Train data |
|---|---|---|---|
| 42169665 | `v2_original_bs32` | 3,720 | native lang only |
| 42169666 | `v2_nllb_bs32` | 3,720 | NLLB-translated |
| 42169667 | `v2_qwen3_bs32` | 3,720 | Qwen3-translated |
| 42169668 | `v2_mixed_bs32` | 6,630 | orig + NLLB |
| 42169669 | `v2_triple_mixed_bs32` | 9,540 | orig + NLLB + Qwen3 |

> All 5 jobs completed. Eval set: `eval_26files_challenge_repr.jsonl` (1,110 questions, challenge-representative distribution).

#### 7.2.1 Round 2 results

| Variant | Train Q | Best eval loss | Last eval loss | Last train loss | Gap | **Test acc (1,110q)** |
|---|---|---|---|---|---|---|
| `v2_original_bs32` | 3,720 | — | — | — | — | 91.17% |
| `v2_nllb_bs32` | 3,720 | 0.1623 | 0.1823 | 0.0803 | −0.102 | 91.17% |
| `v2_qwen3_bs32` | 3,720 | — | — | — | — | 90.45% |
| `v2_mixed_bs32` | 6,630 | **0.1461** | 0.1879 | 0.0414 | −0.147 | **92.34%** |
| `v2_triple_mixed_bs32` | 9,540 | — | — | — | — | 91.98% |

**Findings**: `v2_mixed_bs32` is the best variant (+1.17pp over `v2_nllb`, +1.89pp over `v2_qwen3`). The Round 1 ranking holds: mixed > triple > original ≈ NLLB > qwen3 on the held-out eval. `v2_mixed_bs32` is used as the baseline for all Round 3 ablations.

---

### 7.3 Round 3 — Component & LR Ablation on v2_mixed_bs32

Baseline: `voxtral_mcq_v2_mixed_bs32_4gpu` (train_124files_mixed.jsonl, bs=32, lr=5e-5, all components LoRA r=16/α=32, eval=test=eval_26files_challenge_repr.jsonl).

#### 7.3.0 Collar Sweep + Remap (submitted 2026-06-20)

**Motivation**: Audio crop analysis revealed two issues in the baseline:
1. `--remap-timestamps-after-crop` was missing → 51% of questions have timestamp refs (ranges like `382.659-383.997`, point refs like `at 16.546`) whose values are silently wrong after cropping. All sweep variants include remap.
2. `--crop-collar-seconds 30` may be too tight: median ref span is 4.7s, so collar=30 gives only 30s of conversational run-up around the referenced moment. Testing 15/30/45/60 to find the optimal context window.

**Data statistics** (train_124files_mixed.jsonl, 6630 questions):
- 51% questions have timestamp refs → timestamp-based crop path
- 49% questions without refs → fallback `--random-crop-seconds 300` (5 min)
- Clamped crop sizes at collar=30: median=65s, p90=81s, p95=109s, p99=143s
- At collar=60: median=125s, p90=138s, p95=166s, p99=255s (comfortably below 300s fallback)
- 14/3353 timestamp questions produce near-full-file crops (false-positive range parses) — no OOM risk due to rarity and clamping

| SLURM job | collar | remap | Δ vs baseline | Expected crop (median / p99) |
|---|---|---|---|---|
| 42175740 | 15s | ✓ | `--crop-collar-seconds 15 --remap-timestamps-after-crop` | 35s / 83s |
| 42175741 | 30s | ✓ | `--crop-collar-seconds 30 --remap-timestamps-after-crop` | 65s / 143s |
| 42175742 | 45s | ✓ | `--crop-collar-seconds 45 --remap-timestamps-after-crop` | 95s / 200s |
| 42175743 | 60s | ✓ | `--crop-collar-seconds 60 --remap-timestamps-after-crop` | 125s / 255s |

**Results**:

| Experiment | collar | remap | Best eval loss | Last eval loss | Last train loss | Gap | **Test acc** | Challenge |
|---|---|---|---|---|---|---|---|---|
| `v2_mixed_bs32` *(baseline, no remap)* | 30s | ✗ | 0.1461 | 0.1879 | 0.0414 | −0.147 | 92.34% | pending |
| `v2_mixed_bs32_collar15` | 15s | ✓ | 0.1339 | 0.1517 | 0.0581 | −0.094 | 92.16% | — |
| `v2_mixed_bs32_collar30` | 30s | ✓ | 0.1380 | 0.1685 | 0.0698 | −0.099 | **92.88%** | — |
| `v2_mixed_bs32_collar45` | 45s | ✓ | 0.1348 | 0.1858 | 0.0667 | −0.119 | 91.80% | — |
| `v2_mixed_bs32_collar60` | 60s | ✓ | 0.1434 | 0.1912 | 0.0514 | −0.140 | 91.62% | — |

**Findings**: collar=30s is optimal (92.88%, tied best overall). Shorter collar (15s) slightly underperforms; longer collars (45–60s) progressively hurt, likely due to irrelevant context diluting the attended audio region. Remap alone vs. baseline (both collar=30): 92.88% vs 92.34% (+0.54pp), suggesting timestamp remapping provides a small but consistent benefit.

#### 7.3.1 P1 Experiments (submitted 2026-06-20)

| SLURM job | Script | Δ vs baseline | Research question |
|---|---|---|---|
| 42173219 | `frz_encoder` | `--freeze-encoder` | Does frozen encoder preserve acoustic repr? Standard speech-LLM recipe |
| 42173220 | `frz_enc_conn` | `--freeze-encoder-connector` | LLM-only LoRA: is the gap a purely language-side problem? |
| 42177469 | `frz_enc_llm` | `--freeze-encoder --freeze-llm` | **Connector-only** training: isolates whether the encoder↔LLM bridge is the bottleneck |
| 42173221 | `lr_1e5` | `--learning-rate 1e-5` | Is 5e-5 too aggressive for 6.6k MCQ examples? |
| 42173222 | `frz_enc_lr2e5` | `--freeze-encoder --learning-rate 2e-5` | Frozen encoder + gentler LR on connector+LLM |

#### 7.3.1b P1v2 Experiments — +remap (submitted 2026-06-20)

Identical to P1 but adding `--remap-timestamps-after-crop`. Discovery: 51% of questions contain explicit timestamp refs (ranges like `382.659-383.997`, point refs like `at 16.546`); without remap these are silently wrong after cropping. P1 vs P1v2 comparison directly isolates the remap effect under each configuration.

| SLURM job | Script | Δ vs P1 counterpart |
|---|---|---|
| 42175023 | `frz_encoder_remap` | + `--remap-timestamps-after-crop` |
| 42175024 | `frz_enc_conn_remap` | + `--remap-timestamps-after-crop` |
| 42175025 | `lr_1e5_remap` | + `--remap-timestamps-after-crop` |
| 42175026 | `frz_enc_lr2e5_remap` | + `--remap-timestamps-after-crop` |

#### 7.3.1c P1v3 Experiments — +crop600 (submitted 2026-06-20)

Identical to P1 but with `--random-crop-seconds 600` (10 min) instead of 300 (5 min). Targets the 49% of questions without timestamp refs that currently get a random 5-min fallback crop — doubling the context window gives these questions access to more of the conversation without changing the timestamp-crop path.

| SLURM job | Script | Δ vs P1 counterpart |
|---|---|---|
| 42175066 | `frz_encoder_crop600` | `--random-crop-seconds 600` |
| 42175067 | `frz_enc_conn_crop600` | `--random-crop-seconds 600` |
| 42175068 | `lr_1e5_crop600` | `--random-crop-seconds 600` |
| 42175069 | `frz_enc_lr2e5_crop600` | `--random-crop-seconds 600` |

#### 7.3.2 P2 Experiments (pending P1 results)

| Script | Δ vs baseline | Research question |
|---|---|---|
| `triple_lr` | `--encoder-learning-rate 1e-5 --connector-learning-rate 5e-5 --llm-learning-rate 5e-5` | Low encoder LR, full LR on connector+LLM |
| `lora_r32` | `--lora-r 32 --lora-alpha 64` | More LoRA capacity for a harder task |

#### 7.3.3 P3 Experiments (pending P2 results)

| Script | Δ vs baseline | Research question |
|---|---|---|
| `frz_llm` | `--freeze-llm` | Acoustic-only adaptation (closes ablation triangle for paper) |
| `lr_2e5` | `--learning-rate 2e-5` | Fills LR grid between 1e-5 and 5e-5 |

#### 7.3.4 Results

| Experiment | remap | crop_s | Best eval loss | Last eval loss | Last train loss | Gap | **Test acc** | Challenge |
|---|---|---|---|---|---|---|---|---|
| `v2_mixed_bs32` *(baseline)* | ✗ | 300 | 0.1461 | 0.1879 | 0.0414 | −0.147 | 92.34% | pending |
| `frz_encoder` | ✗ | 300 | 0.1443 | 0.2052 | 0.0817 | −0.123 | 92.70% | — |
| `frz_enc_conn` | ✗ | 300 | — | — | — | — | **92.88%** | — |
| `frz_enc_llm` | ✗ | 300 | — | — | — | — | ⏳ running | — |
| `lr_1e5` | ✗ | 300 | — | — | — | — | 87.84% | — |
| `frz_enc_lr2e5` | ✗ | 300 | — | — | — | — | 91.17% | — |
| `frz_encoder_remap` | ✓ | 300 | 0.1431 | 0.1815 | 0.1207 | −0.061 | 91.53% | — |
| `frz_enc_conn_remap` | ✓ | 300 | 0.1521 | 0.2011 | 0.0486 | −0.153 | 91.08% | — |
| `lr_1e5_remap` | ✓ | 300 | — | — | — | — | 88.38% | — |
| `frz_enc_lr2e5_remap` | ✓ | 300 | 0.1499 | 0.1551 | 0.0874 | −0.068 | 90.09% | — |
| `frz_encoder_crop600` | ✗ | 600 | 0.1393 | 0.1975 | 0.0596 | −0.138 | 92.52% | — |
| `frz_enc_conn_crop600` | ✗ | 600 | — | — | — | — | ⏳ running | — |
| `lr_1e5_crop600` | ✗ | 600 | — | — | — | — | ⏳ running | — |
| `frz_enc_lr2e5_crop600` | ✗ | 600 | 0.1514 | 0.1719 | 0.0743 | −0.098 | 89.55% | — |

**Findings**:

1. **Freeze strategy**: `frz_enc_conn` (LLM-only LoRA) = 92.88% (tied best), `frz_encoder` = 92.70%, baseline = 92.34%. Freezing the encoder does not hurt — it slightly helps, likely by stabilising the audio representations. Connector-only (`frz_enc_llm`, running) is expected to perform worse since the LLM cannot adapt.

2. **Remap effect is negative** across all variants: remap consistently hurts vs. no-remap counterparts:
   - `frz_encoder` 92.70% → `frz_encoder_remap` 91.53% (−1.17pp)
   - `frz_enc_conn` 92.88% → `frz_enc_conn_remap` 91.08% (−1.80pp)
   This is counterintuitive given that 51% of questions reference timestamps. Possible explanation: remapping introduces small floating-point errors in the timestamp values, and the baseline model may have learned to ignore inaccurate timestamps rather than rely on them.

3. **Longer random crop (600s) helps mildly**: `frz_encoder_crop600` = 92.52% vs `frz_encoder` = 92.70% (−0.18pp, marginal). Doubling the fallback crop window from 5 to 10 minutes provides no benefit for frozen-encoder configs, possibly because the audio encoder attention already saturates on the relevant region.

4. **Learning rate 1e-5 is strongly suboptimal** (87.84%, −4.5pp). 5e-5 is the right scale for this task and dataset size.

---

### 7.4 Round 4 — Prompt Format Ablation (submitted 2026-06-20)

**Motivation**: All Round 2–3 experiments used a training format mismatch: `_encode_chat_sample` called `processor.apply_transcription_request(language="en")` which places the question *outside* `[INST]`:

```
<s>[INST][BEGIN_AUDIO][AUDIO]×N[/INST]lang:en
[TRANSCRIBE]
question → A
```

But evaluation always used `processor.apply_chat_template(conversation)` which places the question *inside* `[INST]`:

```
<s>[INST][BEGIN_AUDIO][AUDIO]×N question[/INST] → A
```

Three new formats were implemented and tested:

| Format flag | Train format | Eval format | Description |
|---|---|---|---|
| *(none — old)* | `apply_transcription_request` → question outside `[INST]` | `apply_chat_template` → question inside `[INST]` | **Mismatched** — all Round 2/3 experiments |
| `--use-chat-template-for-training` | `apply_chat_template` → question inside `[INST]` | same | **Matched** |
| `--use-transcription-hint-format` | `apply_chat_template` with `lang:en\n[TRANSCRIBE]\n` prefix inside `[INST]` | same | **Matched + ASR hint** |

Additionally, a pre-training variant merges a pre-trained ASR LoRA into the base model before applying MCQ LoRA (`--base-adapter-path`).

#### 7.4.1 Experiment summary

| SLURM job | Experiment | Flags | Base |
|---|---|---|---|
| 42178820 | `v2_mixed_chat_tmpl` | `--use-chat-template-for-training` | Voxtral base |
| 42180014 | `v2_mixed_trans_hint` | `--use-transcription-hint-format` | Voxtral base |
| 42181387 | `v2_mixed_asr_base` | `--use-transcription-hint-format --base-adapter-path experiments/mlc25_mlc26_continue_4gpu_fixed_bothsides_20260530_120641/final_model` | ASR-merged |

ASR LoRA used for `asr_base`: `mlc25_mlc26_continue_4gpu` (r=8, α=32, targets=[q,k,v,o]_proj), WER=12.06% on held-out ASR eval.

#### 7.4.2 Format comparison ablation (job 42179845)

Eval-only job applying both old (transcription-request) and new (chat_template) eval formats to the `v2_mixed_bs32` final model on the 1,110-question eval set. Output: `experiments/ablation_format_comparison/metrics.json`.

| Eval format | Accuracy (1,110q) | Note |
|---|---|---|
| `apply_chat_template` (eval format, always used) | **90.9%** | Question inside `[INST]`, model outputs A/B/C/D directly |
| `apply_transcription_request` (training format) | **37.5%** | Model generates transcript first; eval code misreads first non-ABCD token as answer |

**Interpretation**: With the transcription-request format at inference, the model generates a transcription of the audio before reaching the question+answer — so the first generated tokens are transcript words, not A/B/C/D. The eval extractor picks those up as wrong answers, yielding near-random performance (37.5% vs 25% chance for 4-class MCQ). This definitively confirms that the training format was unusable at inference time and that the model's 92.34% baseline came entirely from cross-format generalisation to `apply_chat_template`.

> Note: `eval_format_accuracy` here (90.9%) is slightly below the canonical `v2_mixed_bs32` result (92.34%) — likely a minor implementation difference in the comparison script.

#### 7.4.3 Zero-audio ablation

To quantify how much the model relies on audio vs. language priors, `input_features` were zeroed before `generate()` on the `v2_mixed_bs32` model:

| Condition | Test acc (1,110q) | Δ vs audio |
|---|---|---|
| With audio (standard) | 92.34% | — |
| Zero audio (text-only) | **81.71%** | −10.63pp |

The model uses audio for ~10.6pp of its accuracy. The remaining ~82% reflects language-model priors from question text alone.

#### 7.4.4 Results

| Experiment | Best eval loss | Last eval loss | Last train loss | Gap | Steps | **Test acc** |
|---|---|---|---|---|---|---|
| `v2_mixed_bs32` *(baseline, old format)* | 0.1461 | 0.1879 | 0.0414 | −0.147 | — | 92.34% |
| `v2_mixed_chat_tmpl` | **0.1439** | 0.1863 | 0.0497 | −0.136 | 275 (ES) | **92.97%** ✅ |
| `v2_mixed_trans_hint` | **0.1403** | 0.1609 | 0.0544 | −0.107 | 275 (ES) | **92.70%** ✅ |
| `v2_mixed_asr_base` | 0.1718 | — | — | — | 275 (ep 1.3) | ⏳ running |

> ES = early-stopped at step 275 (epoch 1.32, patience=3).

**Findings**:
- **`chat_tmpl` (92.97%) > `trans_hint` (92.70%)**: despite `trans_hint` having the lower eval loss during training (0.1403 vs 0.1439), it does not convert to higher accuracy. The transcription-hint prefix (`lang:en\n[TRANSCRIBE]\n` inside `[INST]`) does not help and may slightly confuse the model at inference.
- Both new-format models improve over the old-format baseline (92.34%), confirming the training format mismatch was the dominant bottleneck.
- **`asr_base`** still running at ep 1.3 with best_el=0.1718 — significantly worse than both format-corrected models at the same stage. Merging the ASR LoRA before MCQ training appears to hurt initialisation.
- **Format comparison ablation** (§7.4.2) shows the training (transcription-request) format yields only 37.5% at inference, while eval (chat_template) format gives 90.9% on the same model — conclusively proving the mismatch was the root cause of the training signal corruption.

---

### 7.5 Round 5 — English-Only Training Strategy (submitted 2026-06-20)

**Motivation**: Rounds 2–4 trained on mixed-language questions (NLLB cross-lang + original same-lang). The challenge eval however always presents questions in the audio's native language. A cleaner strategy: train exclusively with English questions on all audio (translate non-EN questions via NLLB), so that at inference time all challenge questions are also translated to English before being fed to the model. This removes the language mismatch and lets the LLM focus on audio reasoning in a single language.

#### 7.5.1 New Data Split (all 150 files)

Previous splits (124/26) mixed same-lang and cross-lang records in the eval set, making them incompatible with the English-only strategy. New split uses all 150 files with English-only questions throughout.

**Test set design**: 1 audio file per language group (21 groups: 5 EN varieties + 16 non-EN). Selected as the last file alphabetically within each group → no overlap with training.

| File | Records | Questions | Source |
|---|---|---|---|
| `train_129files_nllb.jsonl` | 129 audio files | 3,870 | NLLB-translated EN (from `mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl`) |
| `train_129files_qwen3.jsonl` | 129 audio files | 3,870 | Qwen3-translated EN |
| `train_129files_nllb_qwen3.jsonl` | 258 records (129×2) | 7,740 | NLLB + Qwen3 combined (EN audio appears twice, identical) |
| `eval_21files_en_only.jsonl` | 21 audio files | 630 | English questions only — 1 file/language, no train overlap |

> Note: `eval_21files_en_only.jsonl` (630q) is not comparable to prior experiments (1,110q eval_26files). Numbers cannot be compared across §7.2–§7.4 and §7.5.

#### 7.5.2 Experiments (submitted 2026-06-20)

Both experiments use best settings from Round 4 (`frz_enc_conn` + `chat_tmpl`), combined with the new English-only data. No remap (evidence from §7.3.4 shows remap consistently hurts). lr=5e-5, bs=32 (4GPU×1×grad_accum=8), collar=30s.

| SLURM job | Experiment | Train data | Eval data | Steps (est.) |
|---|---|---|---|---|
| 42186114 | `voxtral_mcq_v2_nllb_frz_enc_conn_chat_tmpl_4gpu` | `train_129files_nllb.jsonl` (3,870q) | `eval_21files_en_only.jsonl` (630q) | ~363 (3 ep) |
| 42186145 | `voxtral_mcq_v2_nllb_qwen3_frz_enc_conn_chat_tmpl_4gpu` | `train_129files_nllb_qwen3.jsonl` (7,740q) | `eval_21files_en_only.jsonl` (630q) | ~726 (3 ep) |

Flags both: `--freeze-encoder-connector --use-chat-template-for-training`

#### 7.5.3 Results

Actual experiment dirs (timestamped): `..._20260620_225426` (nllb) and `..._20260620_225643` (nllb+qwen3). A first aborted run (`_20260620_224237`) produced no metrics.

| Experiment | Best eval loss | Last eval loss | Steps | **Test acc (630q)** | **Challenge Ph1 (NLLB inf.)** | **Challenge Ph1 (orig. inf.)** |
|---|---|---|---|---|---|---|
| `nllb_frz_enc_conn_chat_tmpl` | **0.1370** | 0.1658 | 175 | **91.27%** | 0.7229 | **0.7344** |
| `nllb_qwen3_frz_enc_conn_chat_tmpl` | 0.1540 | 0.1574 | 125 | 89.84% | ⏳ pending | ⏳ pending |

> Eval losses are on `eval_21files_en_only.jsonl` (630q held-out). Not comparable to §7.1–§7.4 which used the organisers' eval sets.

**Observations**:
- `nllb` (NLLB-only train) outperforms `nllb_qwen3` (NLLB+Qwen3 combined) on both eval loss and test accuracy, consistent with Round 1 findings where adding more augmentation beyond NLLB hurts (diluted training signal per step).
- Both converge quickly (125–175 steps) and early-stop — the 21-file held-out eval is a stricter stopping criterion than the organisers' eval, resulting in fewer steps than Round 1 (575 steps).
- Challenge leaderboard: `nllb_frz_enc_conn_chat_tmpl` + **NLLB inference** = 0.7229; same model + **original-language inference** = **0.7344** (+1.15pp). Original inference is clearly better — NLLB question translation at test time hurts. Compared to Round 1 `nllb_bs16` + original inference = **0.7382** (best overall), Round 5 trails by 0.38pp despite freezing encoder-connector and using chat template, suggesting those choices did not help for the EN-only training regime.

---

### 7.6 Challenge Dataset Translation (2026-06-20)

To enable English-only inference on the challenge datasets, both Phase 1 and Phase 2 question JSONL files were translated to English using the same NLLB-200 pipeline as the training data.

**Challenge data format**: flat JSONL (one record per question): `session_id`, `question_id`, `question_stem`, `options`. Different from the nested training format — handled by new `asr_merging/translate_challenge_nllb.py` (imports shared logic from `translate_to_english_nllb.py`).

**Translation job**: SLURM 42186269 (1 GPU, fp32 to match training translations).

| Output file | Total records | Translated | Kept EN | EN rate after |
|---|---|---|---|---|
| `task2_phase1_questions_options_nllb_en.jsonl` | 4,985 | 1,141 (22.9%) | 3,844 | **99.6%** (1,137/1,141) |
| `task2_phase2_questions_options_nllb_en.jsonl` | 9,470 | 2,341 (24.7%) | 7,129 | **99.7%** (2,335/2,341) |

Remaining non-EN after translation (~4–6 questions each): Japanese questions where a quoted verbatim audio span dominates the text — the quoted part is correctly kept in Japanese (per quote-preservation logic), and the non-quoted question stem is translated. Detected as non-EN by langdetect due to the Japanese quote dominating.

Empty translated options (17 Phase 1 / 29 Phase 2): single-word quoted audio references like `「英語」` — entirely within quotes, correctly kept verbatim, flagged only because length < 5 chars.

**Translation precision note**: fp32 used for both training and challenge translations (the original `translate_to_english_nllb.py` used `dtype=torch.float16` which was silently ignored by `from_pretrained`, resulting in fp32). The `challenge_nllb` script explicitly loads without dtype override to match exactly.

#### 7.6.1 Challenge Eval Script

New script `deploy/run_voxtral_challenge_eval_phase_4gpu.sh` supports Phase 1/2 and original/NLLB selection via environment variables:

```bash
MODEL_ADAPTER_PATH=experiments/.../final_model \
PHASE=1 USE_NLLB=1 \
sbatch deploy/run_voxtral_challenge_eval_phase_4gpu.sh
```

Variables: `PHASE` (1|2), `USE_NLLB` (1|0), `EVAL_AUDIO_ROOT`, `PROMPT_LANGUAGE`.
Output: `experiments/<EXPERIMENT>/challenge_phase<N>[_nllb|_orig]_hyp.txt`

The script dynamically shards the selected JSONL into 4 parts at runtime (the old `run_voxtral_challenge_eval_4gpu.sh` used pre-built shards covering only Phase 1 original).

---

### 7.7 Round 6 — Best-of-all-ablations (submitted 2026-06-21)

**Hypothesis**: Combine every validated improvement from ablations (§7.7) and round experiments (§7.1–§7.5) in a single run on the largest available EN-only training set.

**Configuration** (job 42200867, script `deploy/run_voxtral_mcq_train_nllb150_r32a64_duallr_chat_tmpl_4gpu.sh`):

| Parameter | Value | Evidence |
|---|---|---|
| Train data | `mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl` (150 files, 4,500q) | Same as `nllb_bs16` (0.7382 challenge) |
| Eval/Test | `organisers_balanced_en_translated.jsonl` / `organisers_en_translated.jsonl` | Consistent with Round 1 (150-file train → org eval) |
| LoRA | r=32, α=64 | §7.7.1: best rank (+3pp over r=16/α=32) |
| LR enc+connector | 1e-5 | §7.7.2 A3: lower enc LR +1.7pp |
| LR LLM | 5e-5 | §7.7.2 A3: optimal LLM LR |
| Chat template | ✅ | §7.4.4: +0.63pp |
| Effective batch size | 32 (4GPU × 1 × grad_accum=8) | §7.1.3: bs32 > bs16 uniformly |
| collar | 30s | §7.3.0: optimal |
| random-crop | 300s | §7.3.1c: crop600 no benefit |
| remap | ❌ | §7.3.4: remap consistently hurts |
| freeze-encoder-connector | ❌ | §7.7.2 A3 (no freeze + diff LR) > §7.7.3 B1 (freeze); best challenge score (nllb_bs16) had no freeze |
| early-stopping patience | 5 | r=32 converges slower than r=16; r32_a64 ran 423 steps in ablation |

**Expected steps**: ~422 (4,500q / bs=32 × 3 epochs) — should train to near-full convergence as `nllb_bs32` did in Round 1.

#### 7.7.1 Results

| Experiment | Best eval loss | Last eval loss | Steps | Test acc (n=300) | **Challenge Ph1 (orig)** |
|---|---|---|---|---|---|
| `nllb150_r32a64_duallr_chat_tmpl` (42200867) | 0.0539 | 0.0539 | 400 | ⏳ | **0.7087** |

**Challenge result (2026-06-21)**: Original-language inference scored **0.7087** — down **−0.0257** from Round 5's 0.7344. Answer distribution analysis shows Round 6 has slightly higher A-bias overall (62.8% vs 60.5%) and lower C-rate (9.5% vs 12.0%); CJK/TL A-bias is essentially unchanged (83.0% vs 84.3%). The C-rate drop is the main regression driver. Both rounds show the same structural weakness: 80–84% A-predictions for CJK/Tagalog/Thai languages (model defaults to A when it cannot parse native-script MCQ text). This motivates CL Phase 2: the native-language (A) and full-mix (D/E) variants should restore CJK/TL competence.

---

### 7.8 Curriculum Learning Phase 2 — Experiment Design (2026-06-21)

**Motivation**: Phase 1 establishes an EN-only foundation (Round 6). Phase 2 fine-tunes from that checkpoint to inject multilingual signal without catastrophic forgetting. Nine dataset variants ablate: (1) signal type (same-language vs cross-language), (2) NLLB translation path quality (round-trip vs clean EN-origin), (3) EN replay amount, and (4) anti-memorization option shuffling.

#### 7.8.1 Round 6 Base (Phase 1 foundation)

| Field | Value |
|---|---|
| Job | 42200867 |
| Experiment dir | `experiments/voxtral_mcq_nllb150_r32a64_duallr_chat_tmpl_4gpu_20260621_091853/` |
| Final model | `…/final_model` (= checkpoint-400) |
| Best checkpoint | step 400, epoch 2.839, eval_loss = **0.05394** |
| Challenge eval job | 42203738 (Phase 1, original-language inference, USE_NLLB=0) |
| Config | r=32, α=64, enc_conn_lr=1e-5, llm_lr=5e-5, chat_tmpl, 3 epochs |

#### 7.8.2 CL Phase 2 Dataset Inventory

All CL datasets start from `final_model` above and run 1 epoch, halved LR (enc_conn=5e-6, llm=2e-5), r=32/α=64.

**Ingredients**:
- `EN`: `mlcslm_2nd_dev_qa_successed_opensource_en_translated.jsonl` — 150 records, 4,500q (all English questions)
- `NLLB×8`: `mlcslm_2nd_dev_qa_en_to_{por,fra,spa,rus,vie,tur,tgl,deu}.jsonl` — 150 records each, translated from EN (path: EN→lang); 113/150 records involve a round-trip for non-EN audio (native→EN→lang)
- `NLLB-clean×8`: same files but filtered to the **37 originally-EN audio records** — path is strictly EN→lang, no round-trip
- `native`: non-EN dominant records from `mlcslm_2nd_dev_qa_successed_opensource.jsonl` — 113 records, 3,390q, original human-written questions in native languages

**Original answer label distribution (EN data)**: A=34%, B=41%, C=23%, D=2% — severely skewed.  
**After per-question-per-language shuffling (NLLB records)**: ~25% per label — balanced and breaks positional memorisation.

| ID | Name | Dataset file | Records | Questions | EN% | Steps/ep | Signal types | NLLB path |
|---|---|---|---|---|---|---|---|---|
| Base | Round 6 | `en_translated` | 150 | 4,500 | 100% | ~141 | EN-only | — |
| **A** | native | `cl_native` | 263 | 7,890 | 57% | ~247 | same-lang | none |
| **B** | nllb_up | `cl_nllb_upsample` | 1,350 | 40,500 | 11% | ~1,265 | cross-lang | mixed (round-trips) |
| **C** | nllb_safe | `cl_nllb_safe` | 1,500 | 45,000 | 20% | ~1,406 | cross-lang | mixed + 2× EN |
| **D** | full_mix | `cl_full_mix` | 1,463 | 43,890 | 10% | ~1,371 | same + cross | mixed |
| **E** | full_mix_safe | `cl_full_mix_safe` | 1,613 | 48,390 | 19% | ~1,512 | same + cross | mixed + 2× EN |
| **F** | nllb_clean | `cl_nllb_clean` | 333 | 9,990 | 11% | ~312 | cross-lang | clean EN→lang only |
| **G** | nllb_clean_native | `cl_nllb_clean_native` | 446 | 13,380 | 8% | ~418 | same + cross | clean EN→lang only |
| **B_sh** | nllb_up_shuffled | `cl_nllb_up_shuffled` | 1,350 | 40,500 | 11% | ~1,265 | cross-lang | mixed, options shuffled |
| **E_sh** | fms_shuffled | `cl_full_mix_safe_shuffled` | 1,613 | 48,390 | 19% | ~1,512 | same + cross | mixed, options shuffled |

#### 7.8.3 Training Scripts

All scripts in `deploy/` accept `BASE_ADAPTER_PATH` env var. **Eval/test sets** (multilingual, original language):
- `--eval-jsonl data/mlc26_task2/organisers_balanced.jsonl` (EN=25%, PT=11%, FR=11%, JA=7%, …)
- `--test-jsonl data/mlc26_task2/organisers.jsonl`

> ⚠️ **Pilot batch (jobs 42203702–42203708, submitted 2026-06-21)** used `organisers_balanced_en_translated.jsonl` as eval — EN-only, not ideal for multilingual CL. Internal metrics are comparable across A–G and Round 6 base, but early stopping was triggered by EN-only signal. A second batch with corrected multilingual eval is running.

| Script | Opt |
|---|---|
| `run_voxtral_mcq_cl_native_finetune_4gpu.sh` | A |
| `run_voxtral_mcq_cl_nllb_upsample_finetune_4gpu.sh` | B |
| `run_voxtral_mcq_cl_nllb_safe_finetune_4gpu.sh` | C |
| `run_voxtral_mcq_cl_full_mix_finetune_4gpu.sh` | D |
| `run_voxtral_mcq_cl_full_mix_safe_finetune_4gpu.sh` | E |
| `run_voxtral_mcq_cl_nllb_clean_finetune_4gpu.sh` | F |
| `run_voxtral_mcq_cl_nllb_clean_native_finetune_4gpu.sh` | G |
| `run_voxtral_mcq_cl_nllb_up_shuffled_finetune_4gpu.sh` | B_sh |
| `run_voxtral_mcq_cl_full_mix_safe_shuffled_finetune_4gpu.sh` | E_sh |

**Launch block**:
```bash
BAP=experiments/voxtral_mcq_nllb150_r32a64_duallr_chat_tmpl_4gpu_20260621_091853/final_model
for s in deploy/run_voxtral_mcq_cl_{native,nllb_upsample,nllb_safe,full_mix,full_mix_safe,nllb_clean,nllb_clean_native,nllb_up_shuffled,full_mix_safe_shuffled}_finetune_4gpu.sh; do
  BASE_ADAPTER_PATH=$BAP sbatch $s
done
```

#### 7.8.4 Comparison Strategy

All experiments (Base + A–G + B_sh + E_sh) share the same eval/test sets → **eval_loss and test_acc are directly comparable**. Leaderboard score is the ultimate ground truth.

**Primary ablations**:

| Comparison | What it isolates |
|---|---|
| Base → A | same-language injection (no cross-lang) |
| Base → B | cross-language injection (EN audio + NLLB questions), with round-trips |
| Base → F | cross-language injection, clean EN-origin only (no round-trips) |
| B vs F | effect of round-trip NLLB translations (quality concern) |
| B vs C | EN replay amount (11% → 20%) for cross-lang forgetting |
| B vs D | same-lang signal on top of cross-lang |
| D vs E | EN replay amount for full-mix forgetting |
| D vs G | round-trip NLLB vs clean NLLB in the full-mix setting |
| B vs B_sh | option shuffling: breaks positional memorisation in cross-lang data |
| E vs E_sh | option shuffling: breaks positional memorisation in kitchen-sink data |
| B_sh vs E_sh | same-lang native signal, controlled for memorisation |

**Decision rule**: if any CL variant improves test_acc on `organisers.jsonl` vs Base → submit that checkpoint's challenge eval. If B_sh or E_sh beat B/E → shuffling helps, apply to all future CL experiments.

---

### 7.9 Preliminary Ablation Experiments (LoRA rank, LR, Freeze)

These experiments predate Rounds 2–5 and were conducted on the **Round 1 data** using the organisers' splits. Results are **not directly comparable** to Rounds 2–5 (different train/eval/test data), but provide clean ablation signals for hyperparameter selection.

**Important data context**:
- **Train**: `mlcslm_2nd_dev_qa_successed_opensource_balanced.jsonl` — balanced subset of the full 150-file organiser dataset (Round 1 data, NOT the NLLB-translated 129-file split used in Rounds 2–5)
- **Eval** (early stopping): `data/mlc26_task2/organisers_balanced.jsonl` — organisers' balanced eval set
- **Test** (reported accuracy): `data/mlc26_task2/organisers.jsonl` — organisers' full test set (n=300, all-language)
- **Architecture default**: LoRA r=16, α=32 (the `--lora-r`/`--lora-alpha` defaults in `voxtral_train_MCQ.py`). All A/B experiments rely on these defaults (no explicit LoRA flags in run commands).
- **No `--use-chat-template-for-training`**, no freeze flags (unless noted). Rounds 2–5 added chat template and freeze-encoder-connector.

#### 7.9.1 LoRA Rank/Alpha Ablation (`voxtral_mcq_lora_r*_a*`)

Sweeps LoRA `r` and `α` with all other settings fixed: lr=5e-5 (uniform, both components), grad-accum=8 (bs=8), fp16.

| Experiment | r | α | scale (α/r) | BestEvalL | TrainL (final) | Steps | **Test acc** |
|---|---|---|---|---|---|---|---|
| `lora_r8_a16` | 8 | 16 | 2.0 | 0.0838 | 0.1039 | 200 | 94.67% |
| `lora_r16_a16` | 16 | 16 | 1.0 | 0.0875 | 0.1094 | 200 | 93.00% |
| `lora_r16_a32` *(default)* | 16 | 32 | 2.0 | 0.0726 | 0.1144 | 200 | 95.00% |
| `lora_r16_a64` | 16 | 64 | 4.0 | 0.0774 | 0.0722 | 200 | 94.67% |
| **`lora_r32_a64`** | **32** | **64** | **2.0** | **0.0435** | **0.0034** | **423** | **97.00%** |

All stopped by early-stopping (patience=3, eval every 25 steps) except `r32_a64`, which trained to 423 steps — the same step count as `nllb_bs32` in Round 1 (grad-accum=8 on same dataset), indicating it converged fully rather than early-stopping. The dramatically lower train loss (0.0034 vs ≥0.07 for others) and lower best eval loss (0.0435) confirm higher-rank LoRA has substantially more fitting capacity.

**Key findings**:
- r=32, α=64 (scale=2.0) is clearly best (+2pp over r=16, α=32 at the same scale).
- Scale α/r=2.0 consistently outperforms scale=1.0 (r16_a16) at the same rank — effective LR for LoRA updates scales with α/r, so α=2r is a better initialisation than α=r.
- Doubling α further to scale=4.0 (r16_a64) does not help and slightly hurts, suggesting over-scaling destabilises training.
- Higher rank (r=32) gives more model capacity and allows full-epoch convergence where r=16 stalls at step 200.

#### 7.9.2 Learning Rate Ablation — Group A (`voxtral_mcq_lr_A*`)

Tests differential per-component LRs: `--encoder-connector-learning-rate` vs `--llm-learning-rate`. Uses default LoRA (r=16, α=32), grad-accum=4 (bs=4), fp16. Encoder-connector = audio encoder + cross-modal connector; LLM = Mistral decoder.

| Experiment | enc_conn LR | LLM LR | BestEvalL | Steps | **Test acc** |
|---|---|---|---|---|---|
| **`lr_A3`** | **1e-5** | **5e-5** | **0.0782** | **400** | **96.33%** |
| `lr_A1` | 5e-5 | 5e-5 *(equal)* | 0.0623 | 200 | 94.67% |
| `lr_A4` | 5e-5 | 1e-4 | 0.0697 | 400 | 94.00% |
| `lr_A2` | 1e-4 | 5e-5 | 0.0803 | 200 | 94.67% |
| `lr_A5` | 5e-5 | 1e-5 | 0.0800 | 400 | 92.00% |

**Key findings**:
- A3 (enc_conn=1e-5, llm=5e-5) is the best at **96.33%** — a 5× lower LR on the encoder-connector yields +1.7pp over uniform 5e-5 (A1). The audio encoder is already well pre-trained; aggressive updates destabilise its representations. Applying a gentler LR lets the LLM adapt while preserving encoder quality.
- A2 (enc_conn=1e-4): higher enc_conn LR does not help over 5e-5 (same 94.67%) — the encoder saturates quickly.
- A4 (llm=1e-4): raising LLM LR slightly hurts (94.00%) — standard 5e-5 is already near-optimal for the LLM.
- A5 (llm=1e-5): lowering LLM LR is the worst setting (92.00%) — the LLM needs sufficient LR to adapt to the MCQ format from audio context.
- A3 trained for 400 steps (vs 200 for A1/A2): lower enc_conn LR leads to slower convergence, requiring more steps — but the model avoids local minima that plague aggressive enc_conn updates.
- Notably, A3 (96.33%, default r=16/α=32) slightly outperforms the explicit `lora_r16_a32` run (95.00%) — the sole difference is the differential LR, confirming enc_conn LR reduction is a reliable gain.

#### 7.9.3 Freeze Strategy Ablation — Group B (`voxtral_mcq_lr_B*`)

Tests freezing major model components. Uses default LoRA (r=16, α=32), uniform lr=5e-5, grad-accum=4, fp16. The LoRA adapters are applied only to unfrozen components.

| Experiment | Frozen component | BestEvalL | Steps | **Test acc** |
|---|---|---|---|---|
| `lr_B1_freeze_enc` | encoder + connector | 0.0820 | 200 | **95.00%** |
| `lr_B2_freeze_llm` | LLM (Mistral decoder) | 0.1749 | 400 | **79.67%** ← catastrophic |

**Key findings**:
- Freezing the encoder+connector (B1, 95.00%) is nearly as good as training everything with uniform LR (A1, 94.67%), despite having fewer trainable parameters. This validates the `--freeze-encoder-connector` strategy used in Round 5 — the encoder/connector representations are transferable without fine-tuning.
- Freezing the LLM (B2, 79.67%) is catastrophic: a 15pp drop. The LLM must adapt to interpret audio-grounded MCQ context; without LLM fine-tuning, the model cannot learn the task format regardless of how well the connector is tuned.
- Comparison of B1 vs A3: B1 (freeze enc, 95.00%) vs A3 (differential LR enc_conn=1e-5, 96.33%). The differential LR approach is strictly better — it allows gentle encoder adaptation rather than complete freezing, gaining an additional 1.3pp.

#### 7.9.4 Cross-Ablation Summary and Implications for Future Experiments

| Strategy | Best config | Test acc | Current use in Rounds 2–5 |
|---|---|---|---|
| LoRA rank | r=32, α=64 | **97.00%** | ❌ Not yet (R2–5 all use r=16, α=32) |
| Differential LR | enc_conn=1e-5, llm=5e-5 | **96.33%** | ❌ Not yet (R2–5 use uniform LR) |
| Freeze strategy | freeze enc+connector | **95.00%** | ✅ Used in Round 5 |

The highest-value untested combination is **r=32, α=64 + differential LR (enc_conn=1e-5)**, which could compound both gains. Round 5 adds `--use-chat-template-for-training` on top (validated in §7.4), so the full optimal configuration would be: `r=32, α=64` + `enc_conn_lr=1e-5` + `llm_lr=5e-5` + `freeze-encoder-connector` + `chat_tmpl` + `train_129files_nllb.jsonl`.

---

## 8. Leaderboard Results and CL Experiments (2026-06-21)

### 8.1 Current Leaderboard Standing

| Round | Experiment | LB score | Notes |
|---|---|---|---|
| R6 | `voxtral_mcq_nllb_translated_4gpu_20260620_112000` | **0.7382** 🥇 | Best so far. 150-file NLLB-EN train, baseline LoRA r=16 |
| R7 | `voxtral_mcq_cl_bs16_a_ft_4gpu_20260621_123553` | 0.7287 | CL fine-tune on R6 base, native-only data, r=16/α=32 |
| R7 | `voxtral_mcq_cl_fms_shuf_ft_4gpu_20260621_113708` | 0.7073 | CL fine-tune on corr-base, full-mix-safe-shuffled, r=32/α=64 |
| R8 | `voxtral_mcq_v2_mixed_chat_tmpl_4gpu_20260620_190745` | 0.7344 | clean 124-file train, chat_tmpl, bs=32 — below R6_base |
| R9 | `voxtral_mcq_nllb129_r32a64_duallr_chat_tmpl_balanced_4gpu_20260622_191816` | 0.7235 | 129-file NLLB-EN, r=32/α=64, dual-lr, chat_tmpl, shuffled labels; A=46% B=34% C=12% D=8% |
| R9 | `voxtral_mcq_nllb150_r32a64_duallr_chat_tmpl_balanced_4gpu_20260622_191816` | ⏳ pending | 150-file NLLB-EN, r=32/α=64, dual-lr, chat_tmpl, shuffled labels; A=57% B=32% C=9% D=2% |

### 8.9 Round 9 — Analysis: n129 shuffled-labels result (2026-06-22)

**Configuration**: r=32/α=64, dual-lr (enc_conn=1e-5, llm=5e-5), chat_tmpl, 129-file NLLB-EN train, **option labels shuffled per question** (correct answer uniformly distributed A/B/C/D≈25% in training).

**Key observations**:

1. **First valid crop-at-eval submission**: n129/n150 are the first challenge submissions using `crop_from_question_refs=True` at eval time. All prior successful submissions (R6–R8) used full session audio. The 1.5pp gap to R6 has too many confounders to isolate the crop effect, but the result is essentially a wash — consistent with the §7.3.4 finding that cropping/remap does not reliably help on local eval either. The audio encoder appears to localise to the relevant region in full session audio without explicit cropping.

2. **Label shuffling is a null result, not a negative**: The 1.5pp gap to R6 (0.7235 vs 0.7382) cannot be attributed to shuffling alone — there are 5 confounding differences (fewer files, higher rank, dual-lr, chat_tmpl, bs=32). The shuffled-label prediction distribution (A=46%) closely matches the GT challenge distribution (A≈45%), confirming the model is doing genuine audio-content reasoning rather than defaulting to A. The gap to R6 is small and explainable by the 21 fewer training files alone.

3. **Label shuffling ≠ downsampling**: Options were randomly reordered per question (correct answer reassigned to a random letter). This breaks positional memorisation without changing data quantity or question difficulty. The model cannot learn "when uncertain, pick A" from training data where A=25%. At challenge time (GT A=45%), this removes the free positional prior — but the 1.5pp cost shows the prior is worth very little.

4. **n150 A-bias (57%) is a warning sign**: Despite identical training procedure to n129, n150 predicts A=57% — well above GT 45%. With only 21 extra training files, this suggests the 150-file distribution has more A-answer questions in those extra files, or the model overfit differently. Expected to score lower than n129 (label spread too skewed, D=2% is near-collapsed).

**Conclusions**:
- Audio cropping at eval adds complexity with no measurable gain → consider disabling `crop_from_question_refs` for future challenge eval runs
- Label shuffling is safe and does not hurt; whether it helps requires a controlled experiment (same data, only shuffling toggled)
- The R6 formula (150 files, r=16, unbalanced, original ordering) remains the strongest baseline; improvements must come from data quality or model capacity, not training tricks

### 8.2 Round 7 — Curriculum Learning (CL) Fine-tuning on R6 Base

**Motivation**: Fine-tune the best R6 model further on augmented multilingual data to improve non-English performance. Two batch families were tested.

#### 8.2.1 CL-bs16 batch (r=16/α=32, fine-tuned on R6 base = `voxtral_mcq_nllb_translated_4gpu_20260620_112000`)

| Variant | Dataset | Best eval_loss@step | Dev acc (contaminated) |
|---|---|---|---|
| **cl_bs16_a** ⭐ | native-only | 0.0549@450 | 96.0% |
| cl_bs16_b | nllb upsampled | — | 92.3% |
| cl_bs16_c | nllb safe | 0.1119@25 | 92.3% |
| cl_bs16_d | full mix | 0.0844@700 | RUNNING (resumed ckpt-750) |
| cl_bs16_e | full mix safe | 0.0866@300 | 93.7% |
| cl_bs16_f | nllb clean | 0.1026@125 | 92.3% |
| cl_bs16_g | nllb clean+nat | 0.0997@25 | 93.3% |

#### 8.2.2 CL-corr batch (r=32/α=64, fine-tuned on corr-base = `voxtral_mcq_nllb150_r32a64_duallr_chat_tmpl_4gpu_20260621_091853`)

| Variant | Dataset | Best eval_loss@step | Dev acc (contaminated) |
|---|---|---|---|
| cl_native | native | 0.0864@200 | 94.0% |
| cl_full_mix | full mix | 0.0453@400 | 95.0% |
| cl_full_mix_safe | full mix safe | — | 95.0% |
| cl_native_shuf | native shuf | 0.1093@247 | 93.0% |
| **cl_fms_shuf** ⭐ | full mix safe shuf | 0.0795@400 | 96.0% |
| cl_nllb_clean | nllb clean | 0.1205@75 | 94.0% |
| cl_nllb_clean_nat | nllb clean+nat | 0.0911@225 | 94.3% |
| cl_nllb_safe | nllb safe | 0.0995@150 | 95.7% |
| cl_nllb_safe_shuf | nllb safe shuf | 0.0937@200 | 94.7% |
| cl_nllb_up | nllb upsampled | 0.0918@125 | 94.0% |
| cl_nllb_up_shuf | nllb up shuf | 0.1031@175 | 93.7% |

#### 8.2.3 CL Findings and Root Cause Analysis

**Critical finding**: Both CL submissions **regressed** vs R6_base (0.7382) on the leaderboard despite showing 96% on the local dev set.

**Root cause: contaminated checkpoint selection.** The eval JSONL used for early stopping and checkpoint selection was `organisers_balanced.jsonl`, which contains all 150 audio files — the same 150 files used for training. Overlap with `eval_26files_challenge_repr.jsonl` (the intended held-out set): **26/26 files** (100%). The "best checkpoint" was selected by optimising loss on training data, causing silent overfitting.

**Label distribution analysis (challenge Phase 1 predictions):**

| Model | LB score | Non-EN A% | Non-EN B% | Non-EN C% | Non-EN D% |
|---|---|---|---|---|---|
| cl_bs16_a | 0.7287 | **57.9%** ⚠️ | 28.8% | 10.7% | **2.6%** |
| cl_fms_shuf | 0.7073 | 46.0% | 32.9% | 13.0% | 8.1% |

`cl_bs16_a` collapses to A=58%/D=3% on non-English audio — the model defaults to A when it cannot understand the audio, masking poor multilingual comprehension behind a spurious label bias. Despite the bias, `cl_bs16_a` outperforms `cl_fms_shuf` on the leaderboard, suggesting the challenge Phase 1 non-English questions have an above-average A-answer rate, or that `cl_bs16_a`'s English comprehension advantage dominates.

### 8.3 Contamination Analysis — Local Eval Sets

| Experiment group | Training files | Overlap with eval_26files (26 files) | Local eval valid? |
|---|---|---|---|
| R6_base, all CL experiments | 150-file variants | **26/26** ❌ | No |
| `voxtral_mcq_v2_*` (most) | `train_124files_*.jsonl` (124 files) | **0/26** ✅ | Yes |
| `voxtral_mcq_v2_nllb_frz_enc_conn_..._225*`, `v2_nllb_qwen3*` | `train_129files_*.jsonl` (129 files) | 26/26 ❌ | No |

**The only valid clean local eval is `eval_26files_challenge_repr.jsonl` (1,110 Q) for `voxtral_mcq_v2_*` models using `train_124files_*`.**

### 8.4 v2 Clean Eval Summary (eval_26files_challenge_repr.jsonl, 1110 Q, never seen in training)

| Config | Overall | English | Non-EN | Notes |
|---|---|---|---|---|
| `v2_mixed_chat_tmpl` | **93.0%** | 96.7% | 91.6% | best; chat_tmpl = correct challenge eval encoding |
| `v2_mixed_frz_enc_conn` (×2) | 92.9% | 96.3% | 91.6% | LLM-only LoRA |
| `v2_mixed_bs32_collar30` | 92.9% | 96.0% | 91.7% | best non-EN |
| `v2_mixed_frz_encoder` | 92.7% | 95.3% | 91.7% | |
| `v2_mixed_trans_hint` | 92.7% | **97.0%** | 91.1% | best English |
| `v2_mixed_bs32` (baseline) | 92.3% | 95.7% | 91.1% | |
| `v2_mixed_frz_enc_llm` (connector-only) | **81.4%** ❌ | 86.0% | 79.8% | worst — connector-only insufficient |

**Key insight**: `voxtral_forgetting_eval.py` (challenge scorer) **always** calls `processor.apply_chat_template()`. All non-chat_tmpl models have a train/eval encoding mismatch at challenge time. `v2_mixed_chat_tmpl` is the only model with correct alignment → best candidate for R8 submission.

### 8.5 Pending Actions

- [x] **R8**: `voxtral_mcq_v2_mixed_chat_tmpl_4gpu_20260620_190745` → **0.7344** (below R6_base 0.7382; clean eval + chat_tmpl insufficient to beat EN-only 150-file bs=16)
- [ ] **Monolingual**: 13 jobs (42216148–42216160) running; evaluate on leaderboard when done
- [ ] **cl_bs16_d**: Resuming from checkpoint-750 (job 42215472); evaluate when done
- [ ] **Next experiment direction**: retrain CL variants using `eval_26files_challenge_repr.jsonl` for checkpoint selection (requires restricting training to 124-file split to avoid contamination), with `--use-chat-template-for-training`

---

## 9. MCQ Model Input Format

### 9.1 Text Prompt

The MCQ text prompt is built by `_build_mcq_prompt()` in `voxtral_train_MCQ.py`:

```
Choose the most suitable answer from the options below.
You must respond with only one label from: A, B, C, D.

Question: {question_stem}

A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
```

### 9.2 Full Model Input (collator)

The collator (`_run_audio_instruction`, `_encode_chat_sample`) constructs a single-turn conversation and calls `processor.apply_chat_template()`:

```python
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "path": str(audio_source)},   # → audio tokens
            {"type": "text",  "text": prompt_text},          # → MCQ prompt
        ],
    },
    {
        "role": "assistant",
        "content": answer_label,                             # e.g. "B"
    },
]
```

This produces the token sequence:

```
<s>[INST] [BEGIN_AUDIO][AUDIO]×N {MCQ_prompt_text} [/INST] B </s>
```

- Audio tokens and the MCQ text are **both inside the `[INST]` block**, with audio tokens appearing first.
- At inference the assistant turn is omitted; the model autoregressively generates the answer label.
- The `--use-chat-template-for-training` flag (active from Round 4 onward) ensures training and inference use identical tokenisation.

### 9.3 Optional Transcription-Hint Prefix

When `--use-transcription-hint-format` is set, `prompt_text` is prefixed with `lang:en\n[TRANSCRIBE]\n` before being placed in the `[INST]` block:

```
<s>[INST] [BEGIN_AUDIO][AUDIO]×N lang:en
[TRANSCRIBE]
{MCQ_prompt_text} [/INST] B </s>
```

This was tested in Round 4 (`v2_mixed_trans_hint`, §7.4.4) and shown to be slightly worse than plain chat-template (92.70% vs 92.97%), so it is not used in subsequent rounds.

---

## 10. Round 10 — Transcript Augmentation, Generalization Gap, Mixed Data & Weighted Loss (2026-06-23)

### 10.1 New LB Best: nllb_transcript best_gen_gap step75

**Experiment**: `voxtral_mcq_nllb_transcript_4gpu_20260623_003920`  
**Config**: LoRA r=16, α=32, lr=5e-5, 4×H100-64GB, fp16, `--best-ckpt-strategy min_generalization_gap`, `--early-stopping-patience 0` (saves **all** LoRA checkpoints, ~100 MB each).  
**Train data**: `mlcslm_2nd_dev_qa_successed_opensource_mixed.jsonl` — 150 NLLB-translated + original CL records (same files used in rounds 1 and 6), **with Whisper transcripts injected** via `--transcript-dir data/transcripts/` (300 transcripts total, 93.7% clean).  
**Eval data**: `organisers_balanced_mixed.jsonl` (organisers' balanced eval set, mixed language).

| Challenge submission | Step | EVAL_CROP | A% | **LB score** |
|---|---|---|---|---|
| best_gen_gap, full audio | 75 | 0 | 63.5% | 0.7283 |
| **best_gen_gap, cropped** | **75** | **1** | **63.0%** | **0.7446** ← new best |
| final_model (overfit) | 525+ | — | — | not submitted |

**New leaderboard standing**:

| Round | Model | LB |
|---|---|---|
| R6 | nllb_bs16, 150-file, full audio | 0.7382 |
| R8 | v2_mixed_chat_tmpl | 0.7344 |
| R10 | **nllb_transcript best_gen_gap step75 crop** | **0.7446** 🥇 |
| R10 | nllb_transcript best_gen_gap step75 full | 0.7283 |
| R9 | nllb129 r32a64 duallr (overfit final_model) | 0.7235 |

**EVAL_CROP=1 gain (+0.0163 vs full audio)**: Consistent with the known 100% audio overlap between the 150-file train set and the challenge audio → cropping to question-referenced segments reduces "memorised full-session" advantage of overfit checkpoints while preserving relevant content for the best (generalizing) checkpoint.

---

### 10.2 Generalization Gap Analysis: Why Step 75 is the True Best

**Finding**: The `best_gen_gap` checkpoint strategy (`--best-ckpt-strategy min_generalization_gap`) selected step 75 as the checkpoint with the smallest `abs(eval_loss − avg_train_loss_since_last_eval)`.

**Key code detail** (voxtral_train_MCQ.py line ~872):
```python
gen_gap = abs(eval_loss - avg_train_loss_since_last_eval)
```

**Training curve interpretation**:
- Steps 1–75: model learns genuine audio-MCQ patterns, train ≈ eval (small abs gap)
- Steps 75–525: train loss keeps falling while eval loss plateaus or rises → large gen_gap → overfitting
- Step 525 (final): memorisation of training audio, near-zero train loss but high eval loss

**Contrast with nllb129 experiment (R9)**: that experiment used an older non-abs gen_gap formula (`gen_gap = eval_loss - avg_train_loss`, signed). Min of a signed gap picked step 25 (`gen_gap ≈ −3.26`) which is the earliest eval (large negative), not the generalizing checkpoint. The final_model (step 363, overfit) was submitted → scored 0.7235.

**Implication**: `--best-ckpt-strategy min_generalization_gap` requires `abs()` in the gen_gap formula (already fixed in current code). All future experiments use this strategy with `--early-stopping-patience 0` to keep every checkpoint.

---

### 10.3 Audio Overlap Discovery

**Finding**: All 149 unique audio files appear in **both** the 150-file training set and the challenge Phase 1 evaluation set → 100% train/challenge audio overlap.

**Impact on local eval accuracy**: local eval accuracy exceeds 90% on the contaminated eval sets (organisers_balanced_*.jsonl), because the model has seen all audio files during training. Local accuracy is not a reliable measure of generalisation.

**Impact on gen_gap strategy**: The gen_gap strategy remains useful even with 100% overlap because it computes `abs(eval_loss − avg_train_loss)` — a low gap at step 75 means the model is not yet memorising (train and eval losses are close), while a large gap at step 525 signals memorisation even though eval loss is computed on training audio.

**Fix for future experiments**: Created a **128-file non-overlap split**:
- Train: 128 audio files (149 unique - 21 held-out = 128; removed `0318_001_phone` which appeared in both candidate sets)
- Eval: 21 audio files (0 audio-file overlap with training)

---

### 10.4 Label Imbalance Analysis

**Finding**: The original cross-lingual data has a severe answer-label imbalance:

| Label | Original 150-file mixed | Challenge GT (Phase 1) |
|---|---|---|
| A | ~34% | ~25% |
| B | ~41% | ~25% |
| C | ~23% | ~25% |
| **D** | **~2%** | **~25%** |

D is almost absent (2.3%) in the original organiser questions + NLLB mixed data. The balanced organiser file (`mlcslm_2nd_dev_qa_successed_opensource_balanced.jsonl`) has perfect 25% per label.

**Fix**: Balanced JSONL files use the **balanced organiser file** (25% each) as the NLLB source (replacing the unbalanced source), combined with the full original CL data as-is. After balancing, D rises from 2% → 12–14% in training data.

---

### 10.5 Weighted Loss Implementation

**Motivation**: Different question types have different difficulty/value signals:
- `en/en` (English audio, English question): easy baseline, weight=1
- `nllb non-en/en` (NLLB-translated questions for non-EN audio): cross-lingual signal, weight=2 (emphasised — this is what the challenge tests)
- `orig CL non-en/non-en` (original organiser questions in native language): same-language signal, weight=3 (highest emphasis — rare in training, common in challenge)

**Implementation in `voxtral_train_MCQ.py`**:
1. `load_jsonl_audio_mcq`: reads `"sample_weight"` field from each JSONL row (`float`, default=1.0)
2. `_samples_to_hf_dataset`: propagates `"sample_weight"` as a top-level HF dataset column
3. `VoxtralMCQCollator.__call__`: packs weights into batch dict as `"sample_weight"` tensor
4. New `WeightedLossTrainer(Trainer)` class: overrides `compute_loss` to pop `sample_weight` from inputs and scale loss: `loss = loss * weights.mean()`
5. CLI arg `--weighted-loss` (store_true): selects `WeightedLossTrainer` instead of base `Trainer`
6. Trainer selection: `TripleLR > DualLR > WeightedLoss > Trainer`

**JSONL weight assignment** (per entry):
- `sample_weight=1.0`: EN-audio records (original English questions)
- `sample_weight=2.0`: NLLB-translated non-EN-audio records (cross-lingual)
- `sample_weight=3.0`: original non-EN-audio records in native language (same-language)

---

### 10.6 Round 10 Experiment Grid (submitted 2026-06-23)

All 8 experiments: LoRA r=16, α=32, lr=5e-5, 2 epochs, `--best-ckpt-strategy min_generalization_gap`, `--early-stopping-patience 0` (keeps all checkpoints), `--transcript-dir data/transcripts/`, EVAL_CROP=1 for challenge evals.

#### 10.6.1 Unbalanced variants (Jobs 42329768–42329771)

| Job | Script | Train JSONL | Records | Questions | Weighted | Audio overlap | Status |
|---|---|---|---|---|---|---|---|
| 42329768 | `mixed_transcript` | `mlcslm_2nd_dev_qa_successed_opensource_mixed.jsonl` | 300 | 7,890 Q | No | Full (150) | PENDING |
| 42329769 | `128mixed_transcript` | `train_128files_mixed_balanced.jsonl` | 256 | 7,680 Q | No | **None** | PENDING |
| 42329770 | `mixed_weighted` | `mlcslm_mixed_weighted_150.jsonl` | 300 | 9,000 Q | **Yes** (1/2/3) | Full (150) | PENDING |
| 42329771 | `128mixed_weighted` | `train_128files_mixed_weighted.jsonl` | 256 | 7,680 Q | **Yes** (1/2/3) | **None** | PENDING |

> `mlcslm_mixed_weighted_150.jsonl`: 150 NLLB-translated (w=2) + 150 original CL (w=3), 9,000 Q total.  
> `train_128files_mixed_balanced.jsonl`: 128 non-overlap audio files, mixed NLLB+orig, 7,680 Q.

#### 10.6.2 Balanced variants (Jobs 42330209–42330212)

Same as unbalanced variants but using **balanced NLLB source** (`mlcslm_2nd_dev_qa_successed_opensource_balanced.jsonl`, 25% per label) instead of the unbalanced organiser file. Original CL data kept as-is (100% non-EN audio, native-language questions, naturally skewed labels).

| Job | Script | Train JSONL | Records | Questions | D% | Weighted | Audio overlap | Status |
|---|---|---|---|---|---|---|---|---|
| 42330209 | `mixed_bal_transcript` | `train_150_balanced_mixed.jsonl` | 300 | 9,000 Q | **14%** | No | Full (150) | PENDING |
| 42330210 | `128mixed_bal_transcript` | `train_128_balanced_mixed.jsonl` | 256 | 7,680 Q | **12%** | No | **None** | PENDING |
| 42330211 | `mixed_bal_weighted` | `train_150_balanced_mixed_weighted.jsonl` | 300 | 9,000 Q | **14%** | **Yes** (1/2/3) | Full (150) | PENDING |
| 42330212 | `128mixed_bal_weighted` | `train_128_balanced_mixed_weighted.jsonl` | 256 | 7,680 Q | **12%** | **Yes** (1/2/3) | **None** | PENDING |

#### 10.6.3 Ablation design

| Comparison | Isolates |
|---|---|
| mixed vs 128mixed (same weighting) | Effect of audio overlap (100% vs 0%) on gen_gap checkpoint quality |
| unweighted vs weighted (same data) | Effect of sample_weight=1/2/3 loss scaling |
| unbalanced vs balanced (same data shape) | Effect of D-label imbalance (2% → 12–14%) |
| mixed_bal_weighted vs all others | Best-of-all: no-overlap + balanced + weighted |

#### 10.6.4 JSONL file inventory (created 2026-06-23)

| File | Records | Questions | Label dist. | Weights |
|---|---|---|---|---|
| `data/mlc26_task2/train_128files_mixed_balanced.jsonl` | 256 | 7,680 | A31/B34/C23/D12 | uniform (1.0) |
| `data/mlc26_task2/eval_21files_mixed_balanced.jsonl` | 42 | 1,260 | — | — |
| `data/mlc26_task2/mlcslm_mixed_weighted_150.jsonl` | 300 | 9,000 | A34/B41/C23/D2 (unbalncd) | 2.0/3.0 per pair type |
| `data/mlc26_task2/train_128files_mixed_weighted.jsonl` | 256 | 7,680 | — | 2.0/3.0 per pair type |
| `data/mlc26_task2/eval_21files_mixed_weighted.jsonl` | 42 | 1,260 | — | 1.0 (eval) |
| `data/mlc26_task2/train_150_balanced_mixed.jsonl` | 300 | 9,000 | A30/B33/C24/D**14** | uniform |
| `data/mlc26_task2/train_128_balanced_mixed.jsonl` | 256 | 7,680 | A31/B34/C23/D**12** | uniform |
| `data/mlc26_task2/train_150_balanced_mixed_weighted.jsonl` | 300 | 9,000 | A30/B33/C24/D**14** | 2.0/3.0 per pair type |
| `data/mlc26_task2/train_128_balanced_mixed_weighted.jsonl` | 256 | 7,680 | A31/B34/C23/D**12** | 2.0/3.0 per pair type |

---

### 10.7 Challenge Eval Process for Round 10 Models

When any Round 10 job finishes:

```bash
# Identify best_gen_gap checkpoint (saved by trainer automatically)
MODEL_ADAPTER_PATH=experiments/EXPERIMENT_NAME/best_gen_gap_checkpoint \
  EVAL_CROP=1 \
  sbatch deploy/run_voxtral_challenge_eval_4gpu.sh
```

Use EVAL_CROP=1 consistently (gained +0.0163 LB vs full audio for the nllb_transcript model).  
Reference best: `experiments/voxtral_mcq_nllb_transcript_4gpu_20260623_003920/best_gen_gap_checkpoint/` → LB=0.7446.
