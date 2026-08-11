# ASR-merging

Research codebase for multilingual ASR fine-tuning and multiple-choice spoken
question answering (MCQA), developed for the **MLC-SLM 2026 challenge**
(Interspeech 2026) by the **Eloquence team** (Telefónica Scientific Research,
Fondazione Bruno Kessler, CNR). The Eloquence team placed **5th** in the final
ranking.

The task (Task 2) requires selecting the correct answer (A–D) for questions
grounded in multilingual spoken conversations spanning 21 languages.

---

## MLC-SLM 2026 Challenge: Submitted Systems

Three independent systems were developed and submitted. The paper describing all
three is in `tex/mlc-challenge/main.tex` (also published at Interspeech 2026).

### System 1 — Fine-tuned Voxtral-Mini-3B (LoRA + data augmentation)
**Best Phase 1 accuracy: 0.7446** | Phase 2: 0.70

The fine-tuning pipeline has four components:

**1a. Cross-lingual data augmentation** (`translate_*.py`, `deploy/run_translate_*.sh`)

The MLC26 evaluation set contains ~52% cross-lingual question pairs (non-English
audio, English question), while training data has 0% coverage. All non-English
training questions are translated to English via **NLLB-200-distilled-1.3B**,
masking audio quotations before translation. A *mixed* dataset concatenating
original and translated copies of each non-English audio file best approximates
the challenge distribution (original 0% cross-lingual → mixed 41.9% → target
52%):

```bash
bash deploy/run_translate_challenge_nllb_1gpu.sh   # translate challenge questions
bash deploy/run_translate_training_nllb_1gpu.sh    # translate training set
bash deploy/run_translate_training_qwen3_1gpu.sh   # alternative: Qwen3 translator
```

**1b. ASR transcript generation** (`transcribe_sessions.py`, `deploy/run_voxtral_transcribe_*.sh`)

A dedicated Voxtral-Mini-3B LoRA ASR model is trained in two sequential stages:

- **Seed model**: trained on MLC25 data (~29k utterances, 11 languages, 5 epochs,
  lr=5e-5). WER: 11.8% overall on MLC25 test set. Unseen MLC26 languages fail
  badly (e.g. Urdu 140.1%, Turkish 88.8%).
- **Adapted model**: continued from Seed on the full MLC25+MLC26 corpus (~1.93M
  utterances, 17 languages, 4×H100). The 6 new MLC26 languages are oversampled
  1.5× for the first 25% of training steps (2-stage adaptive schedule). MLC26
  overall WER drops from 26.2% to 15.2% (e.g. Urdu −127.1 pp, Turkish −67.6 pp,
  Tagalog −29.1 pp). All prompts use English-mode decoding (`language_dropout=1.0`).

Audio sessions are split into 1-minute chunks with 2-second overlap. Hallucination
loops (5-gram repetitions in a 150-word window) are detected and the transcript
truncated at the onset. 93.7% of challenge sessions pass quality filtering.

```bash
bash deploy/run_voxtral_transcribe_all_4gpu.sh       # transcribe all sessions
bash deploy/run_voxtral_transcribe_retry_1min_4gpu.sh # retry failed sessions
```

**1c. MCQ fine-tuning** (`voxtral_train_MCQ.py`, `deploy/run_voxtral_mcq_*.sh`)

Voxtral-Mini-3B-2507 is fine-tuned with LoRA (r=16, α=32, all linear modules)
on 4×H100-64GB, effective batch 32, lr=5×10⁻⁵ with linear decay:

- Each instance is a single-turn conversation: audio + MCQ text in the user turn;
  the model learns to emit a single letter (A–D) as the assistant turn.
- ASR transcripts are prepended as `"Transcript: {text}"` to provide a textual anchor.
- ~51% of questions contain explicit timestamp references; audio is **cropped to
  ±30s** around detected timestamps to focus the encoder on the relevant segment.
- **Generalisation-gap checkpoint selection**: since all 150 training audio files
  appear in the challenge evaluation set, extended training causes session
  memorisation. Every LoRA checkpoint (~100MB, every 25 gradient steps) is saved
  and the one minimising `|loss_eval − mean_loss_train|` is selected (step 75
  of ~525 total steps).

Key training ablation (Phase 1, 4,985 questions):

| Data mix | Transcript | Crop | Acc. |
|---|---|---|---|
| NLLB-only | No | No | 0.7382 |
| Mixed | Yes | No | 0.7283 |
| Mixed | Yes | 30s | **0.7446** |

```bash
bash deploy/run_voxtral_mcq_train_mixed_bal_transcript_4gpu.sh
bash deploy/run_voxtral_mcq_train_bs64_mixed_bal_trx_4gpu.sh
bash deploy/run_voxtral_mcq_lora_r16_a32_4gpu.sh  # LoRA rank/alpha sweep
```

**1d. Label-bias mitigation** (`analysis/analyze_debias_detail.py`, `analysis/apply_calibration_to_challenge.py`)

A systematic class-A prediction bias is present across all fine-tuned 3B variants
(~57% class A, <2% class D; reference distribution: 43%/6%). The bias comes from
skewed training answer labels (class A: ~34%, class D: ~2%). Two strategies are
evaluated:

- **Cyclic debias**: rotates option content across all label slots (N rotations),
  accumulates probabilities back onto content, picks the highest-mean content.
- **Prior calibration (CBU)**: subtracts null-prompt log-probabilities to remove
  positional priors (2 forward passes per question).

For the fine-tuned 3B model the bias is deeper than a positional preference and
neither fully compensates; balanced answer-label sampling during training is the
primary open fix.

---

### System 2 — Frozen Voxtral-24B + In-Context Learning
**Best Phase 2 accuracy: 0.8095** (best result overall)

A systematic class-A bias (63% class A, 2% class D) is present even in the
out-of-the-box 24B model. Prepending solved MCQ examples before the target
question (few-shot ICL) corrects the bias with zero parameter updates:

- **ICL-2 text**: 2 text-only shots → class A shifts from 63% to 44%, +5.1pp accuracy.
- **ICL-6 multimodal**: 6 shots each with a 30s audio clip, audio-focus prefix,
  MCQ text, and bare answer letter → +1.5pp additional, **0.8095**.
- For 4-option questions, shots cover one instance of each label (A,B,C,D) drawn
  from held-out audio. For 2-option questions, 2 shots are used.
- **Prior calibration on top of ICL-2** reaches the same accuracy as plain ICL-2
  (~0.795), confirming ICL alone corrects the positional bias for the 24B model.

The ICL logic is implemented inside `voxtral_train_MCQ.py` (inference path) and
controlled via the `--icl-shots` and `--icl-multimodal` flags. Launch scripts:

```bash
bash deploy/run_voxtral_mcq_nllb_icl_crop30_4gpu.sh               # ICL-6 mm, no freeze
bash deploy/run_voxtral_mcq_nllb_icl_crop30_frzenc_only_4gpu.sh   # ICL-6 mm, frozen encoder
```

---

### System 3 — Training-Free Voice-Anchored Retrieval
**Phase 2 accuracy: 0.68** (no training on challenge corpus)

The third system treats Task 2 as a **retrieval problem** using a persistent
three-layer voice-anchored memory. The architecture is implemented across
`voxtral_train_router.py` and `voxtral_eval_router.py` (enrollment + query path):

- **L_ac – Acoustic identity**: 192-d TitaNet-Large speaker embeddings stored in
  ChromaDB. Supports top-k cosine speaker identification across segments.
- **L_sem – Semantic content**: each utterance is transcribed and tagged (language,
  gender, age, acoustic emotion, textual emotion) in a single Voxtral call; text
  is encoded with all-MiniLM-L6-v2 and stored under the same UUID as L_ac.
- **L_kg – Knowledge graph**: NetworkX directed graph with structural edges
  (SAID, CONTAINS, HAS_PARTICIPANT, KNOWS) added at ingestion without LLM calls,
  and semantic edges (MENTIONS, typed FACT with valid-from/valid-to timestamps)
  extracted by a local Qwen3-14B-AWQ served via vLLM. Facts that contradict
  existing ones close prior validity intervals rather than overwriting them.

At query time a lightweight question classifier selects one of three reading modes
(profile / dialogue / entity) from L_kg, the retrieval layer assembles a **fact
sheet** (speaker identity + relevant utterances + graph context + emotion), and a
frozen Qwen3-14B-AWQ emits the single option letter. Sessions are enrolled once
and all questions answered without re-processing audio.

```bash
bash deploy/run_voxtral_router_train_mlc26_continue_4gpu.sh
bash deploy/run_voxtral_router_train_mlc26_full_train_continuation_4gpu.sh
bash deploy/run_voxtral_challenge_eval_4gpu.sh
```

---

## ASR system summary

| Stage | Data | WER (MLC25 test) | WER (MLC26 dev) |
|---|---|---|---|
| Seed | MLC25 only | 11.8% | 26.2% |
| Adapted | MLC25 + MLC26 | 12.0% | **15.2%** |

Largest MLC26 improvements from Seed → Adapted: Urdu −127pp, Turkish −68pp,
Tagalog −29pp, French-Canadian −16pp, Brazilian Portuguese −8pp.

---

## Repository structure

```
asr_merging/
  voxtral_train_MCQ.py        Systems 1 & 2: MCQ fine-tuning and ICL inference
  voxtral_train_router.py     System 3: voice-anchored memory enrollment and retrieval
  voxtral_eval_router.py      System 3: evaluation / answer generation
  voxtral_forgetting_eval.py  Multi-phase forgetting evaluation across training stages
  transcribe_sessions.py      ASR transcript generation (System 1b)
  translate_challenge_nllb.py NLLB-200 translation of challenge questions (System 1a)
  translate_to_english_nllb.py / translate_from_english_nllb.py  Training-set translation
  translate_to_english_qwen3.py  Alternative Qwen3-based translator
  eval_zero_audio.py          Zero-audio ablation evaluation
  scripts/split_challenge_jsonl.py  Challenge data preparation

configuration/
  mlc26_train_eval_*.json     Train+eval configs (fine-tuning System 1)
  mlc26_eval_*.json           Eval-only configs (all systems on dev/test sets)
  voxtral_eval_*.json         Voxtral-specific eval configs (MLC25/MLC26)

deploy/
  run_voxtral_mcq_*.sh        System 1 & 2 MCQ runs (data mix, LoRA, ICL sweeps)
  run_voxtral_router_*.sh     System 3 router train/eval runs
  run_voxtral_challenge_*.sh  Challenge submission eval runs
  run_voxtral_transcribe_*.sh ASR transcript generation runs
  run_translate_*.sh          NLLB / Qwen3 translation pipeline runs
  submit_voxtral_*.sh         Phase 2 multi-job sweep submissions

analysis/
  analyze_debias_detail.py             Label-bias analysis
  analyze_train_balanced_debias.py     Balanced-training debias study
  apply_calibration_to_challenge.py    CBU / cyclic-debias post-processing
  plot_*.py                            Class distribution, ICL, sweep result plots
```

---

## Setup (HPC / Singularity)

This project runs on SLURM-managed GPU clusters via Singularity containers.

```bash
pip install -e .
# or on HPC nodes:
pip install -r requirements-hpc.txt
```

---

## Model guides

- [Voxtral router guide](asr_merging/README_voxtral_train_router.md)
- [Seamless M4T router guide](asr_merging/README_seamless_train_router.md)
- [Whisper-Turbo router guide](asr_merging/README_whisper_turbo_train_router.md)
