# ASR-merging

Research codebase for multilingual ASR fine-tuning, model merging, and evaluation,
developed for the **MLC-SLM 2026 challenge** at Telefónica Scientific Research.

The main backbone is **Voxtral-Mini-3B** (Mistral AI), fine-tuned with LoRA for
multilingual speech recognition and multiple-choice question answering (MCQ) on
11-language challenge data.

---

## Repository structure

```
asr_merging/          Core Python package
  voxtral_train_MCQ.py        MCQ fine-tuning with LoRA, transcript hints, audio-focus prompts
  voxtral_train_router.py     Routing-based fine-tuning (language/domain router)
  voxtral_eval_router.py      Evaluation for router models
  voxtral_forgetting_eval.py  Catastrophic forgetting evaluation across training phases
  transcribe_sessions.py      Batch session transcription utility
  translate_*.py              Translation utilities (NLLB-200, Qwen3)
  eval_zero_audio.py          Zero-audio ablation evaluation
  scripts/                    Data preparation utilities (split_challenge_jsonl, etc.)

configuration/        JSON run configs for training and evaluation experiments
  mlc26_train_eval_*.json     MLC 2026 train+eval configurations
  mlc26_eval_*.json           MLC 2026 evaluation-only configurations
  voxtral_eval_*.json         Voxtral-specific eval configs (MLC25/MLC26 dev/test)

deploy/               SLURM launch scripts for all experiments
  run_voxtral_mcq_*.sh        MCQ fine-tuning runs (language sweeps, LoRA sweeps, data ablations)
  run_voxtral_router_*.sh     Router training and evaluation runs
  run_voxtral_challenge_*.sh  Challenge submission eval runs
  run_voxtral_transcribe_*.sh Batch transcription runs
  run_translate_*.sh          Translation pipeline runs
  submit_voxtral_*.sh         Multi-job submission scripts (phase 2 experiments)

analysis/             Post-hoc analysis and plotting scripts
  analyze_*.py                Training dynamics and debias analysis
  plot_*.py                   Class distribution, ICL, and per-epoch result plots
  apply_calibration_to_challenge.py  Calibration post-processing for challenge submission

notebooks/            Exploratory notebooks for dataset building and model prototyping
```

---

## Setup (HPC / Singularity)

This project runs on SLURM-managed GPU clusters via Singularity containers.
See `requirements-hpc.txt` for the pinned HPC environment.

```bash
pip install -e .
# or on HPC nodes:
pip install -r requirements-hpc.txt
```

---

## Training

### MCQ fine-tuning (main track)

```bash
# Example: balanced mixed data, 4 GPUs
bash deploy/run_voxtral_mcq_train_balanced_4gpu.sh

# LoRA sweep
bash deploy/run_voxtral_mcq_lora_sweep.sh
```

All MCQ run scripts map to a JSON config in `configuration/`. Key hyperparameters
(LoRA rank, learning rate, freeze strategy, transcript hints, data mix) are set
there.

### Router training

```bash
bash deploy/run_voxtral_router_train_mlc26_continue_4gpu.sh
bash deploy/run_voxtral_router_train_mlc26_full_train_continuation_4gpu.sh
```

See [asr_merging/README_voxtral_train_router.md](asr_merging/README_voxtral_train_router.md)
for full details on data layout and config options.

---

## Evaluation

```bash
# Challenge eval (4 GPUs, latest checkpoint)
bash deploy/run_voxtral_challenge_eval_4gpu.sh

# Forgetting evaluation across phases
bash deploy/run_voxtral_train_balanced_debias_eval_4gpu.sh

# MLC26 dev set
bash deploy/run_voxtral_eval_mlc26_new_test.sh
```

---

## Data utilities

```bash
# Transcribe all sessions (4 GPUs)
bash deploy/run_voxtral_transcribe_all_4gpu.sh

# Translate challenge data (NLLB-200)
bash deploy/run_translate_challenge_nllb_1gpu.sh

# Translate to English (Qwen3)
bash deploy/run_translate_training_qwen3_1gpu.sh
```

---

## Model guides

- [Voxtral router guide](asr_merging/README_voxtral_train_router.md)
- [Seamless M4T router guide](asr_merging/README_seamless_train_router.md)
- [Whisper-Turbo router guide](asr_merging/README_whisper_turbo_train_router.md)
