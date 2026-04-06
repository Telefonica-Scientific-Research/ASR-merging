# Voxtral Training Router

This guide explains how to use `asr_merging/voxtral_train_router.py`.

The script is designed from the notebook workflow in `notebooks/voxtral-FT-QLORA.ipynb` and keeps the same ideas:
- canonical Voxtral dataset schema (`audio` + `text`)
- cache-first dataset recovery
- VAD/config controls in one place
- train/evaluation routing by CLI arguments
- baseline vs adapter model modes
- experiment folder/config snapshot per run
- optional TensorBoard tracking

## 1) Run location

Run commands from the project root:

```bash
cd /home/id02619@hi.inet/repos/eloquence/eloquence/WP4/ASR-merging
```

## 2) Main command pattern

```bash
python -m asr_merging.voxtral_train_router \
  [--config-json <path>] \
  --source <vc|mlc|openslr> \
  [--language <sr|mt|da>] \
  [--do-train] [--do-eval] \
  [--model-mode <baseline|adapter>] \
  [--adapter-path <path>] \
  [--train-set <name>] [--valid-set <name>] [--evaluation-set <name>] \
  [--timestamped-exp-dir] [--experiment-name <name>] [--output-root <dir>] \
  [--output-dir <dir>] [--tf-tracking]
```

Evaluation behavior:
- If `--do-train` is not passed, the script runs evaluation by default on `--evaluation-set`.
- `--test-set` is still accepted as a backward-compatible alias of `--evaluation-set`.
- If `--config-json` is provided, values are loaded from JSON first and then overridden by explicit CLI arguments.

JSON templates:
- Root folder: `configuriation/`
- Provided examples:
  - `configuriation/mlc_train_eval_baseline.json`
  - `configuriation/mlc_eval_only_adapter.json`

## 3) Dataset sources and split names

This section describes both the CLI names and the notebook-origin dataset objects/caches.

| Source | Notebook section | Canonical cache root | Typical split names |
|---|---|---|---|
| VC (`sr`,`mt`,`da`) | `Common Voice Serbian/Maltese/Danish` + `Save/Load Processed Datasets (canonical Voxtral-first)` | `data/cache/processed_cv_datasets/voxtral_prompt_aligned/<lang>/` | `train`, `valid`, `test` |
| OpenSLR108 (ES) | `OpenSLR108` + cache recovery cells | `data/cache/processed_openslr/voxtral_prompt_aligned/pipe=voxtral__...` | `test` |
| MLC-SLM | `MLC-SLM Workshop (11 Languages)` + `From MLC` recovery | `data/cache/voxtral/mlc_slm_*/` | `train`, `dev`, `test`, and optional clean/subset splits |

### VC source
- `--source vc --language sr` for Serbian
- `--source vc --language mt` for Maltese
- `--source vc --language da` for Danish
- Available split names: `train`, `valid`, `test`

### OpenSLR source
- `--source openslr`
- Available split names: `test`

### MLC source
- `--source mlc`
- Available split names from recovered cache:
- `train`, `dev`, `test`
- if clean index cache exists: `train_clean`, `dev_clean`, `test_clean`
- if train subset cache exists: `train_eval`, `train_finetune`

Notebook mapping note:
- `train_eval` corresponds to the held-out small training subset (`mlc_train_eval_subset_processed`, about 29k samples).
- `train_finetune` corresponds to `mlc_train_finetune_processed`.

### MLC split details (router name -> notebook object)

| Router split | Notebook object | Description |
|---|---|---|
| `train` | `mlc_splits_processed['train']` | Original processed train split |
| `dev` | `mlc_splits_processed['dev']` | Original processed dev split |
| `test` | `mlc_splits_processed['test']` | Original processed test split |
| `train_clean` | `mlc_splits_clean['train']['processed']` | Cleaned train after removed indices |
| `dev_clean` | `mlc_splits_clean['dev']['processed']` | Cleaned dev |
| `test_clean` | `mlc_splits_clean['test']['processed']` | Cleaned test |
| `train_eval` | `mlc_train_eval_subset_processed` | Small held-out subset (~29k), from `train_subset` cache |
| `train_finetune` | `mlc_train_finetune_processed` | Remaining train subset used for finetuning |

## 4) Examples aligned to notebook usage

### A) Train on MLC small set (~29k) and evaluate on MLC clean test

```bash
python -m asr_merging.voxtral_train_router \
  --source mlc \
  --do-train --do-eval \
  --model-mode baseline \
  --train-set train_eval \
  --valid-set dev_clean \
  --evaluation-set test_clean \
  --timestamped-exp-dir \
  --experiment-name mlc_train_eval_29k
```

### B) Train on MLC finetune split and evaluate

```bash
python -m asr_merging.voxtral_train_router \
  --source mlc \
  --do-train --do-eval \
  --model-mode baseline \
  --train-set train_finetune \
  --valid-set train_eval \
  --evaluation-set test_clean \
  --timestamped-exp-dir \
  --experiment-name mlc_train_finetune
```

### C) VC Serbian train and evaluate

```bash
python -m asr_merging.voxtral_train_router \
  --source vc --language sr \
  --do-train --do-eval \
  --model-mode baseline \
  --train-set train --valid-set valid --evaluation-set test \
  --timestamped-exp-dir \
  --experiment-name vc_sr
```

### D) OpenSLR evaluation with an adapter

```bash
python -m asr_merging.voxtral_train_router \
  --source openslr \
  --do-eval \
  --model-mode adapter \
  --adapter-path experiments/my_adapter/checkpoint-1000 \
  --evaluation-set test \
  --timestamped-exp-dir \
  --experiment-name openslr_adapter_eval
```

### E) Eval-only mode (no train/valid splits provided)

```bash
python -m asr_merging.voxtral_train_router \
  --source mlc \
  --model-mode baseline \
  --evaluation-set test_clean \
  --timestamped-exp-dir \
  --experiment-name mlc_eval_only
```

### F) Enable TensorBoard tracking

```bash
python -m asr_merging.voxtral_train_router \
  --source mlc \
  --do-train --do-eval \
  --train-set train_eval \
  --valid-set dev_clean \
  --evaluation-set test_clean \
  --tf-tracking \
  --timestamped-exp-dir \
  --experiment-name mlc_with_tb
```

### G) Run from JSON config and override one field via CLI

```bash
python -m asr_merging.voxtral_train_router \
  --config-json configuriation/mlc_train_eval_baseline.json \
  --learning-rate 3e-5
```

## 5) Key options

- `--model-mode baseline`
Loads base Voxtral model.

- `--model-mode adapter --adapter-path <path>`
Loads a LoRA adapter for test (and for continued training mode when used with `--do-train`).

- `--config-json <path>`
Loads run arguments and config values from a JSON file.

- Priority order
Defaults < JSON config < explicit CLI arguments.

- `--evaluation-set <split>`
Selects the split used to compute WER/CER.

- Eval-only default behavior
If no `--do-train` is provided, the script evaluates on `--evaluation-set`.

- `--timestamped-exp-dir --experiment-name <name> --output-root <dir>`
Creates an experiment directory with timestamp under `output-root`.

- `--output-dir <dir>`
Explicit output dir override (disables timestamp naming unless you pass it yourself).

- `--tf-tracking`
Enables TensorBoard logging via `TrainingArguments(report_to='tensorboard')`.

- `--no-use-vad`
Disable VAD filtering in the preprocessing path.

- `--num-epochs`, `--train-batch-size`, `--eval-batch-size`, `--grad-accum-steps`, `--learning-rate`, `--weight-decay`
Training knobs equivalent to notebook-style configuration.

## 6) Outputs

The script writes outputs under the resolved experiment folder:
- `experiment_config.json` with CLI/config/split metadata
- `input_config.json` copy of the JSON config used for this run (if `--config-json` was provided)
- training artifacts under `final_model/` (when `--do-train`)
- evaluation metrics in `eval_metrics.json` (when evaluation runs)
- TensorBoard logs under `tensorboard/` (when `--tf-tracking` is enabled)

## 7) Troubleshooting

- If MLC split names like `train_eval` are missing:
Run the notebook MLC recovery flow first to generate clean-index and train-subset caches.

- If adapter load fails:
Verify `--adapter-path` points to a valid PEFT adapter directory with adapter files.

- If cache not found errors appear:
Confirm canonical cache roots exist under:
- `data/cache/processed_cv_datasets/voxtral_prompt_aligned`
- `data/cache/processed_openslr/voxtral_prompt_aligned`
- `data/cache/voxtral`

## 8) How to rerun datasets from notebooks

Notebook file:
- `notebooks/voxtral-FT-QLORA.ipynb`

Recommended rerun flow to rebuild caches used by the router:

### A) VC (Serbian/Maltese/Danish)
1. Run configuration/helpers at the top (model config, VAD helper functions, canonical helpers).
2. Run `Common Voice Serbian/Maltese/Danish` processing section.
3. Run `Save/Load Processed Datasets (canonical Voxtral-first)` section to persist canonical caches.
4. Verify canonical aliases exist (`*_train_canonical`, `*_valid_canonical`, `*_test_canonical`) and cache folders are present under `data/cache/processed_cv_datasets/voxtral_prompt_aligned/<lang>/`.

### B) OpenSLR108 (Spanish)
1. Run `OpenSLR108` raw loading/preprocessing section.
2. Run canonical OpenSLR cache save/recover cells (`canonical Voxtral-first` path).
3. Verify `openslr108_test_canonical` schema is `['audio','text']` and cache exists under `data/cache/processed_openslr/voxtral_prompt_aligned/pipe=voxtral__...`.

### C) MLC-SLM
1. Run `MLC-SLM Workshop (11 Languages)` cache-builder section (canonical `audio`/`text`).
2. Run `From MLC` recovery section to rebuild/resolve cleaned indices and subset splits.
3. Ensure the following variables are available in the notebook session:
  - `mlc_splits_processed`, `mlc_splits_raw`, `mlc_splits_refs`
  - `mlc_splits_clean`
  - `mlc_train_eval_subset_processed`, `mlc_train_finetune_processed`
4. Confirm MLC cache paths exist under `data/cache/voxtral/mlc_slm_*/` and, when used, `clean_index_cache/train_subset/`.

### D) After notebook rerun
Run router commands again. The script is cache-first and will pick up regenerated canonical datasets/splits without notebook runtime state.
