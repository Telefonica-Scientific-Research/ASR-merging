# Seamless Training Router

This guide explains how to use `asr_merging/seamless_train_router.py`.

The script follows the workflow in `notebooks/seamless-FT-QLORA.ipynb` and keeps a similar interface to the Voxtral router:
- cache-first split recovery for VC, MLC, and OpenSLR
- baseline and adapter training/evaluation modes
- JSON config support with CLI-overrides
- optional auto validation split from the train split
- experiment tracking with `experiment_config.json` and `input_config.json`
- optional TensorBoard logging

## 1) Run location

```bash
cd /home/id02619@hi.inet/repos/eloquence/eloquence/WP4/ASR-merging
```

## 2) Main command pattern

```bash
python -m asr_merging.seamless_train_router \
  [--config-json <path>] \
  --source <vc|mlc|openslr> \
  [--language <sr|mt|da>] \
  [--target-lang <srp|mlt|dan|eng|spa|...>] \
  [--do-train] [--do-eval] \
  [--model-mode <baseline|adapter>] \
  [--adapter-path <path>] \
  [--train-set <name>] [--valid-set <name>] [--evaluation-set <name>] \
  [--validation-split-ratio <float>] [--validation-split-seed <int>] \
  [--timestamped-exp-dir] [--experiment-name <name>] [--output-root <dir>] \
  [--output-dir <dir>] [--tf-tracking]
```

Priority order:
- defaults < JSON config < explicit CLI arguments

## 3) Cache roots aligned with the Seamless notebook

- VC cache root: `cache/processed_cv_datasets/<lang>/`
- OpenSLR cache root: `cache/processed_openslr/`
- MLC cache root: `cache/mlc_slm_*/`

Expected processed schema per split:
- `input_features`
- `labels`

## 4) Source and split names

- VC (`--source vc --language sr|mt|da`): `train`, `valid`, `test`
- OpenSLR (`--source openslr`): `test`
- MLC (`--source mlc`): `train`, `dev`, `test`, optional `train_clean`, `dev_clean`, `test_clean`, `train_eval`, `train_finetune`

## 5) Configuration JSON templates

Templates were added under `configuration/`:
- `configuration/seamless_mlc_train_eval_baseline.json`
- `configuration/seamless_mlc_eval_adapter.json`
- `configuration/seamless_vc_sr_train_baseline.json`

Example with config + CLI override:

```bash
python -m asr_merging.seamless_train_router \
  --config-json configuration/seamless_mlc_train_eval_baseline.json \
  --learning-rate 3e-5
```

## 6) Auto validation split

If `--do-train` is enabled and `--valid-set` is omitted or null in JSON:
- train split is internally partitioned using `train_test_split`
- default ratio: `0.1`
- default seed: `42`

Control with:
- `--validation-split-ratio`
- `--validation-split-seed`

## 7) Output artifacts

Under the resolved experiment folder:
- `experiment_config.json`
- `input_config.json` (if `--config-json` is provided)
- `final_model/` (after training)
- `eval_metrics.json` (after evaluation)
- `tensorboard/` (when `--tf-tracking` is enabled)

## 8) Notes on target language

Seamless generation uses `--target-lang` for decoding.
Use appropriate language tags for your evaluation scenario, for example:
- Serbian CV: `srp`
- Maltese CV: `mlt`
- Danish CV: `dan`
- English: `eng`
- Spanish: `spa`
