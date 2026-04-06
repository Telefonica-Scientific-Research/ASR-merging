# asr_merging

[![codecov](https://codecov.io/gh/Telefonica-Scientific-Research/ASR-merging/branch/main/graph/badge.svg?token=ASR-merging_token_here)](https://codecov.io/gh/Telefonica-Scientific-Research/ASR-merging)
[![CI](https://github.com/Telefonica-Scientific-Research/ASR-merging/actions/workflows/main.yml/badge.svg)](https://github.com/Telefonica-Scientific-Research/ASR-merging/actions/workflows/main.yml)


## How to use the code

### Install it from PyPI

```bash
pip install asr_merging
```

### Usage

```py
from asr_merging import BaseClass
from asr_merging import base_function

BaseClass().base_method()
base_function()
```

```bash
$ python -m asr_merging
#or
$ asr_merging
```

### Voxtral Router Guide

For notebook-aligned training/evaluation on VC, OpenSLR, and MLC datasets using canonical Voxtral paths, see:

- [asr_merging/README_voxtral_train_router.md](asr_merging/README_voxtral_train_router.md)

### Seamless Router Guide

For notebook-aligned training/evaluation on VC, OpenSLR, and MLC datasets using Seamless M4T processed caches, see:

- [asr_merging/README_seamless_train_router.md](asr_merging/README_seamless_train_router.md)

### Whisper-Turbo Router Guide

For notebook-aligned training/evaluation on VC, OpenSLR, and MLC datasets using Whisper-Turbo processed caches, see:

- [asr_merging/README_whisper_turbo_train_router.md](asr_merging/README_whisper_turbo_train_router.md)

### Notebook Section Index

Main notebook:
- [notebooks/voxtral-FT-QLORA.ipynb](notebooks/voxtral-FT-QLORA.ipynb)

Recommended sections to rerun when rebuilding dataset caches:
- Common Voice: `Common Voice Serbian/Maltese/Danish`
- OpenSLR: `OpenSLR108`
- MLC: `MLC-SLM Workshop (11 Languages)` and `From MLC`
- Cache persistence/recovery: `Save/Load Processed Datasets (canonical Voxtral-first)`


### Requirements

```bash
numpy==1.25.2
pandas==2.1.0
scikit-learn==1.1.2
tqdm==4.64.1
torch==1.13.0+cu117
torch-geometric==2.2.0
xgboost==1.7.2
```


## Citation

If you use this codebase, please cite our work:

```bib
@article{authorYearTitle,
    title={title},
    author={author},
    year={year},
    journal={journal},
    url={url}
}
```
