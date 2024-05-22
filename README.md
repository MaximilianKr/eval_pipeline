# Eval Pipeline

Prototyping an evaluation pipeline for language models on discourse connectives.

## Overview

- [Eval Pipeline](#eval-pipeline)
  - [Overview](#overview)
  - [Models](#models)
    - [EleutherAI-Pythia](#eleutherai-pythia)
    - [AI2-OLMo](#ai2-olmo)
  - [Setup](#setup)
    - [venv](#venv)
    - [conda](#conda)
  - [Evaluation materials](#evaluation-materials)
  - [Running experiments](#running-experiments)
  - [ToDo](#todo)
  - [Author](#author)

## Models

### EleutherAI-Pythia

- [arXiv Technical Report](https://arxiv.org/abs/2304.01373)
- [HuggingFace Pythia Scaling Suite](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1)
- [Github pythia](https://github.com/EleutherAI/pythia)
- [EleutherAI](https://www.eleuther.ai/)

### AI2-OLMo

- [arXiv Technical Report](https://arxiv.org/abs/2402.00838)
- [HuggingFace OLMo Suite](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778)
- [Github OLMo](https://github.com/allenai/OLMo)
- [AI2](https://allenai.org/)

Both models were released in different parameter sizes and with intermediate checkpoints available. The weights, training code, and data are all fully accessible.

## Setup

- requires GPU with `cuda >= 12.1` support (smaller models can run on CPU, but not recommended)

### venv

- use [uv package manager](https://github.com/astral-sh/uv) for a fast setup

```shell
uv venv
```

```shell
# macOS / Linux
source .venv/bin/activate
```

```shell
# Windows
.venv\Scripts\activate
```

```shell
uv pip install -r requirements.txt
```

### conda

```shell
conda env create -f environment.yml
```

```shell
conda activate pipe
```

## Evaluation materials

- **ToDo**: add datasets, preprocessing, documentation, sources

>Evaluation datasets can be found in the [`datasets`](datasets) folder.
>Please refer to the README in that folder for more details on how the stimuli were assembled and formatted.

## Running experiments

Run the bash script and specify the [dataset](data/README.md), `model` and optionally `revision`. [^1]

```shell
# Template
bash run_eval.sh {dataset} {model} {optional: revision}
```

- [dtfit](data/dtfit/README.md)

```shell
bash run_eval.sh dtfit EleutherAI/pythia-14m
```

- [connectives](data/connectives/README.md)

```shell
bash run_eval.sh connectives allenai/OLMo-1B-hf
```

[^1]: `main` as default corresponds to the final model checkpoint for all available models. Check either [Pythia](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) or [OLMo](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) suites on Huggingface, select a model and branch to access different (earlier) checkpoints.

## ToDo

- [ ] separate running experiment from running evaluation of results

- [ ] visualize results / use notebook

- [ ] add datasets and corresponding documentation

- [ ] test batching - only single instances passed to the model, possible improvements achievable (especially for larger models)

- [ ] add OpenAI support as upper bound for commercial models?

## Author

- Maximilian Krupop

[Back to Top](#eval-pipeline)
