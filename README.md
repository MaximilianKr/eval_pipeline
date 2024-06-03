# Eval Pipeline

Prototyping an evaluation pipeline for language models on discourse connectives.

## Overview

- [Eval Pipeline](#eval-pipeline)
  - [Overview](#overview)
  - [Models](#models)
  - [Setup](#setup)
    - [venv](#venv)
    - [conda](#conda)
  - [Datasets for evaluation](#datasets-for-evaluation)
  - [Running experiments](#running-experiments)
  - [ToDo](#todo)
  - [Author](#author)

## Models

| AI2-OLMo                                  | EleutherAI-Pythia                              |
|-------------------------------------------|------------------------------------------------|
| [Huggingface Suite](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) | [Huggingface Suite](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) |
| [Github](https://github.com/allenai/OLMo) | [Github](https://github.com/EleutherAI/pythia) |
| [Technical Report](https://arxiv.org/abs/2402.00838) | [Technical Report](https://arxiv.org/abs/2304.01373) |
| [Website](https://allenai.org/) | [Website](https://www.eleuther.ai/) |

Both models were released in different parameter sizes at different checkpoints / training steps (revisions).

## Setup

- requires GPU with `cuda >= 12.1` support (smaller models can run on CPU, but not recommended)

### venv

- recommended: use [uv package manager](https://github.com/astral-sh/uv) for a fast setup

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

## Datasets for evaluation

The datasets for evaluation can be found in the [`data`](data) folder.
Please refer to the [README.md](data/README.md) in that folder for more details on how the stimuli were assembled and formatted. Additional datasets can easily be integrated.

## Running experiments

Run the bash script and specify the [dataset](data/README.md), `model` and optionally `revision` (defaults to `main`, final checkpoint for all models).

To access different checkpoints, check either [Pythia](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) or [OLMo](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) suites on Huggingface, select a model and choose a corresponding branch.

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

## ToDo

- [ ] add test coverage / `pytest`

- [ ] change bash scripts to `argparse` interface

- [ ] test [HPC cluster](https://docs.hpc.uni-potsdam.de/overview/index.html) for larger models
  - `pythia-6.9B`, `pythia-12B`, `pythia-6.9B-deduped`, `pythia-12B-deduped`
    - at various checkpoints
  - `OLMo-1.7-7B-hf`
    - at various checkpoints
  - [model parallelism](https://huggingface.co/docs/transformers/v4.13.0/en/parallelism) for V100 nodes?

- [ ] test `OLMo` model checkpoints for non-`hf` versions?
  - requires additional code
  - `OLMo-1B-hf`, `OLMo-7B-hf`

- [ ] add more datasets and corresponding documentation
  - [DisSent](https://github.com/windweller/DisExtract)

- [ ] extend visualization of results (notebook)

- [ ] add OpenAI support as upper bound for commercial models?

- [ ] extract & analyze [contextual word embeddings](https://github.com/kanishkamisra/minicons/blob/master/examples/word_representations.md)

- [ ] test other open models with checkpoints?
  - `togethercomputer/RedPajama-INCITE-7B-Base`
  - `TinyLlama/TinyLlama-1.1B`
  - `Zyphra/Zamba-7b`
  - [Ablation Models](https://huggingface.co/collections/HuggingFaceFW/ablation-models-662457b0d213e8c14fe47f32)?
    - checkpoints available for different common datasets for pretraining

## Author

- Maximilian Krupop

[Back to Top](#eval-pipeline)
