# Minimal Pairs Eval Pipeline

An evaluation pipeline for autoregressive language models using direct probability measurement for minimal pairs.

This pipeline evaluates language models by reading out the conditional log probabilities for minimal pairs of sentences. In each pair, one sentence is considered *correct*, while the other contains a minimal violation. The model is expected to assign a lower probability to the *incorrect* sentence.

By using a sufficient number of test items targeting specific linguistic phenomena, the accuracy of the model’s probability assignments provides an indication of its linguistic capabilities and understanding of these phenomena. Assessing models at different training checkpoints allows for analyzing learning dynamics of selected phenomena.

## Overview

- [Minimal Pairs Eval Pipeline](#minimal-pairs-eval-pipeline)
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

Both models were released in different parameter sizes at different intermediate training checkpoints (revisions).
This makes it possible to test for emerging capabilities across parameter scale and training time.

## Setup

- tested on Python `3.12.x`, `3.11.x`, `3.10.x`
- requires GPU with `CUDA >= 12.1` support (smaller models can run on CPU, but not recommended)

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

An example dataset for testing can be found in the [`data`](data) folder.
Additional datasets can easily be integrated and tested.
Please refer to the corresponding [README.md](data/README.md) in the folder for more details.

## Running experiments

Run the Python script and specify the [dataset](data/README.md), `model` and
optionally `revision` (defaults to `main`, final checkpoint for all models).

To access different intermediate training checkpoints (revisions), check either [Pythia](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1) or [OLMo](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) suites on Huggingface, select a model's *Files and versions* and choose a corresponding branch.

```shell
# Template
python run_eval.py {dataset} {model} {optional: revision}
```

- [dtfit](data/dtfit/README.md)

  ```shell
  python run_eval.py dtfit EleutherAI/pythia-14m
  ```

## ToDo

- [ ] **Performance**
  - [ ] fix batch support
- [ ] **Optional**
  - [ ] add support for commercial APIs as upper bound
  - [ ] extract & analyze [contextual word embeddings](https://github.com/kanishkamisra/minicons/blob/master/examples/word_representations.md)
  - [ ] test other open models with checkpoints?
    - `togethercomputer/RedPajama-INCITE-7B-Base`
    - `TinyLlama/TinyLlama-1.1B`
    - `Zyphra/Zamba-7b`
    - [Ablation Models](https://huggingface.co/collections/HuggingFaceFW/ablation-models-662457b0d213e8c14fe47f32)?
      - checkpoints available for different common datasets for pretraining

## Author

- Maximilian Krupop

[Back to Top](#minimal-pairs-eval-pipeline)
