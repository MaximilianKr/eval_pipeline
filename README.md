# Eval Pipeline

Prototyping an evaluation pipeline for language models on discourse connectives. 

## Overview

- [Models](#models)
  - [EleutherAI/Pythia](#eleutherai-pythia)
  - [AI2/OLMo](#ai2-olmo)
- [Setup](#setup)
  - [venv](#venv)
  - [conda](#conda)
- [Evaluation materials](#evaluation-materials)
- [Evaluation scripts](#evaluation-scripts)
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
conda activate eval_pipeline
```

## Evaluation materials

- **ToDo**

>Evaluation datasets can be found in the [`datasets`](datasets) folder.
>Please refer to the README in that folder for more details on how the stimuli were assembled and formatted.

## Evaluation scripts

### Pythia & OLMo

- **ToDo**

```shell
# Template
bash run_eval.sh {dataset} {model} {optional: revision} {optional: quantization}
```

```shell
bash run_eval.sh EleutherAI/pythia-70m-deduped step3000 8bit
```

- `revision`

  - `main` corresponds to the final model checkpoint. Must be set when using quantization. Check either [Pythia](https://huggingface.co/EleutherAI/pythia-70m-deduped) or [OLMo](https://huggingface.co/allenai/OLMo-1.7-7B-hf) model cards on Huggingface for details on how to access different (earlier) checkpoints.

- `quantization`
  
  - `8bit` or `4bit`, running with less precision also requires less VRAM. Loading checkpoint shards can take longer than with full precision (*quantized OLMo models load fine, Pythia models very slow*). Must set revision to use (e.g., use `main`).

### OpenAI

- **ToDo** (optional)

There are still 2 base models available via *OpenAI*'s API (`babbage-002`/replacement for GPT-3 `ada` and `babbage` base models and `davinci-002`/replacement for GPT-3 `curie` and `davinci` base models).

For more details on the base models still available read the [official documentation](https://platform.openai.com/docs/models/gpt-base).

## ToDo

- [ ] test [minicons](https://github.com/kanishkamisra/minicons) implementation

- [ ] visualize results / use notebook

- [ ] add batching support - only single instances passed to the model, possible improvements achievable (especially for larger models)

- [ ] add OpenAI support?

## Author

- Maximilian Krupop

[Back to Top](#eval-pipeline)
