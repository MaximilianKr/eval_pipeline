# Datasets

This folder contains the datasets and stimuli used for the experiments.

| Dataset      | Reference    | Description |
|--------------|--------------|-------------|
| [`connectives`](connectives) | [Drenhaus et al. (2014)](https://escholarship.org/uc/item/9q88v0zh) / [Pandia et al. (2021)](https://aclanthology.org/2021.conll-1.29/) | Minimal pairs testing causal vs concessive connectives |
| [`dtfit`](dtfit) | [Vassallo et al. (2018)](https://hal.science/hal-01724286) | Minimal pairs testing semantic plausibility |

---

New datasets can be added as a folder containing a `corpus.csv` with minimal pairs in the following structure:

| item_id      | prefix            | good_continuation | bad_continuation |
|--------------|-------------------|-------------------|------------------|
| 1            | This example ends | good              | bad              |
| 2            | ...               | ...               | ...              |
