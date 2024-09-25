# Datasets

This folder contains the datasets and stimuli used for the experiments.

| Dataset      | Reference    | Description |
|--------------|--------------|-------------|
| [`dtfit`](dtfit) | [Vassallo et al. (2018)](https://hal.science/hal-01724286) | Minimal pairs testing semantic plausibility |

---

New datasets can be added as a folder containing a `corpus.csv` with minimal pairs in the following structure:

| item_id      | prefix            | good_continuation | bad_continuation |
|--------------|-------------------|-------------------|------------------|
| 1            | This example ends | good              | bad              |
| 2            | ...               | ...               | ...              |
