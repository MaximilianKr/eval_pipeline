# dtfit - Minimal pairs testing semantic plausibility

The dataset contains **397** instances, processed as minimal pairs with a prefix, a *good* and a *bad* continuation, based on **plausibility** of the resulting phrase.

For example:

- The actor won the {award=*good*, battle=*bad*}

The code, data and documentation inside this folder was forked from [Hu, Jennifer & Levy, Roger (2023)](https://github.com/jennhu/metalinguistic-prompting). Please refer to their original repository for additional information.

The content of the original [README.md](https://github.com/jennhu/metalinguistic-prompting/tree/master/datasets/exp2/dtfit) is attached below.

---

The original file `clean_DTFit_human_dat.csv` was downloaded from [here](https://github.com/carina-kauf/lm-event-knowledge/blob/main/analyses/clean_data/clean_DTFit_human_dat.csv) on March 22, 2023.

Please see [Kauf, Ivanova et al. (2022)](https://arxiv.org/abs/2212.01488) for more details.

Here is a description from their paper about human scores:

>Human judgments for Dataset 2 had been previously collected by Vassallo et al. (2018) on Prolific, a web-based platform for collecting behavioral data. Participants in this experiment answered questions of the form “How common is it for an actor to win an award?” on a Likert scale from 1 (very atypical) to 7 (very typical).

The data are originally from [Vassallo et al. (2018)](http://lrec-conf.org/workshops/lrec2018/W9/pdf/5_W9.pdf).

The final `corpus.csv` file was created by running the script `python make_corpus.py`.
