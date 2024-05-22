# connectives - Minimal pairs testing causal and concessive discourse connectives

The dataset contains **60** instances, processed as minimal pairs with a prefix, a *good* and a *bad* continuation, based on the discourse relation between the prefix and the continuation (either causal or concessive).

For example:

>John is thinking about going to see the latest movie or to listen to some famous arias. He would like to hear some great tenors and sopranos.

- **causal** relation

    | good | bad |
    | -------- | ------- |
    | *Therefore* he buys tickets for an opera in the city center | *Nevertheless* he buys tickets for an opera in the city center |

- **concessive** relation

    | good | bad |
    | -------- | ------- |
    | *Nevertheless* he buys tickets for a cinema in the city center | *Therefore* he buys tickets for a cinema in the city center |

---

The original file `english_stimuli_connector_2014_drenhaus_et_al.json` was provided by the authors of [Pandia et al. (2021)](https://aclanthology.org/2021.conll-1.29/).

Please see [Drenhaus et al. (2014)](https://escholarship.org/uc/item/9q88v0zh) for their inspiration and data and [Pandia et al. (2021)](https://aclanthology.org/2021.conll-1.29/) for further processing of the data for more details.

For the purpose of this evaluation, the stimuli, originally created for testing masked language models, were transformed into minimal pairs testing either a causal or concessive connective.

The final `corpus.csv` file was created by running the script `python make_corpus.py`.

Additional details are provided in the Jupyter Notebook `interactive_make_corpus.ipynb`.
