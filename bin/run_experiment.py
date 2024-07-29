"""
Module to run experiments for evaluating language models.

This module provides functions to run experiments using a specified language 
model on a dataset, and save the results to a JSON file with metadata.
"""

import sys
import pandas as pd
from tqdm import tqdm
from minicons import scorer
from bin.io import initialize_model, timestamp, dict2json


def run_experiment(
    model: scorer.IncrementalLMScorer, meta_data: dict, file_out: str
) -> None:
    """
    Run the experiment for the given model and dataset and save the results to
    a JSON file with metadata.

    Args:
        model (scorer.IncrementalLMScorer): The model to evaluate.
        meta_data (dict): Metadata about the model and dataset.
        file_out (str): The path to the output file.

    Returns:
        None
    """
    dataset = meta_data["dataset"]
    df = pd.read_csv(f"./data/{dataset}/corpus.csv")

    results = []

    print("Running experiment...")

    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
        good_instance = f"{row.prefix} {row.good_continuation}"
        bad_instance = f"{row.prefix} {row.bad_continuation}"
        stimuli = [good_instance, bad_instance]

        # Sequence Log-probability, normalized by number of tokens
        # reduction = lambda x: x.mean(0).item()
        # see https://github.com/kanishkamisra/minicons
        logprobs = model.sequence_score(
            stimuli, reduction=lambda x: x.mean(0).item()
            )

        res = {
            "item_id": row.item_id,
            "prefix": row.prefix,
            "good_continuation": row.good_continuation,
            "bad_continuation": row.bad_continuation,
            "logprob_of_good_continuation": logprobs[0],
            "logprob_of_bad_continuation": logprobs[1],
            "relation": row.category,
        }

        # For cases where the dataset has a 'type' column for inter- or intra- 
        # sentential connective continuations
        if 'type' in row.index:
            res['type'] = row.type

        results.append(res)

    output = {"meta": meta_data, "results": results}

    dict2json(output, file_out)


def main() -> None:
    """
    Main function to run the evaluation. It should be called from 'run_eval.py'
    in the parent directory with following command line arguments:
        - model_name: The name of the model to evaluate.
        - revision: The revision of the model to evaluate.
        - dataset: The dataset to evaluate the model on.
        - file_out: The path to the output file.

    Args:
        None

    Returns:
        None
    """
    if len(sys.argv) < 4:
        print(
            "Usage: Incorrect call. Check the documentation on how to run the \
            evaluation."
        )
        sys.exit(1)

    model_name, revision, dataset, file_out = sys.argv[1:5]

    # minicons IncrementalLMScorer
    model = initialize_model(model_name, revision)

    meta_data = {
        "model": model_name,
        "revision": revision,
        "dataset": dataset,
        "timestamp": timestamp(),
    }

    run_experiment(model, meta_data, file_out)


if __name__ == "__main__":
    main()
