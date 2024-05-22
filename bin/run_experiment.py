import sys
import pandas as pd
from tqdm import tqdm
from bin.io import initialize_model, timestamp, dict2json
from minicons import scorer


def run_experiment(model: scorer.IncrementalLMScorer, meta_data: dict, file_out: str):
    # Read corpus data
    dataset = meta_data["dataset"]
    df = pd.read_csv(f"./data/{dataset}/corpus.csv")

    results = []

    print("Running experiment...")
    # TODO: Run batched (all instances passed at once to minicons)
    # Run row-by-row with tqdm
    for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
        good_instance = f"{row.prefix} {row.good_continuation}"
        bad_instance = f"{row.prefix} {row.bad_continuation}"
        stimuli = [good_instance, bad_instance]

        # Sequence Log-probability
        # reduction = lambda x: x.sum(0).item()
        # see https://github.com/kanishkamisra/minicons
        logprobs = model.sequence_score(stimuli, reduction = lambda x: x.sum(0).item())
        
        # Store results in dictionary
        res = {
            "item_id": row.item_id,
            "prefix": row.prefix,
            "good_continuation": row.good_continuation,
            "bad_continuation": row.bad_continuation,
            "logprob_of_good_continuation": logprobs[0],
            "logprob_of_bad_continuation": logprobs[1]
        }
        
        # Record results for current item
        results.append(res)

    # Combine meta information with model results into one dict
    output = {
        "meta": meta_data,
        "results": results
    }

    # Export results
    dict2json(output, file_out)


def main():
    if len(sys.argv) < 4:
        print("Usage: Incorrect call. Check the documentation on how to run the evaluation.") 
        sys.exit(1)

    model_name, revision, dataset, file_out = sys.argv[1:5]

    # minicons IncrementalLMScorer
    model = initialize_model(model_name, revision)

    meta_data = {
        "model": model_name,
        "revision": revision,
        "dataset": dataset,
        "timestamp": timestamp()
    }

    run_experiment(model, meta_data, file_out)


if __name__ == '__main__':
    main()