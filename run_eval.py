"""
Module for running evaluation scripts with specified datasets and model.

This script takes command line arguments for one or more datasets, a model,
and an optional revision, constructs the necessary paths, checks if the dataset
directories exist, sets up the environment variables, and runs the experiment
using the provided arguments.
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    """
    Run the evaluation script with specified datasets and model.

    This function takes command line arguments for one or more datasets, model,
    and optional revision. It constructs the necessary paths, checks if the
    dataset directories exist, and sets up the environment variables. Finally,
    it runs the experiment using the provided arguments.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Run evaluation script with specified datasets and model."
    )
    parser.add_argument(
        "dataset",
        type=str,
        nargs="+",
        help="Dataset(s) to be used. Must be directories in the 'data' folder\
              each containing a 'corpus.csv' with the test items.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Huggingface model to be used in the format 'namespace/modelname'\
              e.g., 'EleutherAI/pythia-14m'.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Optional: model revision or specific checkpoint, default is\
              'main'. Check model-specific revision naming on Huggingface.",
    )

    args = parser.parse_args()

    datasets = args.dataset
    model = args.model
    revision = args.revision

    if "/" in model:
        save_name = model.split("/")[-1]
    else:
        print("Error: MODEL should be in the format 'namespace/modelname'.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir

    python_script = os.path.join(script_dir, "bin", "run_experiment.py")

    # Verify dataset existence before running the experiment
    valid_datasets = []
    for dataset in datasets:
        dataset_dir = os.path.join(script_dir, "data", dataset)

        if not os.path.isdir(dataset_dir):
            print(
                f"Warning: Dataset directory '{dataset_dir}' does not exist. \
                  Skipping this dataset."
            )
        else:
            valid_datasets.append(dataset)

            # Create results directory if it doesn't exist
            result_dir = os.path.join(script_dir, "results", dataset)
            os.makedirs(result_dir, exist_ok=True)

    if not valid_datasets:
        print("Error: No valid dataset directories found. Exiting.")
        sys.exit(1)

    # Output paths with placeholders for dataset names
    output_template = os.path.join(
        "results", "{dataset}", f"{save_name}_{revision}.json"
        )

    command = ["python", python_script, model, revision, \
               *valid_datasets, output_template]

    subprocess.run(command, env=env, check=True)

    print("All experiments completed successfully.")


if __name__ == "__main__":
    main()
