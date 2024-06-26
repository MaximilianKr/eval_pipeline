"""
Module for running evaluation scripts with specified dataset and model.

This script takes command line arguments for the dataset, model, and optional 
revision, constructs the necessary paths, checks if the dataset directory 
exists, sets up the environment variables, and runs the experiment using the
provided arguments.
"""

import argparse
import os
import subprocess
import sys


def main() -> None:
    """
    Run the evaluation script with specified dataset and model.

    This function takes command line arguments for the dataset, model, and
    optional revision. It constructs the necessary paths, checks if the dataset
    directory exists, and sets up the environment variables. Finally, it runs
    the experiment using the provided arguments.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Run evaluation script with specified dataset and model."
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset to be used. Must be a directory in the 'data' folder\
            containing a 'corpus.csv' with the test items.",
    )
    parser.add_argument(
        "model",
        type=str,
        help="Huggingface model to be used in the format 'namespace/modelname'\
            e.g., 'EleutherAI/pythia-14m'.",
    )
    parser.add_argument(
        "revision",
        type=str,
        nargs="?",
        default="main",
        help="Optional: model revision or specific checkpoint, default is\
            'main'. Check the model specific revision naming on Huggingface.",
    )

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    revision = args.revision

    if "/" in model:
        save_name = model.split("/")[1]
    else:
        print("Error: MODEL should be in the format 'namespace/modelname'.")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "data", dataset)

    if not os.path.isdir(dataset_dir):
        print(
            f"Error: Dataset directory '{dataset_dir}' does not exist. \
              Either check your input or add a new dataset."
        )
        sys.exit(1)

    python_script = os.path.join(script_dir, "bin", "run_experiment.py")

    result_dir = os.path.join("results", dataset)
    os.makedirs(result_dir, exist_ok=True)

    file_out = os.path.join(result_dir, f"{save_name}_{revision}.json")
    print(f"Results will be saved to: {file_out}")

    env = os.environ.copy()
    env["PYTHONPATH"] = script_dir

    subprocess.run(
        ["python", python_script, model, revision, dataset, file_out],
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
