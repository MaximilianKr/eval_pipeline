"""
Module for initializing models and handling JSON operations.

This module provides utility functions to initialize language models
for scoring and to handle JSON operations such as writing dictionaries
to JSON files.
"""

import json
from datetime import datetime
from minicons import scorer
import torch


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)


def timestamp() -> str:
    """
    Returns the local current timestamp as a formatted string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dict2json(input_dict: dict, out_file: str) -> None:
    """
    Writes a dictionary to a JSON file.

    Args:
        input_dict (dict): The dictionary to write.
        out_file (str): The path to the output file.
    """
    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(input_dict, file, indent=2)


def initialize_model(model_name: str, revision: str) -> scorer.IncrementalLMScorer:
    """
    Initializes the model for scoring. Supports all models supported by the
    current Huggingface Transformers library.

    Args:
        model_name (str): The name of the model.
        revision (str): The revision of the model.

    Returns:
        scorer.IncrementalLMScorer: The initialized model scorer.
    """
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = 'auto'  # passed as 'device_map' to IncrementalLMScorer
            print("Multiple GPUs detected! Using all available GPUs.")
        else:
            device = torch.device("cuda")
            print("Set device to CUDA.")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA unavailable); adjust your expectations.")

    model = scorer.IncrementalLMScorer(
            model=model_name, device=device, revision=revision,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )

    return model
