import json
from datetime import datetime
from minicons import scorer
import torch


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.set_grad_enabled(False)


def timestamp() -> str:
    """
    Returns the local current timestamp as a formatted string.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dict2json(d: dict, out_file: str) -> None:
    """
    Writes a dictionary to a JSON file.

    Args:
        d (dict): The dictionary to write.
        out_file (str): The path to the output file.
    """
    with open(out_file, "w") as fp:
        json.dump(d, fp, indent=2)


def initialize_model(model_name: str, revision: str) -> scorer.IncrementalLMScorer:
    """
    Initializes the model for scoring. Currently supported model suites are
        - EleutherAI/pythia-*
        - allenai/OLMo-*-hf (only Huggingface variants)
    Consult the documentation to access the available models from each suite.

    Args:
        model_name (str): The name of the model.
        revision (str): The revision of the model.

    Returns:
        scorer.IncrementalLMScorer: The initialized model scorer.
    
    Raises:
        ValueError: If the model name is not supported.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Set device to CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA unvailable); adjust your expectations") 
    if "pythia" in model_name or "allenai" in model_name:
        model = scorer.IncrementalLMScorer(
            model=model_name, 
            device=device, 
            revision=revision
            )
    else:
        raise ValueError(
            f"Model not (yet) supported! (Your model: {model_name})"
        )
    
    return model
