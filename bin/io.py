import json
from time import gmtime, strftime
from minicons import scorer
import torch

torch.set_grad_enabled(False)


def timestamp():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def dict2json(d: dict, out_file: str):
    with open(out_file, "w") as fp:
        json.dump(d, fp, indent=2)

def initialize_model(model_name: str, revision: str):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Set device to CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU (CUDA unvailable); adjust your expectations") 
    if "pythia" or "allenai" in model_name:
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
