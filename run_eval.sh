#!/bin/bash

# Template
# bash run_eval.sh {dataset} {model} {optional: revision} {optional: quantization}

# TODO: Add dataset?
DATASET="dtfit"

# Required, e.g., "EleutherAI/pythia-14m", "allenai/OLMo-1B-hf"
MODEL=$1

# Optional, default "main" / see intermediate checkpoints for models
REVISION=${2:-"main"}

# TODO: Validate revision input?

# Optional, default "full" precision, possible "4bit" or "8bit"
QUANTIZATION=${3:-"full"}

# Validate quantization input
if [[ "$QUANTIZATION" != "full" && "$QUANTIZATION" != "4bit" && "$QUANTIZATION" != "8bit" ]]; then
    echo "Error: QUANTIZATION must be either unspecified (defaults to full precision), '4bit', or '8bit'."
    exit 1
fi

# Extract the second part of the model name
if [[ $MODEL == */* ]]; then
    SAFEMODEL=$(echo $MODEL | cut -d '/' -f 2)
else
    echo "Error: MODEL should be in the format 'namespace/modelname'."
    exit 1
fi

# Determine the directory where the script is located
SCRIPT_DIR=$(dirname "$0")

# Construct path relative to the script's location
PYTHON_SCRIPT="$SCRIPT_DIR/bin/run_eval.py"

RESULTDIR="results/$DATASET"
mkdir -p $RESULTDIR

FILE_OUT="$RESULTDIR/${SAFEMODEL}_${REVISION}_${QUANTIZATION}.json"
echo "Results will be saved to: $FILE_OUT"

# Set PYTHONPATH to include the parent directory
export PYTHONPATH=$SCRIPT_DIR

python "$PYTHON_SCRIPT" $MODEL $REVISION $QUANTIZATION $DATASET $FILE_OUT
