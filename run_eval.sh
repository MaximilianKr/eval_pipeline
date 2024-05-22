#!/bin/bash

# Template
# bash run_eval.sh {dataset} {model} {optional: revision} {optional: quantization}

# TODO: Add dataset?
DATASET="dtfit"

# Required, e.g., "EleutherAI/pythia-14m", "allenai/OLMo-1B-hf"
MODEL=$1

# Optional, default "main" / see intermediate checkpoints for models
# TODO: Validate revision input?
REVISION=${2:-"main"}

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
PYTHON_SCRIPT="$SCRIPT_DIR/bin/run_experiment.py"

RESULTDIR="results/$DATASET"
mkdir -p $RESULTDIR

FILE_OUT="$RESULTDIR/${SAFEMODEL}_${REVISION}.json"
echo "Results will be saved to: $FILE_OUT"

# Set PYTHONPATH to include the parent directory
export PYTHONPATH=$SCRIPT_DIR

# Run experiment
python "$PYTHON_SCRIPT" $MODEL $REVISION $DATASET $FILE_OUT

# Run eval?