#!/bin/bash

# Template
# bash run_eval.sh {dataset} {model} {optional: revision}

# Required, e.q., "dtfit"
DATASET=$1

# Required, e.g., "EleutherAI/pythia-14m", "allenai/OLMo-1B-hf"
MODEL=$2

# Optional, default "main" / see intermediate checkpoints for models
# TODO: Validate revision input?
REVISION=${3:-"main"}

# Extract the second part of the model name
if [[ $MODEL == */* ]]; then
    SAFEMODEL=$(echo $MODEL | cut -d '/' -f 2)
else
    echo "Error: MODEL should be in the format 'namespace/modelname'."
    exit 1
fi

# Determine the directory where the script is located
SCRIPT_DIR=$(dirname "$0")

# Construct path to the dataset directory relative to the script's location
DATASET_DIR="$SCRIPT_DIR/data/$DATASET"

# Check if the dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory '$DATASET_DIR' does not exist. Either check your input or add a new dataset."
    exit 1
fi

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