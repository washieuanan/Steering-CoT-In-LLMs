#!/bin/bash

# Script to run 3 models on ARC dataset with 250 examples each
# Models: Mistral-7B, Qwen2.5-7B, Llama-3.1-8B

set -e  # Exit on error

# Define the models
MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Common parameters
DATASET="arc"
NUM_EXAMPLES=250
TAG="nov_ten_arc_only"
MIN_COT_TOKENS=128

echo "=========================================="
echo "Running ARC experiments on 3 models"
echo "Dataset: ${DATASET}"
echo "Number of examples: ${NUM_EXAMPLES}"
echo "Tag: ${TAG}"
echo "Min CoT tokens: ${MIN_COT_TOKENS}"
echo "=========================================="
echo ""

# Run each model
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Starting model: ${MODEL}"
    echo "Time: $(date)"
    echo "=========================================="
    echo ""
    
    python3 -m experiments.phase_a_step1_generate \
        --model_name "${MODEL}" \
        --datasets "${DATASET}" \
        --num_examples ${NUM_EXAMPLES} \
        --tag "${TAG}" \
        --min_cot_tokens ${MIN_COT_TOKENS}
    
    echo ""
    echo "Completed model: ${MODEL}"
    echo "Time: $(date)"
    echo ""
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "All 3 models completed successfully!"
echo "Final time: $(date)"
echo "=========================================="
