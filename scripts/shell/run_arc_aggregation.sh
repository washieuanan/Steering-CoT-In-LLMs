#!/bin/bash

# Script to run Phase A Aggregation across ARC results
# This combines per-dataset screening results into robust layer rankings

set -e  # Exit on error

# Parameters
RESULTS_ROOT="results/phase_a"
TAG="nov_ten_arc_only"
MODELS="Llama-3.1-8B-Instruct,Mistral-7B-Instruct-v0.3,Qwen2.5-7B-Instruct"
DATASETS="arc"
PRIMARY_METRIC="auc"
TOPK=8
WEIGHT_BY="n_examples"

echo "=========================================="
echo "Running Phase A Aggregation"
echo "Tag: ${TAG}"
echo "Models: ${MODELS}"
echo "Datasets: ${DATASETS}"
echo "Parameters:"
echo "  primary_metric: ${PRIMARY_METRIC}"
echo "  topk: ${TOPK}"
echo "  weight_by: ${WEIGHT_BY}"
echo "=========================================="
echo ""

python3 -m experiments.phase_a_aggregate \
    --results_root "${RESULTS_ROOT}" \
    --tag "${TAG}" \
    --models "${MODELS}" \
    --datasets "${DATASETS}" \
    --primary_metric "${PRIMARY_METRIC}" \
    --topk ${TOPK} \
    --weight_by "${WEIGHT_BY}" \
    --save_consensus_directions

echo ""
echo "=========================================="
echo "Aggregation completed successfully!"
echo "Time: $(date)"
echo ""
echo "Results saved in:"
echo "  ${RESULTS_ROOT}/<MODEL>__${TAG}/aggregation/"
echo "=========================================="
