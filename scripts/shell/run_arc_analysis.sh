#!/bin/bash

# Script to run Phase A Step 2 (analysis) on all ARC model results
# This performs offline parsing, screening, and direction extraction

set -e  # Exit on error

# Define the model result directories
RESULT_DIRS=(
    "results/phase_a/Llama-3.1-8B-Instruct__nov_ten_arc_only/arc"
    "results/phase_a/Mistral-7B-Instruct-v0.3__nov_ten_arc_only/arc"
    "results/phase_a/Qwen2.5-7B-Instruct__nov_ten_arc_only/arc"
)

# Analysis parameters (adjust as needed)
PROBE_C=1.0
PROBE_MAX_ITER=1000
TOPK_LAYERS=8
DIRECTION_METHOD="dense_normalized"
NUMERIC_TOL=1e-6

echo "=========================================="
echo "Running Phase A Step 2 Analysis"
echo "Dataset: ARC"
echo "Parameters:"
echo "  probe_C: ${PROBE_C}"
echo "  probe_max_iter: ${PROBE_MAX_ITER}"
echo "  topk_layers: ${TOPK_LAYERS}"
echo "  direction_method: ${DIRECTION_METHOD}"
echo "  numeric_tol: ${NUMERIC_TOL}"
echo "=========================================="
echo ""

# Run analysis for each model result directory
for RUN_DIR in "${RESULT_DIRS[@]}"; do
    echo "=========================================="
    echo "Analyzing: ${RUN_DIR}"
    echo "Time: $(date)"
    echo "=========================================="   
    echo ""
    
    if [ ! -d "${RUN_DIR}" ]; then
        echo "⚠️  WARNING: Directory not found: ${RUN_DIR}"
        echo "   Skipping..."
        echo ""
        continue
    fi
    
    python3 -m experiments.phase_a_step2_analyze \
        --run_dir "${RUN_DIR}" \
        --probe_C ${PROBE_C} \
        --probe_max_iter ${PROBE_MAX_ITER} \
        --topk_layers ${TOPK_LAYERS} \
        --direction_method "${DIRECTION_METHOD}" \
        --numeric_tol ${NUMERIC_TOL}
    
    echo ""
    echo "Completed analysis for: ${RUN_DIR}"
    echo "Time: $(date)"
    echo ""
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "All analyses completed successfully!"
echo "Final time: $(date)"
echo ""
echo "Results saved in each directory under 'screening/' subfolder:"
for RUN_DIR in "${RESULT_DIRS[@]}"; do
    if [ -d "${RUN_DIR}" ]; then
        echo "  - ${RUN_DIR}/screening/"
    fi
done
echo "=========================================="
