#!/bin/bash

# =============================================================================
# Phase A with GPT Grading - Full Pipeline Script
# =============================================================================
# Runs Phase A (generation + GPT annotation + analysis) for:
#   - 3 models: Mistral-7B, Qwen2.5-7B, Llama-3.1-8B
#   - 3 datasets: arc, mmlu_pro, gsm8k
#
# Total: 9 combinations, run sequentially
#
# Prerequisites:
#   - Set OPENAI_API_KEY environment variable
#   - GPU with ~40GB memory recommended
# =============================================================================

set -e  # Exit on error

# Check for API key
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set!"
    echo "Please run: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# =============================================================================
# Configuration
# =============================================================================

MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

DATASETS=(
    "arc"
    "mmlu_pro"
    "gsm8k"
)

# Common parameters
NUM_EXAMPLES=100
TAG="phase_a_gpt_graded"
MIN_COT_TOKENS=128
MAX_NEW_TOKENS=512
SEED=42
JUDGE_MODEL="gpt-5-nano"
OUTPUT_DIR="results/phase_a"

# =============================================================================
# Main Loop
# =============================================================================

echo "=========================================="
echo "Phase A with GPT Grading - Full Pipeline"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Examples per run: ${NUM_EXAMPLES}"
echo "Tag: ${TAG}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Start time: $(date)"
echo "=========================================="
echo ""

TOTAL_COMBOS=$((${#MODELS[@]} * ${#DATASETS[@]}))
CURRENT=0

for MODEL in "${MODELS[@]}"; do
    # Extract short model name for path construction
    MODEL_SHORT=$(basename "${MODEL}")
    
    for DATASET in "${DATASETS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "=========================================="
        echo "[${CURRENT}/${TOTAL_COMBOS}] ${MODEL_SHORT} / ${DATASET}"
        echo "=========================================="
        echo "Start: $(date)"
        echo ""
        
        # Construct the run directory path
        # Phase A Step 1 creates: results/phase_a/{MODEL_SHORT}__{TAG}/{DATASET}/
        RUN_DIR="${OUTPUT_DIR}/${MODEL_SHORT}__${TAG}/${DATASET}"
        
        # ---------------------------------------------------------------------
        # Step 1: Generate (pooled.npz, generations.csv)
        # ---------------------------------------------------------------------
        echo ">>> Step 1: Generation"
        echo "    Model: ${MODEL}"
        echo "    Dataset: ${DATASET}"
        echo "    Examples: ${NUM_EXAMPLES}"
        echo ""
        
        python3 -m experiments.phase_a_step1_generate \
            --model_name "${MODEL}" \
            --datasets "${DATASET}" \
            --split test \
            --num_examples ${NUM_EXAMPLES} \
            --tag "${TAG}" \
            --min_cot_tokens ${MIN_COT_TOKENS} \
            --max_new_tokens ${MAX_NEW_TOKENS} \
            --temperature 0.0 \
            --seed ${SEED} \
            --output_dir "${OUTPUT_DIR}" \
            --stop_on_final_answer
        
        echo ""
        echo ">>> Step 1 complete"
        echo ""
        
        # ---------------------------------------------------------------------
        # Step 1.5: GPT Annotation (generations_with_judgments.csv)
        # ---------------------------------------------------------------------
        echo ">>> Step 1.5: GPT Annotation"
        echo "    Run dir: ${RUN_DIR}"
        echo "    Judge: ${JUDGE_MODEL}"
        echo ""
        
        python3 -m scripts.annotate_reasoning_with_gpt \
            --run_dir "${RUN_DIR}" \
            --judge_model "${JUDGE_MODEL}"
        
        echo ""
        echo ">>> Step 1.5 complete"
        echo ""
        
        # ---------------------------------------------------------------------
        # Step 2: Analyze with GPT labels (screening/layer_to_U.npz)
        # ---------------------------------------------------------------------
        echo ">>> Step 2: Analysis with GPT labels"
        echo "    Run dir: ${RUN_DIR}"
        echo ""
        
        python3 -m experiments.phase_a_step2_analyze \
            --run_dir "${RUN_DIR}" \
            --use_gpt_labels \
            --topk_layers 8 \
            --probe_C 1.0 \
            --probe_max_iter 1000 \
            --direction_method dense_normalized
        
        echo ""
        echo ">>> Step 2 complete"
        echo ""
        
        echo "=========================================="
        echo "[${CURRENT}/${TOTAL_COMBOS}] ${MODEL_SHORT} / ${DATASET} DONE"
        echo "End: $(date)"
        echo "=========================================="
        echo ""
        
    done
    
    echo ""
    echo "=========================================="
    echo "Completed all datasets for: ${MODEL_SHORT}"
    echo "=========================================="
    echo ""
    
done

echo ""
echo "=========================================="
echo "ALL RUNS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Total combinations: ${TOTAL_COMBOS}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Final time: $(date)"
echo ""
echo "Next steps:"
echo "  1. Review results in ${OUTPUT_DIR}/"
echo "  2. Run Phase B interventions on the generated directions"
echo "=========================================="
