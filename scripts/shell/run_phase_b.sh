#!/bin/bash

# =============================================================================
# Phase B: Causal Interventions - Full Pipeline Script
# =============================================================================
# Runs Phase B interventions using Phase A artifacts for:
#   - 3 models: Mistral-7B, Qwen2.5-7B, Llama-3.1-8B
#   - 3 datasets: arc, mmlu_pro, gsm8k
#
# Modes:
#   - add: Alpha sweep from -2 to +2
#   - ara: Add-Remove-Add toggle test
#   - random: Random subspace control
#
# Total: 9 combinations, run sequentially
#
# Prerequisites:
#   - Set OPENAI_API_KEY environment variable
#   - Phase A with GPT grading completed
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
    # "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen2.5-7B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
)

DATASETS=(
    "arc"
    "mmlu_pro"
    "gsm8k"
)

# Common parameters
NUM_EXAMPLES=20
PHASE_A_TAG="phase_a_gpt_graded"
PHASE_A_DIR="results/phase_a"
PHASE_B_DIR="results/phase_b"
SEED=42
JUDGE_MODEL="gpt-5-nano"

# Alpha grid (negative = suppression, positive = amplification)
ALPHA_GRID="-2,-1,-0.5,0,0.5,1,2"
ALPHA_GRID_1="1"
ALPHA_GRID_RANDOM="-2,-1,0,1,2"  # Smaller grid for random control

# Layers to intervene (use top-k from Phase A)
TOPK_LAYERS=1
LAYERS=31

# =============================================================================
# Main Loop
# =============================================================================

echo "=========================================="
echo "Phase B: Causal Interventions - Full Pipeline"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Examples per run: ${NUM_EXAMPLES}"
echo "Top-k layers: ${TOPK_LAYERS}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Alpha grid: ${ALPHA_GRID}"
echo "Phase A dir: ${PHASE_A_DIR}"
echo "Phase B dir: ${PHASE_B_DIR}"
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
        
        # Construct Phase A run directory path
        PHASE_A_RUN="${PHASE_A_DIR}/${MODEL_SHORT}__${PHASE_A_TAG}/${DATASET}"
        
        # Check if Phase A run exists
        if [ ! -d "${PHASE_A_RUN}" ]; then
            echo "WARNING: Phase A run not found: ${PHASE_A_RUN}"
            echo "Skipping this combination. Run Phase A first!"
            echo ""
            continue
        fi
        
        # Check if directions exist
        if [ ! -f "${PHASE_A_RUN}/screening/layer_to_U.npz" ]; then
            echo "WARNING: No directions found: ${PHASE_A_RUN}/screening/layer_to_U.npz"
            echo "Skipping this combination. Run Phase A analysis first!"
            echo ""
            continue
        fi
        
        echo "Phase A run: ${PHASE_A_RUN}"
        echo ""
        
        # ---------------------------------------------------------------------
        # Mode 1: add (Alpha Sweep)
        # ---------------------------------------------------------------------
        echo ">>> Mode 1: add (Alpha Sweep)"
        echo "    Alpha grid: ${ALPHA_GRID}"
        echo ""
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DATASET}" \
            --split test \
            --num_examples ${NUM_EXAMPLES} \
            --topk_layers ${TOPK_LAYERS} \
            --modes add \
            --alpha_grid=${ALPHA_GRID} \
            --prompt_mode phase_b \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens 256 \
            --seed ${SEED} \
            --out_dir "${PHASE_B_DIR}"
        
        echo ""
        echo ">>> Mode 1 (add) complete"
        echo ""
        
        # ---------------------------------------------------------------------
        # Mode 2: ara (Add-Remove-Add Toggle Test)
        # ---------------------------------------------------------------------
        echo ">>> Mode 2: ara (Add-Remove-Add Toggle Test)"
        echo "    Alpha: 1"
        echo ""
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DATASET}" \
            --split test \
            --num_examples ${NUM_EXAMPLES} \
            --topk_layers ${TOPK_LAYERS} \
            --modes ara \
            --alpha_grid=${ALPHA_GRID_1} \
            --prompt_mode phase_b \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens 256 \
            --seed ${SEED} \
            --out_dir "${PHASE_B_DIR}"
        
        echo ""
        echo ">>> Mode 2 (ara) complete"
        echo ""
        
        # ---------------------------------------------------------------------
        # Mode 3: random (Control Experiment)
        # ---------------------------------------------------------------------
        echo ">>> Mode 3: random (Control Experiment)"
        echo "    Alpha grid: ${ALPHA_GRID_RANDOM}"
        echo ""
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DATASET}" \
            --split test \
            --num_examples ${NUM_EXAMPLES} \
            --topk_layers ${TOPK_LAYERS} \
            --modes random \
            --alpha_grid="${ALPHA_GRID_RANDOM}" \
            --prompt_mode phase_b \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens 256 \
            --seed ${SEED} \
            --out_dir "${PHASE_B_DIR}"
        
        echo ""
        echo ">>> Mode 3 (random) complete"
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
echo "ALL PHASE B RUNS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Total combinations: ${TOTAL_COMBOS}"
echo "Output directory: ${PHASE_B_DIR}"
echo "Final time: $(date)"
echo ""
echo "Results structure:"
echo "  ${PHASE_B_DIR}/{MODEL}__intv_{TIMESTAMP}/{DATASET}/"
echo "    - run_config.json"
echo "    - grid.csv          (add/random mode summaries)"
echo "    - ara_summary.csv   (ara mode summaries)"
echo "    - runs/             (per-config paired outputs)"
echo ""
echo "Next steps:"
echo "  1. Aggregate results across models/datasets"
echo "  2. Generate plots for alpha sweep effects"
echo "  3. Analyze ARA consistency/recovery rates"
echo "  4. Compare learned vs random subspace effects"
echo "=========================================="
