#!/bin/bash
# Job Flags 
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH -t 16:00:00
#SBATCH --mem=128G

# export OPENAI_API_KEY='your-api-key-here'  # Set this before running
echo "Loading the conda enviornment"
module load miniforge/24.3.0-0
source  activate nlp_venv
echo $CONDA_DEFAULT_ENV
echo "Activate nlp enviornment"
#
# Phase B SCALED Experiment: CoT Locality Only
# 
# This script runs Phase B interventions with:
# - Locality: cot (inject during CoT generation, before answer phrase)
# - Layer: 25 only
# - n = 100 examples
# - Alpha grid: 0.5, 1, 2
# - Gamma grid: 0.5, 1, 2
# - Beta grid: 0.5, 1, 2
# - GPT grading: gpt-5-nano
#
# Phase A results are used from existing run (commented out).
#

set -e  # Exit on error

echo "=========================================="
echo "Phase B SCALED: CoT Locality Experiments"
echo "Layer 25 Only | n=100 | Full Parameter Grid"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# =============================================================
# CONFIGURATION
# =============================================================

# Model (Qwen2.5 only)
MODEL="Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT="Qwen2.5-7B-Instruct"

# Datasets
DATASETS="mmlu_pro"

# Run tag (for Phase A reference)
TAG="qwen_causality_v3"

# Phase A Settings (for reference - Phase A is commented out)
PHASE_A_OUT="results/phase_a"

# Phase B Settings - SCALED
NUM_EXAMPLES_PHASE_B=100
LAYER=25                  # Focus on Layer 25 only
ADD_MODE="constant"       # RECOMMENDED: constant addition
ALPHA_GRID="0.5,1,2"      # Alpha values for add mode
GAMMA_GRID="0.5 1 2"      # Gamma values for lesion/rescue (space-separated for loop)
BETA_GRID="0.5 1 2"       # Beta values for rescue (space-separated for loop)
JUDGE_MODEL="gpt-5-nano"
PHASE_B_OUT="results/phase_b_scaled"
MAX_GEN_TOKENS=256
SEED=42

# Locality setting - THIS IS THE KEY DIFFERENCE
LOCALITY="cot"

# =============================================================
# PRE-FLIGHT CHECKS
# =============================================================

echo ">>> Pre-flight checks..."

# Check for OPENAI_API_KEY
if [ -z "${OPENAI_API_KEY}" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set!"
    echo "Please run: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi
echo "  ✓ OPENAI_API_KEY is set"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "  ✓ Found ${GPU_COUNT} GPU(s)"
else
    echo "  WARNING: nvidia-smi not found, GPU may not be available"
fi

# Check Python packages
python3 -c "import torch; import transformers; import openai" 2>/dev/null && \
    echo "  ✓ Required Python packages found" || \
    { echo "ERROR: Missing required packages (torch, transformers, openai)"; exit 1; }

echo ""

# =============================================================
# PHASE A - COMMENTED OUT (Using existing results)
# =============================================================

# Phase A Step 1 and Step 2 are commented out.
# Using existing Phase A results from:
#   results/phase_a/Qwen2.5-7B-Instruct__qwen_causality_v3/mmlu_pro

# =============================================================
# PHASE B - CoT Locality Scaled Experiments
# =============================================================

echo "=========================================="
echo "PHASE B - CoT Locality SCALED Experiments"
echo "=========================================="
echo "Locality: ${LOCALITY} (inject during CoT generation phase)"
echo "Layer: ${LAYER}"
echo "Num examples: ${NUM_EXAMPLES_PHASE_B}"
echo "Alpha grid: ${ALPHA_GRID}"
echo "Gamma grid: ${GAMMA_GRID}"
echo "Beta grid: ${BETA_GRID}"
echo "Judge model: ${JUDGE_MODEL}"
echo ""

# Generate a single tag for this run - all modes will share the same directory
PHASE_B_TAG="${LOCALITY}_locality_L${LAYER}_scaled_$(date +%Y%m%d_%H%M%S)"
echo ">>> Phase B tag: ${PHASE_B_TAG}"
echo ""

for DS in ${DATASETS//,/ }; do
    PHASE_A_RUN="${PHASE_A_OUT}/${MODEL_SHORT}__${TAG}/${DS}"
    
    echo "=========================================="
    echo "Dataset: ${DS}, Locality: ${LOCALITY}, Layer: ${LAYER}"
    echo "=========================================="
    
    # ---------------------------------------------------------
    # Mode: add (Sufficiency test)
    # Alpha grid: 0.5, 1, 2
    # ---------------------------------------------------------
    echo ""
    echo ">>> Mode: add (Sufficiency test)"
    echo "    Alpha grid: ${ALPHA_GRID}"
    echo "    Layer: ${LAYER}"
    
    python3 -m experiments.phase_b_runner \
        --model_name "${MODEL}" \
        --phase_a_run "${PHASE_A_RUN}" \
        --datasets "${DS}" \
        --split test \
        --num_examples ${NUM_EXAMPLES_PHASE_B} \
        --layers "${LAYER}" \
        --modes add \
        --add_mode ${ADD_MODE} \
        --locality ${LOCALITY} \
        --alpha_grid="${ALPHA_GRID}" \
        --prompt_mode cot \
        --judge_model "${JUDGE_MODEL}" \
        --max_new_tokens ${MAX_GEN_TOKENS} \
        --seed ${SEED} \
        --out_dir "${PHASE_B_OUT}" \
        --phase_b_tag "${PHASE_B_TAG}"
    
    echo "    ✓ Completed add mode for ${DS} (locality=${LOCALITY}, layer=${LAYER})"
    
    # ---------------------------------------------------------
    # Mode: lesion (Necessity test)
    # Gamma grid: 0.5, 1, 2
    # ---------------------------------------------------------
    echo ""
    echo ">>> Mode: lesion (Necessity test)"
    echo "    Gamma grid: ${GAMMA_GRID}"
    echo "    Layer: ${LAYER}"
    
    for GAMMA in ${GAMMA_GRID}; do
        echo ""
        echo "    Running lesion with gamma=${GAMMA}..."
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DS}" \
            --split test \
            --num_examples ${NUM_EXAMPLES_PHASE_B} \
            --layers "${LAYER}" \
            --modes lesion \
            --locality ${LOCALITY} \
            --gamma ${GAMMA} \
            --prompt_mode cot \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens ${MAX_GEN_TOKENS} \
            --seed ${SEED} \
            --out_dir "${PHASE_B_OUT}" \
            --phase_b_tag "${PHASE_B_TAG}"
        
        echo "    ✓ Completed lesion mode (gamma=${GAMMA}) for ${DS}"
    done
    
    echo "    ✓ Completed all lesion modes for ${DS} (locality=${LOCALITY}, layer=${LAYER})"
    
    # ---------------------------------------------------------
    # Mode: rescue (Causal triad test)
    # Full grid: gamma x beta = 3x3 = 9 combinations
    # ---------------------------------------------------------
    echo ""
    echo ">>> Mode: rescue (Causal triad test)"
    echo "    Gamma x Beta grid: (${GAMMA_GRID}) x (${BETA_GRID}) = 9 combinations"
    echo "    Layer: ${LAYER}"
    
    for GAMMA in ${GAMMA_GRID}; do
        for BETA in ${BETA_GRID}; do
            echo ""
            echo "    Running rescue with gamma=${GAMMA}, beta=${BETA}..."
            
            python3 -m experiments.phase_b_runner \
                --model_name "${MODEL}" \
                --phase_a_run "${PHASE_A_RUN}" \
                --datasets "${DS}" \
                --split test \
                --num_examples ${NUM_EXAMPLES_PHASE_B} \
                --layers "${LAYER}" \
                --modes rescue \
                --locality ${LOCALITY} \
                --gamma ${GAMMA} \
                --beta ${BETA} \
                --prompt_mode cot \
                --judge_model "${JUDGE_MODEL}" \
                --max_new_tokens ${MAX_GEN_TOKENS} \
                --seed ${SEED} \
                --out_dir "${PHASE_B_OUT}" \
                --phase_b_tag "${PHASE_B_TAG}"
            
            echo "    ✓ Completed rescue mode (gamma=${GAMMA}, beta=${BETA}) for ${DS}"
        done
    done
    
    echo "    ✓ Completed all rescue modes for ${DS} (locality=${LOCALITY}, layer=${LAYER})"
    echo ""
done

echo "✓ Completed all modes for locality=${LOCALITY}"

# =============================================================
# Mode: random (Control experiment) - COMMENTED OUT FOR LATER
# =============================================================

# Uncomment below to run random subspace control experiments
#
# echo ""
# echo ">>> Mode: random (Control experiment)"
# echo "    Alpha grid: ${ALPHA_GRID}"
# echo "    Layer: ${LAYER}"
#
# for DS in ${DATASETS//,/ }; do
#     PHASE_A_RUN="${PHASE_A_OUT}/${MODEL_SHORT}__${TAG}/${DS}"
#     
#     python3 -m experiments.phase_b_runner \
#         --model_name "${MODEL}" \
#         --phase_a_run "${PHASE_A_RUN}" \
#         --datasets "${DS}" \
#         --split test \
#         --num_examples ${NUM_EXAMPLES_PHASE_B} \
#         --layers "${LAYER}" \
#         --modes random \
#         --locality ${LOCALITY} \
#         --alpha_grid="${ALPHA_GRID}" \
#         --prompt_mode cot \
#         --judge_model "${JUDGE_MODEL}" \
#         --max_new_tokens ${MAX_GEN_TOKENS} \
#         --seed ${SEED} \
#         --out_dir "${PHASE_B_OUT}" \
#         --phase_b_tag "${PHASE_B_TAG}"
#     
#     echo "    ✓ Completed random mode for ${DS} (locality=${LOCALITY}, layer=${LAYER})"
# done

echo ""
echo "✓ Phase B SCALED complete for locality=${LOCALITY}"
echo ""

# =============================================================
# SUMMARY
# =============================================================

echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Configuration:"
echo "  Locality: ${LOCALITY}"
echo "  Layer: ${LAYER}"
echo "  Num examples: ${NUM_EXAMPLES_PHASE_B}"
echo "  Alpha grid: ${ALPHA_GRID}"
echo "  Gamma grid: ${GAMMA_GRID}"
echo "  Beta grid: ${BETA_GRID}"
echo ""
echo "Results location:"
echo "  ${PHASE_B_OUT}/${MODEL_SHORT}__${PHASE_B_TAG}/${DS}/"
echo ""
echo "Key output files:"
echo "  - grid.csv: Summary of add/lesion results"
echo "  - rescue_summary.csv: Summary of rescue results"
echo "  - runs/: Individual experiment CSVs"
echo ""
echo "Experiment runs:"
echo "  - Add mode: 3 alpha values (0.5, 1, 2)"
echo "  - Lesion mode: 3 gamma values (0.5, 1, 2)"  
echo "  - Rescue mode: 9 gamma x beta combinations"
echo "  - Random mode: COMMENTED OUT (for later)"
echo ""
echo "=========================================="
