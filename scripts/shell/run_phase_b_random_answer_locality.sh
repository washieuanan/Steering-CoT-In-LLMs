#!/bin/bash
# Job Flags 
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 4
#SBATCH -t 06:00:00
#SBATCH --mem=128G

# export OPENAI_API_KEY='your-api-key-here'  # Set this before running
echo "Loading the conda enviornment"
module load miniforge/24.3.0-0
source  activate nlp_venv
echo $CONDA_DEFAULT_ENV
echo "Activate nlp enviornment"
#
# Phase B RANDOM SUBSPACE CONTROL: Answer Locality
# 
# This script runs Phase B interventions using RANDOM ORTHOGONAL SUBSPACE
# as a control experiment to verify effects are specific to learned direction.
#
# Key difference from main experiments:
# - Uses --use_random_subspace flag
# - All interventions (add, lesion, rescue) use random directions
#
# Settings:
# - Locality: answer (inject during answer generation, after answer phrase)
# - Layer: 25 only
# - n = 100 examples
# - Alpha grid: 0.5, 1, 2
# - Gamma grid: 0.5, 1, 2
# - Beta grid: 0.5, 1, 2
# - GPT grading: gpt-5-nano
#

set -e  # Exit on error

echo "=========================================="
echo "Phase B RANDOM CONTROL: Answer Locality"
echo "Layer 25 | n=100 | Random Orthogonal Subspace"
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

# Phase A Settings (for reference - Phase A provides subspace dimensions)
PHASE_A_OUT="results/phase_a"

# Phase B Settings - RANDOM CONTROL
NUM_EXAMPLES_PHASE_B=100
LAYER=25                  # Focus on Layer 25 only
ADD_MODE="constant"       # RECOMMENDED: constant addition
ALPHA_GRID="0.5,1,2"      # Alpha values for add mode
GAMMA_GRID="0.5 1 2"      # Gamma values for lesion/rescue (space-separated for loop)
BETA_GRID="0.5 1 2"       # Beta values for rescue (space-separated for loop)
JUDGE_MODEL="gpt-5-nano"
PHASE_B_OUT="results/phase_b_random"
MAX_GEN_TOKENS=256
SEED=42

# Locality setting
LOCALITY="answer"

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
# PHASE B - Random Subspace Control Experiments (Answer Locality)
# =============================================================

echo "=========================================="
echo "PHASE B - RANDOM SUBSPACE CONTROL"
echo "=========================================="
echo "⚠️  CONTROL EXPERIMENT: Using random orthogonal subspace"
echo "    This tests whether effects are specific to learned direction"
echo ""
echo "Locality: ${LOCALITY} (inject during answer generation phase)"
echo "Layer: ${LAYER}"
echo "Num examples: ${NUM_EXAMPLES_PHASE_B}"
echo "Alpha grid: ${ALPHA_GRID}"
echo "Gamma grid: ${GAMMA_GRID}"
echo "Beta grid: ${BETA_GRID}"
echo "Judge model: ${JUDGE_MODEL}"
echo ""

# Generate a single tag for this run - all modes will share the same directory
PHASE_B_TAG="random_${LOCALITY}_locality_L${LAYER}_$(date +%Y%m%d_%H%M%S)"
echo ">>> Phase B tag: ${PHASE_B_TAG}"
echo ""

for DS in ${DATASETS//,/ }; do
    PHASE_A_RUN="${PHASE_A_OUT}/${MODEL_SHORT}__${TAG}/${DS}"
    
    echo "=========================================="
    echo "Dataset: ${DS}, Locality: ${LOCALITY}, Layer: ${LAYER}"
    echo "CONTROL: Random Subspace"
    echo "=========================================="
    
    # ---------------------------------------------------------
    # Mode: add with random subspace (Control for sufficiency)
    # Alpha grid: 0.5, 1, 2
    # ---------------------------------------------------------
    echo ""
    echo ">>> Mode: add with RANDOM subspace (Control for sufficiency)"
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
        --phase_b_tag "${PHASE_B_TAG}" \
        --use_random_subspace
    
    echo "    ✓ Completed random add mode for ${DS} (locality=${LOCALITY}, layer=${LAYER})"
    
    # ---------------------------------------------------------
    # Mode: lesion with random subspace (Control for necessity)
    # Gamma grid: 0.5, 1, 2
    # ---------------------------------------------------------
    echo ""
    echo ">>> Mode: lesion with RANDOM subspace (Control for necessity)"
    echo "    Gamma grid: ${GAMMA_GRID}"
    echo "    Layer: ${LAYER}"
    
    for GAMMA in ${GAMMA_GRID}; do
        echo ""
        echo "    Running random lesion with gamma=${GAMMA}..."
        
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
            --phase_b_tag "${PHASE_B_TAG}" \
            --use_random_subspace
        
        echo "    ✓ Completed random lesion mode (gamma=${GAMMA}) for ${DS}"
    done
    
    echo "    ✓ Completed all random lesion modes for ${DS} (locality=${LOCALITY}, layer=${LAYER})"
    
    # ---------------------------------------------------------
    # Mode: rescue with random subspace (Control for causal triad)
    # Full grid: gamma x beta = 3x3 = 9 combinations
    # ---------------------------------------------------------
    echo ""
    echo ">>> Mode: rescue with RANDOM subspace (Control for causal triad)"
    echo "    Gamma x Beta grid: (${GAMMA_GRID}) x (${BETA_GRID}) = 9 combinations"
    echo "    Layer: ${LAYER}"
    
    for GAMMA in ${GAMMA_GRID}; do
        for BETA in ${BETA_GRID}; do
            echo ""
            echo "    Running random rescue with gamma=${GAMMA}, beta=${BETA}..."
            
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
                --phase_b_tag "${PHASE_B_TAG}" \
                --use_random_subspace
            
            echo "    ✓ Completed random rescue mode (gamma=${GAMMA}, beta=${BETA}) for ${DS}"
        done
    done
    
    echo "    ✓ Completed all random rescue modes for ${DS} (locality=${LOCALITY}, layer=${LAYER})"
    echo ""
done

echo "✓ Completed all random control modes for locality=${LOCALITY}"
echo ""

# =============================================================
# SUMMARY
# =============================================================

echo "=========================================="
echo "RANDOM CONTROL PIPELINE COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Configuration:"
echo "  CONTROL: Random orthogonal subspace"
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
echo "  - grid.csv: Summary of add/lesion results with random subspace"
echo "  - rescue_summary.csv: Summary of rescue results with random subspace"
echo "  - runs/: Individual experiment CSVs"
echo ""
echo "Experiment runs:"
echo "  - Random Add mode: 3 alpha values (0.5, 1, 2)"
echo "  - Random Lesion mode: 3 gamma values (0.5, 1, 2)"  
echo "  - Random Rescue mode: 9 gamma x beta combinations"
echo ""
echo "Expected results for CONTROL:"
echo "  - Delta values should be near zero (no systematic effect)"
echo "  - This validates that learned direction effects are specific"
echo ""
echo "=========================================="
