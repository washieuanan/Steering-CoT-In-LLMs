#!/bin/bash
#
# Qwen2.5 Full Pipeline: Phase A + Phase B with Pre-CoT vs Post-CoT Locality
#
# This implements causal interventions with locality control:
# - cot locality: Inject during CoT generation (before answer phrase)
# - answer locality: Inject during answer generation (after answer phrase)
#
# Modes:
# - Add (Sufficiency): h' = h + α * u
# - Lesion (Necessity): h' = h - γ * P_S(h)
# - Rescue: h' = h + (β - γ) * P_S(h)
# - Random: Control with random subspace
#

set -e  # Exit on error

echo "=========================================="
echo "Qwen2.5 Full Pipeline: Phase A + Phase B"
echo "Pre-CoT vs Post-CoT Locality Experiments"
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
DATASETS="arc,gsm8k,mmlu_pro"

# Run tag
TAG="qwen_causality_v1"

# Phase A Settings
NUM_EXAMPLES=100
NUM_EXAMPLES_PHASE_B=25
MIN_COT_TOKENS=128
MAX_GEN_TOKENS=256
TEMPERATURE=0.0
SEED=42
PHASE_A_OUT="results/phase_a"

# Phase A Step 2 Settings
PROBE_C=1.0
PROBE_MAX_ITER=1000
TOPK_LAYERS=3
DIRECTION_METHOD="dense_normalized"

# Phase B Settings
ADD_MODE="constant"      # RECOMMENDED: constant addition
ALPHA_GRID="0.5,1,2"     # Positive alphas only (lesion handles removal)
GAMMA=1.0                # Lesion strength (full removal)
BETA=1.0                 # Rescue restore strength
JUDGE_MODEL="gpt-4.1-mini"
PHASE_B_OUT="results/phase_b_new"

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
# PHASE A - STEP 1: Generate CoT traces (uncomment to run)
# =============================================================

# echo "=========================================="
# echo "PHASE A - Step 1: Generate CoT traces"
# echo "=========================================="
# echo "Model: ${MODEL}"
# echo "Datasets: ${DATASETS}"
# echo "Examples per dataset: ${NUM_EXAMPLES}"
# echo ""

# python3 -m experiments.phase_a_step1_generate \
#     --model_name "${MODEL}" \
#     --datasets "${DATASETS}" \
#     --num_examples ${NUM_EXAMPLES} \
#     --min_cot_tokens ${MIN_COT_TOKENS} \
#     --max_new_tokens ${MAX_GEN_TOKENS} \
#     --temperature ${TEMPERATURE} \
#     --tag "${TAG}" \
#     --seed ${SEED} \
#     --output_dir "${PHASE_A_OUT}" \
#     --dtype bfloat16 \
#     --device auto \
#     --stop_on_final_answer

# echo ""
# echo "✓ Phase A Step 1 complete"
# echo ""

# =============================================================
# PHASE A - STEP 2: Analyze & extract directions (uncomment to run)
# =============================================================

# echo "=========================================="
# echo "PHASE A - Step 2: Analyze & extract directions"
# echo "=========================================="
# echo "Direction method: ${DIRECTION_METHOD}"
# echo "Top-k layers: ${TOPK_LAYERS}"
# echo ""

# for DS in ${DATASETS//,/ }; do
#     RUN_DIR="${PHASE_A_OUT}/${MODEL_SHORT}__${TAG}/${DS}"
    
#     echo ">>> Processing dataset: ${DS}"
#     echo "    Run directory: ${RUN_DIR}"
    
#     python3 -m experiments.phase_a_step2_analyze \
#         --run_dir "${RUN_DIR}" \
#         --probe_C ${PROBE_C} \
#         --probe_max_iter ${PROBE_MAX_ITER} \
#         --topk_layers ${TOPK_LAYERS} \
#         --direction_method "${DIRECTION_METHOD}"
    
#     echo "    ✓ Completed analysis for ${DS}"
#     echo ""
# done

# echo "✓ Phase A Step 2 complete"
# echo ""

# =============================================================
# PHASE B - Run for both localities (cot and answer)
# =============================================================

echo "=========================================="
echo "PHASE B - Pre-CoT vs Post-CoT Locality Experiments"
echo "=========================================="
echo "Localities to test: cot, answer"
echo "Modes: add, lesion, rescue, random"
echo "Alpha grid: ${ALPHA_GRID}"
echo ""

for LOCALITY in cot answer; do
    echo ""
    echo "###################################################"
    echo "# LOCALITY: ${LOCALITY}"
    echo "# (inject during ${LOCALITY} generation phase)"
    echo "###################################################"
    echo ""
    
    for DS in ${DATASETS//,/ }; do
        PHASE_A_RUN="${PHASE_A_OUT}/${MODEL_SHORT}__${TAG}/${DS}"
        
        echo "=========================================="
        echo "Dataset: ${DS}, Locality: ${LOCALITY}"
        echo "=========================================="
        
        # ---------------------------------------------------------
        # Mode: add (Sufficiency test)
        # ---------------------------------------------------------
        echo ""
        echo ">>> Mode: add (Sufficiency test)"
        echo "    Alpha grid: ${ALPHA_GRID}"
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DS}" \
            --split test \
            --num_examples ${NUM_EXAMPLES_PHASE_B} \
            --topk_layers ${TOPK_LAYERS} \
            --modes add \
            --add_mode ${ADD_MODE} \
            --locality ${LOCALITY} \
            --alpha_grid="${ALPHA_GRID}" \
            --prompt_mode cot \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens ${MAX_GEN_TOKENS} \
            --seed ${SEED} \
            --out_dir "${PHASE_B_OUT}"
        
        echo "    ✓ Completed add mode for ${DS} (locality=${LOCALITY})"
        
        # ---------------------------------------------------------
        # Mode: lesion (Necessity test)
        # ---------------------------------------------------------
        echo ""
        echo ">>> Mode: lesion (Necessity test)"
        echo "    Gamma (lesion strength): ${GAMMA}"
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DS}" \
            --split test \
            --num_examples ${NUM_EXAMPLES_PHASE_B} \
            --topk_layers ${TOPK_LAYERS} \
            --modes lesion \
            --locality ${LOCALITY} \
            --gamma ${GAMMA} \
            --prompt_mode cot \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens ${MAX_GEN_TOKENS} \
            --seed ${SEED} \
            --out_dir "${PHASE_B_OUT}"
        
        echo "    ✓ Completed lesion mode for ${DS} (locality=${LOCALITY})"
        
        # ---------------------------------------------------------
        # Mode: rescue (Causal triad test)
        # ---------------------------------------------------------
        echo ""
        echo ">>> Mode: rescue (Causal triad test)"
        echo "    Gamma: ${GAMMA}, Beta: ${BETA}"
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DS}" \
            --split test \
            --num_examples ${NUM_EXAMPLES_PHASE_B} \
            --topk_layers ${TOPK_LAYERS} \
            --modes rescue \
            --locality ${LOCALITY} \
            --gamma ${GAMMA} \
            --beta ${BETA} \
            --prompt_mode cot \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens ${MAX_GEN_TOKENS} \
            --seed ${SEED} \
            --out_dir "${PHASE_B_OUT}"
        
        echo "    ✓ Completed rescue mode for ${DS} (locality=${LOCALITY})"
        
        # ---------------------------------------------------------
        # Mode: random (Control experiment)
        # ---------------------------------------------------------
        echo ""
        echo ">>> Mode: random (Control experiment)"
        echo "    Alpha grid: ${ALPHA_GRID}"
        
        python3 -m experiments.phase_b_runner \
            --model_name "${MODEL}" \
            --phase_a_run "${PHASE_A_RUN}" \
            --datasets "${DS}" \
            --split test \
            --num_examples ${NUM_EXAMPLES_PHASE_B} \
            --topk_layers ${TOPK_LAYERS} \
            --modes random \
            --locality ${LOCALITY} \
            --alpha_grid="${ALPHA_GRID}" \
            --prompt_mode cot \
            --judge_model "${JUDGE_MODEL}" \
            --max_new_tokens ${MAX_GEN_TOKENS} \
            --seed ${SEED} \
            --out_dir "${PHASE_B_OUT}"
        
        echo "    ✓ Completed random mode for ${DS} (locality=${LOCALITY})"
        echo ""
    done
    
    echo "✓ Completed all modes for locality=${LOCALITY}"
done

echo ""
echo "✓ Phase B complete for both localities"
echo ""

# =============================================================
# SUMMARY
# =============================================================

echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results locations:"
echo "  Phase A: ${PHASE_A_OUT}/${MODEL_SHORT}__${TAG}/"
echo "  Phase B: ${PHASE_B_OUT}/"
echo ""
echo "Key output files:"
echo "  - Phase A directions: <dataset>/screening/layer_to_U.npz"
echo "  - Phase B grid summary: grid.csv (per locality)"
echo "  - Phase B rescue summary: rescue_summary.csv (per locality)"
echo ""
echo "Experiment structure:"
echo "  - Qwen2.5-7B-Instruct__cot_locality_<timestamp>/<dataset>/"
echo "  - Qwen2.5-7B-Instruct__answer_locality_<timestamp>/<dataset>/"
echo ""
echo "Analysis questions:"
echo "  1. Does adding reasoning vector during CoT help? (locality=cot, mode=add)"
echo "  2. Does adding it during answer-only help/hurt? (locality=answer, mode=add)"
echo "  3. Is reasoning vector necessary during CoT? (locality=cot, mode=lesion)"
echo "  4. Can we recover after lesioning? (locality=cot/answer, mode=rescue)"
echo "  5. Are effects specific to learned vector? (mode=random should show ~0 effect)"
echo ""
echo "Next steps:"
echo "  1. Run experiments/phase_b_analysis.ipynb to visualize results"
echo "  2. Compare grid.csv across localities (cot vs answer)"
echo "  3. Check rescue_summary.csv for causal triad evidence"
echo "=========================================="
