#!/bin/bash
#
# Full Pipeline Script: Step 1 → Step 2 → Aggregation
# Runs Phase A complete workflow for all models and datasets
#

set -e  # Exit on error

echo "=========================================="
echo "Starting Full Pipeline Execution"
echo "=========================================="
echo ""

# ============================================================
# CONFIGURABLE PARAMETERS
# ============================================================

# Datasets to process (comma-separated)
export DSETS="arc,gsm8k,mmlu_pro"

# Models to process (array)
export MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.3"
  "Qwen/Qwen2.5-7B-Instruct"
  "meta-llama/Llama-3.1-8B-Instruct"
)

# Tag for this run
export TAG="test_run"

# Output root directory
export OUT_ROOT="results/phase_a"

# Step 1 parameters
export NUM=100          # examples per dataset
export MIN_COT=128        # min CoT tokens
export MAX_GEN=256        # max new tokens to generate
export TEMP=0.0           # deterministic
export SEED=42

# Step 2 parameters
export PROBE_C=1.0
export PROBE_MAX_ITER=1000
export TOPK_LAYERS=8
export DIRECTION_METHOD="dense_normalized"
export NUMERIC_TOL=1e-6

# Aggregation parameters
export PRIMARY_METRIC="auc"
export WEIGHT_BY="n_examples"

# ============================================================
# STEP 1: GENERATE (All models × all datasets)
# ============================================================

echo "=========================================="
echo "STEP 1: GENERATING OUTPUTS"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Datasets: ${DSETS}"
echo "Tag: ${TAG}"
echo ""

for MODEL in "${MODELS[@]}"; do
  echo "================================================"
  echo "==> STEP 1 | MODEL: ${MODEL}"
  echo "================================================"
  
  python3 -m experiments.phase_a_step1_generate \
    --model_name "$MODEL" \
    --datasets "$DSETS" \
    --num_examples $NUM \
    --min_cot_tokens $MIN_COT \
    --max_new_tokens $MAX_GEN \
    --temperature $TEMP \
    --tag "$TAG" \
    --seed $SEED \
    --output_dir "$OUT_ROOT" \
    --dtype bfloat16 \
    --device auto \
    --stop_on_final_answer
  
  echo ""
  echo "✓ Completed Step 1 for ${MODEL}"
  echo ""
done

echo "✓ STEP 1 COMPLETE: All models processed"
echo ""

# ============================================================
# STEP 2: ANALYZE (Screening & directions for each model × dataset)
# ============================================================

echo "=========================================="
echo "STEP 2: ANALYZING (SCREENING & DIRECTIONS)"
echo "=========================================="
echo ""

for MODEL in "${MODELS[@]}"; do
  # Extract model directory name (remove org prefix)
  MODEL_DIR=$(basename "$MODEL")
  
  echo "================================================"
  echo "==> STEP 2 | MODEL: ${MODEL_DIR}"
  echo "================================================"
  
  # Process each dataset
  for DS in ${DSETS//,/ }; do
    RUN_DIR="$OUT_ROOT/${MODEL_DIR}__${TAG}/${DS}"
    
    echo "  → Processing dataset: ${DS}"
    echo "    Run directory: ${RUN_DIR}"
    
    python3 -m experiments.phase_a_step2_analyze \
      --run_dir "$RUN_DIR" \
      --probe_C $PROBE_C \
      --probe_max_iter $PROBE_MAX_ITER \
      --topk_layers $TOPK_LAYERS \
      --direction_method "$DIRECTION_METHOD" \
      --numeric_tol $NUMERIC_TOL
    
    echo "    ✓ Completed analysis for ${DS}"
    echo ""
  done
  
  echo "✓ Completed Step 2 for ${MODEL_DIR}"
  echo ""
done

echo "✓ STEP 2 COMPLETE: All analyses finished"
echo ""

# ============================================================
# STEP 3: AGGREGATE (Across datasets per model)
# ============================================================

echo "=========================================="
echo "STEP 3: AGGREGATING RESULTS"
echo "=========================================="
echo ""

# Build models list (basename only, comma-separated)
MODEL_NAMES=""
for MODEL in "${MODELS[@]}"; do
  MODEL_DIR=$(basename "$MODEL")
  if [ -z "$MODEL_NAMES" ]; then
    MODEL_NAMES="$MODEL_DIR"
  else
    MODEL_NAMES="${MODEL_NAMES},${MODEL_DIR}"
  fi
done

echo "Models for aggregation: ${MODEL_NAMES}"
echo "Datasets: ${DSETS}"
echo "Primary metric: ${PRIMARY_METRIC}"
echo "Top-K layers: ${TOPK_LAYERS}"
echo ""

python3 -m experiments.phase_a_aggregate \
  --results_root "$OUT_ROOT" \
  --tag "$TAG" \
  --models "$MODEL_NAMES" \
  --datasets "$DSETS" \
  --primary_metric $PRIMARY_METRIC \
  --topk $TOPK_LAYERS \
  --weight_by $WEIGHT_BY \
  --save_consensus_directions

echo ""
echo "✓ STEP 3 COMPLETE: Aggregation finished"
echo ""

# ============================================================
# SUMMARY
# ============================================================

echo "=========================================="
echo "PIPELINE EXECUTION COMPLETE"
echo "=========================================="
echo ""
echo "Results location: ${OUT_ROOT}"
echo ""
echo "Key output locations:"
echo "  - Step 1 outputs: ${OUT_ROOT}/<MODEL_DIR>__${TAG}/<dataset>/"
echo "  - Step 2 outputs: ${OUT_ROOT}/<MODEL_DIR>__${TAG}/<dataset>/screening/"
echo "  - Aggregation:    ${OUT_ROOT}/<MODEL_DIR>__${TAG}/aggregation/"
echo ""
echo "Next steps:"
echo "  - Review summary_layers.csv in each model's aggregation directory"
echo "  - Check consensus_top_layers.txt for selected layers"
echo "  - Verify layer_to_U_consensus.npz for consensus directions"
echo ""
echo "Pipeline completed successfully!"
