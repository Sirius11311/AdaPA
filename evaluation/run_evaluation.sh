#!/bin/bash
# ============================================================
# AdaPA-Agent CRS Evaluation Script
# ============================================================
# This script runs AdaPA-Agent for conversational recommendation
# system evaluation with preference arithmetic.
#
# USAGE:
#   ./run_evaluation.sh                    # Use default settings
#   ./run_evaluation.sh --gpu 0            # Specify GPU
#   ./run_evaluation.sh --use_pa           # Enable preference arithmetic
# ============================================================

# Default Configuration
GPU_ID="0"
USE_PREFERENCE_ARITHMETIC=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --use_pa|--use_preference_arithmetic)
            USE_PREFERENCE_ARITHMETIC=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${GPU_ID}"

# API Configuration (replace with your own API key)
API_KEY="${ADAPA_API_KEY:-your-api-key}"
API_BASE="${ADAPA_API_BASE:-https://api.openai.com/v1/}"

# Model IDs
USER_MODEL_ID=2  # gpt-4o-mini
CRS_MODEL_ID=2   # gpt-4o-mini

# Experiment Parameters
SAMPLE_SIZE=50
CANDIDATE_SIZE=5
REC_NUM=3
MAX_WORKERS=10
SEEDS="1 10 100"
TURNS="3 5 7"

# AdaPA Parameters
K=3
ALPHA=1.0

# Preference Arithmetic Configuration (if enabled)
# Replace with your local model path
GENERATION_MODEL="${ADAPA_GENERATION_MODEL:-/path/to/Llama-2-7b-chat-hf}"
HF_TOKEN="${ADAPA_HF_TOKEN:-your-hf-token}"

# Data Paths
CONVERSATION_TRACE_PATH="../data/processed/filtered_test_data_K_5_M_5.jsonl"

# ============================================================
echo "============================================================"
echo "AdaPA-Agent CRS Evaluation"
echo "============================================================"
echo "Starting time: $(date)"
echo ""

cd "$(dirname "$0")"

# Build command
CMD="python user_simulator.py \
    --user_model_id ${USER_MODEL_ID} \
    --crs_model_id ${CRS_MODEL_ID} \
    --api_key \"${API_KEY}\" \
    --api_base \"${API_BASE}\" \
    --sample_size ${SAMPLE_SIZE} \
    --candidate_size ${CANDIDATE_SIZE} \
    --rec_num ${REC_NUM} \
    --max_workers ${MAX_WORKERS} \
    --seeds ${SEEDS} \
    --turns ${TURNS} \
    --K ${K} \
    --alpha ${ALPHA} \
    --conversation_trace_path \"${CONVERSATION_TRACE_PATH}\""

# Add preference arithmetic flags if enabled
if [ "$USE_PREFERENCE_ARITHMETIC" = true ]; then
    echo "Preference Arithmetic: ENABLED"
    echo "Generation Model: ${GENERATION_MODEL}"
    
    # Set offline mode for local models
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    
    CMD="${CMD} \
        --use_preference_arithmetic \
        --generation_model \"${GENERATION_MODEL}\" \
        --hf_token \"${HF_TOKEN}\""
else
    echo "Preference Arithmetic: DISABLED (using API-based generation)"
fi

echo ""

# Run evaluation
eval $CMD "$@"

echo ""
echo "============================================================"
echo "Evaluation completed at: $(date)"
echo "============================================================"

