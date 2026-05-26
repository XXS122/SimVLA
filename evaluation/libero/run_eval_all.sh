#!/bin/bash
# =============================================================================
# SimVLA LIBERO Evaluation Script (parallel 4 task suites)
# =============================================================================

set -e

# =============================================================================
# LIBERO Environment Setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

echo "LIBERO Environment:"
echo "   LIBERO_ROOT: $LIBERO_ROOT"
echo "   PYTHONPATH: $PYTHONPATH"
echo ""

# Default arguments
PORT=${1:-8089}
NUM_TRIALS=${2:-10}
OUTPUT_PREFIX=${3:-"eval_simvla"}
GPUS=${4:-"4 5 6 7"}  # Default GPUs: 4 5 6 7
ALGO=${5:-"simvla"}

# Parse GPU list
read -ra GPU_ARRAY <<< "$GPUS"
if [ ${#GPU_ARRAY[@]} -lt 4 ]; then
    echo "ERROR: Need at least 4 GPUs, got ${#GPU_ARRAY[@]}"
    echo "   Usage: $0 <port> <num_trials> <output_prefix> \"<gpu1> <gpu2> <gpu3> <gpu4>\""
    exit 1
fi

GPU_SPATIAL=${GPU_ARRAY[0]}
GPU_OBJECT=${GPU_ARRAY[1]}
GPU_GOAL=${GPU_ARRAY[2]}
GPU_10=${GPU_ARRAY[3]}

# Output directory (timestamped subdirs created per suite inside this folder)
OUTPUT_DIR="./${OUTPUT_PREFIX}_${PORT}"
mkdir -p "$OUTPUT_DIR"

echo "Starting LIBERO evaluation..."
echo "   Server Port: $PORT"
echo "   Num Trials: $NUM_TRIALS"
echo "   Output Prefix: $OUTPUT_PREFIX"
echo "   Output Dir: $OUTPUT_DIR"
echo "   GPUs: spatial=$GPU_SPATIAL, object=$GPU_OBJECT, goal=$GPU_GOAL, 10=$GPU_10"
echo ""


# Run 4 task suites in parallel
echo "Launching 4 evaluation tasks..."

CUDA_VISIBLE_DEVICES=$GPU_SPATIAL python -u libero_client.py \
    --host 127.0.0.1 --port $PORT \
    --client_type websocket \
    --task_suite libero_spatial \
    --num_trials $NUM_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --algo "$ALGO" > "${OUTPUT_DIR}/${OUTPUT_PREFIX}_spatial.txt" 2>&1 &
PID_SPATIAL=$!
echo "   [PID $PID_SPATIAL] libero_spatial  (GPU $GPU_SPATIAL)"

CUDA_VISIBLE_DEVICES=$GPU_OBJECT python -u libero_client.py \
    --host 127.0.0.1 --port $PORT \
    --client_type websocket \
    --task_suite libero_object \
    --num_trials $NUM_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --algo "$ALGO" > "${OUTPUT_DIR}/${OUTPUT_PREFIX}_object.txt" 2>&1 &
PID_OBJECT=$!
echo "   [PID $PID_OBJECT] libero_object  (GPU $GPU_OBJECT)"

CUDA_VISIBLE_DEVICES=$GPU_GOAL python -u libero_client.py \
    --host 127.0.0.1 --port $PORT \
    --client_type websocket \
    --task_suite libero_goal \
    --num_trials $NUM_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --algo "$ALGO" > "${OUTPUT_DIR}/${OUTPUT_PREFIX}_goal.txt" 2>&1 &
PID_GOAL=$!
echo "   [PID $PID_GOAL] libero_goal  (GPU $GPU_GOAL)"

CUDA_VISIBLE_DEVICES=$GPU_10 python -u libero_client.py \
    --host 127.0.0.1 --port $PORT \
    --client_type websocket \
    --task_suite libero_10 \
    --num_trials $NUM_TRIALS \
    --output_dir "$OUTPUT_DIR" \
    --algo "$ALGO" > "${OUTPUT_DIR}/${OUTPUT_PREFIX}_10.txt" 2>&1 &
PID_10=$!
echo "   [PID $PID_10] libero_10  (GPU $GPU_10)"

echo ""
echo "Waiting for all evaluations to complete..."
echo "   Monitor progress with: tail -f ${OUTPUT_PREFIX}_*.txt"
echo ""

# Wait for all tasks
wait $PID_SPATIAL $PID_OBJECT $PID_GOAL $PID_10

echo ""
echo "All evaluations completed!"
echo ""
echo "Results summary:"
echo "=========================================="
for suite in spatial object goal 10; do
    log="${OUTPUT_DIR}/${OUTPUT_PREFIX}_${suite}.txt"
    json_file=$(ls "${OUTPUT_DIR}/libero_${suite}_"*/results.json 2>/dev/null | head -1)
    echo "--- libero_${suite} ---"
    if [ -n "$json_file" ]; then
        python3 -c "
import json, sys
d = json.load(open('$json_file'))
print(f'  success_rate : {d[\"total_success_rate\"]*100:.1f}%  ({d[\"total_successes\"]}/{d[\"total_episodes\"]})')
print(f'  duration     : {d[\"duration_seconds\"]}s')
print(f'  output       : $json_file')
" 2>/dev/null || echo "  (see $log)"
    else
        echo "  (no results.json found, see $log)"
    fi
done
echo "=========================================="
echo "All outputs in: $OUTPUT_DIR"
