#!/bin/bash
# 串行评估脚本：在 2 张 GPU 上依次跑 4 个 LIBERO 任务套件
# 服务器用 GPU 0，客户端用 GPU 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 读取 paths.env
PATHS_ENV="${PROJECT_ROOT}/paths.env"
if [ ! -f "$PATHS_ENV" ]; then
    echo "ERROR: paths.env not found at $PATHS_ENV"
    exit 1
fi
source "$PATHS_ENV"

CHECKPOINT="$SIMVLA_CHECKPOINTS"
SMOLVLM_MODEL="$SIMVLA_SMOLVLM_MODEL"
NORM_STATS="${PROJECT_ROOT}/norm_stats/libero_norm.json"

# LIBERO 环境
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

# 参数
PORT=${1:-8102}
NUM_TRIALS=${2:-50}
OUTPUT_PREFIX=${3:-"eval_simvla"}
GPU_SERVER=${4:-0}
GPU_CLIENT=${5:-1}

OUTPUT_DIR="./eval_simvla_${PORT}"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "SimVLA 串行评估"
echo "  Checkpoint:   $CHECKPOINT"
echo "  SmolVLM:      $SMOLVLM_MODEL"
echo "  Norm stats:   $NORM_STATS"
echo "  Port:         $PORT"
echo "  Num trials:   $NUM_TRIALS"
echo "  GPU server:   $GPU_SERVER  |  GPU client: $GPU_CLIENT"
echo "  Output dir:   $OUTPUT_DIR"
echo ""

# 启动推理服务器（后台）
echo "启动推理服务器..."
CUDA_VISIBLE_DEVICES=$GPU_SERVER python -u "${SCRIPT_DIR}/serve_smolvlm_libero.py" \
    --checkpoint "$CHECKPOINT" \
    --norm_stats "$NORM_STATS" \
    --smolvlm_model "$SMOLVLM_MODEL" \
    --port "$PORT" > "${OUTPUT_DIR}/server.log" 2>&1 &
SERVER_PID=$!
echo "  服务器 PID: $SERVER_PID，日志: ${OUTPUT_DIR}/server.log"

# 等待服务器就绪
echo "等待服务器就绪..."
for i in $(seq 1 60); do
    if grep -q "listening on" "${OUTPUT_DIR}/server.log" 2>/dev/null; then
        echo "  服务器已就绪（${i}s）"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: 服务器进程已退出，查看日志："
        cat "${OUTPUT_DIR}/server.log"
        exit 1
    fi
    sleep 1
done

cleanup() {
    echo ""
    echo "关闭服务器 (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# 串行跑 4 个套件
SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for SUITE in "${SUITES[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始评估: $SUITE"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=$GPU_CLIENT python -u "${SCRIPT_DIR}/libero_client.py" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --client_type websocket \
        --task_suite "$SUITE" \
        --num_trials "$NUM_TRIALS" \
        --video_out "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_PREFIX}_${SUITE}.txt"
done

echo ""
echo "=========================================="
echo "全部评估完成，结果汇总："
echo "=========================================="
for SUITE in "${SUITES[@]}"; do
    FILE="${OUTPUT_PREFIX}_${SUITE}.txt"
    echo "--- $SUITE ---"
    grep -E "Total success rate|success rate" "$FILE" 2>/dev/null | tail -1 || echo "  (见 $FILE)"
done
echo "=========================================="
