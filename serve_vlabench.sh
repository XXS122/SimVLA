#!/bin/bash
# SimVLA VLABench 推理服务器启动脚本（simvla 环境）
#
# 用法：
#   bash serve_vlabench.sh <checkpoint_path> [port] [ode_steps]
#
# 示例：
#   bash serve_vlabench.sh ./simvla_output/simvla_hiphys/ckpt-100000
#   bash serve_vlabench.sh ./simvla_output/simvla_hiphys/ckpt-100000 8001 20
#
# 注意：需要在 simvla 环境中运行（conda activate simvla）

set -e

CHECKPOINT=${1:-""}
PORT=${2:-8001}
ODE_STEPS=${3:-20}

# =============================================================================
# 参数检查
# =============================================================================
if [ -z "$CHECKPOINT" ]; then
    echo "用法: bash serve_vlabench.sh <checkpoint_path> [port] [ode_steps]"
    echo ""
    echo "示例:"
    echo "  bash serve_vlabench.sh ./simvla_output/simvla_hiphys/ckpt-100000"
    echo "  bash serve_vlabench.sh ./simvla_output/simvla_hiphys/ckpt-100000 8001 20"
    echo ""
    echo "可用 checkpoint:"
    ls -d ./simvla_output/*/ckpt-* 2>/dev/null | sort -t- -k2 -n | tail -10 || echo "  (未找到 checkpoint)"
    exit 1
fi

if [ ! -d "$CHECKPOINT" ]; then
    echo "错误: checkpoint 目录不存在: $CHECKPOINT"
    exit 1
fi

# =============================================================================
# 路径配置（从 paths.env 加载，不存在则用默认值）
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/paths.env" ] && source "${SCRIPT_DIR}/paths.env"

export CUDA_VISIBLE_DEVICES="${SIMVLA_CUDA_DEVICES:-0}"
# 推理服务器只用单卡（取第一块）
export CUDA_VISIBLE_DEVICES="${SIMVLA_CUDA_DEVICES%%,*}"

NORM_STATS_PATH="${SCRIPT_DIR}/norm_stats/vlabench_norm.json"
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-/root/model/smolvlm-500M}"

# =============================================================================
# 启动服务器
# =============================================================================
echo "============================================================"
echo "SimVLA VLABench 推理服务器"
echo "  checkpoint: $CHECKPOINT"
echo "  port:       $PORT"
echo "  ode_steps:  $ODE_STEPS"
echo "  GPU:        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  norm_stats: $NORM_STATS_PATH"
echo "  smolvlm:    $SMOLVLM_MODEL"
echo "============================================================"

python "${SCRIPT_DIR}/evaluation/vlabench/serve_smolvlm_vlabench.py" \
    --checkpoint "$CHECKPOINT" \
    --norm_stats "$NORM_STATS_PATH" \
    --smolvlm_model "$SMOLVLM_MODEL" \
    --port "$PORT" \
    --ode-steps "$ODE_STEPS"
