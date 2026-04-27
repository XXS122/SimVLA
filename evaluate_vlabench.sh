#!/bin/bash
# SimVLA VLABench 评估脚本（vlabench 环境）
#
# 用法：
#   bash evaluate_vlabench.sh [track] [n_episode] [port]
#
# 示例：
#   bash evaluate_vlabench.sh track_1_in_distribution
#   bash evaluate_vlabench.sh track_1_in_distribution 10 8001
#   bash evaluate_vlabench.sh track_5_long_horizon 50 8001
#
# 支持的 track：
#   track_1_in_distribution
#   track_2_cross_category
#   track_3_common_sense
#   track_4_semantic_instruction
#   track_5_long_horizon
#   track_6_unseen_texture
#
# 注意：需要在 vlabench 环境中运行（conda activate vlabench）

set -e

EVAL_TRACK=${1:-track_1_in_distribution}
N_EPISODE=${2:-10}
PORT=${3:-8001}

# =============================================================================
# 路径配置（从 paths.env 加载，不存在则用默认值）
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/paths.env" ] && source "${SCRIPT_DIR}/paths.env"

VLABENCH_CODE="${SIMVLA_VLABENCH_CODE:-/root/code/VLABench}"
SAVE_DIR="${SIMVLA_EVAL_RESULTS:-/root/eval_results}"

# =============================================================================
# 参数检查
# =============================================================================
VALID_TRACKS=(
    track_1_in_distribution
    track_2_cross_category
    track_3_common_sense
    track_4_semantic_instruction
    track_5_long_horizon
    track_6_unseen_texture
)

VALID=false
for t in "${VALID_TRACKS[@]}"; do
    [ "$EVAL_TRACK" = "$t" ] && VALID=true && break
done

if [ "$VALID" = false ]; then
    echo "错误: 无效的 eval-track '$EVAL_TRACK'"
    echo ""
    echo "支持的 track:"
    for t in "${VALID_TRACKS[@]}"; do echo "  $t"; done
    exit 1
fi

if [ ! -d "$VLABENCH_CODE" ]; then
    echo "错误: VLABench 代码目录不存在: $VLABENCH_CODE"
    echo "请在 paths.env 中设置 SIMVLA_VLABENCH_CODE"
    exit 1
fi

# =============================================================================
# 启动评估
# =============================================================================
echo "============================================================"
echo "SimVLA VLABench 评估"
echo "  eval_track: $EVAL_TRACK"
echo "  n_episode:  $N_EPISODE"
echo "  port:       $PORT"
echo "  save_dir:   $SAVE_DIR"
echo "  vlabench:   $VLABENCH_CODE"
echo "============================================================"

cd "$VLABENCH_CODE"

python "${SCRIPT_DIR}/evaluation/vlabench/evaluate_simvla.py" \
    --eval-track "$EVAL_TRACK" \
    --n-episode "$N_EPISODE" \
    --port "$PORT" \
    --save-dir "$SAVE_DIR"
