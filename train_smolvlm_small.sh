#!/bin/bash
# SimVLA Training Script for LIBERO (Small Model)
#
# 用法:
#   source paths.env          # 加载路径环境变量
#   bash train_smolvlm_small.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
#
# 依赖环境变量 (paths.env):
#   SIMVLA_SMOLVLM_MODEL  — SmolVLM 模型路径
#   LIBERO_DATASETS       — 原始 LIBERO HDF5 数据集根目录
#   SIMVLA_CHECKPOINTS    — SimVLA checkpoint 路径 (resume 用)
#   CUDA_DEVICES          — 使用的 GPU 编号 (默认 "0")

set -e

# =============================================================================
# 命令行参数 (带默认值)
# =============================================================================
BATCH_SIZE=${1:-48}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_small}
RESUME_CKPT=${4:-"${SIMVLA_RESUME_CKPT:-}"}

# =============================================================================
# 路径配置 (优先读取环境变量)
# =============================================================================
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"
LIBERO_DATA_DIR="${LIBERO_DATASETS:?请先 source paths.env 或设置 LIBERO_DATASETS}"
TRAIN_METAS_PATH="./datasets/metas/libero_small_train.json"
NORM_STATS_PATH="./datasets/metas/libero_small_norm.json"
# 空格分隔，按需增减：libero_goal libero_spatial libero_object libero_10
SUBSETS="libero_goal"

# GPU 配置
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"

# WandB：有 API key 则启用，否则关闭
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
    export WANDB_PROJECT="${WANDB_PROJECT:-simvla}"
    export WANDB_DISABLED=false
else
    export WANDB_DISABLED=true
    export WANDB_MODE=disabled
fi

# 关闭 TF 日志
export TF_CPP_MIN_LOG_LEVEL=2

echo "============================================================"
echo " SimVLA Training (Small) — LIBERO"
echo "============================================================"
echo "  SmolVLM backbone : $SMOLVLM_MODEL"
echo "  LIBERO 数据目录  : $LIBERO_DATA_DIR"
echo "  元数据文件       : $TRAIN_METAS_PATH"
echo "  归一化统计       : $NORM_STATS_PATH"
echo "  输出目录         : $OUTPUT_DIR"
echo "  Batch size       : $BATCH_SIZE"
echo "  Learning coef    : $LEARNING_COEF"
echo "  GPU              : $CUDA_VISIBLE_DEVICES (共 $NUM_GPUS 卡)"
echo "  Resume           : ${RESUME_CKPT:-'无 (从头训练)'}"
echo "============================================================"

# =============================================================================
# Step 1: 生成训练元数据 (如果不存在)
# =============================================================================
mkdir -p "$(dirname $TRAIN_METAS_PATH)"
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "[Step 1] 生成训练元数据..."
    python create_libero_meta.py \
        --data_dir "$LIBERO_DATA_DIR" \
        --subsets $SUBSETS \
        --output "$TRAIN_METAS_PATH"
else
    echo "[Step 1] 元数据已存在，跳过: $TRAIN_METAS_PATH"
fi

# =============================================================================
# Step 2: 计算归一化统计量 (如果不存在)
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "[Step 2] 计算归一化统计量..."
    python compute_libero_norm_stats.py \
        --data_dir "$LIBERO_DATA_DIR" \
        --subsets $SUBSETS \
        --output "$NORM_STATS_PATH"
else
    echo "[Step 2] 归一化统计已存在，跳过: $NORM_STATS_PATH"
fi

# =============================================================================
# Step 3: 训练超参数
# =============================================================================
LEARNING_RATE=1e-4
NUM_ACTIONS=10
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=2
MAX_GRAD_NORM=1.0

# 模型架构 (Small)
HIDDEN_SIZE=768
DEPTH=12
NUM_HEADS=12

# 创新模块 (默认全开)
USE_CTAF=true
USE_PSCA=true

# =============================================================================
# Step 4: 构建训练参数
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM}"

[ "${USE_CTAF}" = true ] && ARGS="${ARGS} --use_ctaf"
[ "${USE_PSCA}" = true ] && ARGS="${ARGS} --use_psca"

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
fi

# =============================================================================
# Step 5: 启动训练
# =============================================================================
echo "[Step 3] 开始训练..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_GPUS} \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "训练完成！输出目录: ${OUTPUT_DIR}"
