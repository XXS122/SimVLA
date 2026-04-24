#!/bin/bash
# SimVLA Training Script for VLABench (Small Model)

set -e

BATCH_SIZE=${1:-1}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./simvla_output/simvla_vlabench_small}
RESUME_CKPT=${4:-""}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"

export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# Path configuration（从 paths.env 加载机器特定路径，不存在则用默认值）
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/paths.env" ] && source "${SCRIPT_DIR}/paths.env"

# GPU 配置（从 paths.env 读取，默认双卡 6,7）
export CUDA_VISIBLE_DEVICES="${SIMVLA_CUDA_DEVICES:-6,7}"
NUM_PROCESSES="${SIMVLA_NUM_GPUS:-2}"

VLABENCH_DATA_DIR="${SIMVLA_VLABENCH_DATA:-/root/dataset/vlabench-data/1.0.0}"
NORM_STATS_PATH="./norm_stats/vlabench_norm.json"
TRAIN_METAS_PATH="./datasets/metas/vlabench_train.json"
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-/root/model/smolvlm-500M}"

# ===============================================================
# Training hyperparameters
# =============================================================================
LEARNING_RATE=1e-4
NUM_ACTIONS=10
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0

HIDDEN_SIZE=768
DEPTH=12
NUM_HEADS=12
USE_ADALN=false

# 损失函数与时间步采样（新增）
USE_HUBER_LOSS=true       # Huber loss 替代 MSE
GRIPPER_WEIGHT=5.0        # gripper 维度损失权重
TIME_SAMPLING="logit_normal"  # 时间步采样策略：beta | logit_normal | cosine

# =============================================================================
# Step 1: Create training metadata (if not exists)
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata..."
    python create_vlabench_meta.py \
        --data_dir $VLABENCH_DATA_DIR \
        --output $TRAIN_METAS_PATH
fi

# =============================================================================
# Step 2: Compute normalization statistics (if not exists)
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics..."
    python compute_vlabench_norm_stats.py \
        --data_dir $VLABENCH_DATA_DIR \
        --output $NORM_STATS_PATH
fi

# =============================================================================
# Step 3: Start training
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode vlabench_joint \
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

if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

if [ "${USE_HUBER_LOSS}" = true ]; then
    ARGS="${ARGS} --use_huber_loss --gripper_weight ${GRIPPER_WEIGHT}"
fi

ARGS="${ARGS} --time_sampling ${TIME_SAMPLING}"

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
fi

echo "============================================================"
echo "Starting SimVLA Training on VLABench (Small Action Transformer)"
echo "Action mode: vlabench_joint"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
