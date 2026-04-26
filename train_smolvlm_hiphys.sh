#!/bin/bash
# HiPhys-VLA Training Script（HistoryEncoder + PhysicsPredicateDecoder）
# 用法：bash train_smolvlm_hiphys.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
# 示例：bash train_smolvlm_hiphys.sh 16 0.1 ./simvla_output/simvla_hiphys

set -e

BATCH_SIZE=${1:-16}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./simvla_output/simvla_hiphys}
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

# GPU 配置（从 paths.env 读取，默认单卡 0）
export CUDA_VISIBLE_DEVICES="${SIMVLA_CUDA_DEVICES:-0}"
NUM_PROCESSES="${SIMVLA_NUM_GPUS:-1}"

VLABENCH_DATA_DIR="${SIMVLA_VLABENCH_DATA:-/root/dataset/vlabench-data/1.0.0}"
NORM_STATS_PATH="./norm_stats/vlabench_norm.json"
TRAIN_METAS_PATH="./datasets/metas/vlabench_train.json"
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-/root/model/smolvlm-500M}"

# =============================================================================
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

# 损失函数
USE_HUBER_LOSS=true
GRIPPER_WEIGHT=5.0
TIME_SAMPLING="logit_normal"

# HiPhys-VLA 专用参数
USE_ADALN=true
USE_HISTORY_ENCODER=true
HISTORY_SEQ_LEN=4
SWITCH_LOSS_WEIGHT=0.05
USE_PHYSICS_COT=true
PHYSICS_WEIGHT=0.01

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
    --max_grad_norm ${MAX_GRAD_NORM} \
    --use_adaln \
    --history_seq_len ${HISTORY_SEQ_LEN} \
    --switch_loss_weight ${SWITCH_LOSS_WEIGHT} \
    --physics_weight ${PHYSICS_WEIGHT} \
    --time_sampling ${TIME_SAMPLING}"

if [ "${USE_HISTORY_ENCODER}" = true ]; then
    ARGS="${ARGS} --use_history_encoder"
fi

if [ "${USE_PHYSICS_COT}" = true ]; then
    ARGS="${ARGS} --use_physics_cot"
fi

if [ "${USE_HUBER_LOSS}" = true ]; then
    ARGS="${ARGS} --use_huber_loss --gripper_weight ${GRIPPER_WEIGHT}"
fi

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
fi

echo "============================================================"
echo "Starting HiPhys-VLA Training"
echo "  HistoryEncoder: GRU K=${HISTORY_SEQ_LEN} frames, switch_loss=${SWITCH_LOSS_WEIGHT}"
echo "  PhysicsDecoder: 5 predicates, physics_loss=${PHYSICS_WEIGHT}"
echo "  Action mode: vlabench_joint"
echo "  SmolVLM: ${SMOLVLM_MODEL}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port 29505 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
