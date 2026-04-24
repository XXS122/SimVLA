#!/bin/bash
# SimVLA Training Script - CVAE 子目标潜变量（SubgoalVAE）
#
# 用法：
#   bash train_smolvlm_subgoal.sh [batch_size] [learning_coef] [output_dir] [resume_ckpt]
#
# 示例：
#   bash train_smolvlm_subgoal.sh 32 0.1 ./runs/simvla_subgoal
#   bash train_smolvlm_subgoal.sh 16 0.1 ./runs/simvla_subgoal ./runs/simvla_subgoal/ckpt-10000

set -e

# =============================================================================
# 命令行参数（带默认值）
# =============================================================================
BATCH_SIZE=${1:-32}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./simvla_output/simvla_subgoal}
RESUME_CKPT=${4:-""}

echo "============================================================"
echo "SimVLA Training - CVAE SubgoalVAE"
echo "============================================================"
echo "   batch_size:    $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir:    $OUTPUT_DIR"
echo "   resume_ckpt:   ${RESUME_CKPT:-'None (training from scratch)'}"

# =============================================================================
# 环境变量
# =============================================================================
export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# 路径配置（从 paths.env 加载机器特定路径，不存在则用默认值）
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
# 训练超参数
# =============================================================================
LEARNING_RATE=1e-4
NUM_ACTIONS=10
ITERS=100000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0

# 模型架构（Small）
HIDDEN_SIZE=768
DEPTH=12
NUM_HEADS=12

# CVAE 子目标潜变量参数
SUBGOAL_LATENT_DIM=64
KL_WEIGHT=0.001
KL_WARMUP_STEPS=10000

# Latent Diffusion Model（z 空间 Flow Matching）
LATENT_FLOW_STEPS=5
LATENT_FM_WEIGHT=1.0

# 损失函数与时间步采样（新增）
USE_HUBER_LOSS=true
GRIPPER_WEIGHT=5.0
TIME_SAMPLING="logit_normal"

# =============================================================================
# Step 1: 生成训练元数据（不存在时自动创建）
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata..."
    python create_vlabench_meta.py \
        --data_dir $VLABENCH_DATA_DIR \
        --output $TRAIN_METAS_PATH
fi

# =============================================================================
# Step 2: 计算归一化统计量（不存在时自动计算）
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics..."
    python compute_vlabench_norm_stats.py \
        --data_dir $VLABENCH_DATA_DIR \
        --output $NORM_STATS_PATH
fi

# =============================================================================
# Step 3: 构建训练参数
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
    --use_subgoal_vae \
    --subgoal_latent_dim ${SUBGOAL_LATENT_DIM} \
    --kl_weight ${KL_WEIGHT} \
    --kl_warmup_steps ${KL_WARMUP_STEPS} \
    --use_latent_flow \
    --latent_flow_steps ${LATENT_FLOW_STEPS} \
    --latent_fm_weight ${LATENT_FM_WEIGHT} \
    --use_huber_loss \
    --gripper_weight ${GRIPPER_WEIGHT} \
    --time_sampling ${TIME_SAMPLING}"

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 4: 启动训练
# =============================================================================
echo "============================================================"
echo "Model:        SmolVLM-500M + AdaLN + SubgoalVAE + LatentFlowNet (LDM)"
echo "Data:         VLABench (vlabench_joint)"
echo "GPUs:         CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Arch:         hidden=${HIDDEN_SIZE}, depth=${DEPTH}, heads=${NUM_HEADS}"
echo "CVAE:         latent_dim=${SUBGOAL_LATENT_DIM}, kl_weight=${KL_WEIGHT}, kl_warmup=${KL_WARMUP_STEPS}"
echo "LDM:          latent_flow_steps=${LATENT_FLOW_STEPS}, latent_fm_weight=${LATENT_FM_WEIGHT}"
echo "Output:       ${OUTPUT_DIR}"
echo "============================================================"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port 29507 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"
