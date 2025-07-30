#!/bin/bash
set -euo pipefail

# Config options

# Model ID
QWEN3_DENSE="Qwen/Qwen3-0.6B"
QWEN3_MOE="assets/qwen3_moe_4layer_16experts" #"assets/qwen3_moe_4layer" #"Qwen/Qwen3-30B-A3B"

MODEL_ID="${QWEN3_MOE}"

# Parallelism Config
TP=2
PP=2
CP=1
EP=2
ETP=1 # Explicitly set ETP to 1, otherwise Megatron will default to TP size
VPP_SIZE=None

# See Megatron parallel_state initialization logic for how they implement MoE parallel folding
# TLDR: parallel folding ~ separate parallel groups for MoE / non-MoE layers: MoE layers uses EP / ETP process groups while non-MoE uses TP / CP process groups 
WORLD_SIZE_NON_MOE=$((TP * CP * PP))
WORLD_SIZE_MOE=$((EP * ETP * PP))

if [[ "${WORLD_SIZE_NON_MOE}" -gt "${WORLD_SIZE_MOE}" ]]; then
    WORLD_SIZE=${WORLD_SIZE_NON_MOE}
else
    WORLD_SIZE=${WORLD_SIZE_MOE}
fi

# Distributed launch
# Use fake for debugging -- can't run any comms when using fake
BACKEND="nccl"
DIST_LAUNCH="torchrun --nproc-per-node ${WORLD_SIZE}"
LOCAL_LAUNCH="python"
RANK=0

if [[ ${BACKEND} == "fake" ]]; then
    LAUNCHER=${LOCAL_LAUNCH}
else
    LAUNCHER=${DIST_LAUNCH}
fi

# Model init
INIT_META="--init-model-with-meta-device"
INIT_CPU="--use-cpu-initialization"
INIT_METHOD=${INIT_META}

# Checkpoint saving
SAVE_CHECKPOINT=0
SAVE_DIR="mcore_chkpts"
CKPT_FORMAT="torch" 

# Logits checking
CHECK_LOGITS=1
LOGITS_SAVE_PATH="logits_check"
TOPK=5

mkdir -p ${SAVE_DIR}

ARGS="--model-id ${MODEL_ID} \
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size ${PP} \
--context-parallel-size ${CP} \
--expert-model-parallel-size ${EP} \
--expert-tensor-parallel-size ${ETP} \
--check-logits \
--logits-save-path logits_check \
--topk 5 \
--save ${SAVE_DIR} \
--save-interval 1 \
--ckpt-format ${CKPT_FORMAT}"
# --save-checkpoint"

if [[ ${INIT_METHOD} == ${INIT_CPU} || ${INIT_METHOD} == ${INIT_META} ]]; then
    ARGS+=" ${INIT_METHOD}"
fi

if [[ "${VPP_SIZE}" != "None" && "${VPP_SIZE}" -gt 1 ]]; then
    if [[ "${PP}" -le 1 ]]; then
        echo "ERROR: PP must be > 1 when VPP_SIZE not None"
        exit 1
    fi 
    ARGS+=" --num-virtual-stages-per-pipeline-rank ${VPP_SIZE}"
fi

# Add world_size and rank for local launches
if [[ "${LAUNCHER}" == "${LOCAL_LAUNCH}" ]]; then
    ARGS+=" --backend fake --world_size ${WORLD_SIZE} --rank ${RANK}"
else
    ARGS+=" --backend ${BACKEND}"
fi

PY_EXEC="convert_model.py"
LAUNCH_CMD="${LAUNCHER} ${PY_EXEC}"
CMD="${LAUNCH_CMD} ${ARGS}"

echo "${CMD}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
eval "${CMD}"

