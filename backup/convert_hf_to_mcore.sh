#!/bin/bash
set -euo pipefail

QWEN3_DENSE="Qwen/Qwen3-0.6B"
QWEN3_MOE="assets/qwen3_moe_4layer_16experts" #"assets/qwen3_moe_4layer" #"Qwen/Qwen3-30B-A3B"

MODEL_ID="${QWEN3_MOE}"

TP=2
PP=1
CP=1
EP=2
ETP=1 # Explicitly set ETP to 1, otherwise Megatron will default this to TP size
VPP_SIZE=None

# See Megatron parallel_state initialization logic for how they implement MoE parallel folding
# Parallel folding ~ separate parallel groups for MoE / non-MoE layers: MoE layers uses EP / ETP while non-MoE uses TP / CP 
WORLD_SIZE_NON_MOE=$((TP * CP * PP))
WORLD_SIZE_MOE=$((EP * ETP * PP))

if [[ "${WORLD_SIZE_NON_MOE}" -gt "${WORLD_SIZE_MOE}" ]]; then
    WORLD_SIZE=${WORLD_SIZE_NON_MOE}
else
    WORLD_SIZE=${WORLD_SIZE_MOE}
fi

BACKEND="nccl"
DIST_LAUNCH="torchrun --nproc-per-node ${WORLD_SIZE}"
LOCAL_LAUNCH="python"
RANK=0

if [[ ${BACKEND} == "fake" ]]; then
    LAUNCHER=${LOCAL_LAUNCH}
else
    LAUNCHER=${DIST_LAUNCH}
fi


INIT_META="--init-model-with-meta-device"
INIT_CPU="--use-cpu-initialization"
INIT_CUDA="cuda"

INIT_METHOD=${INIT_META}

SAVE_DIR="mcore_chkpts"
CKPT_FORMAT="torch" # torch_dist

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
--ckpt-format ${CKPT_FORMAT} \
--save-checkpoint"

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

