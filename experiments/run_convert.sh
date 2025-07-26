#!/bin/bash
set -euo pipefail

QWEN3_DENSE="Qwen/Qwen3-0.6B"
QWEN3_MOE="assets/qwen3_moe_1layer" #"assets/qwen3_moe_4layer" #"Qwen/Qwen3-30B-A3B"

MODEL_ID="${QWEN3_DENSE}"

TP=1
PP=1
CP=1
EP=1
ETP=1
VPP_SIZE=None

WORLD_SIZE_NON_MOE=$((TP * CP * PP))
WORLD_SIZE_MOE=$((EP * ETP * PP))
if [[ "${WORLD_SIZE_NON_MOE}" -gt "${WORLD_SIZE_MOE}" ]]; then
    WORLD_SIZE=${WORLD_SIZE_NON_MOE}
else
    WORLD_SIZE=${WORLD_SIZE_MOE}
fi

DIST_LAUNCH="torchrun --nproc-per-node ${WORLD_SIZE}"
LOCAL_LAUNCH="python"
LAUNCHER=${DIST_LAUNCH}

BACKEND="nccl"
RANK=0
INIT_META="--init-model-with-meta-device"
INIT_CPU="--use-cpu-initialization"

INIT_METHOD=${INIT_META}

LAUNCH_CMD="${LAUNCHER} hf_to_mcore_config.py"
SAVE_DIR="mcore_chkpts"
CKPT_FORMAT="torch" # torch_dist

mkdir -p ${SAVE_DIR}

ARGS="--model-id ${MODEL_ID} \
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size ${PP} \
--context-parallel-size ${CP} \
--expert-model-parallel-size ${EP} \
--expert-tensor-parallel-size ${ETP} \
--save ${SAVE_DIR} \
--save-interval 1 \
--ckpt-format ${CKPT_FORMAT}"

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

# Correctly combine the launch command and arguments
CMD="${LAUNCH_CMD} ${ARGS}"


echo "${CMD}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
#export WORLD_SIZE=${WORLD_SIZE}

eval "${CMD}"

#VPP_STAGE_SIZE=1
# LAYERS_PER_RANK = NUM_LAYERS // PP 
# when using VPP, LAYERS_PER_RANK_PER_STAGE = LAYERS_PER_RANK // VPP
# Each rank gets LAYERS_PER_RANK_PER_STAGE layers in round robin fashion
# VP_STAGES = NUM_LAYERS // (PP * VPP)
# VP_STAGE virtual ranks
# Virtual rank ==> real rank = VIRTUAL_RANK % pp_size
# For Qwen 0.6B, 28 layers
# for PP=2, VPP=2, 28 // 2 = 14 stages per rank, interleaved such that Rank 0: [1-7],[15-21] | Rank1: [8-14],[22-28]
# Note that layer_id's start at 1 in decoder layers
#--num_layers_per_virtual_pipeline_stage ${VPP_STAGE_SIZE} \
# model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
