#!/bin/bash
set -euo pipefail

MODEL_ID="Qwen/Qwen3-0.6B"
TP=1
PP=2
CP=1
EP=1
ETP=1
VPP_SIZE=2

WORLD_SIZE=$((TP * CP * PP))
BACKEND="fake"
#VPP_STAGE_SIZE=1
# LAYERS_PER_RANK = NUM_LAYERS // PP 
# when using VPP, LAYERS_PER_RANK_PER_STAGE = LAYERS_PER_RANK // VPP
# Each rank gets LAYERS_PER_RANK_PER_STAGE layers in round robin fashion
# VP_STAGES = NUM_LAYERS // (PP * VPP)
# VP_STAGE virtual ranks
# Virtual rank ==> real rank = VIRTUAL_RANK % pp_size
#--num_layers_per_virtual_pipeline_stage ${VPP_STAGE_SIZE} \
# model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIST_LAUNCH="torchrun --nproc-per-node ${WORLD_SIZE}"
PY_EXEC="python"
LAUNCHER=${PY_EXEC}

export WORLD_SIZE

CMD="${LAUNCHER} hf_to_mcore_config.py \
--model-id ${MODEL_ID} \
--backend ${BACKEND} \
--init-model-with-meta-device \
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size ${PP} \
--num-virtual-stages-per-pipeline-rank ${VPP_SIZE} \
--context-parallel-size ${CP} \
--expert-model-parallel-size ${EP} \
--expert-tensor-parallel-size ${ETP} \
--micro-batch-size=1"

echo "${CMD}"
eval "${CMD}"