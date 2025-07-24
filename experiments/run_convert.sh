#!/bin/bash
set -euo pipefail

MODEL_ID="Qwen/Qwen3-0.6B"
TP=1
PP=1
CP=1
EP=1
ETP=1
# model_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
export CUDA_DEVICE_MAX_CONNECTIONS=1

CMD="torchrun --standalone hf_to_mcore_config.py \
--model-id ${MODEL_ID} \
--init-model-with-meta-device \
--tensor-model-parallel-size ${TP} \
--pipeline-model-parallel-size ${PP} \
--context-parallel-size ${CP} \
--expert-model-parallel-size ${EP} \
--expert-tensor-parallel-size ${ETP}"

echo "${CMD}"
eval "${CMD}"