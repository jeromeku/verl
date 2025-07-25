#!/bin/bash
set -euo pipefail

MODEL_ID="Qwen/Qwen3-0.6B" #"Qwen/Qwen3-30B-A3B"
TP=2
PP=1
CP=1
EP=1
ETP=1
WORLD_SIZE=$((TP * CP * PP))
SAVE_DIR="mcore_chkpts"
SAVE_PATH="${SAVE_DIR}/${MODEL_ID##*/}"
BACKEND="gloo"

CMD="debugpy-run -m torch.distributed.run -p 5678 -- \
--nproc-per-node=${WORLD_SIZE} \
convert_multigpu.py \
--model_path ${MODEL_ID} \
--tp=${TP} \
--pp=${PP} \
--ep=${EP} \
--etp=${ETP} \
--cp=${CP} \
--save_path=${SAVE_PATH} \
--backend=${BACKEND}"

echo "$CMD" 
eval "${CMD}"