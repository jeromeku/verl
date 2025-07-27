import itertools
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils.logging import set_verbosity_debug
from safetensors import safe_open
from transformers import AutoConfig
from transformers.utils.hub import cached_file

#set_verbosity_debug()

MODEL_ID = "Qwen/Qwen3-1.7B"
model_cache_dir = Path(snapshot_download(MODEL_ID))
#model_cache_dir = Path(os.path.dirname(cached_file(MODEL_ID, filename="config.json")))
weight_files = list(model_cache_dir.glob("*.safetensors"))
print(list(weight_files))
tensor_index = {}
num_keys = 0
def weights_generator(weight_files: list[str|Path], device: str = "cpu"):

    for wf in weight_files:
        with safe_open(wf, 'pt', device=device) as f:
            for k in f.keys():
                yield k, f.get_tensor(k)
    
wg = weights_generator(weight_files)
k, w = next(wg)
print(k, w.shape)