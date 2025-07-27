import os

import huggingface_hub
from huggingface_hub.utils.logging import set_verbosity_debug

set_verbosity_debug()

MODEL_ID = "Qwen/Qwen3-30B-A3B"
SAVE_DIR = os.path.join("assets", MODEL_ID.split("/")[-1])
huggingface_hub.snapshot_download(MODEL_ID, local_dir=SAVE_DIR)
