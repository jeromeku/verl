import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

NUM_LAYERS = 4
MODEL_ID = "Qwen/Qwen3-30B-A3B"
SAVE_PATH = f"assets/qwen3_moe_{NUM_LAYERS}layer"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(SAVE_PATH)

config = Qwen3MoeConfig(num_hidden_layers=NUM_LAYERS, torch_dtype="bfloat16")
print(config.torch_dtype)
torch.set_default_dtype(config.torch_dtype)
model = Qwen3MoeForCausalLM(config)
assert next(model.parameters()).dtype == config.torch_dtype
model.save_pretrained(SAVE_PATH)
print(f"Model and tokenizer saved to {SAVE_PATH}")

