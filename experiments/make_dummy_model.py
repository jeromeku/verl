import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_PATH = "assets/qwen3_moe_small"
TOKEN_PATH = "assets/qwen3_moe_small"


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(TOKEN_PATH)

config = Qwen3MoeConfig(num_hidden_layers=1, torch_dtype="bfloat16")
print(config.torch_dtype)
torch.set_default_dtype(config.torch_dtype)
model = Qwen3MoeForCausalLM(config)
assert next(model.parameters()).dtype == config.torch_dtype

model.save_pretrained("assets/qwen3_moe_small")

