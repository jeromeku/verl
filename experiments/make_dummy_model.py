import argparse

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_layers", type=int, default=1)
    args = parser.parse_args()

    num_layers = args.num_layers 
    MODEL_ID = "Qwen/Qwen3-30B-A3B"
    SAVE_PATH = f"assets/qwen3_moe_{num_layers}layer"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(SAVE_PATH)

    config: Qwen3MoeConfig = AutoConfig.from_pretrained(MODEL_ID)
    config.num_hidden_layers = num_layers
    print(config)

    torch.set_default_dtype(config.torch_dtype)
    model = Qwen3MoeForCausalLM(config)
    assert next(model.parameters()).dtype == config.torch_dtype
    model.save_pretrained(SAVE_PATH)
    print(f"Model and tokenizer saved to {SAVE_PATH}")

