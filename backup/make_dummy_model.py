import argparse
import os

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

MODEL_ID = "Qwen/Qwen3-30B-A3B"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--save_dir", default="assets")

    args = parser.parse_args()

    model_id = args.model_id
    num_layers = args.num_layers 
    num_experts = args.num_experts
    config: Qwen3MoeConfig = AutoConfig.from_pretrained(model_id)
    num_experts = args.num_experts or config.num_experts

    os.makedirs(args.save_dir, exist_ok=True)    
    SAVE_PATH = f"{args.save_dir}/qwen3_moe_{num_layers}layer_{num_experts}experts"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(SAVE_PATH)

    config.num_hidden_layers = num_layers
    config.num_experts = num_experts
    print(config)

    torch.set_default_dtype(config.torch_dtype)
    model = Qwen3MoeForCausalLM(config)
    assert next(model.parameters()).dtype == config.torch_dtype
    model.save_pretrained(SAVE_PATH)
    print(f"Model and tokenizer saved to {SAVE_PATH}")

