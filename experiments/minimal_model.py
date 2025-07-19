# minimal_megatron_loader.py
import argparse
import json
import os

import torch
from megatron.core import parallel_state as mpu
from transformers import AutoConfig

from verl.models.mcore import (  # step 3‑4 :contentReference[oaicite:0]{index=0}
    hf_to_mcore_config,
    init_mcore_model,
)
from verl.utils.model import (
    load_mcore_dist_weights,  # step 5   :contentReference[oaicite:1]{index=1}
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('hf_path')
    p.add_argument('--checkpoint-path', default=None)
    p.add_argument('--tp', type=int, default=1)      # tensor‑parallel size
    p.add_argument('--pp', type=int, default=1)      # pipeline‑parallel size
    p.add_argument('--dtype', choices=['bfloat16'], default='bfloat16')
    return p.parse_args()

def main():
    args = parse()
    torch_dtype = getattr(torch, args.dtype)
    # ------------------------------------------------------------------ 1‑2
    # (a) Torch distributed bootstrap (we rely on torchrun‑exported envs)
    rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)

    # (b) Megatron parallel topology
    mpu.initialize_model_parallel(
        tensor_model_parallel_size    = args.tp,
        pipeline_model_parallel_size  = args.pp,
    )

    # ------------------------------------------------------------------ 3
    hf_cfg = AutoConfig.from_pretrained(args.hf_path, trust_remote_code=True)
    tf_cfg = hf_to_mcore_config(hf_cfg, dtype=torch_dtype)            # HF → mcore TransformerConfig
    print(tf_cfg)
    if False:
        # ------------------------------------------------------------------ 4
        print(f"[rank{rank}] building empty GPT‑MoE…")
        model = init_mcore_model(
            tf_cfg, hf_cfg,
            pre_process=True,  post_process=True,
            share_embeddings_and_output_weights=True,
        ).to(torch_dtype).cuda()

        # ------------------------------------------------------------------ 5
        if torch.distributed.get_rank() == 0:
            print(">>> streaming distributed ckpt shards from", args.ckpt_dir)
        load_mcore_dist_weights([model], args.ckpt_dir)                   # Megatron shard → GPU

        torch.cuda.synchronize()
        n_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"[rank{rank}] done – {n_params:.2f} B parameters loaded.")

if __name__ == "__main__":
    main()
