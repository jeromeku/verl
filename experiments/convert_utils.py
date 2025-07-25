from argparse import Namespace

import torch
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3_moe import Qwen3MoeConfig


def update_args(
    args: Namespace,
    hf_config: Qwen3MoeConfig | Qwen3Config,
    use_transformer_engine: bool = True,
    **kwargs,
):
    # Required args for MCore args validation
    args.max_position_embeddings = hf_config.max_position_embeddings
    args.num_layers = hf_config.num_hidden_layers
    args.hidden_size = hf_config.hidden_size
    args.num_attention_heads = hf_config.num_attention_heads
    args.seq_length = hf_config.max_position_embeddings
    args.micro_batch_size = 1
    
    args.vocab_size = hf_config.vocab_size
    args.padded_vocab_size = args.vocab_size
    args.untie_embeddings_and_output_weights = not hf_config.tie_word_embeddings
    args.position_embedding_type = "rope"
    args.rotary_percent = 1.0
    args.rotary_base = hf_config.rope_theta
    args.rope_scaling = True if hf_config.rope_scaling is not None else False

    args.no_load_optim = True
    args.no_load_rng = True
    args.perform_initialization = False
    args.no_save_optim = True
    args.no_save_rng = True
    args.mock_data = True

    args.rank = args.rank or torch.distributed.get_rank()
    args.world_size = args.world_size or torch.distributed.get_world_size()

    # use TE for optimized parallel linear, attn, and moe grouped linear
    args.transformer_impl = "transformer_engine" if use_transformer_engine else "local"

    for k, v in kwargs.items():
        setattr(args, k, v)

    return args

