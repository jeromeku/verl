# ruff: noqa
import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pp

import megatron.core as mc

MEGATRON_ROOT = Path(mc.__file__).parents[2]
sys.path.append(MEGATRON_ROOT.resolve().as_posix())
from megatron.training.arguments import _add_distributed_args, _add_moe_args

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig


from transformers.models.qwen3_moe import Qwen3MoeConfig

QWEN3_30B_3B = "Qwen/Qwen3-30B-A3B"
QWEN3_235B_A22B = "Qwen/Qwen3-235B-A22B"


def init_distributed(tp=1, vpp=1, pp=1, cp=1, ep=1, etp=1, seed=0):
    torch.distributed.init_process_group("nccl")

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(seed)


def get_parallelism(sequence_parallel: bool = None, variable_seq_lengths=False):
    return {
        "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
        "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
        "context_parallel_size": mpu.get_data_parallel_world_size(),
        "sequence_parallel": sequence_parallel or mpu.get_tensor_model_parallel_world_size() > 1,
        # Setting this communicates the size of tensors during pp comms.
        # Incurs overhead, should only be set if seqlen varies by microbatch within a global batch.
        "variable_seq_lengths": variable_seq_lengths,
    }

def get_gpt_model_args(hf_config: Qwen3MoeConfig):
    return {
            "vocab_size": hf_config.vocab_size,
            "max_sequence_length": hf_config.max_position_embeddings,
            "position_embedding_type": "rope",
            "rotary_base": hf_config.rope_theta,
    }

def get_moe_config(hf_config: Qwen3MoeConfig):
    moe_config = {
        # Experts
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
        "num_moe_experts": hf_config.num_experts,
        "moe_grouped_gemm": True,  # requires TransformerEngine
        "moe_use_legacy_grouped_gemm": False,  # legacy cutlass grouped gemm
        "moe_shared_expert_intermediate_size": None,  # no shared expert
        # Router
        "moe_router_dtype": torch.float32,
        "moe_router_topk": hf_config.num_experts_per_tok,
        "moe_router_score_function": "softmax",
        "moe_router_pre_softmax": False,
        "moe_token_dispatcher_type": "alltoall",  # TODO: tune
        # auxiliary loss
        "moe_router_load_balancing_type": "aux_loss",
        "moe_expert_capacity_factor": None,  # token choice -> no dropped tokens
        "moe_router_bias_update_rate": 0.001,  # TODO: check whether this is needed for aux_loss
        "moe_aux_loss_coeff": hf_config.router_aux_loss_coef,
        # optimizations
        "moe_enable_deepep": False,
        "moe_deepep_num_sms": 20,  # TODO: tune
        "moe_layer_recompute": True,
        "moe_permute_fusion": False,  # TODO: tune
        "moe_per_layer_logging": True,  # for auxiliary loss
    }
    return moe_config


def get_arch_config(hf_config: Qwen3MoeConfig):
    return {
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "normalization": "RMSNorm",
        "add_bias_linear": False,  # no bias in linear layers (qkv & mlp)
    }


def get_attn_config(hf_config: Qwen3MoeConfig):
    return {
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),  # hidden_size // num_attention_heads
        "qk_layernorm": True,
    }


def get_precision_config(dtype: torch.dtype = torch.bfloat16):
    return {
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,  # Should be true for Qwen3
    }


def get_fusion_config():
    return {
        "masked_softmax_fusion": False,
        "persist_layer_norm": False,
        "bias_activation_fusion": False,
        "bias_dropout_fusion": False,
    }


def get_activation_recompute_config():
    return {
        "recompute_granularity": None,
        "recompute_method": None,
        "recompute_num_layers": None,
        "distribute_saved_activations": None,
        "recompute_modules": None,
    }


def hf_to_mcore(hf_config: Qwen3MoeConfig):
    dtype = hf_config.torch_dtype
    assert dtype == torch.bfloat16

    arch_config = get_arch_config(hf_config)
    attn_config = get_attn_config(hf_config)
    moe_config = get_moe_config(hf_config)

    precision_config = get_precision_config(dtype)
    parallelism_config = get_parallelism()
    fusion_config = get_fusion_config()

    ac_config = get_activation_recompute_config()

    misc_config = {
        "use_cpu_initialization": False,
    }

    final_config = TransformerConfig(
        **arch_config,
        **attn_config,
        **moe_config,
        **precision_config,
        **parallelism_config,
        **fusion_config,
        **ac_config,
    )
    return final_config


if __name__ == "__main__":
    # Not used directly in TransformerConfig, only from CLI args

    parser = argparse.ArgumentParser(
        "HF -> Megatron Config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_distributed_args(parser)
    _add_moe_args(parser)
    args = parser.parse_args()

    hf_config: Qwen3MoeConfig = Qwen3MoeConfig.from_pretrained(QWEN3_30B_3B)


    num_layers_per_pipeline_stage = hf_config.num_hidden_layers // args.pipeline_model_parallel_size
    if args.num_layers_per_virtual_pipeline_stage is not None:
        args.virtual_pipeline_model_parallel_size = (
            num_layers_per_pipeline_stage // int(args.num_layers_per_virtual_pipeline_stage)
        )
    else:
        args.virtual_pipeline_model_parallel_size = None

    init_distributed(
        tp=args.tensor_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        vpp=args.virtual_pipeline_model_parallel_size,
        cp=args.context_parallel_size,
        ep=args.expert_model_parallel_size,
        etp=args.expert_tensor_parallel_size,
    )

    transformer_config = hf_to_mcore(hf_config)

    pp(asdict(transformer_config))

    # tokenizer_config = {
    #     "tokenizer_type": "HuggingFaceTokenizer",
    #     "make-vocab-size-divisible-by": 1187,
    #     "position_embedding_type": "rope",
    # }
