import torch
import torch.nn.functional as F
from megatron.core import mpu
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3_moe import Qwen3MoeConfig

Qwen3ConfigT = Qwen3Config | Qwen3MoeConfig

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


def get_mlp_config(hf_config: Qwen3ConfigT, is_moe: bool = False):
    mlp_config = {
        # Experts
        "gated_linear_unit": True,
        "activation_func": F.silu,
    }

    if is_moe:
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
            "moe_router_pre_softmax": False,  # softmax is applied **after** topk in Qwen3-Moe
            "moe_token_dispatcher_type": "alltoall",  # TODO: tune
            # auxiliary loss
            "moe_router_enable_expert_bias": False,  # aux-loss-free routing, only for sigmoid-scored router
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
    else:
        moe_config = {}

    return {**mlp_config, **moe_config}


def get_arch_config(
    hf_config: Qwen3ConfigT,
    normalization: str = "RMSNorm",
    add_bias_linear: bool = False,
):
    return {
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "normalization": normalization,
        "add_bias_linear": add_bias_linear,
    }

# NOTE: Qwen3Configs specify a head_dim
def get_attn_config(hf_config: Qwen3ConfigT, qk_layernorm: bool = True):

    if isinstance(hf_config, Qwen3ConfigT):
        assert hasattr(hf_config, "head_dim") and hf_config.head_dim is not None

    return {
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),  # hidden_size // num_attention_heads
        "qk_layernorm": qk_layernorm,
    }


def get_precision_config(dtype: torch.dtype = torch.bfloat16):
    return {
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,  # Should be true for Qwen3
    }



def get_fusion_config(
    masked_softmax_fusion: bool = False,
    persist_layer_norm: bool = False,
    bias_activation_fusion: bool = False,
    bias_dropout_fusion: bool = False,
):
    return {
        "masked_softmax_fusion": masked_softmax_fusion,
        "persist_layer_norm": persist_layer_norm,
        "bias_activation_fusion": bias_activation_fusion,
        "bias_dropout_fusion": bias_dropout_fusion,
    }

def get_activation_recompute_config(
    recompute_granularity=None,
    recompute_method=None,
    recompute_num_layers=None,
    distribute_saved_activations=None,
    recompute_modules=None,
):
    return {
        "recompute_granularity": recompute_granularity,
        "recompute_method": recompute_method,
        "recompute_num_layers": recompute_num_layers,
        "distribute_saved_activations": distribute_saved_activations,
        "recompute_modules": recompute_modules,
    }