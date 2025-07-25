import torch
import torch.nn.functional as F
from megatron.core import mpu
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3_moe import Qwen3MoeConfig

Qwen3ConfigT = Qwen3Config | Qwen3MoeConfig

# Dense
QWEN3_600M = "Qwen/Qwen3-0.6B"
QWEN3_4B = "Qwen/Qwen3-4B"

# MoE
QWEN3_30B_3B = "Qwen/Qwen3-30B-A3B"
QWEN3_235B_A22B = "Qwen/Qwen3-235B-A22B"

QWEN3_DENSE_MODELS = [QWEN3_600M, QWEN3_4B]
QWEN3_MOE_MODELS = [QWEN3_30B_3B, QWEN3_235B_A22B]


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


def get_mlp_config(
    hf_config: Qwen3ConfigT,
    is_moe: bool = False,
    # MLP
    gated_linear_unit: bool = True,
    activation_func=F.silu,
    # Experts
    moe_grouped_gemm: bool = True,
    moe_use_legacy_grouped_gemm: bool = False,
    moe_shared_expert_intermediate_size=None,
    # Router
    moe_router_dtype=torch.float32,
    moe_router_score_function: str = "softmax",
    moe_router_pre_softmax: bool = False,
    moe_token_dispatcher_type: str = "alltoall",
    moe_router_enable_expert_bias: bool = False,
    moe_router_load_balancing_type: str = "aux_loss",
    moe_expert_capacity_factor=None,
    moe_router_bias_update_rate: float = 0.001,
    # Optimizations
    moe_enable_deepep: bool = False,
    moe_deepep_num_sms: int = 20,
    moe_layer_recompute: bool = True,
    moe_permute_fusion: bool = False,
    moe_per_layer_logging: bool = True,
):
    if isinstance(hf_config, Qwen3ConfigT):
        assert gated_linear_unit
        assert activation_func == F.silu

    mlp_config = {
        # Experts
        "gated_linear_unit": gated_linear_unit,
        "activation_func": activation_func,
    }

    if is_moe:
        if isinstance(hf_config, Qwen3ConfigT):
            # Checks specific for Qwen3Moe
            assert moe_shared_expert_intermediate_size is None
            assert moe_router_score_function == "softmax"
            assert not moe_router_pre_softmax
            assert not moe_router_enable_expert_bias
            assert moe_router_load_balancing_type == "aux_loss"
            assert moe_expert_capacity_factor is None

        moe_config = {
            # Experts
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "num_moe_experts": hf_config.num_experts,
            "moe_grouped_gemm": moe_grouped_gemm,  # requires TransformerEngine
            "moe_use_legacy_grouped_gemm": moe_use_legacy_grouped_gemm,  # legacy cutlass grouped gemm
            "moe_shared_expert_intermediate_size": moe_shared_expert_intermediate_size,  # no shared expert
            # Router
            "moe_router_dtype": moe_router_dtype,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "moe_router_score_function": moe_router_score_function,
            "moe_router_pre_softmax": moe_router_pre_softmax,  # softmax is applied **after** topk in Qwen3-Moe
            "moe_token_dispatcher_type": moe_token_dispatcher_type,  # TODO: tune
            # auxiliary loss
            "moe_router_enable_expert_bias": moe_router_enable_expert_bias,  # aux-loss-free routing, only for sigmoid-scored router
            "moe_router_load_balancing_type": moe_router_load_balancing_type,
            "moe_expert_capacity_factor": moe_expert_capacity_factor,  # token choice -> no dropped tokens
            "moe_router_bias_update_rate": moe_router_bias_update_rate,  # TODO: check whether this is needed for aux_loss
            "moe_aux_loss_coeff": hf_config.router_aux_loss_coef,
            # optimizations
            "moe_enable_deepep": moe_enable_deepep,
            "moe_deepep_num_sms": moe_deepep_num_sms,  # TODO: tune
            "moe_layer_recompute": moe_layer_recompute,
            "moe_permute_fusion": moe_permute_fusion,  # TODO: tune
            "moe_per_layer_logging": moe_per_layer_logging,  # for auxiliary loss
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


def get_attn_config(hf_config: Qwen3ConfigT, qk_layernorm: bool = True):
    if isinstance(hf_config, Qwen3ConfigT):
        # Qwen3 specifies a head_dim, important for calculating correct qkv proj dims
        assert hasattr(hf_config, "head_dim") and hf_config.head_dim is not None
        assert qk_layernorm

    return {
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),  # hidden_size // num_attention_heads
        "qk_layernorm": qk_layernorm,
    }


def get_precision_config(hf_config: Qwen3ConfigT, dtype: torch.dtype = torch.bfloat16):
    if isinstance(hf_config, Qwen3ConfigT):
        assert dtype == hf_config.torch_dtype

    return {
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,  # Should be true for Qwen3
    }


# Optimizations - disable all when converting checkpoint
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

_DIRECT_MAPPING = {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "decoder.final_layernorm.weight": "model.norm.weight",
    "output_layer.weight": "lm_head.weight",
}
_ATTENTION_MAPPING = {
    "self_attention.linear_proj.weight": [
        "model.layers.{layer_number}.self_attn.o_proj.weight"
    ],
    "self_attention.linear_qkv.layer_norm_weight": [
        "model.layers.{layer_number}.input_layernorm.weight"
    ],
    "self_attention.q_layernorm.weight": [
        "model.layers.{layer_number}.self_attn.q_norm.weight"
    ],
    "self_attention.k_layernorm.weight": [
        "model.layers.{layer_number}.self_attn.k_norm.weight"
    ],
    "self_attention.linear_qkv.weight": [
        "model.layers.{layer_number}.self_attn.q_proj.weight",
        "model.layers.{layer_number}.self_attn.k_proj.weight",
        "model.layers.{layer_number}.self_attn.v_proj.weight",
    ],
    "self_attention.linear_qkv.bias": [
        "model.layers.{layer_number}.self_attn.q_proj.bias",
        "model.layers.{layer_number}.self_attn.k_proj.bias",
        "model.layers.{layer_number}.self_attn.v_proj.bias",
    ],
}
_DENSE_MLP_MAPPING = {
    "mlp.linear_fc1.weight": [
        "model.layers.{layer_number}.mlp.gate_proj.weight",
        "model.layers.{layer_number}.mlp.up_proj.weight",
    ],
    "mlp.linear_fc1.layer_norm_weight": [
        "model.layers.{layer_number}.post_attention_layernorm.weight"
    ],
    "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
}
_MOE_MLP_MAPPING = {
    "shared_experts.linear_fc1.weight": [
        "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
        "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
    ],
    "pre_mlp_layernorm": [
        "model.layers.{layer_number}.post_attention_layernorm.weight"
    ],
    "shared_experts.linear_fc2.weight": [
        "model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
    ],
    "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
    "shared_experts.gate_weight": [
        "model.layers.{layer_number}.mlp.shared_expert_gate.weight"
    ],
    "mlp.experts.linear_fc1": [
        "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
        "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
    ],
    "mlp.experts.linear_fc2": [
        "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
    ],
}


MCORE_TO_HF_PARAM_MAPPINGS = {
    "pre_post_decoder": {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    },
    "attention": {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    },
    "mlp": {
        "dense": {
            "mlp.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.gate_proj.weight",
                "model.layers.{layer_number}.mlp.up_proj.weight",
            ],
            "mlp.linear_fc1.layer_norm_weight": [
                "model.layers.{layer_number}.post_attention_layernorm.weight"
            ],
            "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        },
        "moe": {
            "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
            "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
            "mlp.experts.linear_fc1": [
                "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
            ],
            "mlp.experts.linear_fc2": [
                "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
            ],
        },
    },
}
