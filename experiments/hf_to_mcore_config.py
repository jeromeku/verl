import os

import torch
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from transformers.models.qwen3_moe import Qwen3MoeConfig

QWEN3_30B_3B = "Qwen/Qwen3-30B-A3B"
QWEN3_235B_A22B = "Qwen/Qwen3-235B-A22B"

def init_distributed():

    torch.distributed.init_process_group("nccl")
    
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(0)

def get_parallelism(sequence_parallel: bool = None):
    return {
         "tensor_model_parallel_size": mpu.get_tensor_model_parallel_world_size(),
        "pipeline_model_parallel_size": mpu.get_pipeline_model_parallel_world_size(),
        "expert_model_parallel_size": mpu.get_expert_model_parallel_world_size(),
        "expert_tensor_parallel_size": mpu.get_expert_tensor_parallel_world_size(),
        "virtual_pipeline_model_parallel_size": mpu.get_virtual_pipeline_model_parallel_world_size(),
        "context_parallel_size": mpu.get_data_parallel_world_size(),
        "sequence_parallel": sequence_parallel or mpu.get_tensor_model_parallel_world_size() > 1,
    }


def hf_to_mcore(model_id=QWEN3_30B_3B):
    hf_config = Qwen3MoeConfig.from_pretrained(model_id)
    dtype = hf_config.torch_dtype

    transformer_config = {
        # Model architecture parameters
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),
        "qk_layernorm": True,
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "normalization": "RMSNorm",
    }
    precision_config = {
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,
    }
    parallelism_config = get_parallelism(sequence_parallel=True)

    misc_config = {
        "use_cpu_initialization": False,
        "variable_seq_lengths": True,
        "masked_softmax_fusion": True,
        "moe_token_dispatcher_type": "alltoall",
        "add_bias_linear": False,
    # Other optimizations
    "persist_layer_norm": True,
    "bias_activation_fusion": True,
    "bias_dropout_fusion": True,
    }

    moe_config = {
    # Experts
    "gated_linear_unit": True,
    "activation_func": F.silu,
    "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
    "num_moe_experts": hf_config.num_experts,
    "moe_grouped_gemm": True,
    # Router
    "moe_router_bias_update_rate": 0.001,
    "moe_router_topk": hf_config.num_experts_per_tok,
    "moe_aux_loss_coeff": hf_config.router_aux_loss_coef,
    "moe_router_score_function": "softmax",
    "moe_router_pre_softmax": False,
    "moe_router_load_balancing_type": "none",

}