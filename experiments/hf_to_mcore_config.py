# ruff: noqa
import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pp
from argparse import Namespace
import dataclasses
import megatron.core as mc

MEGATRON_ROOT = Path(mc.__file__).parents[2]
sys.path.append(MEGATRON_ROOT.resolve().as_posix())
from megatron.training.arguments import _add_distributed_args, _add_moe_args

import torch
import torch.nn.functional as F
from megatron.training.arguments import add_megatron_arguments, validate_args

from megatron.core import mpu, tensor_parallel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.enums import ModelType
from megatron.core.transformer.module import Float16Module
from megatron.core.fp8_utils import correct_amax_history_if_needed

from transformers.models.qwen3_moe import Qwen3MoeConfig
from megatron.core.distributed import (
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallelConfig,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

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


def hf_to_mcore(hf_config: Qwen3MoeConfig, **kwargs) -> TransformerConfig:
    dtype = hf_config.torch_dtype
    assert dtype == torch.bfloat16

    arch_config = get_arch_config(hf_config)
    attn_config = get_attn_config(hf_config)
    moe_config = get_moe_config(hf_config)

    precision_config = get_precision_config(dtype)
    parallelism_config = get_parallelism()
    fusion_config = get_fusion_config()

    ac_config = get_activation_recompute_config()

    final_config = TransformerConfig(
        **arch_config,
        **attn_config,
        **moe_config,
        **precision_config,
        **parallelism_config,
        **fusion_config,
        **ac_config,
        **kwargs,
    )

    return final_config


def set_vpp_size(hf_config: Qwen3MoeConfig, args: Namespace):
    num_layers_per_pipeline_stage = hf_config.num_hidden_layers // args.pipeline_model_parallel_size
    if args.num_layers_per_virtual_pipeline_stage is not None:
        args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // int(
            args.num_layers_per_virtual_pipeline_stage
        )
    else:
        args.virtual_pipeline_model_parallel_size = None
    return args


def get_transformer_spec(
    config: TransformerConfig, use_transformer_engine: bool = True, vp_stage: int = None
):
    transformer_layer_spec = get_gpt_decoder_block_spec(
        config, use_transformer_engine=use_transformer_engine, vp_stage=vp_stage
    )
    return transformer_layer_spec


# From pretrain_gpt
def _get_gpt_model(
    config: TransformerConfig,
):
    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
    )


def model_provider(config: TransformerConfig, args: Namespace, use_transformer_engine: bool = True, parallel_output: bool = True):
    def model_provider_func(pre_process: bool, post_process: bool, vp_stage:int = None):
        transformer_layer_spec = get_transformer_spec(config, use_transformer_engine=use_transformer_engine, vp_stage=vp_stage)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            pre_process=pre_process,
            post_process=post_process,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            parallel_output=parallel_output,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            vp_stage=vp_stage,
        )

        return model
    return model_provider_func

def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=False,
    init_on_meta: bool = True,
):
    # Build model.
    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
                post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
                this_model = model_provider_func(
                    pre_process=pre_process, post_process=post_process, vp_stage=i
                )
                this_model.model_type = model_type
                this_model.vp_stage = i
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
            model.model_type = model_type
        return model

    if init_on_meta:
        with torch.device("meta"):
            model = build_model()
    else:
        model = build_model()

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    num_parameters = sum(
        [sum([p.nelement() for p in model_module.parameters()]) for model_module in model]
    )
    if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                num_parameters,
            ),
            flush=True,
        )

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not (args.use_torch_fsdp2 and args.use_cpu_initialization)
        and not args.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    config = model[0].config
    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(config, model_module) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
            DP = torch_FSDP
        elif args.use_custom_fsdp:
            DP = custom_FSDP
        else:
            DP = DDP

        if getattr(args, "use_torch_fsdp2", False):
            reshard_after_forward = getattr(args, "torch_fsdp2_reshard_after_forward", True)
            ddp_config = TorchFullyShardedDataParallelConfig(
                reshard_after_forward=reshard_after_forward
            )
        else:
            kwargs = {}
            for f in dataclasses.fields(DistributedDataParallelConfig):
                if hasattr(args, f.name):
                    kwargs[f.name] = getattr(args, f.name)
            kwargs["grad_reduce_in_fp32"] = args.accumulate_allreduce_grads_in_fp32
            kwargs["check_for_nan_in_grad"] = args.check_for_nan_in_loss_and_grad
            kwargs["check_for_large_grads"] = args.check_for_large_grads
            if args.ddp_num_buckets is not None:
                assert args.ddp_bucket_size is None, (
                    "Cannot specify both --ddp-num-buckets and --ddp-bucket-size"
                )
                assert args.ddp_num_buckets > 0, "--ddp-num-buckets must be greater than 0"
                kwargs["bucket_size"] = num_parameters // args.ddp_num_buckets
            else:
                kwargs["bucket_size"] = args.ddp_bucket_size
            kwargs["pad_buckets_for_high_nccl_busbw"] = args.ddp_pad_buckets_for_high_nccl_busbw
            kwargs["average_in_collective"] = args.ddp_average_in_collective
            if args.use_custom_fsdp and args.use_precision_aware_optimizer:
                kwargs["preserve_fp32_weights"] = False
            ddp_config = DistributedDataParallelConfig(**kwargs)

            # In the custom FSDP and DDP use path, we need to initialize the bucket size.
            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [
            DP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0)
                or args.overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model

def get_gpt_model_args(hf_config: Qwen3MoeConfig):
    return {
        "vocab_size": hf_config.vocab_size,
        "max_sequence_length": hf_config.max_position_embeddings,
        "position_embedding_type": "rope",
        "rotary_base": hf_config.rope_theta,
    }

def update_args_from_hf(args: Namespace, hf_config: Qwen3MoeConfig, use_transformer_engine: bool = True, **kwargs):
    args.vocab_size = hf_config.vocab_size
    args.padded_vocab_size = args.vocab_size
    args.max_position_embeddings = hf_config.max_position_embeddings
    args.untie_embeddings_and_output_weights = not hf_config.tie_word_embeddings
    args.position_embedding_type = "rope"
    args.rotary_percent = 1.0
    args.rotary_base = hf_config.rope_theta
    args.rope_scaling = True if hf_config.rope_scaling is not None else False
    args.transformer_impl = "transformer_engine" if use_transformer_engine else "local"
    
    for k,v in kwargs.items():
        setattr(args, k, v)

    return args

# TODO:
# attention backend, transformer_impl, optimizer config, te config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "HF -> Megatron Config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model_id", help="HF model path, e.g., Qwen/Qwen3-30B-A3B", default=QWEN3_30B_3B
    )
    add_megatron_arguments(parser)
    args = parser.parse_args()

    model_path = args.model_id

    hf_config: Qwen3MoeConfig = Qwen3MoeConfig.from_pretrained(model_path)
    args = update_args_from_hf(args, hf_config, use_transformer_engine=True)
    args = set_vpp_size(hf_config, args)

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
