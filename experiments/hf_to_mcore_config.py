# ruff: noqa
import argparse
from contextlib import contextmanager

from collections import Counter

import os
import sys
from dataclasses import asdict
from pathlib import Path
from pprint import pp
from contextlib import ExitStack, nullcontext
from argparse import Namespace
import dataclasses
import megatron.core as mc

MEGATRON_ROOT = Path(mc.__file__).parents[2]
sys.path.append(MEGATRON_ROOT.resolve().as_posix())
from megatron.training.arguments import _add_distributed_args, _add_moe_args
import itertools
import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.utils.hub import cached_file

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

from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM

from megatron.core.distributed import (
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallelConfig,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP
from megatron.training.utils import unwrap_model
try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

# Dense
QWEN3_600M = "Qwen/Qwen3-0.6B"
QWEN3_4B = "Qwen/Qwen3-4B"

# MoE
QWEN3_30B_3B = "Qwen/Qwen3-30B-A3B"
QWEN3_235B_A22B = "Qwen/Qwen3-235B-A22B"

QWEN3_DENSE_MODELS = [QWEN3_600M, QWEN3_4B]
QWEN3_MOE_MODELS = [QWEN3_30B_3B, QWEN3_235B_A22B]

from contextlib import contextmanager
from unittest.mock import patch
import torch

def weights_generator(model_id: str = None, weight_files: list[str|Path] = None, device: str = "cpu"):
    assert model_id ^ weight_files
    
    if model_id:
        model_cache_dir = Path(snapshot_download(model_id))
        weight_files = list(model_cache_dir.glob("*.safetensors"))
    
    for wf in weight_files:
        with safe_open(wf, 'pt', device=device) as f:
            for k in f.keys():
                yield k, f.get_tensor(k)

@contextmanager
def meta_device_context():
    """
    Force every `torch.cuda.device(...)` or `torch.cuda.current_device()` call
    inside the `with`‑block to point at the *meta* backend instead of a real
    GPU.

    Examples
    --------
    >>> with init_on_meta():
    ...     model = model_provider_func(pre_process=True, post_process=True)
    >>> next(model.parameters()).device
    device(type='meta')
    """

    @contextmanager
    def _meta_device_ctx(*_args, **_kw):
        # Anything created in here inherits the default device = 'meta'
        with torch.device("meta"):
            yield

    import transformer_engine.pytorch.module.base as te_base

    def _noop_reset(*args, **kwargs):
        return

    patches = [
        patch("torch.cuda.device", _meta_device_ctx),
        patch("torch.cuda.current_device", lambda: torch.device("meta")),
    ]

    patches.append(patch.object(te_base, "reset_parameters", _noop_reset, create=True))
    patches.append(
        patch.object(
            getattr(te_base, "TransformerEngineBaseModule"), "reset_parameters", _noop_reset
        )
    )

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with torch.device("meta"):
            yield


def init_distributed(backend="nccl"):
    if backend == "fake":
        from torch.testing._internal.distributed.fake_pg import FakeStore
        store = FakeStore()
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = 0
    else:
        store = None
        world_size = -1
        rank = -1

    torch.distributed.init_process_group(backend=backend, store=store, rank=rank, world_size=world_size)

def init_mpu(tp=1, vpp=1, pp=1, cp=1, ep=1, etp=1, seed=0):
    
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        create_gloo_process_groups=False
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


def get_mlp_config(hf_config: Qwen3MoeConfig, is_moe: bool = False):
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


def hf_to_mcore(hf_config: Qwen3MoeConfig, is_moe: bool = False,  **kwargs) -> TransformerConfig:
    dtype = hf_config.torch_dtype

    assert dtype == torch.bfloat16

    arch_config = get_arch_config(hf_config)
    attn_config = get_attn_config(hf_config)

    mlp_config = get_mlp_config(hf_config, is_moe=is_moe)
    
    precision_config = get_precision_config(dtype)
    parallelism_config = get_parallelism()
    fusion_config = get_fusion_config()

    ac_config = get_activation_recompute_config()

    final_config = TransformerConfig(
        **arch_config,
        **attn_config,
        **mlp_config,
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


def get_model_provider_func(
    config: TransformerConfig, args: Namespace, parallel_output: bool = True
):
    use_transformer_engine = args.transformer_impl == "transformer_engine"

    def model_provider_func(pre_process: bool, post_process: bool, vp_stage: int = None):
        transformer_layer_spec = get_transformer_spec(
            config, use_transformer_engine=use_transformer_engine, vp_stage=vp_stage
        )

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
        with meta_device_context():
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
    if not (args.use_torch_fsdp2 and args.use_cpu_initialization) and not init_on_meta:
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


def update_args(
    args: Namespace,
    hf_config: Qwen3MoeConfig,
    use_transformer_engine: bool = True,
    **kwargs,
):
    
    # Required args for MCore args validation
    args.max_position_embeddings = hf_config.max_position_embeddings
    args.num_layers = hf_config.num_hidden_layers
    args.hidden_size = hf_config.hidden_size
    args.num_attention_heads = hf_config.num_attention_heads
    args.seq_length = hf_config.max_position_embeddings

    args.vocab_size = hf_config.vocab_size
    args.padded_vocab_size = args.vocab_size
    args.untie_embeddings_and_output_weights = not hf_config.tie_word_embeddings
    args.position_embedding_type = "rope"
    args.rotary_percent = 1.0
    args.rotary_base = hf_config.rope_theta
    args.rope_scaling = True if hf_config.rope_scaling is not None else False

    args.rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()

    # Should TE for optimized parallel linear, attn, and moe grouped linear
    args.transformer_impl = "transformer_engine" if use_transformer_engine else "local"

    for k, v in kwargs.items():
        setattr(args, k, v)

    return args


# TODO:
# attention backend, transformer_impl, optimizer config, te config
# modelparallelconfig


def get_model_param_devices(model: torch.nn.Module):
    param_devices = Counter(p.device.type for p in model.parameters())
    return param_devices


@contextmanager
def memory_context():
    def get_memory(prefix: str = ""):
        alloc, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        print(f"{prefix} - Alloc: {alloc / 1e9:.1f}GB Reserved: {reserved / 1e9:.1f}GB", flush=True)
        return alloc, reserved

    get_memory("BEFORE")
    yield
    get_memory("AFTER")

def get_total_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())

def get_module_param_count(model: torch.nn.Module):
    return {n: get_total_params(m) for n,m in model.named_children()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "HF -> Megatron Config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-id",
        help="HF model path, e.g., Qwen/Qwen3-30B-A3B",
        default=QWEN3_30B_3B,
        choices=[*QWEN3_DENSE_MODELS, *QWEN3_MOE_MODELS],
    )
    parser.add_argument("--backend", default="fake", choices=["fake", "gloo", "nccl"])

    add_megatron_arguments(parser)
    args = parser.parse_args()

    model_path = args.model_id
    is_moe = model_path in QWEN3_MOE_MODELS
    model_cls = Qwen3MoeForCausalLM if is_moe else Qwen3ForCausalLM
    hf_config = AutoConfig.from_pretrained(model_path)
    
    init_distributed(backend=args.backend)
    args = update_args(args, hf_config, use_transformer_engine=True)
    validate_args(args)
    
    breakpoint()

    #args = set_vpp_size(hf_config, args)

    init_mpu(
        tp=args.tensor_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        vpp=args.virtual_pipeline_model_parallel_size,
        cp=args.context_parallel_size,
        ep=args.expert_model_parallel_size,
        etp=args.expert_tensor_parallel_size,
    )
    init_on_meta = args.init_model_with_meta_device
    init_on_cpu = args.use_cpu_initialization
    assert not (init_on_meta and init_on_cpu), f"{init_on_meta=} and {init_on_cpu=} both set"

    # weights can't be initialized on meta
    if init_on_meta:
        args.perform_initialization = False

    # Do not initialize so that we can init on meta device
    transformer_config = hf_to_mcore(
        hf_config,
        is_moe=is_moe,
        perform_initialization=args.perform_initialization,
        use_cpu_initialization=args.use_cpu_initialization,
    )
    
    pp(hf_config.to_dict())
    pp(asdict(transformer_config))

    model_provider_func = get_model_provider_func(transformer_config, args)
    model_parts: list[GPTModel] = get_model(model_provider_func, init_on_meta=init_on_meta)
    print(model_parts[0])

    param_devices = sum((get_model_param_devices(m) for m in model_parts), Counter())
    mcore_num_params = sum(len(list(m.parameters())) for m in model_parts)

    if init_on_meta and not param_devices['meta'] == mcore_num_params:
        print(f"WARNING: not all params on 'meta': {param_devices}")

    with torch.device('meta'):
        ref_model: Qwen3ForCausalLM = model_cls(hf_config)
    
    hf_param_count = get_total_params(ref_model)
    mcore_param_count = sum(get_total_params(m) for m in model_parts)

    # TODO: better checks for various parallelisms
    # Param accounting gets complicated with tied weights, pipeline parallel
    breakpoint()
    if torch.distributed.get_world_size() == 1:
        assert hf_param_count == mcore_param_count, f"Param count mismatch: {hf_param_count} != {mcore_param_count}"
    
    from weight_converter import remap_pp
    for m in model_parts:
        ref, test = remap_pp(m)

    breakpoint()
    model_parts = unwrap_model(model_parts)

    # print([type(m) for m in model_parts])
    # for idx, m in enumerate(model_parts):
    #     print(f"Model part {idx}")
    #     pp(m.state_dict().keys())
    breakpoint()
    if False:
        print(f"HF Model total params: {hf_param_count}")
        print(f"MCore total params: {mcore_param_count}")
        print()
        
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3DecoderLayer
        print("HF Model Param Counts")
        pp(get_module_param_count(ref_model))
        qwen3_model = ref_model.model
        qwen3_decoder = qwen3_model.layers[0]
        decoder_counts = get_module_param_count(qwen3_decoder)
        pp(get_module_param_count(qwen3_model))
        pp(get_module_param_count(qwen3_decoder))
        print()
        gpt_model = model_parts[0]
        print("GPT Model Param Counts")
        pp(get_module_param_count(gpt_model))
        pp(get_module_param_count(gpt_model.decoder))
        gpt_decoder = gpt_model.decoder.layers[0]
        gpt_decoder_counts = get_module_param_count(gpt_decoder)
        pp(get_module_param_count(gpt_model.decoder.layers[0]))
        breakpoint()
        ref_mlp = qwen3_decoder.mlp
        test_mlp = gpt_decoder.mlp
        pp(get_module_param_count(ref_mlp))
        pp(get_module_param_count(test_mlp))
    
    # tokenizer_config = {
    #     "tokenizer_type": "HuggingFaceTokenizer",
    #     "make-vocab-size-divisible-by": 1187,
    #     "position_embedding_type": "rope",
    # }
