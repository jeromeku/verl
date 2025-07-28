import os
import re
import time
from argparse import Namespace
from contextlib import ExitStack, contextmanager
from dataclasses import fields
from unittest.mock import patch

import torch
import torch.distributed as dist
from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import (
    DistributedDataParallelConfig,
    TorchFullyShardedDataParallelConfig,
)
from megatron.core.distributed.custom_fsdp import FullyShardedDataParallel as custom_FSDP
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.training.global_vars import get_args
from megatron.training.utils import unwrap_model

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False
from .data_utils import ShardLoader
from .qwen3_configuration import (
    _ATTENTION_MAPPING,
    _DENSE_MLP_MAPPING,
    _DIRECT_MAPPING,
    _MOE_MLP_MAPPING,
    MCORE_TO_HF_PARAM_MAPPINGS,
    Qwen3ConfigT,
    Qwen3MoeConfig,
)


def update_args(
    args: Namespace,
    hf_config: Qwen3ConfigT,
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

    if isinstance(hf_config, Qwen3MoeConfig):
        args.num_experts = hf_config.num_experts

    args.vocab_size = hf_config.vocab_size
    args.padded_vocab_size = args.vocab_size
    args.untie_embeddings_and_output_weights = not hf_config.tie_word_embeddings
    args.position_embedding_type = "rope"
    args.rotary_percent = 1.0
    args.rotary_base = hf_config.rope_theta
    args.rope_scaling = True if hf_config.rope_scaling is not None else False

    args.no_load_optim = True
    args.no_load_rng = True
    args.perform_initialization = not args.init_model_with_meta_device
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


def set_vpp_size(hf_config: Qwen3ConfigT, args: Namespace):
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


def init_distributed(backend="nccl", world_size: int = None, rank: int = None):
    world_size = world_size or os.environ.get("WORLD_SIZE", None)
    rank = rank or os.environ.get("RANK", None)

    assert world_size is not None and rank is not None, (
        "`world_size` and `rank` must be provided when using `fake` backend"
    )

    world_size = int(world_size)
    rank = int(rank)

    if backend == "fake":
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        # world_size = world_size
        # rank = rank
    else:
        store = None
        # world_size = -1
        # rank = -1

    torch.distributed.init_process_group(
        backend=backend, store=store, rank=rank, world_size=world_size
    )


def init_mpu(tp=1, vpp=1, pp=1, cp=1, ep=1, etp=1, seed=0):
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(seed)


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=False,
    init_on_meta: bool = True,
):
    args = get_args()

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
            for f in fields(DistributedDataParallelConfig):
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


def get_gpt_model_args(hf_config: Qwen3ConfigT):
    return {
        "vocab_size": hf_config.vocab_size,
        "max_sequence_length": hf_config.max_position_embeddings,
        "position_embedding_type": "rope",
        "rotary_base": hf_config.rope_theta,
    }


# ---- Weight Conversion ---- #

LAYER_NUMBER_REGEX = re.compile(r"decoder\.layers\.(\d+)\.")
EXPERT_IDX_REGEX = re.compile(r"(?<=\.weight)(\d+)$")

MCORE_ATTN_PAT = "self_attention"
MCORE_QKV_PAT = "linear_qkv"
MCORE_ATTN_QKV_PAT = f"{MCORE_ATTN_PAT}.{MCORE_QKV_PAT}"

MCORE_MLP_PAT = "mlp"
MCORE_MLP_FC_PAT = "linear_fc"
MCORE_MLP_FC1_PAT = f"{MCORE_MLP_FC_PAT}1"
MCORE_MLP_FC2_PAT = f"{MCORE_MLP_FC_PAT}2"
MCORE_EXPERTS_PAT = "experts"
MCORE_EXPERTS_FC_PAT = f"{MCORE_MLP_PAT}.{MCORE_EXPERTS_PAT}.{MCORE_MLP_FC_PAT}"

TE_STATE_PAT = "_extra_state"


def remove_te_keys(keys: list[str]):
    return list(filter(lambda k: k.find(TE_STATE_PAT) < 0, keys))


def remap_pp(model: torch.nn.Module | GPTModel):
    model = unwrap_model(model)

    # Remap from pp layer shard idx -> global layer idx
    # NOTE: "layer_number" starts at 1
    local_to_global = {}
    assert hasattr(model, "decoder")

    for idx, layer in enumerate(model.decoder.layers):
        local_to_global[idx] = layer.layer_number - 1

    def _rename_decoder_layers(param_names: list[str]):
        name_map = {}
        for name in param_names:
            match = LAYER_NUMBER_REGEX.search(name)
            if match:
                local_layer_idx = int(match.group(1))
                global_layer_idx = local_to_global[local_layer_idx]
                new_name = name.replace(f"layers.{local_layer_idx}", f"layers.{global_layer_idx}")
            else:
                new_name = name
            name_map[name] = new_name

        return name_map

    _all_param_names = [k for k in model.state_dict().keys() if "_extra_state" not in k]
    all_param_names = remove_te_keys(model.state_dict().keys())
    assert _all_param_names == all_param_names

    name_map = _rename_decoder_layers(all_param_names)

    ret = {}

    for param_name in all_param_names:
        keyword = "decoder.layers."
        if keyword in param_name:
            layer_idx = int(param_name.split(keyword)[1].split(".")[0])

            global_layer_idx = local_to_global[layer_idx]
            ret[param_name] = param_name.replace(
                f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
            )
        else:
            ret[param_name] = param_name

    assert ret == name_map

    return name_map


def remap_param_names_for_ep_pp(model: GPTModel):
    model = unwrap_model(model)

    # Remap from pp layer shard idx -> global layer idx
    # NOTE: "layer_number" starts at 1
    assert hasattr(model, "decoder")

    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()

    def _remap_pp_layers(param_names: list[str]):
        local_to_global = {}
        for idx, layer in enumerate(model.decoder.layers):
            local_to_global[idx] = layer.layer_number - 1

        name_map = {}
        for name in param_names:
            match = LAYER_NUMBER_REGEX.search(name)
            if match:
                local_layer_idx = int(match.group(1))
                global_layer_idx = local_to_global[local_layer_idx]
                new_name = name.replace(f"layers.{local_layer_idx}", f"layers.{global_layer_idx}")
            else:
                new_name = name
            name_map[name] = new_name

        return name_map

    def _remap_ep_layers(name_map: dict[str, str]):
        num_experts = model.config.num_moe_experts
        num_experts_per_rank = num_experts // ep_size
        local_expert_to_global_expert = {
            i: i + num_experts_per_rank * ep_rank for i in range(num_experts_per_rank)
        }
        for k in name_map.keys():
            v = name_map[k]
            if ".mlp.experts.linear_fc" in v:
                name_prefix, local_expert_id = v.split(".weight")
                global_expert_idx = local_expert_to_global_expert[int(local_expert_id)]
                name_map[k] = f"{name_prefix}.weight{global_expert_idx}"

    all_param_names = remove_te_keys(model.state_dict().keys())
    name_map = _remap_pp_layers(all_param_names)

    if ep_size > 1:
        _remap_ep_layers(name_map)

    return name_map


def _weight_name_mapping_mcore_local_to_global(model: GPTModel) -> dict[str, str]:
    """
    Map local weight names to global weight names, supporting VPP and EP.

    Args:
        model: The model instance

    Returns:
        dict: Mapping from local weight names to global weight names
    """
    # vpp
    local_layer_to_global_layer = {}
    model = unwrap_model(model)
    if hasattr(model, "decoder"):
        for idx, layer in enumerate(model.decoder.layers):
            local_layer_to_global_layer[idx] = layer.layer_number - 1
    all_param_names = [k for k in model.state_dict().keys() if "_extra_state" not in k]
    ret = {}
    for param_name in all_param_names:
        keyword = "decoder.layers."
        if keyword in param_name:
            layer_idx = int(param_name.split(keyword)[1].split(".")[0])
            global_layer_idx = local_layer_to_global_layer[layer_idx]
            ret[param_name] = param_name.replace(
                f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
            )
        else:
            ret[param_name] = param_name

    # ep
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()

    if ep_size > 1:
        num_experts = model.config.num_moe_experts
        num_experts_per_rank = num_experts // ep_size
        local_expert_to_global_expert = {
            i: i + num_experts_per_rank * ep_rank for i in range(num_experts_per_rank)
        }
        for k in ret.keys():
            v = ret[k]
            if ".mlp.experts.linear_fc" in v:
                name_prefix, local_expert_id = v.split(".weight")
                global_expert_idx = local_expert_to_global_expert[int(local_expert_id)]
                ret[k] = f"{name_prefix}.weight{global_expert_idx}"

    return ret




def _weight_name_mapping_attention(name: str) -> list[str]:
    """
    Map attention weight names from MCore to Hugging Face.

    Args:
        name: MCore weight name

    Returns:
        list: Corresponding Hugging Face weight names

    Raises:
        NotImplementedError: If the parameter name is unsupported
    """
    layer_number = name.split(".")[2]
    convert_names = []
    for keyword, mapping_names in _ATTENTION_MAPPING.items():
        if keyword in name:
            convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
            break
    if len(convert_names) == 0:
        raise NotImplementedError(f"Unsupported parameter name: {name}")
    return convert_names


# def _weight_name_mapping_mlp(name: str, is_moe: bool = False) -> list[str]:
#     """
#     Map MLP weight names from MCore to Hugging Face.

#     Args:
#         name: MCore weight name

#     Returns:
#         list: Corresponding Hugging Face weight names

#     Raises:
#         NotImplementedError: If the parameter name is unsupported
#     """
#     mlp_mapping = _MOE_MLP_MAPPING if is_moe else _DENSE_MLP_MAPPING
#     layer_number = name.split(".")[2]
#     convert_names = []
#     for keyword, mapping_names in mlp_mapping.items():
#         if keyword in name:
#             if "{expert_id}" in mapping_names[0]:
#                 assert is_moe
#                 expert_id = name.split("weight")[-1]
#                 convert_names.extend(
#                     [
#                         x.format(layer_number=layer_number, expert_id=expert_id)
#                         for x in mapping_names
#                     ]
#                 )
#             else:
#                 convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
#             break
#     if len(convert_names) == 0:
#         raise NotImplementedError(f"Unsupported parameter name: {name}")
#     return convert_names


def _weight_name_mapping_mlp(name: str, is_moe: bool = False) -> list[str]:
    layer_number = name.split(".")[2]
    convert_names = []
    mlp_mapping = _MOE_MLP_MAPPING if is_moe else _DENSE_MLP_MAPPING
    for keyword, mapping_names in mlp_mapping.items():
        if keyword in name:
            if "{expert_id}" in mapping_names[0]:
                assert is_moe
                expert_id = name.split("weight")[-1]
                convert_names.extend(
                    [
                        x.format(layer_number=layer_number, expert_id=expert_id)
                        for x in mapping_names
                    ]
                )
            else:
                convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
            break
    if len(convert_names) == 0:
        raise NotImplementedError(f"Unsupported parameter name: {name}")
    return convert_names


def _weight_name_mapping_mcore_to_hf(mcore_weights_name: str, is_moe: bool = False) -> list[str]:
    """
    Map MCore weight names to Hugging Face weight names.

    Args:
        mcore_weights_name: MCore weight name

    Returns:
        list: Corresponding Hugging Face weight names
    """
    assert "_extra_state" not in mcore_weights_name, "extra_state should not be loaded"

    if mcore_weights_name in _DIRECT_MAPPING:
        return [_DIRECT_MAPPING[mcore_weights_name]]

    if "self_attention" in mcore_weights_name:
        return _weight_name_mapping_attention(mcore_weights_name)
    elif "mlp" in mcore_weights_name:
        return _weight_name_mapping_mlp(mcore_weights_name, is_moe=is_moe)
    else:
        raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")


def _local_to_hf(local_to_global: dict[str, str], is_moe: bool = False):
    local_to_hf_map = {
        k: _weight_name_mapping_mcore_to_hf(v, is_moe)
        for k, v in local_to_global.items()
        if "_extra_state" not in k
    }
    return local_to_hf_map


def _extract_layer_number(name: str):
    match = LAYER_NUMBER_REGEX.search(name)

    if not match:
        raise ValueError(f"Could not identify layer number in {name}")

    layer_number = int(match.group(1))

    return layer_number


def map_mcore_hf_param_names(
    local_to_global_map: dict[str, str], is_moe: bool = False
) -> dict[str, str]:
    pre_post_decoder_mapping = MCORE_TO_HF_PARAM_MAPPINGS["pre_post_decoder"]
    attention_mapping = MCORE_TO_HF_PARAM_MAPPINGS["attention"]
    mlp_mapping = (
        MCORE_TO_HF_PARAM_MAPPINGS["mlp"]["moe"]
        if is_moe
        else MCORE_TO_HF_PARAM_MAPPINGS["mlp"]["dense"]
    )

    def _map_attn(name: str) -> list[str]:
        layer_number = _extract_layer_number(name)

        mapped_names = []
        for keyword, mapping_names in attention_mapping.items():
            if keyword in name:
                mapped_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                break

        if len(mapped_names) == 0:
            raise ValueError(f"Attention parameter name {name} not recognized")

        return mapped_names

    def _map_mlp(name: str) -> list[str]:
        layer_number = _extract_layer_number(name)

        mapped_names = []
        for mcore_pat, hf_pats in mlp_mapping.items():
            if mcore_pat in name:
                if "expert_id" in hf_pats[0]:
                    assert is_moe

                    match = EXPERT_IDX_REGEX.search(name)
                    if not match:
                        raise ValueError(f"Unable to identify expert id in {name}")
                    expert_id = int(match.group(1))

                    mapped_names.extend(
                        [
                            pat.format(layer_number=layer_number, expert_id=expert_id)
                            for pat in hf_pats
                        ]
                    )
                else:
                    mapped_names.extend([pat.format(layer_number=layer_number) for pat in hf_pats])
                break

        if len(mapped_names) == 0:
            breakpoint()
            raise ValueError(f"MLP parameter name {name} not recognized")

        return mapped_names

    def _mcore_to_hf(name: str) -> list[str]:
        hf_name = pre_post_decoder_mapping.get(name, None)

        if hf_name is None:
            if MCORE_ATTN_PAT in name:
                hf_name = _map_attn(name)
            elif MCORE_MLP_PAT in name:
                hf_name = _map_mlp(name)
            else:
                raise ValueError(f"Param name {name} not recognized")

        # Return list[str] since mcore param could map to multiple hf params
        if not isinstance(hf_name, list):
            hf_name = [hf_name]

        return hf_name

    local_to_hf_map = {
        k: _mcore_to_hf(local_to_global_map[k]) for k in remove_te_keys(local_to_global_map.keys())
    }

    return local_to_hf_map

    # 3 categories of params: embeddings / final norm / output_layer, attn, and mlp


def _weight_to_mcore_format(
    hf_config, mcore_weights_name: str, hf_weights: list[torch.Tensor]
) -> torch.Tensor:
    if len(hf_weights) == 1:
        return hf_weights[0]
    if (
        "self_attention.linear_qkv." in mcore_weights_name
        and "layer_norm" not in mcore_weights_name
    ):
        # merge qkv
        assert len(hf_weights) == 3
        num_key_value_heads = hf_config.num_key_value_heads
        hidden_dim = hf_config.hidden_size
        num_attention_heads = hf_config.num_attention_heads
        head_dim = getattr(hf_config, "head_dim", hidden_dim // num_attention_heads)
        group_dim = head_dim * num_attention_heads // num_key_value_heads
        q, k, v = hf_weights
        # q k v might be tp split
        real_num_key_value_heads = q.shape[0] // group_dim
        q = q.view(
            [
                real_num_key_value_heads,
                group_dim,
                -1,
            ]
        )
        k = k.view([real_num_key_value_heads, head_dim, -1])
        v = v.view([real_num_key_value_heads, head_dim, -1])
        out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

        qkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
        return qkv
    elif "linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name:
        # merge gate_proj and up_proj
        assert len(hf_weights) == 2
        gate, up = hf_weights
        return torch.cat([gate, up], dim=0)

    raise NotImplementedError(f"Unsupported parameter name: {mcore_weights_name}")


def _weight_split_across_tp(
    mcore_weights_name: str,
    mcore_weights: torch.Tensor,
    param: torch.Tensor,
    tp_split_size: int,
) -> list[torch.Tensor]:
    if tp_split_size == 1:
        return [mcore_weights]

    if (
        "self_attention.linear_qkv." in mcore_weights_name
        and "layer_norm" not in mcore_weights_name
    ):
        return mcore_weights.chunk(tp_split_size)
    elif "linear_fc1.weight" in mcore_weights_name or "linear_fc1.bias" in mcore_weights_name:
        gate, up = mcore_weights.chunk(2)
        gates = gate.chunk(tp_split_size)
        ups = up.chunk(tp_split_size)
        ret = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
    elif "mlp.experts.linear_fc2.weight" in mcore_weights_name:  # moe
        ret = mcore_weights.chunk(tp_split_size, dim=1)
    else:
        if param.shape == mcore_weights.shape:
            return [mcore_weights for _ in range(tp_split_size)]
        assert len(param.shape) == len(mcore_weights.shape)
        for partition_dim, (s1, s2) in enumerate(zip(param.shape, mcore_weights.shape)):
            if s1 != s2:
                break

        ret = mcore_weights.chunk(tp_split_size, dim=partition_dim)
    return ret


def _load_hf_weights(
    safetensor_io,
    hf_config: Qwen3ConfigT,
    model: GPTModel,
    local_to_hf_map: dict[str, str],
    scatter_weights: bool = False,
    memory_efficient: bool = False,
    strict: bool = True,
    device: str = "cuda",
):
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    etp_rank = mpu.get_expert_tensor_parallel_rank()
    etp_group = mpu.get_expert_tensor_parallel_group()
    etp_size = mpu.get_expert_tensor_parallel_world_size()

    to_load_from_disk = []
    load_from_disk = not scatter_weights
    meta_device = any(p.device.type == "meta" for p in model.parameters())

    new_sd = {}

    for local_name, hf_names in local_to_hf_map.items():
        if ".mlp.experts.linear_fc" in local_name:
            should_load = load_from_disk or (scatter_weights and etp_rank == 0)
            if should_load:
                to_load_from_disk.extend(hf_names)
        else:
            should_load = load_from_disk or (scatter_weights and tp_rank == 0)
            if should_load:
                to_load_from_disk.extend(hf_names)
            else:
                # special case for lm_head.weight
                # if make value model, every tp rank will load lm_head.weight
                if "lm_head.weight" in hf_names:
                    to_load_from_disk.extend(hf_names)

    # load huggingface weights
    if not memory_efficient:
        hf_weights_map = safetensor_io.load_some_hf_weight(to_load_from_disk)

    # import mcore weights
    for local_name, hf_names in local_to_hf_map.items():
        param = model.state_dict()[local_name]

        # hf format to mcore format
        if set(to_load_from_disk) & set(hf_names):
            if not memory_efficient:
                hf_weights = [hf_weights_map[x] for x in hf_names]
            else:
                hf_weights = [safetensor_io.load_one_hf_weight(x) for x in hf_names]
            mcore_weight = _weight_to_mcore_format(hf_config, local_name, hf_weights)
        else:
            mcore_weight = None

        if hf_names[0] == "lm_head.weight":
            if param.shape[0] == 1 and mcore_weight.shape[0] != 1:
                # skip lm_head.weight when the model is a value model
                continue

        # if param.device.type == "meta":
        #     param = param.new_empty(size=param.size(), device=device)

        param_to_load = torch.empty_like(param)

        if ".mlp.experts.linear_fc" in local_name:
            # split mcore weights across etp
            should_load = load_from_disk or (scatter_weights and etp_rank == 0)

            if should_load:
                mcore_weights_tp_split = _weight_split_across_tp(
                    local_name, mcore_weight, param, etp_size
                )
                mcore_weights_tp_split = list(mcore_weights_tp_split)
                mcore_weights_tp_split = [t.to(device) for t in mcore_weights_tp_split]
            else:
                mcore_weights_tp_split = None

            if scatter_weights:
                torch.distributed.scatter(
                    param_to_load,
                    mcore_weights_tp_split,
                    src=torch.distributed.get_global_rank(etp_group, 0),
                    group=etp_group,
                )
            else:
                param_to_load = mcore_weights_tp_split[etp_rank]
        else:
            should_load = load_from_disk or (scatter_weights and tp_rank == 0)
            # split mcore weights across tp
            if should_load:
                mcore_weights_tp_split = _weight_split_across_tp(
                    local_name, mcore_weight, param, tp_size
                )
                mcore_weights_tp_split = list(mcore_weights_tp_split)
                mcore_weights_tp_split = [t.to(device) for t in mcore_weights_tp_split]
            else:
                mcore_weights_tp_split = None

            if scatter_weights:
                torch.distributed.scatter(
                    param_to_load,
                    mcore_weights_tp_split,
                    src=torch.distributed.get_global_rank(tp_group, 0),
                    group=tp_group,
                )
            else:
                param_to_load = mcore_weights_tp_split[tp_rank]

        new_sd[local_name] = param_to_load
        #    param.copy_(param_to_load)
    # strict must be false because of empty TE states
    model.load_state_dict(new_sd, strict=False, assign=True)




def _interleave_and_merge_qkv(
    num_attn_heads, num_kv_heads, hidden_size, head_dim: int, hf_weights: list[torch.Tensor]
) -> torch.Tensor:
    assert len(hf_weights) == 3
    assert num_attn_heads % num_kv_heads == 0

    query_group_ratio = num_attn_heads // num_kv_heads
    qdim_per_kv_head = head_dim * query_group_ratio

    q, k, v = hf_weights
    q_proj_size = head_dim * num_attn_heads
    k_proj_size = head_dim * num_kv_heads

    assert q.shape[0] == q_proj_size
    assert k.shape[0] == k_proj_size
    assert q.shape[0] // qdim_per_kv_head == num_kv_heads

    q = q.view(
        [
            num_kv_heads,
            qdim_per_kv_head,
            -1,
        ]
    )
    k = k.view([num_kv_heads, head_dim, -1])
    v = v.view([num_kv_heads, head_dim, -1])

    qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_size).contiguous()

    return qkv


def _merge_mlp(src_weights: list[torch.Tensor]) -> torch.Tensor:
    assert len(src_weights) == 2
    gate, up = src_weights
    return torch.cat([gate, up], dim=0)


def _find_partition_dim(src_shape: torch.Size, dst_shape: torch.Size):
    for partition_dim, (s1, s2) in enumerate(zip(src_shape, dst_shape)):
        if s1 != s2:
            break

    return partition_dim


def load_hf_weights(
    weights_loader: ShardLoader,
    hf_config: Qwen3ConfigT,
    model: GPTModel,
    local_to_hf_map: dict[str, str],
    device: str = "cuda",
):
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    etp_rank = mpu.get_expert_tensor_parallel_rank()
    etp_size = mpu.get_expert_tensor_parallel_world_size()

    num_attn_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.head_dim
    hidden_size = hf_config.hidden_size

    new_sd = {}

    def _hf_to_mcore_weights_format(
        mcore_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        if len(hf_weights) == 1:
            return hf_weights[0]
        elif MCORE_ATTN_QKV_PAT in mcore_name and "layer_norm" not in mcore_name:
            return _interleave_and_merge_qkv(
                num_attn_heads=num_attn_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                hf_weights=hf_weights,
            )
        elif MCORE_MLP_FC1_PAT in mcore_name:
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        else:
            raise ValueError(f"{mcore_name} not recognized")

    def _shard_across_tp(
        name: str,
        src_param_mcore: torch.Tensor,
        param_to_load: torch.Tensor,
        tp_size: int,
    ) -> list[torch.Tensor]:
        if tp_size == 1:
            return [src_param_mcore]

        is_qkv = MCORE_ATTN_QKV_PAT in name and "layer_norm" not in name
        is_fc1 = MCORE_MLP_FC1_PAT in name
        is_fc2 = MCORE_MLP_FC2_PAT in name

        if is_qkv:
            sharded_weights = src_param_mcore.chunk(tp_size)
        elif is_fc1:
            gate, up = src_param_mcore.chunk(2)
            gates = gate.chunk(tp_size)
            ups = up.chunk(tp_size)
            sharded_weights = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
        elif is_fc2:
            sharded_weights = src_param_mcore.chunk(tp_size, dim=1)
        else:
            # Remaining non-attn and non-mlp cases
            
            # Replicated params
            if param_to_load.shape == src_param_mcore.shape:
                sharded_weights = [src_param_mcore for _ in range(tp_size)]
            else:
                # Misc
                # TODO: more robust checking for this case
                assert len(param_to_load.shape) == len(src_param_mcore.shape)
                partition_dim = _find_partition_dim(param_to_load.shape, src_param_mcore.shape)
                sharded_weights = src_param_mcore.chunk(tp_size, dim=partition_dim)
        
        return sharded_weights

    for local_name, hf_names in local_to_hf_map.items():
        param_to_load = model.state_dict()[local_name]

        src_params_hf = [weights_loader.get_tensor(n) for n in hf_names]
        src_param_mcore = _hf_to_mcore_weights_format(local_name, src_params_hf)

        if MCORE_EXPERTS_FC_PAT in local_name:
            _tp_size = etp_size
            _tp_rank = etp_rank
        else:
            _tp_size = tp_size
            _tp_rank = tp_rank

        sharded_weights = _shard_across_tp(local_name, src_param_mcore, param_to_load, _tp_size)
        sharded_weights = [w.to(device) for w in sharded_weights]
        new_sd[local_name] = sharded_weights[_tp_rank]

    # strict must be false because of empty TE states, assign must be True when using init on meta
    model.load_state_dict(new_sd, strict=False, assign=True)


def dist_print(*msg, delay: int = 1, rank0_only: bool = False):
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank0_only and rank != 0:
            return
        time.sleep(rank * delay)

    print(f"{rank=}:", *msg, flush=True)

def check_weights(model_path, mcore_model_parts):

    def load_mbridge_ref():
        from mbridge import AutoBridge

        bridge = AutoBridge.from_pretrained(model_path)
        bridge.config.perform_initialization = False
        bridge.config.use_cpu_initialization = True
        ref_models = bridge.get_model(use_cpu_initialization=True)
        bridge.load_weights(ref_models, model_path)
    
        return ref_models
    
    ref_models = load_mbridge_ref()

    for ref_m, test_m in zip(ref_models, mcore_model_parts):
        ref_sd = ref_m.state_dict()
        test_sd = test_m.state_dict()

        if ref_sd.keys() != test_sd.keys():
            breakpoint()

        for k in ref_sd.keys():
            if "_extra_state" in k:
                continue

            expected = ref_sd[k]
            actual = test_sd[k].to(expected.device)

            if expected is None:
                breakpoint()

            if not expected.equal(actual):
                breakpoint()

    dist_print("State dicts match!")