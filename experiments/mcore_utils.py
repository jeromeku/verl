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
from torch.utils.data import DataLoader, TensorDataset

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False
from qwen3_configuration import Qwen3ConfigT

McoreModelT = list[GPTModel]

def patch_mcore_args(
    args: Namespace,
    **kwargs,
):
    # Required args for MCore args validation
    args.micro_batch_size = 1

    # Not needed when loading checkpoints
    args.no_load_optim = True
    args.no_load_rng = True    
    args.no_save_optim = True
    args.no_save_rng = True
    args.mock_data = True

    args.rank = args.rank or torch.distributed.get_rank()
    args.world_size = args.world_size or torch.distributed.get_world_size()

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
    config: TransformerConfig, parallel_output: bool = True
):
    args = get_args()
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
    rank = rank if rank is not None else os.environ.get("RANK", None)
    
    assert world_size is not None and rank is not None, (
        "`world_size` and `rank` must be provided when using `fake` backend"
    )

    world_size = int(world_size)
    rank = int(rank)

    if backend == "fake":
        from torch.testing._internal.distributed.fake_pg import FakeStore
        store = FakeStore()
    else:
        store = None

    torch.distributed.init_process_group(
        backend=backend, store=store, rank=rank, world_size=world_size
    )
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK")))

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


def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=False,
):
    args = get_args()
    init_on_meta = args.init_model_with_meta_device
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


def generate_dataset(vocab_size: int, num_samples: int = 100, seqlen: int = 100, batch_size: int = 2):

    input_ids = torch.randint(0, vocab_size, (num_samples, seqlen))
    position_ids = torch.arange(seqlen).expand(num_samples, -1) 
    attention_mask = torch.ones_like(input_ids)

    dataset = TensorDataset(input_ids, position_ids, attention_mask)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    
    return data_loader

def dist_print(*msg, delay: int = 1, rank0_only: bool = False):
    
    if dist.is_initialized():
        rank = dist.get_rank()
        if rank0_only and rank != 0:
            return
        time.sleep(rank * delay)
    else:
        rank = 0
        
    print(f"{rank=}:", *msg, flush=True)

def dist_breakpoint(rank: int = 0):
    if dist.is_initialized() and rank == dist.get_rank():
        breakpoint()
    dist.barrier()