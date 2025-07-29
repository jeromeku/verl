# ruff: noqa E402
import sys
from pathlib import Path
from pprint import pprint
import megatron.core as mc

# Need include megatron root to resolve modules outside of megatron.core
MEGATRON_ROOT = Path(mc.__file__).parents[2]
sys.path.append(MEGATRON_ROOT.resolve().as_posix())

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

from megatron.training.arguments import add_megatron_arguments, validate_args
from megatron.training.global_vars import set_global_variables, get_args
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.training.utils import unwrap_model
from megatron.core import mpu

from transformers import AutoConfig
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.distributed as dist
import torch
from collections import Counter
from mcore_utils import (
    init_distributed,
    init_mpu,
    patch_mcore_args,
    dist_print,
    get_model_provider_func,
    get_model,
    McoreModelT,
)
from qwen3_configuration import (
    QWEN3_MOE_MODELS,
    Qwen3MCoreConfig,
    Qwen3ConfigT,
    ParallelismConfig,
    Qwen3ModelT,
    is_qwen3_moe_config,
    QWEN3_MODELS,
)
from debugging import get_model_param_devices, get_total_params, get_module_param_count
from ref.convert_utils import _weight_name_mapping_mcore_local_to_global
from weight_conversion import remap_param_names_for_ep_pp, map_mcore_hf_param_names, ShardLoader


def init_megatron(args, seed: int = 1234):
    init_distributed(backend=args.backend, world_size=args.world_size, rank=args.rank)

    args = patch_mcore_args(args)
    args = validate_args(args)

    set_global_variables(args, build_tokenizer=False)

    init_mpu(
        tp=args.tensor_model_parallel_size,
        pp=args.pipeline_model_parallel_size,
        vpp=args.virtual_pipeline_model_parallel_size,
        cp=args.context_parallel_size,
        ep=args.expert_model_parallel_size,
        etp=args.expert_tensor_parallel_size,
    )

    if not (args.use_cpu_initialization or args.init_model_with_meta_device or args.finetune):
        model_parallel_cuda_manual_seed(seed)


def create_mcore_model(config: TransformerConfig, wrap_with_ddp: bool = False) -> McoreModelT:
    model_provider_func = get_model_provider_func(config)
    mcore_model_parts: McoreModelT = get_model(model_provider_func, wrap_with_ddp=wrap_with_ddp)

    param_devices = sum((get_model_param_devices(m) for m in mcore_model_parts), Counter())
    mcore_num_params = sum(len(list(m.parameters())) for m in mcore_model_parts)

    if args.init_model_with_meta_device and not param_devices["meta"] == mcore_num_params:
        print(f"WARNING: not all params on 'meta': {param_devices}")

    return mcore_model_parts


def create_reference_model(config: Qwen3ConfigT, args: Namespace) -> Qwen3ModelT:
    model_cls = Qwen3MoeForCausalLM if is_qwen3_moe_config(config) else Qwen3ForCausalLM
    if args.init_model_with_meta_device:
        device = "meta"
    elif args.use_cpu_initialization:
        device = "cpu"
    else:
        device = "cuda"

    with torch.device(device):
        ref_model: Qwen3ModelT = model_cls(config)

    return ref_model


def update_args_for_model_loading(args: Namespace):
    init_on_meta = args.init_model_with_meta_device
    use_cpu_initialization = args.use_cpu_initialization

    assert not (init_on_meta and use_cpu_initialization), (
        f"{init_on_meta=} and {use_cpu_initialization=} both set"
    )

    # Should disable to prevent initialization on GPU and need for tensor parallel CUDA RNG
    args.perform_initialization = not any(
        [use_cpu_initialization, args.init_model_with_meta_device, args.finetune]
    )

    # This is actually not needed, since `validate_args` will automatically configure vpp_size

    # VPP configuration, see megatron-lm/megatron/training/arguments.py
    # VPP_STAGES = NUM_LAYERS // (PP * VPP)
    # Each rank gets VPP_STAGES layers in round robin fashion
    # E.g., Qwen 0.6B has 28 layers
    # For PP=2, VPP=2, 28 // 2 = 14 stages per rank, interleaved such that Rank 0: [1-7],[15-21] | Rank1: [8-14],[22-28]
    # Without VPP, Rank 0: [1-14], Rank 1: [15-28]
    # Note that layer_id's start at 1 in decoder layers

    assert args.num_layers_per_virtual_pipeline_stage is None, (
        "Use --num_virtual_stages_per_pipeline_rank to set VPP size"
    )
    args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
    return args


def check_param_counts(hf_model: Qwen3ModelT, mcore_model_parts: McoreModelT):
    """
    Quick sanity check only for single device case
    TODO: add distributed checks
    """

    hf_param_count = get_total_params(hf_model)
    mcore_param_count = sum(get_total_params(m) for m in mcore_model_parts)

    # TODO: better checks for various parallelisms
    # Param accounting gets complicated with tied weights, pipeline parallel
    if torch.distributed.get_world_size() == 1:
        assert hf_param_count == mcore_param_count, (
            f"Param count mismatch: {hf_param_count} != {mcore_param_count}"
        )


def create_local_to_global_map(mcore_model_parts: McoreModelT) -> list[dict[str, str]]:
    local_to_global_maps = [remap_param_names_for_ep_pp(m) for m in mcore_model_parts]

    return local_to_global_maps


def create_mcore_hf_mapping(
    local_to_global_maps: list[dict[str, str]], is_moe: bool
) -> dict[str, str]:
    mcore_to_hf_maps = [map_mcore_hf_param_names(m, is_moe=is_moe) for m in local_to_global_maps]

    return mcore_to_hf_maps


def get_model_cache(model_path: str) -> str:
    is_local_dir = not model_path in QWEN3_MODELS
    if not is_local_dir:
        from huggingface_hub import snapshot_download

        model_cache_dir = snapshot_download(model_path)
    else:
        model_cache_dir = model_path

    return model_cache_dir


def load_mcore_model_weights(
    model_path: str,
    mcore_model_parts: McoreModelT,
    mcore_to_hf_maps: list[dict[str, str]],
    device: str = "cuda",
):
    model_cache_dir = get_model_cache(model_path)

    loader = ShardLoader(model_cache_dir)
    assert len(mcore_model_parts) == len(mcore_to_hf_maps)

    loader.load_hf_weights(mcore_model_parts, mcore_to_hf_maps, device=device)

    return mcore_model_parts


def convert_hf_to_mcore(
    qwen_config: Qwen3MCoreConfig, model_path: str, device: str = "cuda", check_params: bool = True
) -> tuple[TransformerConfig, McoreModelT]:
    mcore_config: TransformerConfig = qwen_config.to_mcore()
    hf_config: Qwen3ConfigT = qwen_config.hf_config

    mcore_model_parts = create_mcore_model(mcore_config)
    if check_params:
        hf_model = create_reference_model(hf_config, args)

        check_param_counts(hf_model, mcore_model_parts)

    is_moe = is_qwen3_moe_config(hf_config)

    # Map local (sharded) param names to global full model param names
    local_to_global_maps = create_local_to_global_map(mcore_model_parts)

    # Create param name mapping: mcore < -- > hf
    mcore_to_hf_maps = create_mcore_hf_mapping(local_to_global_maps, is_moe=is_moe)

    # Load per rank model shards
    mcore_model_parts = load_mcore_model_weights(
        model_path=model_path,
        mcore_model_parts=mcore_model_parts,
        mcore_to_hf_maps=mcore_to_hf_maps,
        device=device,
    )

    return mcore_config, mcore_model_parts


def save_local_checkpoint(mcore_model_parts: McoreModelT, iteration: int = 1, flops_count: int = 0):
    from megatron.training.checkpointing import save_checkpoint
    from megatron.core import mpu

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    ep_rank = mpu.get_expert_model_parallel_rank()
    pipeline_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
    expert_parallel = mpu.get_expert_model_parallel_world_size() > 1
    save_checkpoint(
        iteration,
        mcore_model_parts,
        None,  # optimizer
        None,  # scheduler
        num_floating_point_operations_so_far=flops_count,
        pipeline_rank=pp_rank,
        pipeline_parallel=pipeline_parallel,
        expert_rank=ep_rank,
        expert_parallel=expert_parallel,
        tensor_rank=tp_rank,
    )


def reinitialize_rope(mcore_model: McoreModelT, rotary_base: float, device: str = "cuda"):
    for model in mcore_model:
        head_dim = model.config.kv_channels
        if hasattr(model, "rotary_pos_emb") and model.rotary_pos_emb is not None:
            rotary_emb = model.rotary_pos_emb
            if rotary_emb.inv_freq.device.type == "meta":
                rotary_emb.inv_freq = 1 / (
                    rotary_base
                    ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
                )


def forward_step_func(data_iterator, model, device: str):
    def loss_func(output_tensor: torch.Tensor, **kwargs):
        logits = output_tensor.float()

        return {"logits": logits}

    input_ids, position_ids, attention_mask = next(data_iterator)
    output_tensor = model(
        input_ids.to(device),
        position_ids.to(device),
        attention_mask.to(device)
    )

    return output_tensor, loss_func


def check_logits(
    mcore_model_parts: McoreModelT,
    model_path: str,
    num_samples: int = 100,
    prompt_len: int = 100,
    topk: int = 3,
    seed: int = 1234,
):
    from mcore_utils import generate_dataset

    # assert len(mcore_model_parts) == 1, (
    #     f"Logits check not supported for pipeline parallel currently"
    # )

    args = get_args()
    gpt_model = mcore_model_parts[0].cuda()
    device = next(gpt_model.parameters()).device.type
    model_parallel_cuda_manual_seed(seed)

    if args.init_model_with_meta_device:
        reinitialize_rope(mcore_model_parts, rotary_base=args.rotary_base, device=device)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=torch.cuda.current_device()
    )

    torch.manual_seed(seed)
    dataset = generate_dataset(
        vocab_size=args.vocab_size, num_samples=num_samples, seqlen=prompt_len, batch_size=1
    )
    input_ids, position_ids, attention_mask = next(iter(dataset))
    input_ids, position_ids, attention_mask = (
        input_ids.to(device),
        position_ids.to(device),
        attention_mask.to(device),
    )
    # input_ids = torch.randint(0, args.vocab_size, (1, prompt_len), device=device)
    # position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    # attention_mask = torch.ones_like(input_ids).to(input_ids.device)
    rank = dist.get_rank()

    mp_group = mpu.get_model_parallel_group()

    is_pp = mpu.get_pipeline_model_parallel_world_size() > 1
    pp_group = mpu.get_pipeline_model_parallel_group()
    pp_group_rank = dist.get_group_rank(pp_group, rank)
    is_last_stage = mpu.is_pipeline_last_stage()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_group_rank = dist.get_group_rank(tp_group, rank)
    ep_size = mpu.get_expert_model_parallel_world_size()
    etp_size = mpu.get_expert_tensor_parallel_world_size()

    pp_group_ranks = dist.get_process_group_ranks(pp_group)
    tp_group_ranks = dist.get_process_group_ranks(tp_group)
    mp_group_ranks = dist.get_process_group_ranks(mp_group)

    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from functools import partial
    forward_backward_func = get_forward_backward_func()
    dist_print(f"{type(forward_backward_func)=}", rank0_only=True)

    with torch.no_grad():
        outputs = forward_backward_func(
            forward_step_func=partial(forward_step_func, device=device),
            data_iterator=iter(dataset),
            model=mcore_model_parts,
            num_microbatches=1,
            seq_length=prompt_len,
            micro_batch_size=1,
            decoder_seq_length=prompt_len,
            forward_only=True,
            collect_non_loss_data=True,
        )
    
        ref_output = hf_model.forward(input_ids)
    
    if is_last_stage:
        logits = outputs[0]['logits'][0]
    
    ref_logits: torch.Tensor = ref_output.logits[0].float()

    # output = gpt_model(input_ids, position_ids, attention_mask)
#    logits: torch.Tensor = output[0].float()
 
    if tp_size > 1:
        full_logits = torch.zeros(
            *ref_logits.T.shape, device=ref_logits.device, dtype=ref_logits.dtype
        )
        dist.all_gather_into_tensor(full_logits, logits.T.contiguous(), group=tp_group)
        logits = full_logits.T

    if is_last_stage and tp_group_rank == 0:
        diff = (logits - ref_logits).abs().max()
        dist_print(f"logits diff: {diff.item():.4f}")#, rank0_only=True)

        _, topk_ids = logits.topk(topk, dim=-1)
        _, ref_topk_ids = ref_logits.topk(topk, dim=-1)

        num_tokens = logits.shape[0]
        for i, (test, ref) in enumerate(zip(topk_ids, ref_topk_ids)):
            test = test.tolist()
            ref = ref.tolist()
            if set(test) != set(ref):
                dist_print(
                    f"Topk @ {topk} ids mismatch at token position {i + 1} / {num_tokens}: {test} != {ref}",
                    #rank0_only=True,
                )


def main(args: Namespace):
    args = update_args_for_model_loading(args)
    model_path = args.model_id

    hf_config = AutoConfig.from_pretrained(model_path)
    sequence_parallel = args.sequence_parallel or args.tensor_model_parallel_size > 1

    if is_qwen3_moe_config(hf_config) and args.tensor_model_parallel_size > 1:
        sequence_parallel = True

    parallel_config = ParallelismConfig(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    qwen_config = Qwen3MCoreConfig.from_hf(
        hf_config,
        parallelism_config=parallel_config,
        perform_initialization=args.perform_initialization,
        use_cpu_initialization=args.use_cpu_initialization,
        init_model_with_meta_device=args.init_model_with_meta_device,
    )

    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        pprint(qwen_config)

    args = qwen_config.update_mcore_args(args)

    init_megatron(args)

    mcore_config, mcore_model_parts = convert_hf_to_mcore(qwen_config, model_path)

    check_logits(mcore_model_parts, model_path=model_path)
    # save_local_checkpoint(mcore_model_parts)


if __name__ == "__main__":
    parser = ArgumentParser(
        "Convert HF Qwen3 Model to MCore Format", formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HF model path, e.g., Qwen/Qwen3-30B-A3B",
    )
    parser.add_argument("--backend", default="fake", choices=["fake", "gloo", "nccl"])
    parser.add_argument("--rank", default=None, type=int)
    parser.add_argument("--world_size", default=None, type=int)
    parser.add_argument(
        "--no-finetune",
        action="store_false",
        dest="finetune",
        help="Enable finetune by default in order to disable initialization of weights and structs needed only for pretraining",
    )
    add_megatron_arguments(parser)
    args = parser.parse_args()
    assert args.finetune
    main(args)
