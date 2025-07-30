# ruff: noqa E402
import gc
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import tee
from pathlib import Path
from pprint import pprint
from typing import Iterator

import megatron.core as mc
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
from transformers.utils.logging import disable_progress_bar

# Need include megatron root to resolve modules outside of megatron.core
MEGATRON_ROOT = Path(mc.__file__).parents[2]
sys.path.append(MEGATRON_ROOT.resolve().as_posix())

from megatron.core import mpu
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.training.arguments import add_megatron_arguments, validate_args
from megatron.training.global_vars import get_args, set_global_variables
from megatron.training.utils import unwrap_model
from qwen3_configuration import (
    QWEN3_MODELS,
    QWEN3_MOE_MODELS,
    MoeOptConfig,
    ParallelismConfig,
    Qwen3ConfigT,
    Qwen3MCoreConfig,
    Qwen3ModelT,
    is_qwen3_moe_config,
)
from debugging import get_model_param_devices, get_module_param_count, get_total_params
from mcore_utils import (
    McoreModelT,
    dist_print,
    get_model,
    get_model_provider_func,
    init_distributed,
    init_mpu,
    patch_mcore_args,
)
from weight_conversion import ShardLoader, map_mcore_hf_param_names, remap_param_names_for_ep_pp
from testing import ModelCheckResult, postprocess_logits, PromptDataset, get_test_prompts

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
    from megatron.core import mpu
    from megatron.training.checkpointing import save_checkpoint

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

    inputs = next(data_iterator)
    input_ids = inputs["input_ids"].to(device)
    position_ids = inputs["position_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output_tensor = model(input_ids, position_ids, attention_mask)

    return output_tensor, loss_func


@torch.no_grad
def run_hf(model_path: str, data_iter: Iterator, topk: int):
    device = torch.cuda.current_device()
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)

    results = []
    for d in data_iter:
        input_ids = d["input_ids"].to(device)
        output = hf_model.forward(input_ids)
        logits = output.logits[0].float()
        topk_scores, topk_ids = logits.topk(topk, dim=-1)
        results.append(
            ModelCheckResult(
                logits=logits.cpu().numpy(),
                topk_ids=topk_ids.tolist(),
                topk_scores=topk_scores.tolist(),
                input_ids=input_ids[0].tolist(),
                prompt=d["prompt"],
            )
        )

    del hf_model
    torch.cuda.empty_cache()
    gc.collect()

    return results


@torch.no_grad
def run_mc(mcore_model_parts: McoreModelT, data_iter: Iterator, topk: int):
    rank = dist.get_rank()

    pp_group = mpu.get_pipeline_model_parallel_group()
    is_last_stage = mpu.is_pipeline_last_stage()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()

    forward_backward_func = get_forward_backward_func()
    results = []
    device = torch.cuda.current_device()

    # Pass single batch, single sample at a time to avoid having to deal with variable seqlen
    for d in data_iter:
        input_ids = d["input_ids"]
        prompt_len = len(input_ids[0])

        if rank == 0:
            print(f"Processing prompt {input_ids=}, {prompt_len=}")

        outputs = forward_backward_func(
            forward_step_func=partial(forward_step_func, device=device),
            data_iterator=iter([d]),
            model=mcore_model_parts,
            num_microbatches=1,
            seq_length=prompt_len,
            micro_batch_size=1,
            decoder_seq_length=prompt_len,
            forward_only=True,
            collect_non_loss_data=True,
        )

        if is_last_stage:
            logits = outputs[0]["logits"][0].float()

        # When using pp, only last pp stage produces logits
        # When using tp, outputs are parallel, split across vocab dim (1)
        if tp_size > 1 and is_last_stage:
            # all_gather only supports gather_dim=0 => transpose -> gather -> transpose
            output_shape = (logits.shape[1] * tp_size, logits.shape[0])
            full_logits = torch.zeros(*output_shape, device=logits.device, dtype=logits.dtype)
            dist.all_gather_into_tensor(full_logits, logits.T.contiguous(), group=tp_group)
            logits = full_logits.T

        if is_last_stage:
            topk_scores, topk_ids = logits.topk(topk, dim=-1)
            results.append(
                ModelCheckResult(
                    logits=logits.cpu().numpy(),
                    input_ids=input_ids[0].tolist(),
                    topk_scores=topk_scores.tolist(),
                    topk_ids=topk_ids.tolist(),
                )
            )

    return results


def process_results(
    hf_results: list[ModelCheckResult],
    mcore_results: list[ModelCheckResult],
    model_path: str,
    save_dir: Path,
):
    logits_df, token_topk_df = postprocess_logits(hf_results, mcore_results)

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    etp_size = mpu.get_expert_tensor_parallel_world_size()

    base_name = model_path.split("/")[-1] + "__" + f"pp{pp_size}tp{tp_size}ep{ep_size}etp{etp_size}"

    def save_df(df: pd.DataFrame, label: str):
        file_stem = base_name + "__" + label
        save_path = (save_dir / file_stem).resolve().as_posix() + ".csv"
        df.to_csv(save_path)
        print(f"{label} df saved to {save_path}")

    save_df(logits_df, "logits")
    save_df(token_topk_df, "token_topk")

    # Format for printing
    pd.set_option("display.float_format", "{:.4f}".format)  # Set float precision
    pd.set_option("display.max_columns", None)  # Display all columns

    # Logits stats
    print(logits_df)

    # Topk stats
    token_topk_df["hf_topk_scores"] = token_topk_df["hf_topk_scores"].apply(
        lambda scores: [f"{x:.4f}" for x in scores]
    )
    token_topk_df["mcore_topk_scores"] = token_topk_df["mcore_topk_scores"].apply(
        lambda scores: [f"{x:.4f}" for x in scores]
    )
    print(token_topk_df)


def check_logits(
    mcore_model_parts: McoreModelT,
    model_path: str,
    topk: int = 3,
    seed: int = 1234,
):
    args = get_args()
    gpt_model = mcore_model_parts[0].cuda()
    device = next(gpt_model.parameters()).device.type
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    if args.init_model_with_meta_device:
        reinitialize_rope(mcore_model_parts, rotary_base=args.rotary_base, device=device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = get_test_prompts()
    dataset = PromptDataset(prompts, tokenizer, device="cuda")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rank = dist.get_rank()

    # PP comms, TODO: add VPP
    is_last_stage = mpu.is_pipeline_last_stage()

    # TP comms
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_group_rank = dist.get_group_rank(tp_group, rank)

    # Expert comms
    ep_group = mpu.get_expert_model_parallel_group()
    ep_group_rank = dist.get_group_rank(ep_group, rank)
    etp_group = mpu.get_expert_tensor_parallel_group()
    etp_group_rank = dist.get_group_rank(etp_group, rank)

    data_iter = iter(data_loader)
    hf_data, mcore_data = tee(data_iter, 2)

    hf_results = run_hf(model_path, hf_data, topk)
    mcore_results = run_mc(mcore_model_parts, mcore_data, topk)

    # Only last pp stage has valid results, other conditions are to remove duplicate printing
    should_post_process = (
        is_last_stage and tp_group_rank == 0 and ep_group_rank == 0 and etp_group_rank == 0
    )

    if should_post_process:
        process_results(
            hf_results, mcore_results, model_path=model_path, save_dir=args.logits_save_path
        )

    dist.barrier()


def main(args: Namespace):
    args = update_args_for_model_loading(args)
    model_path = args.model_id
    disable_progress_bar()

    hf_config = AutoConfig.from_pretrained(model_path)
    sequence_parallel = args.sequence_parallel or args.tensor_model_parallel_size > 1

    is_moe = is_qwen3_moe_config(hf_config)

    if is_moe and args.tensor_model_parallel_size > 1:
        sequence_parallel = True

    parallel_config = ParallelismConfig.from_args(args, sequence_parallel=sequence_parallel)

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

    check_logits(mcore_model_parts, model_path=model_path, topk=args.topk)

    if args.save_checkpoint:
        save_local_checkpoint(mcore_model_parts)


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
    parser.add_argument("--check-logits", action="store_true")
    parser.add_argument(
        "--topk", type=int, default=10, help="topk logits / token ids when comparing model outputs"
    )
    parser.add_argument(
        "--logits-save-path",
        type=Path,
        default="logits_results",
        help="If checking logits, where to save results",
    )
    parser.add_argument(
        "--save-checkpoint", action="store_true", help="Store converted Megatron checkpoint"
    )
    add_megatron_arguments(parser)
    args = parser.parse_args()

    if args.check_logits:
        os.makedirs(args.logits_save_path, exist_ok=True)

    main(args)

    dist.barrier()
    dist.destroy_process_group()
