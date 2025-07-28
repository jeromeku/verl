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
from megatron.training.global_vars import set_global_variables
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.training.utils import unwrap_model

from transformers import AutoConfig
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

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
)
from qwen3_configuration import (
    QWEN3_MOE_MODELS,
    Qwen3MCoreConfig,
    Qwen3ConfigT,
    ParallelismConfig,
    Qwen3ModelT,
    is_qwen3_moe_config,
    QWEN3_MODELS
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


def create_mcore_model(config: TransformerConfig, wrap_with_ddp: bool = False):
    model_provider_func = get_model_provider_func(config)
    mcore_model_parts: list[GPTModel] = get_model(model_provider_func, wrap_with_ddp=wrap_with_ddp)

    param_devices = sum((get_model_param_devices(m) for m in mcore_model_parts), Counter())
    mcore_num_params = sum(len(list(m.parameters())) for m in mcore_model_parts)

    if args.init_model_with_meta_device and not param_devices["meta"] == mcore_num_params:
        print(f"WARNING: not all params on 'meta': {param_devices}")

    return mcore_model_parts


def create_reference_model(config: Qwen3ConfigT, args: Namespace):
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


def check_param_counts(hf_model: Qwen3ModelT, mcore_model_parts: list[GPTModel]):
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

def create_local_to_global_map(mcore_model_parts: list[GPTModel]):
    name_maps = []

    for m in mcore_model_parts:
        test = remap_param_names_for_ep_pp(m)
        ref = _weight_name_mapping_mcore_local_to_global(m)
        if test != ref:
            key_diff = set(ref.keys()) - set(ref.keys())
            val_diff = set(ref.values()) - set(test.values())
            print(f"{key_diff=}")
            print(f"{val_diff=}")
            assert False

        name_maps.append(test)

    return name_maps

def create_mcore_hf_mapping(mcore_model_parts: list[GPTModel], local_to_global_map: dict[str, str], is_moe: bool):
    from ref.convert_utils import _local_to_hf
#    gpt_model: GPTModel = unwrap_model(mcore_model_parts)
    # layer: TransformerLayer = gpt_model.decoder.layers[0]
    # mlp = layer.mlp

    # is_moe = isinstance(hf_config, Qwen3MoeConfig)

    mcore_to_hf_maps = []
    for m in local_to_global_map:
        ref = _local_to_hf(m, is_moe=is_moe)
        test = map_mcore_hf_param_names(m, is_moe=is_moe)

        assert ref == test
        mcore_to_hf_maps.append(test)

    return mcore_to_hf_maps

def get_model_cache(model_path: str):
    is_local_dir = not model_path in QWEN3_MODELS
    if not is_local_dir:
        from huggingface_hub import snapshot_download
        model_cache_dir = snapshot_download(model_path)
    else:
        model_cache_dir = model_path

    return model_cache_dir

def load_mcore_model_weights(model_path: str, mcore_model_parts: list[GPTModel], mcore_to_hf_maps: list[dict[str, str]], device: str = "cuda"):
    model_cache_dir = get_model_cache(model_path)

    loader = ShardLoader(model_cache_dir)
    assert len(mcore_model_parts) == len(mcore_to_hf_maps)

    loader.load_hf_weights(mcore_model_parts, mcore_to_hf_maps, device=device)
    from ref.convert_utils import check_weights

    check_weights(model_path, mcore_model_parts=mcore_model_parts)

    return mcore_model_parts

def main(args: Namespace):
    args = update_args_for_model_loading(args)
    model_path = args.model_id

    sequence_parallel = args.sequence_parallel or args.tensor_model_parallel_size > 1

    parallel_config = ParallelismConfig(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        sequence_parallel=sequence_parallel
    )

    hf_config = AutoConfig.from_pretrained(model_path)

    qwen_config = Qwen3MCoreConfig.from_hf(
        hf_config,
        parallelism_config=parallel_config,
        perform_initialization=args.perform_initialization,
        use_cpu_initialization=args.use_cpu_initialization,
    )

    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        pprint(qwen_config)

    args = qwen_config.update_mcore_args(args)
    d = qwen_config.to_dict(transformer_config_only=True)

    mcore_config: TransformerConfig = qwen_config.to_mcore()
    pprint(mcore_config)

    init_megatron(args)

    mcore_model_parts = create_mcore_model(mcore_config)
    hf_model = create_reference_model(hf_config, args)

    check_param_counts(hf_model, mcore_model_parts)
    is_moe = is_qwen3_moe_config(hf_config)
    local_to_global_maps = create_local_to_global_map(mcore_model_parts)
    mcore_to_hf_maps = create_mcore_hf_mapping(mcore_model_parts, local_to_global_maps, is_moe=is_moe)

    load_mcore_model_weights(model_path=model_path, mcore_model_parts=mcore_model_parts, mcore_to_hf_maps=mcore_to_hf_maps)
    # from mbridge.core.safetensor_io import SafeTensorIO

    # is_local_dir = not model_path in QWEN3_MODELS
    # if not is_local_dir:
    #     from huggingface_hub import snapshot_download
    #     model_cache_dir = snapshot_download(model_path)
    # else:
    #     model_cache_dir = model_path

    # def load_mbridge_ref():
    #     from mbridge import AutoBridge

    #     bridge = AutoBridge.from_pretrained(model_path)
    #     bridge.config.perform_initialization = False
    #     bridge.config.use_cpu_initialization = args.use_cpu_initialization
    #     ref_models = bridge.get_model(use_cpu_initialization=args.use_cpu_initialization)
    #     bridge.load_weights(ref_models, model_path)
    
    #     return ref_models
    
    # from ref.convert_utils import load_hf_weights
    # from ref.data_utils import ShardLoader
    
    # loader = ShardLoader(model_cache_dir)

    # assert len(mcore_model_parts) == len(mcore_to_hf_maps)

    # ref_models = load_mbridge_ref()
    # device_type = next(ref_models[0].parameters()).device.type

    # for model, map in zip(mcore_model_parts, mcore_to_hf_maps):
    #     load_hf_weights(loader, hf_config=hf_config, model=model, local_to_hf_map=map, device=device_type)


    # for ref_m, test_m in zip(ref_models, mcore_model_parts):
    #     ref_devices = get_model_param_devices(ref_m)
    #     test_devices = get_model_param_devices(test_m)
    #     ref_sd = ref_m.state_dict()
    #     test_sd = test_m.state_dict()

    #     if ref_sd.keys() != test_sd.keys():
    #         breakpoint()

    #     for k in ref_sd.keys():
    #         if "_extra_state" in k:
    #             continue

    #         expected = ref_sd[k]
    #         actual = test_sd[k].to(expected.device)

    #         assert expected is not None
    #         assert expected.nonzero().sum() > 0
    #         assert actual is not None
    #         assert actual.nonzero().sum() > 0

    #         if not expected.equal(actual):
    #             breakpoint()

    # dist_print("State dicts match!")

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
