# ruff: noqa E402
import sys
from pathlib import Path

import megatron.core as mc

MEGATRON_ROOT = Path(mc.__file__).parents[2]
sys.path.append(MEGATRON_ROOT.resolve().as_posix())

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from pprint import pp

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.training.arguments import add_megatron_arguments, validate_args
from megatron.training.global_vars import set_global_variables
from megatron.training.utils import unwrap_model
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3 import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.utils.hub import cached_file


from convert_utils import (
    update_args,
    init_distributed,
    init_mpu,
    get_model,
    get_model_provider_func,
    remap_param_names_for_ep_pp,
    _weight_name_mapping_mcore_local_to_global,
    dist_print
)
from qwen3_configuration import (
    get_activation_recompute_config,
    get_arch_config,
    get_attn_config,
    get_fusion_config,
    get_mlp_config,
    get_parallelism,
    get_precision_config,
    QWEN3_DENSE_MODELS,
    QWEN3_MOE_MODELS,
    QWEN3_600M,
)

from contextlib import contextmanager
from unittest.mock import patch

import torch


def weights_generator(
    model_id: str = None, weight_files: list[str | Path] = None, device: str = "cpu"
):
    assert model_id ^ weight_files

    if model_id:
        model_cache_dir = Path(snapshot_download(model_id))
        weight_files = list(model_cache_dir.glob("*.safetensors"))

    for wf in weight_files:
        with safe_open(wf, "pt", device=device) as f:
            for k in f.keys():
                yield k, f.get_tensor(k)


def hf_to_mcore(hf_config: Qwen3MoeConfig, is_moe: bool = False, **kwargs) -> TransformerConfig:
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
    return {n: get_total_params(m) for n, m in model.named_children()}

def check_params(ref_model, test_model, check_tp: bool = False):
    import torch.distributed as dist
    import torch.distributed.distributed_c10d as c10d
    import torch.distributed.collective_utils as collectives
    import torch.distributed._functional_collectives as funcol
    from convert_utils import dist_print, _extract_layer_number
    from megatron.core.tensor_parallel.layers import _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS

    hf_param_count = get_total_params(ref_model)
    mcore_param_count = sum(get_total_params(m) for m in test_model)

    dist_print(f"mcore param count: {mcore_param_count}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    wg = c10d._get_default_group()
    mcore_param_counts = [None] * world_size

    dist.all_gather_object(mcore_param_counts, mcore_param_count)
    total_params = sum(mcore_param_counts)

    dist_print(mcore_param_counts, rank0_only=True)
    dist_print(f"{hf_param_count=} {total_params=}", rank0_only=True)

    
    if check_tp:
        etp_size = mpu.get_expert_tensor_parallel_world_size() 
        tp_size =  mpu.get_tensor_model_parallel_world_size()

        if rank == 0:
            print(f"{etp_size=} {tp_size=}")
            param_count = 0
            causal_lm_params = {}
            attn_params = {}
            mlp_params = {}
            other_params = {}
            for m in test_model:
                m: GPTModel = unwrap_model(m)
                for name, param in m.named_parameters():
                    param_attr = {
                        "tp": param.tensor_model_parallel,
                        "numel": param.numel(),
                        "shape": param.shape,
                    }
                    if "decoder" not in name:
                        causal_lm_params[name] = param_attr
                    elif "attention" in name:
                        attn_params[name] = param_attr
                    elif "mlp." in name:
                        mlp_params[name] = param_attr
                    else:
                        other_params[name] = param_attr
                
                    if param.tensor_model_parallel:
                        group_size = etp_size if "mlp" in name else tp_size
                        param_count += param.numel() * group_size
                    else:
                        param_count += param.numel()
                
                hf_params = {
                    "other": {n: {"numel": p.numel(), "shape": p.shape} for n,p in ref_model.named_parameters() if ("attn" not in n and "mlp" not in n)},
                    "attn": {n: {"numel": p.numel(), "shape": p.shape} for n, p in ref_model.named_parameters() if "attn" in n},
                    "mlp": {n: {"numel": p.numel(), "shape": p.shape} for n, p in ref_model.named_parameters() if "mlp" in n},
                }

def get_param_shape_and_numel(param: torch.nn.Parameter):
    return { "shape": param.shape, "numel": param.numel() }

def _get_model_param_by_pattern(model_parts: list[torch.nn.Module], pat: str):
    param_attr = {}
    for m in model_parts:
        for name, param in m.named_parameters():
            if pat in name:
                param_attr[name] = get_param_shape_and_numel(param)

    return param_attr

def get_attn_params(model_parts: list[torch.nn.Module]):
    return _get_model_param_by_pattern(model_parts, "attn")

def get_mlp_params(model_parts: list[torch.nn.Module]):
    experts, router, other = {}, {}, {}
    base = _get_model_param_by_pattern(model_parts, "mlp")
    for k, v in base.items():
        if "experts" in k:
            experts[k] = v
        elif "router" in k:
            router[k] = v
        else:
            other[k] = v
    return v

def main(args):
    model_path = args.model_id
    is_moe = model_path in QWEN3_MOE_MODELS or "moe" in model_path.lower()
    model_cls = Qwen3MoeForCausalLM if is_moe else Qwen3ForCausalLM
    hf_config = AutoConfig.from_pretrained(model_path)

    init_distributed(backend=args.backend, world_size=args.world_size, rank=args.rank)
    args = update_args(args, hf_config, use_transformer_engine=True)
    args = validate_args(args)
    set_global_variables(args, build_tokenizer=False)

    # args = set_vpp_size(hf_config, args)

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

    transformer_config = hf_to_mcore(
        hf_config,
        is_moe=is_moe,
        perform_initialization=args.perform_initialization,
        use_cpu_initialization=args.use_cpu_initialization,
    )

    # pp(hf_config.to_dict())
    # pp(asdict(transformer_config))

    
    model_provider_func = get_model_provider_func(transformer_config, args)
    model_parts: list[GPTModel] = get_model(model_provider_func, init_on_meta=init_on_meta)

    # print(model_parts[0])

    param_devices = sum((get_model_param_devices(m) for m in model_parts), Counter())
    mcore_num_params = sum(len(list(m.parameters())) for m in model_parts)

    if init_on_meta and not param_devices["meta"] == mcore_num_params:
        print(f"WARNING: not all params on 'meta': {param_devices}")

    with torch.device("meta"):
        ref_model: Qwen3ForCausalLM = model_cls(hf_config)

    hf_param_count = get_total_params(ref_model)
    mcore_param_count = sum(get_total_params(m) for m in model_parts)
    check_params(ref_model, model_parts, check_tp=False)
    
    # TODO: better checks for various parallelisms
    # Param accounting gets complicated with tied weights, pipeline parallel
    if torch.distributed.get_world_size() == 1:
        assert hf_param_count == mcore_param_count, (
            f"Param count mismatch: {hf_param_count} != {mcore_param_count}"
        )

    name_maps = []
    for m in model_parts:
        test = remap_param_names_for_ep_pp(m)
        ref = _weight_name_mapping_mcore_local_to_global(m)
        if test != ref:
            key_diff = set(ref.keys()) - set(ref.keys())
            val_diff = set(ref.values()) - set(test.values())
            print(f"{key_diff=}")
            print(f"{val_diff=}")
            assert False

        name_maps.append(test)

    # for map in name_maps:
    #     pp(map)

    from convert_utils import map_mcore_hf_param_names, _local_to_hf
    from megatron.core.transformer import TransformerLayer

    gpt_model: GPTModel = unwrap_model(model_parts[0])
    layer: TransformerLayer = gpt_model.decoder.layers[0]
    mlp = layer.mlp

    is_moe = isinstance(hf_config, Qwen3MoeConfig)

    local_to_hf_maps = []
    for m in name_maps:
        ref = _local_to_hf(m, is_moe=is_moe)
        test = map_mcore_hf_param_names(m, is_moe=is_moe)

        assert ref == test
        local_to_hf_maps.append(test)

    from mbridge.core.safetensor_io import SafeTensorIO

    is_local_dir = not model_path in [*QWEN3_DENSE_MODELS, *QWEN3_MOE_MODELS]
    if not is_local_dir:
        from huggingface_hub import snapshot_download

        model_cache_dir = snapshot_download(model_path)
    else:
        model_cache_dir = model_path


    def load_mbridge_ref():
        from mbridge import AutoBridge

        bridge = AutoBridge.from_pretrained(model_path)
        ref_models = bridge.get_model(use_cpu_initialization=True)
        bridge.load_weights(ref_models, model_path)
    
        return ref_models
    
    from convert_utils import load_hf_weights, dist_print #_load_hf_weights
    from data_utils import ShardLoader

    loader = ShardLoader(model_cache_dir)
    #safetensor_io = SafeTensorIO(model_cache_dir)
#    use_TE = args.transformer_impl == "transformer_engine"

    assert len(model_parts) == len(local_to_hf_maps)

    ref_models = load_mbridge_ref()
    device_type = next(ref_models[0].parameters()).device.type

    for model, map in zip(model_parts, local_to_hf_maps):
        load_hf_weights(loader, hf_config=hf_config, model=model, local_to_hf_map=map, device=device_type)


    for ref_m, test_m in zip(ref_models, model_parts):
        ref_devices = get_model_param_devices(ref_m)
        test_devices = get_model_param_devices(test_m)
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
    return

    if args.use_cpu_initialization:
        from megatron.training.checkpointing import save_checkpoint, load_checkpoint

        save_checkpoint(1, model_parts, None, None, 0)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        args.load = args.save

        load_checkpoint(model_parts, None, None, strict=True)

    # print([type(m) for m in model_parts])
    # for idx, m in enumerate(model_parts):
    #     print(f"Model part {idx}")
    #     pp(m.state_dict().keys())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "HF -> Megatron Config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model-id",
        help="HF model path, e.g., Qwen/Qwen3-30B-A3B",
        default=QWEN3_600M,
        #  choices=[*QWEN3_DENSE_MODELS, *QWEN3_MOE_MODELS],
    )
    parser.add_argument("--backend", default="fake", choices=["fake", "gloo", "nccl"])
    parser.add_argument("--rank", default=None, type=int)
    parser.add_argument("--world_size", default=None, type=int)

    add_megatron_arguments(parser)
    args = parser.parse_args()
    main(args)

    if False:
        print(f"HF Model total params: {hf_param_count}")
        print(f"MCore total params: {mcore_param_count}")
        print()

        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3Model

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
