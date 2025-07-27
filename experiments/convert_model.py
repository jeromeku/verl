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

from transformers import AutoConfig
from transformers.models.qwen3 import Qwen3ForCausalLM
from transformers.models.qwen3_moe import Qwen3MoeForCausalLM

import torch.distributed as dist
import torch

from mcore_utils import (
    init_distributed,
    init_mpu,
    patch_mcore_args,
    dist_print,
    get_model_provider_func,
    get_model,
)
from qwen3_configuration import QWEN3_MOE_MODELS, Qwen3MCoreConfig, Qwen3ConfigT, ParallelismConfig


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


def create_mcore_model(transformer_config: TransformerConfig, wrap_with_ddp: bool = False):
    model_provider_func = get_model_provider_func(transformer_config)
    model_parts: list[GPTModel] = get_model(model_provider_func, wrap_with_ddp=wrap_with_ddp)
    return model_parts


def main(args: Namespace):
    model_path = args.model_id
    is_moe = model_path in QWEN3_MOE_MODELS or "moe" in model_path.lower()

    model_cls = Qwen3MoeForCausalLM if is_moe else Qwen3ForCausalLM
    assert args.num_layers_per_virtual_pipeline_stage is None, (
        "Use --num_virtual_stages_per_pipeline_rank to set VPP size"
    )
    args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
    parallel_config = ParallelismConfig(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
    )
    hf_config = AutoConfig.from_pretrained(model_path)
    use_cpu_initialization = args.use_cpu_initialization
    
    # Should disable to prevent initialization on GPU and need for tensor parallel CUDA RNG
    perform_initialization = not any([use_cpu_initialization, args.init_model_with_meta_device, args.finetune])
    assert not perform_initialization

    qwen_config = Qwen3MCoreConfig.from_hf(
        hf_config,
        parallelism_config=parallel_config,
        perform_initialization=perform_initialization,
        use_cpu_initialization=use_cpu_initialization,
    )

    if dist.is_initialized() and dist.get_rank() == 0 or not dist.is_initialized():
        pprint(qwen_config)

    args = qwen_config.update_mcore_args(args)
    d = qwen_config.to_dict(transformer_config_only=True)

    pprint(d)
    assert not qwen_config.perform_initialization
    mcore_config: TransformerConfig = qwen_config.to_mcore()
    pprint(mcore_config)
    assert not mcore_config.perform_initialization

    init_megatron(args)

    init_on_meta = args.init_model_with_meta_device
    init_on_cpu = args.use_cpu_initialization
    assert not (init_on_meta and init_on_cpu), f"{init_on_meta=} and {init_on_cpu=} both set"
    assert not mcore_config.perform_initialization 

    mcore_model = create_mcore_model(mcore_config)
    print(mcore_model)


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
