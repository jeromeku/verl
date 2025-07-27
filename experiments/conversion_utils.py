import torch
from megatron.core.transformer import TransformerConfig
from qwen3_configs import (
    Qwen3ConfigT,
    get_activation_recompute_config,
    get_arch_config,
    get_attn_config,
    get_fusion_config,
    get_mlp_config,
    get_parallelism,
    get_precision_config,
    is_qwen3_moe,
)


def _hf_to_mcore(hf_config: Qwen3ConfigT, is_moe: bool = False, **kwargs) -> TransformerConfig:
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

class Qwen3MCoreConfig:    
    @staticmethod
    def from_hf(config: Qwen3ConfigT, **kwargs) -> TransformerConfig:
        return _hf_to_mcore(config, is_moe=is_qwen3_moe(config), **kwargs)
