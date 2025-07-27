from argparse import Namespace
from dataclasses import asdict, dataclass, field, is_dataclass

import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.core.transformer import TransformerConfig
from transformers.models.qwen3 import Qwen3Config
from transformers.models.qwen3_moe import Qwen3MoeConfig

Qwen3ConfigT = Qwen3Config | Qwen3MoeConfig

# Dense
QWEN3_600M = "Qwen/Qwen3-0.6B"
QWEN3_4B = "Qwen/Qwen3-4B"

# MoE
QWEN3_30B_3B = "Qwen/Qwen3-30B-A3B"
QWEN3_235B_A22B = "Qwen/Qwen3-235B-A22B"

QWEN3_DENSE_MODELS = [QWEN3_600M, QWEN3_4B]
QWEN3_MOE_MODELS = [QWEN3_30B_3B, QWEN3_235B_A22B]


def is_qwen3_moe_config(config: Qwen3ConfigT):
    return isinstance(config, Qwen3MoeConfig)


@dataclass
class ConfigBase:
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ParallelismConfig(ConfigBase):
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    variable_seq_lengths: bool = False


@dataclass
class MLPConfig(ConfigBase):
    # Common MLP features
    gated_linear_unit: bool = True
    activation_func: object = F.silu


@dataclass
class MoeConfig(MLPConfig):
    # Experts
    moe_ffn_hidden_size: int | None = None
    num_moe_experts: int | None = None
    moe_shared_expert_intermediate_size: int | None = None

    # Router
    moe_router_topk: int | None = None
    moe_router_score_function: str = "softmax"
    moe_router_pre_softmax: bool = False
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float | None = None

    # Not used by Qwen3Moe
    moe_router_enable_expert_bias: bool = False
    moe_expert_capacity_factor: float | None = None
    moe_router_bias_update_rate: float = 0.001

    @classmethod
    def from_hf(cls, config: Qwen3MoeConfig):
        return cls(
            moe_ffn_hidden_size=config.moe_intermediate_size,
            num_moe_experts=config.num_experts,
            moe_router_topk=config.num_experts_per_tok,
            moe_aux_loss_coeff=config.router_aux_loss_coef,
        )


@dataclass
class MoeOptConfig(ConfigBase):
    """
    Moe optimization config
    """

    # Expert computation
    moe_grouped_gemm: bool = True
    moe_use_legacy_grouped_gemm: bool = False

    # Router
    moe_router_dtype: torch.dtype = torch.float32
    moe_token_dispatcher_type: str = "alltoall"

    # Optimizations
    moe_enable_deepep: bool = False
    moe_deepep_num_sms: int = 20
    moe_layer_recompute: bool = True
    moe_permute_fusion: bool = False
    moe_per_layer_logging: bool = False


@dataclass
class ArchConfig(ConfigBase):
    num_layers: int
    hidden_size: int
    layernorm_epsilon: float
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False

    @classmethod
    def from_hf(cls, config: Qwen3ConfigT):
        return cls(
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            layernorm_epsilon=config.rms_norm_eps,
            normalization="RMSNorm",
            add_bias_linear=False,
        )


@dataclass
class AttnConfig(ConfigBase):
    num_attention_heads: int
    num_query_groups: int
    ffn_hidden_size: int
    attention_dropout: float
    hidden_dropout: float = 0.0
    kv_channels: int | None = None
    qk_layernorm: bool = True

    @classmethod
    def from_hf(cls, config: Qwen3ConfigT):
        if isinstance(config, Qwen3ConfigT):
            # Qwen3 specifies a head_dim, important for calculating correct qkv proj dims
            assert hasattr(config, "head_dim") and config.head_dim is not None

        return cls(
            num_attention_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
            ffn_hidden_size=config.intermediate_size,
            attention_dropout=config.attention_dropout,
            hidden_dropout=getattr(config, "hidden_dropout", 0.0),
            kv_channels=config.head_dim,
            qk_layernorm=True,  # Qwen3 always uses qk norm
        )


@dataclass
class PrecisionConfig(ConfigBase):
    pipeline_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True

    @classmethod
    def from_hf(
        cls,
        config: Qwen3ConfigT,
        pipeline_dtype: torch.dtype = None,
        params_dtype: torch.dtype = None,
        bf16: bool = None,
        fp16: bool = None,
    ):
        default_dtype = config.torch_dtype

        if bf16 is not None and fp16 is not None:
            assert bf16 ^ fp16
        if fp16 is not None and fp16:
            print(f"WARNING: {fp16=} does not match default HF dtype {default_dtype}")
        if pipeline_dtype is not None and pipeline_dtype != default_dtype:
            print(
                f"WARNING: pipeline_dtype != HF default dtype: {pipeline_dtype} != {default_dtype}"
            )
        if params_dtype is not None and params_dtype != default_dtype:
            print(f"WARNING: params_dtype != HF default dtype: {params_dtype} != {default_dtype}")

        # Qwen3 / MoE uses bfloat16 by default
        if isinstance(config, Qwen3ConfigT):
            assert config.torch_dtype == torch.bfloat16

        pipeline_dtype = pipeline_dtype or default_dtype
        params_dtype = params_dtype or default_dtype
        bf16 = bf16 or default_dtype == torch.bfloat16
        fp16 = not bf16

        return cls(pipeline_dtype=pipeline_dtype, params_dtype=params_dtype, bf16=bf16, fp16=fp16)


@dataclass
class FusionConfig(ConfigBase):
    masked_softmax_fusion: bool = False
    persist_layer_norm: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False


@dataclass
class ActivationRecomputeConfig(ConfigBase):
    recompute_granularity: str = None
    recompute_method: str = None
    recompute_num_layers: int = None
    distribute_saved_activations: bool = None
    recompute_modules: list[str] = None


@dataclass(kw_only=True)
class Qwen3MCoreConfig:
    # Args from megatron.training.arguments
    vocab_size: int
    untie_embeddings_and_output_weights: bool
    seq_length: int
    padded_vocab_size: int = None

    # RoPE
    rotary_base: float
    rope_scaling: bool
    position_embedding_type = "rope"
    rotary_percent = 1.0

    # Deprecated but needed to initialize Megatron
    max_position_embeddings: int = None

    # Args used to instantiate TransformerConfig
    arch_config: ArchConfig
    attn_config: AttnConfig
    mlp_config: MLPConfig | MoeConfig

    parallelism_config: ParallelismConfig
    precision_config: PrecisionConfig

    # Disable gpu init and tensor parallel CUDA RNG tracker    
    perform_initialization: bool = False
    use_cpu_initialization: bool = False

    # Optional configs, primarily for optimization
    moe_opt_config: MoeOptConfig = None
    fusion_config: FusionConfig = field(default_factory=FusionConfig)
    activation_recompute_config: ActivationRecomputeConfig = field(
        default_factory=ActivationRecomputeConfig
    )
    use_transformer_engine: bool = True

    @classmethod
    def from_hf(
        cls,
        config: Qwen3ConfigT,
        # General Mcore options not encapsulated by TransformerConfig and not derived from HF QwenConfig
        padded_vocab_size: int = None,
        use_transformer_engine: bool = True,
        # TransformerConfig
        moe_opt_config: MoeOptConfig = MoeOptConfig(),
        parallelism_config: ParallelismConfig = ParallelismConfig(),
        fusion_config: FusionConfig = FusionConfig(),
        activation_recompute_config: ActivationRecomputeConfig = ActivationRecomputeConfig(),
        precision_config: PrecisionConfig = PrecisionConfig(),
        perform_initialization: bool = False,
        use_cpu_initialization: bool = False,
    ):
        arch_config = ArchConfig.from_hf(config)
        if is_qwen3_moe_config(config):
            mlp_config = MoeConfig.from_hf(config)
        else:
            mlp_config = MLPConfig()

        attn_config = AttnConfig.from_hf(config)
        return Qwen3MCoreConfig(
            # MCore args that live outside TransformerConfig
            vocab_size=config.vocab_size,
            padded_vocab_size=padded_vocab_size or config.vocab_size,
            untie_embeddings_and_output_weights=not config.tie_word_embeddings,
            seq_length=config.max_position_embeddings,
            max_position_embeddings=config.max_position_embeddings,
            rotary_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
            use_transformer_engine=use_transformer_engine,
            perform_initialization=perform_initialization,
            use_cpu_initialization=use_cpu_initialization,
            # Transformer Config
            arch_config=arch_config,
            attn_config=attn_config,
            mlp_config=mlp_config,
            parallelism_config=parallelism_config,
            precision_config=precision_config,
            moe_opt_config=moe_opt_config,
            fusion_config=fusion_config,
            activation_recompute_config=activation_recompute_config,
        )

    def to_dict(self, transformer_config_only: bool = False):
        merged = {}

        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)

            if is_dataclass(value):
                merged.update(asdict(value))
            else:
                if transformer_config_only:
                    continue
                merged[field_name] = value

        return merged

    def to_mcore(self, **kwargs):
        return TransformerConfig(
            **self.to_dict(transformer_config_only=True),
            perform_initialization=self.perform_initialization,
            use_cpu_initialization=self.use_cpu_initialization,
            **kwargs,
        )

    def update_mcore_args(self, args: Namespace):
        args.max_position_embeddings = self.max_position_embeddings
        args.num_layers = self.arch_config.num_layers
        args.hidden_size = self.arch_config.hidden_size
        args.num_attention_heads = self.attn_config.num_attention_heads
        args.seq_length = self.max_position_embeddings

        if isinstance(self.mlp_config, MoeConfig):
            args.num_experts = self.mlp_config.num_moe_experts
            args.moe_router_topk = self.mlp_config.moe_router_topk

        args.vocab_size = self.vocab_size
        args.padded_vocab_size = self.vocab_size
        args.untie_embeddings_and_output_weights = self.untie_embeddings_and_output_weights
        args.position_embedding_type = "rope"
        args.rotary_percent = 1.0
        args.rotary_base = self.rotary_base
        args.rope_scaling = True if self.rope_scaling is not None else False

        args.perform_initialization = self.perform_initialization
        args.use_cpu_initialization = self.use_cpu_initialization

        return args


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


def get_mlp_config(
    hf_config: Qwen3ConfigT,
    is_moe: bool = False,
    # MLP
    gated_linear_unit: bool = True,
    activation_func=F.silu,
    # Experts
    moe_grouped_gemm: bool = True,
    moe_use_legacy_grouped_gemm: bool = False,
    moe_shared_expert_intermediate_size=None,
    # Router
    moe_router_dtype=torch.float32,
    moe_router_score_function: str = "softmax",
    moe_router_pre_softmax: bool = False,
    moe_token_dispatcher_type: str = "alltoall",
    moe_router_enable_expert_bias: bool = False,
    moe_router_load_balancing_type: str = "aux_loss",
    moe_expert_capacity_factor=None,
    moe_router_bias_update_rate: float = 0.001,
    # Optimizations
    moe_enable_deepep: bool = False,
    moe_deepep_num_sms: int = 20,
    moe_layer_recompute: bool = True,
    moe_permute_fusion: bool = False,
    moe_per_layer_logging: bool = True,
):
    if isinstance(hf_config, Qwen3ConfigT):
        assert gated_linear_unit
        assert activation_func == F.silu

    mlp_config = {
        # Experts
        "gated_linear_unit": gated_linear_unit,
        "activation_func": activation_func,
    }

    if is_moe:
        if isinstance(hf_config, Qwen3ConfigT):
            # Checks specific for Qwen3Moe
            assert moe_shared_expert_intermediate_size is None
            assert moe_router_score_function == "softmax"
            assert not moe_router_pre_softmax
            assert not moe_router_enable_expert_bias
            assert moe_router_load_balancing_type == "aux_loss"
            assert moe_expert_capacity_factor is None

        moe_config = {
            # Experts
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "num_moe_experts": hf_config.num_experts,
            "moe_grouped_gemm": moe_grouped_gemm,  # requires TransformerEngine
            "moe_use_legacy_grouped_gemm": moe_use_legacy_grouped_gemm,  # legacy cutlass grouped gemm
            "moe_shared_expert_intermediate_size": moe_shared_expert_intermediate_size,  # no shared expert
            # Router
            "moe_router_dtype": moe_router_dtype,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "moe_router_score_function": moe_router_score_function,
            "moe_router_pre_softmax": moe_router_pre_softmax,  # softmax is applied **after** topk in Qwen3-Moe
            "moe_token_dispatcher_type": moe_token_dispatcher_type,  # TODO: tune
            # auxiliary loss
            "moe_router_enable_expert_bias": moe_router_enable_expert_bias,  # aux-loss-free routing, only for sigmoid-scored router
            "moe_router_load_balancing_type": moe_router_load_balancing_type,
            "moe_expert_capacity_factor": moe_expert_capacity_factor,  # token choice -> no dropped tokens
            "moe_router_bias_update_rate": moe_router_bias_update_rate,  # TODO: check whether this is needed for aux_loss
            "moe_aux_loss_coeff": hf_config.router_aux_loss_coef,
            # optimizations
            "moe_enable_deepep": moe_enable_deepep,
            "moe_deepep_num_sms": moe_deepep_num_sms,  # TODO: tune
            "moe_layer_recompute": moe_layer_recompute,
            "moe_permute_fusion": moe_permute_fusion,  # TODO: tune
            "moe_per_layer_logging": moe_per_layer_logging,  # for auxiliary loss
        }
    else:
        moe_config = {}

    return {**mlp_config, **moe_config}


def get_arch_config(
    hf_config: Qwen3ConfigT,
    normalization: str = "RMSNorm",
    add_bias_linear: bool = False,
):
    return {
        "num_layers": hf_config.num_hidden_layers,
        "hidden_size": hf_config.hidden_size,
        "layernorm_epsilon": hf_config.rms_norm_eps,
        "normalization": normalization,
        "add_bias_linear": add_bias_linear,
    }


def get_attn_config(hf_config: Qwen3ConfigT, qk_layernorm: bool = True):
    if isinstance(hf_config, Qwen3ConfigT):
        # Qwen3 specifies a head_dim, important for calculating correct qkv proj dims
        assert hasattr(hf_config, "head_dim") and hf_config.head_dim is not None

    return {
        "num_attention_heads": hf_config.num_attention_heads,
        "num_query_groups": hf_config.num_key_value_heads,
        "ffn_hidden_size": hf_config.intermediate_size,
        "attention_dropout": hf_config.attention_dropout,
        "hidden_dropout": getattr(hf_config, "hidden_dropout", 0.0),
        "kv_channels": getattr(hf_config, "head_dim", None),  # hidden_size // num_attention_heads
        "qk_layernorm": qk_layernorm,
    }


def get_precision_config(hf_config: Qwen3ConfigT, dtype: torch.dtype = torch.bfloat16):
    if isinstance(hf_config, Qwen3ConfigT):
        assert dtype == hf_config.torch_dtype

    return {
        "pipeline_dtype": dtype,
        "params_dtype": dtype,
        "bf16": dtype is torch.bfloat16,  # Should be true for Qwen3
    }


# Optimizations - disable all when converting checkpoint
def get_fusion_config(
    masked_softmax_fusion: bool = False,
    persist_layer_norm: bool = False,
    bias_activation_fusion: bool = False,
    bias_dropout_fusion: bool = False,
):
    return {
        "masked_softmax_fusion": masked_softmax_fusion,
        "persist_layer_norm": persist_layer_norm,
        "bias_activation_fusion": bias_activation_fusion,
        "bias_dropout_fusion": bias_dropout_fusion,
    }


def get_activation_recompute_config(
    recompute_granularity=None,
    recompute_method=None,
    recompute_num_layers=None,
    distribute_saved_activations=None,
    recompute_modules=None,
):
    return {
        "recompute_granularity": recompute_granularity,
        "recompute_method": recompute_method,
        "recompute_num_layers": recompute_num_layers,
        "distribute_saved_activations": distribute_saved_activations,
        "recompute_modules": recompute_modules,
    }


# ---- Param Name Mappings ---- #

_DIRECT_MAPPING = {
    "embedding.word_embeddings.weight": "model.embed_tokens.weight",
    "decoder.final_layernorm.weight": "model.norm.weight",
    "output_layer.weight": "lm_head.weight",
}
_ATTENTION_MAPPING = {
    "self_attention.linear_proj.weight": ["model.layers.{layer_number}.self_attn.o_proj.weight"],
    "self_attention.linear_qkv.layer_norm_weight": [
        "model.layers.{layer_number}.input_layernorm.weight"
    ],
    "self_attention.q_layernorm.weight": ["model.layers.{layer_number}.self_attn.q_norm.weight"],
    "self_attention.k_layernorm.weight": ["model.layers.{layer_number}.self_attn.k_norm.weight"],
    "self_attention.linear_qkv.weight": [
        "model.layers.{layer_number}.self_attn.q_proj.weight",
        "model.layers.{layer_number}.self_attn.k_proj.weight",
        "model.layers.{layer_number}.self_attn.v_proj.weight",
    ],
    "self_attention.linear_qkv.bias": [
        "model.layers.{layer_number}.self_attn.q_proj.bias",
        "model.layers.{layer_number}.self_attn.k_proj.bias",
        "model.layers.{layer_number}.self_attn.v_proj.bias",
    ],
}
_DENSE_MLP_MAPPING = {
    "mlp.linear_fc1.weight": [
        "model.layers.{layer_number}.mlp.gate_proj.weight",
        "model.layers.{layer_number}.mlp.up_proj.weight",
    ],
    "mlp.linear_fc1.layer_norm_weight": [
        "model.layers.{layer_number}.post_attention_layernorm.weight"
    ],
    "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
}
_MOE_MLP_MAPPING = {
    "shared_experts.linear_fc1.weight": [
        "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
        "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
    ],
    "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
    "shared_experts.linear_fc2.weight": [
        "model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
    ],
    "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
    "shared_experts.gate_weight": ["model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
    "mlp.experts.linear_fc1": [
        "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
        "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
    ],
    "mlp.experts.linear_fc2": [
        "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
    ],
}


MCORE_TO_HF_PARAM_MAPPINGS = {
    "pre_post_decoder": {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    },
    "attention": {
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight"
        ],
        "self_attention.linear_qkv.layer_norm_weight": [
            "model.layers.{layer_number}.input_layernorm.weight"
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight"
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight"
        ],
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_qkv.bias": [
            "model.layers.{layer_number}.self_attn.q_proj.bias",
            "model.layers.{layer_number}.self_attn.k_proj.bias",
            "model.layers.{layer_number}.self_attn.v_proj.bias",
        ],
    },
    "mlp": {
        "dense": {
            "mlp.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.gate_proj.weight",
                "model.layers.{layer_number}.mlp.up_proj.weight",
            ],
            "mlp.linear_fc1.layer_norm_weight": [
                "model.layers.{layer_number}.post_attention_layernorm.weight"
            ],
            "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        },
        "moe": {
            "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
            "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
            "mlp.experts.linear_fc1": [
                "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
            ],
            "mlp.experts.linear_fc2": [
                "model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
            ],
        },
    },
}
