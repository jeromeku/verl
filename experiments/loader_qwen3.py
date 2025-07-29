"""
Improved Qwen-3 HuggingFace loader with better error handling and organization
"""

import json
import os
import sys
from pathlib import Path

import megatron.core as mc

MEGATRON_ROOT = Path(mc.__file__).parents[2].resolve().as_posix()

sys.path.append(MEGATRON_ROOT)

import types

import torch
from tqdm import tqdm

try:
    import transformers
except ImportError:
    raise ImportError("The 'transformers' package is not installed.")

from tools.checkpoint.utils import _ConverterFakeProcessGroup


def _add_distributed_args(parser):
    group = parser.add_argument_group(title="distributed")

    group.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Degree of tensor model parallelism.",
    )
    group.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Degree of pipeline model parallelism.",
    )
    group.add_argument(
        "--virtual-pipeline-model-parallel-size",
        type=int,
        default=None,
        help="Number of virtual pipeline stages per pipeline parallelism rank",
    )
    group.add_argument("--init-model-with-meta-device", action="store_true")
    group.add_argument(
        "--context-parallel-size", type=int, default=1, help="Degree of context parallelism."
    )
    group.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        help="Degree of expert model parallelism.",
    )
    group.add_argument(
        "--expert-tensor-parallel-size",
        type=int,
        default=1,
        help="Degree of expert model parallelism. Default is None, which will be set to the value of --tensor-model-parallel-size.",
    )
    

def _add_moe_args(parser):
    group = parser.add_argument_group(title="MoE")

    group.add_argument(
        "--num-experts", type=int, default=None, help="Number of Experts in MoE (None means no MoE)"
    )
    group.add_argument(
        "--moe-ffn-hidden-size",
        type=int,
        default=None,
        help="The hidden size of each expert's feed-forward network (ffn). "
        "If not specified, defaults to the ffn_hidden_size.",
    )
    group.add_argument(
        "--moe-shared-expert-intermediate-size",
        type=int,
        default=None,
        help="Shared expert total ffn hidden size. "
        'It should be equal to "num_shared_experts * ffn_size_of_each_shared_expert" if there are multiple shared experts. '
        "None means no shared expert.",
    )
    group.add_argument(
        "--moe-grouped-gemm",
        action="store_true",
        help="When there are multiple experts per rank, launch multiple local GEMM kernels in multiple streams to improve the utilization and performance with GroupedLinear in TransformerEngine.",
    )
    group.add_argument(
        "--moe-use-legacy-grouped-gemm",
        action="store_true",
        help="Use legacy GroupedMLP rather than TEGroupedMLP. Note: The legacy one will be deprecated soon.",
    )
    group.add_argument(
        "--moe-layer-recompute",
        action="store_true",
        help="Enable checkpointing for moe_layer, should be used when memory is not sufficient. "
        'Deprecated. Use "--recompute-granularity selective --recompute-modules moe" instead.',
    )
    # Router arguments
    group.add_argument(
        "--moe-router-dtype",
        type=str,
        choices=["fp32", "fp64"],
        default=None,
        help="Data type for routing computation and expert output weighted averaging. "
        "Fp32/fp64 enhances numerical stability, especially with numerous experts. "
        "The perf impact should be negligible when used with permute fusion. "
        "None means no changes for dtype.",
    )
    group.add_argument(
        "--moe-router-topk",
        type=int,
        default=2,
        help="Number of experts to route to for each token. The default is 2.",
    )
    group.add_argument(
        "--moe-router-pre-softmax",
        action="store_true",
        help="Enable pre-softmax routing for MoE, which means softmax is before the top-k selection. By default, softmax is done after top-k.",
    )
    group.add_argument(
        "--moe-router-group-topk",
        type=int,
        default=None,
        help="Number of selected groups for group-limited routing.",
    )
    # Token dispatcher arguments
    group.add_argument(
        "--moe-token-dispatcher-type",
        type=str,
        choices=["allgather", "alltoall", "flex"],
        default="allgather",
        help="The type of token dispatcher to use. The default is 'allgather'. Options are 'allgather', 'alltoall'. We recommend using 'alltoall' when applying expert parallelism. For more information, please refer to the documentation in core/moe/README.",
    )
    group.add_argument(
        "--moe-enable-deepep",
        action="store_true",
        help="[Experimental] Enable DeepSeek/DeepEP for efficient token dispatching and combine in MoE models. Only works with flex token dispatcher by setting --moe-token-dispatcher-type=flex.",
    )
    group.add_argument(
        "--moe-deepep-num-sms", type=int, default=20, help="Number of SMs to use for DeepEP."
    )

    

def add_arguments(parser):
    """Add Qwen-3 specific arguments to parser."""
    group = parser.add_argument_group(title="Qwen-3 HuggingFace loader.")

    group.add_argument(
        "--model-name", type=str, required=True, help="HuggingFace model name (e.g., Qwen/Qwen3-4B)"
    )
    # group.add_argument('--bf16', action='store_true',
    #                    help='Whether to load weights in bf16.')
    # group.add_argument('--fp16', action='store_true',
    #                    help='Whether to load weights in fp16.')
    # group.add_argument('--true-vocab-size', type=int, default=None,
    #                    help='Original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument("--vocab-file", type=str, default=None, help="Path to the vocab file.")
    group.add_argument("--tokenizer-model", type=str, default=None, help="Tokenizer model file.")
    # group.add_argument('--megatron-path', type=str, default=None,
    #                    help='Base directory of Megatron repository')

    _add_distributed_args(parser)
    _add_moe_args(parser)
    
def validate_model_config(config):
    """Validate that the HuggingFace config has required attributes."""
    required_attrs = [
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
    ]
    missing_attrs = [attr for attr in required_attrs if not hasattr(config, attr)]

    if missing_attrs:
        raise ValueError(f"Model config missing required attributes: {missing_attrs}")

    # Validate MoE configuration if present
    if hasattr(config, "num_experts") and config.num_experts:
        if not hasattr(config, "num_experts_per_tok"):
            print("Warning: MoE model detected but num_experts_per_tok not found, assuming 2")


def get_dtype(args):
    """Determine the appropriate data type for model weights."""
    if args.bf16:
        return torch.bfloat16
    elif args.fp16:
        return torch.float16
    else:
        return torch.float32


def load_hf_model(args):
    """Load HuggingFace model and configuration with error handling."""
    from transformers import AutoConfig, AutoModelForCausalLM

    try:
        print(f"Loading HuggingFace model: {args.model_name}")
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

        # Validate model architecture
        validate_model_config(config)

        # Determine data type
        dtype = get_dtype(args)

        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name, torch_dtype=dtype, trust_remote_code=True, device_map="cpu"
        )

        hf_state_dict = hf_model.state_dict()

        return config, hf_model, hf_state_dict, dtype

    except Exception as e:
        print(f"Error loading HuggingFace model {args.model_name}: {e}")
        raise


def setup_megatron_args(args):
    """Set up Megatron arguments from HuggingFace config."""
    # Set up minimal Megatron args for model creation
    sys.argv = [
        "script.py",
        "--use-mcore-models",  # Use mcore models
        "--disable-bias-linear",  # Disable bias in linear layers
        "--qk-layernorm",  # Enable Q/K layernorm for Qwen-3
        "--no-masked-softmax-fusion",
        "--no-bias-gelu-fusion",
        "--no-bias-dropout-fusion",
        "--no-async-tensor-model-parallel-allreduce",
        "--no-gradient-accumulation-fusion",  # Disable gradient accumulation fusion
        "--use-cpu-initialization",
        "--micro-batch-size",
        "1",
        "--no-load-optim",
        "--no-load-rng",
        "--no-save-optim",
        "--no-save-rng",
        "--mock-data",
        "--no-initialization",
        "--transformer-impl",
        "transformer_engine",  # Required for proper tensor layouts
        "--load",
        "dummy",  # Required by some Megatron components
        "--no-one-logger",
    ]

    # if args.bf16:
    #     sys.argv.append('--bf16')
    # elif args.fp16:
    #     sys.argv.append('--fp16')

    return sys.argv


def configure_megatron_args(margs, config, dtype):
    """Configure Megatron args from HuggingFace config."""
    # Set model configuration from HF config
    from megatron.core.enums import ModelType

    margs.model_type = ModelType.encoder_or_decoder

    # Core model dimensions
    margs.num_layers = config.num_hidden_layers
    margs.hidden_size = config.hidden_size
    margs.ffn_hidden_size = config.intermediate_size
    margs.num_attention_heads = config.num_attention_heads
    margs.num_query_groups = config.num_key_value_heads
    # Qwen-3 uses fixed per-head dimension defined by `head_dim` in its config.
    margs.kv_channels = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )

    # Sequence and position settings
    margs.seq_length = getattr(config, "max_position_embeddings", 40960)
    margs.max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
    margs.position_embedding_type = "rope"
    margs.add_position_embedding = False
    margs.use_rotary_position_embeddings = True
    margs.rotary_base = getattr(config, "rope_theta", 1000000)
    margs.rotary_percent = 1.0
    margs.rotary_interleaved = False

    # Vocab settings
    margs.vocab_size = config.vocab_size
    margs.padded_vocab_size = config.vocab_size
    margs.make_vocab_size_divisible_by = 1
    margs.tokenizer_type = "HuggingFaceTokenizer"

    # Attention and activation settings
    margs.group_query_attention = config.num_key_value_heads < config.num_attention_heads
    margs.swiglu = getattr(config, "hidden_act", "silu") == "silu"
    margs.normalization = "RMSNorm"
    margs.norm_epsilon = getattr(config, "rms_norm_eps", 1e-6)

    # Enable Q/K layernorm for Qwen-3 compatibility
    margs.qk_layernorm = True

    # Bias settings
    margs.add_bias_linear = getattr(config, "mlp_bias", False)
    margs.add_qkv_bias = getattr(config, "attention_bias", False)
    margs.disable_bias_linear = True  # Override - disable bias for modern transformers

    # Training and optimization settings
    margs.apply_query_key_layer_scaling = False
    margs.attention_dropout = 0.0
    margs.hidden_dropout = 0.0
    margs.squared_relu = False
    margs.apply_layernorm_1p = False
    margs.untie_embeddings_and_output_weights = False  # Qwen-3 has tied embeddings

    # Required fields from working loaders
    margs.bert_binary_head = False
    margs.iteration = 1  # Required: '0' and 'release' don't work
    margs.global_batch_size = 1024

    # Model parallel configuration
    margs.tensor_model_parallel_size = 1
    margs.pipeline_model_parallel_size = 1
    margs.expert_model_parallel_size = 1
    margs.virtual_pipeline_model_parallel_size = None
    margs.sequence_parallel = True  # Enable like working loaders - affects tensor layouts

    # MoE configuration
    margs.num_experts = getattr(config, "num_experts", None)
    if margs.num_experts:
        margs.moe_layer_freq = 1
        margs.moe_ffn_hidden_size = getattr(config, "moe_intermediate_size", None)
        margs.moe_router_topk = getattr(config, "num_experts_per_tok", 2)
        margs.moe_token_dispatcher_type = "allgather"

    # Training metadata (fixed duplicate iteration assignment)
    margs.consumed_train_samples = 0
    margs.consumed_valid_samples = 0

    # Data types
    margs.params_dtype = dtype
    margs.fp16 = dtype == torch.float16
    margs.bf16 = dtype == torch.bfloat16

    return margs


def build_metadata(config, dtype):
    """Build metadata for checkpoint conversion."""
    md = types.SimpleNamespace()
    md.model_type = "GPT"
    md.num_layers = config.num_hidden_layers
    md.hidden_size = config.hidden_size
    md.seq_length = getattr(config, "max_position_embeddings", 40960)
    md.num_attention_heads = config.num_attention_heads
    md.kv_channels = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    md.num_query_groups = config.num_key_value_heads
    md.ffn_hidden_size = config.intermediate_size
    md.max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
    md.position_embedding_type = "rope"
    md.tokenizer_type = "HuggingFaceTokenizer"
    md.iteration = 1
    md.params_dtype = dtype
    md.bert_binary_head = False
    md.output_layer = False  # Qwen-3 has tied embeddings
    md.linear_bias = getattr(config, "mlp_bias", False)
    md.qkv_bias = getattr(config, "attention_bias", False)
    md.norm_has_bias = False  # RMSNorm doesn't have bias
    md.swiglu = getattr(config, "hidden_act", "silu") == "silu"
    # Enable Q/K layernorm for Qwen-3
    md.qk_layernorm = True
    md.previous_tensor_parallel_size = 1
    md.previous_pipeline_parallel_size = 1
    md.true_vocab_size = config.vocab_size
    md.vocab_size = config.vocab_size
    md.padded_vocab_size = config.vocab_size
    md.make_vocab_size_divisible_by = 1
    md.consumed_train_samples = 0
    md.consumed_valid_samples = 0
    md.num_experts = getattr(config, "num_experts", None) or 0
    md.checkpoint_args = None  # Will be set by caller if needed

    return md


def merge_qkv_weight(q, k, v, *, num_heads, num_query_groups):
    """Interleave-concatenate Q, K, V exactly like Qwen-3 + Megatron expect."""
    head_size = q.shape[0] // num_heads
    hidden_size = q.shape[1]
    heads_per_group = num_heads // num_query_groups

    # sanity checks
    assert k.shape[0] // num_query_groups == head_size
    assert v.shape == k.shape
    assert k.shape[1] == hidden_size

    # reshape to (H, D) and (G, D)
    q = q.view(num_heads, head_size, hidden_size)
    k = k.view(num_query_groups, head_size, hidden_size)
    v = v.view(num_query_groups, head_size, hidden_size)

    pieces = []
    for g in range(num_query_groups):
        # Q-heads that belong to this group
        pieces.append(q[g * heads_per_group : (g + 1) * heads_per_group])
        # one K and one V for the whole group
        pieces.append(k[g : g + 1])
        pieces.append(v[g : g + 1])

    qkv = torch.cat(pieces, dim=0)
    qkv = qkv.reshape(head_size * (num_heads + 2 * num_query_groups), hidden_size)
    return qkv


def merge_qkv_bias(qb, kb, vb, *, num_heads, num_query_groups):
    """Re-implement the original Qwen3 bias merge."""
    head_size = qb.shape[0] // num_heads

    qb = qb.view(num_heads, head_size)
    kb = kb.view(num_query_groups, head_size)
    vb = vb.view(num_query_groups, head_size)

    heads_per_group = num_heads // num_query_groups
    qkv_bias = []
    for g in range(num_query_groups):
        qkv_bias.append(qb[g * heads_per_group : (g + 1) * heads_per_group])
        qkv_bias.append(kb[g : g + 1])
        qkv_bias.append(vb[g : g + 1])
    return torch.cat(qkv_bias, dim=0).reshape(-1)


def merge_fc1(gate: torch.Tensor, up: torch.Tensor):
    """Merge gate and up proj into concatenated fc1."""
    return torch.cat((gate, up), dim=0)


def add_layer_norms(layer_msg, hf_state_dict, layer_idx):
    """Add layer normalization weights to message."""
    # Input layer norm
    layer_msg["input norm weight"] = hf_state_dict[
        f"model.layers.{layer_idx}.input_layernorm.weight"
    ]
    if f"model.layers.{layer_idx}.input_layernorm.bias" in hf_state_dict:
        layer_msg["input norm bias"] = hf_state_dict[
            f"model.layers.{layer_idx}.input_layernorm.bias"
        ]

    # Post attention layer norm
    layer_msg["post norm weight"] = hf_state_dict[
        f"model.layers.{layer_idx}.post_attention_layernorm.weight"
    ]
    if f"model.layers.{layer_idx}.post_attention_layernorm.bias" in hf_state_dict:
        layer_msg["post norm bias"] = hf_state_dict[
            f"model.layers.{layer_idx}.post_attention_layernorm.bias"
        ]


def add_attention_weights(layer_msg, hf_state_dict, config, layer_idx):
    """Add attention weights to message."""
    # Merged QKV projection
    q_weight = hf_state_dict[f"model.layers.{layer_idx}.self_attn.q_proj.weight"]
    k_weight = hf_state_dict[f"model.layers.{layer_idx}.self_attn.k_proj.weight"]
    v_weight = hf_state_dict[f"model.layers.{layer_idx}.self_attn.v_proj.weight"]

    merged_qkv = merge_qkv_weight(
        q_weight,
        k_weight,
        v_weight,
        num_heads=config.num_attention_heads,
        num_query_groups=config.num_key_value_heads,
    )
    layer_msg["qkv weight"] = merged_qkv

    # QKV bias (if present)
    q_bias_key = f"model.layers.{layer_idx}.self_attn.q_proj.bias"
    k_bias_key = f"model.layers.{layer_idx}.self_attn.k_proj.bias"
    v_bias_key = f"model.layers.{layer_idx}.self_attn.v_proj.bias"
    if all(key in hf_state_dict for key in [q_bias_key, k_bias_key, v_bias_key]):
        merged_qkv_bias = merge_qkv_bias(
            hf_state_dict[q_bias_key],
            hf_state_dict[k_bias_key],
            hf_state_dict[v_bias_key],
            num_heads=config.num_attention_heads,
            num_query_groups=config.num_key_value_heads,
        )
        layer_msg["qkv bias"] = merged_qkv_bias

    # Q/K normalization weights (critical for Qwen-3)
    q_norm_key = f"model.layers.{layer_idx}.self_attn.q_norm.weight"
    k_norm_key = f"model.layers.{layer_idx}.self_attn.k_norm.weight"
    if q_norm_key in hf_state_dict and k_norm_key in hf_state_dict:
        layer_msg["q norm weight"] = hf_state_dict[q_norm_key]
        layer_msg["k norm weight"] = hf_state_dict[k_norm_key]
        print(f"  Added Q/K norm weights for layer {layer_idx}")
    else:
        print(f"  Warning: Q/K norm weights not found for layer {layer_idx}")

    # Output projection
    layer_msg["dense weight"] = hf_state_dict[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]

    # Output projection bias (if present)
    o_bias_key = f"model.layers.{layer_idx}.self_attn.o_proj.bias"
    if o_bias_key in hf_state_dict:
        layer_msg["dense bias"] = hf_state_dict[o_bias_key]


def add_dense_mlp_weights(layer_msg, hf_state_dict, config, layer_idx):
    """Add dense MLP weights to message."""
    swiglu_activation = getattr(config, "hidden_act", "silu") == "silu"

    gate_weight = hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
    up_weight = hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"]

    if swiglu_activation:
        # For SwiGLU, provide separate W and V weights
        layer_msg["mlp l0 weight W"] = gate_weight  # Gate projection
        layer_msg["mlp l0 weight V"] = up_weight  # Up projection
    else:
        # For non-SwiGLU, merge gate and up projections
        layer_msg["mlp l0 weight"] = merge_fc1(gate_weight, up_weight)

    # MLP output projection
    layer_msg["mlp l1 weight"] = hf_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"]

    # MLP biases (if present)
    gate_bias_key = f"model.layers.{layer_idx}.mlp.gate_proj.bias"
    up_bias_key = f"model.layers.{layer_idx}.mlp.up_proj.bias"
    down_bias_key = f"model.layers.{layer_idx}.mlp.down_proj.bias"

    if swiglu_activation:
        # For SwiGLU, provide separate W and V biases
        if gate_bias_key in hf_state_dict:
            layer_msg["mlp l0 bias W"] = hf_state_dict[gate_bias_key]
        if up_bias_key in hf_state_dict:
            layer_msg["mlp l0 bias V"] = hf_state_dict[up_bias_key]
    else:
        # For non-SwiGLU, merge biases
        if gate_bias_key in hf_state_dict and up_bias_key in hf_state_dict:
            layer_msg["mlp l0 bias"] = merge_fc1(
                hf_state_dict[gate_bias_key], hf_state_dict[up_bias_key]
            )

    if down_bias_key in hf_state_dict:
        layer_msg["mlp l1 bias"] = hf_state_dict[down_bias_key]


def add_moe_weights(layer_msg, hf_state_dict, config, layer_idx):
    """Add MoE weights to message."""
    num_experts = config.num_experts
    swiglu_activation = getattr(config, "hidden_act", "silu") == "silu"

    expert_gate_weights = []
    expert_up_weights = []
    expert_down_weights = []

    for expert_idx in range(num_experts):
        expert_gate_weights.append(
            hf_state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"]
        )
        expert_up_weights.append(
            hf_state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"]
        )
        expert_down_weights.append(
            hf_state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"]
        )

    # Stack experts - this matches the expected format
    if swiglu_activation:
        # For SwiGLU, provide separate W and V weights for all experts
        layer_msg["mlp l0 weight W"] = torch.stack(expert_gate_weights, dim=0)
        layer_msg["mlp l0 weight V"] = torch.stack(expert_up_weights, dim=0)
    else:
        # For non-SwiGLU, merge and stack expert weights
        expert_fc1_weights = [
            merge_fc1(gate, up) for gate, up in zip(expert_gate_weights, expert_up_weights)
        ]
        layer_msg["mlp l0 weight"] = torch.stack(expert_fc1_weights, dim=0)

    layer_msg["mlp l1 weight"] = torch.stack(expert_down_weights, dim=0)

    # Router weight
    layer_msg["mlp router weight"] = hf_state_dict[f"model.layers.{layer_idx}.mlp.gate.weight"]


def update_args_for_model_loading(margs, args):
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

    # assert args.num_layers_per_virtual_pipeline_stage is None, (
    #     "Use --num_virtual_stages_per_pipeline_rank to set VPP size"
    # )
    # args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank

    return args


def load_checkpoint(queue, args):
    """Load Qwen-3 HuggingFace checkpoint and convert to Megatron format."""
    # Set required environment variable for tensor parallelism
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    # # Search in directory above this
    # sys.path.append(os.path.abspath(
    #     os.path.join(os.path.dirname(__file__),
    #                  os.path.pardir,
    #                  os.path.pardir)))
    # if args.megatron_path is not None:
    #     sys.path.insert(0, args.megatron_path)

    try:
        from megatron.core import mpu
        from megatron.core.enums import ModelType
        from megatron.legacy import fused_kernels
        from megatron.legacy.model import module
        from megatron.training.arguments import parse_args, validate_args
        from megatron.training.global_vars import set_global_variables
    except ModuleNotFoundError as e:
        print(
            f"Unable to import required Megatron modules: {e}. Please check installation or specify --megatron-path. Exiting."
        )
        queue.put("exit")
        exit(1)

    # config, hf_model, hf_state_dict, dtype = load_hf_model(args)
    from mcore_utils import patch_mcore_args
    from qwen3_configuration import (
        MoeOptConfig,
        ParallelismConfig,
        Qwen3MCoreConfig,
        is_qwen3_moe_config,
    )
    from transformers import AutoConfig

    sys.argv = setup_megatron_args(args)
    margs = parse_args()

    for k,v in vars(args).items():
        if hasattr(margs, k):
            setattr(margs, k, v)

#    margs.virtual_pipeline_model_parallel_size = args.virtual_pipeline_model_parallel_size
    
    model_path = args.model_name
    hf_config = AutoConfig.from_pretrained(model_path)

    is_moe = is_qwen3_moe_config(hf_config)

    sequence_parallel = margs.sequence_parallel or margs.tensor_model_parallel_size > 1

    if is_moe and args.tensor_model_parallel_size > 1:
        sequence_parallel = True

    # parallel_config = ParallelismConfig(
    #     tensor_model_parallel_size=margs.tensor_model_parallel_size,
    #     pipeline_model_parallel_size=margs.pipeline_model_parallel_size,
    #     virtual_pipeline_model_parallel_size=margs.virtual_pipeline_model_parallel_size,
    #     context_parallel_size=margs.context_parallel_size,
    #     expert_model_parallel_size=margs.expert_model_parallel_size,
    #     expert_tensor_parallel_size=margs.expert_tensor_parallel_size,
    #     sequence_parallel=sequence_parallel,
    # )
    vpp_size = args.virtual_pipeline_model_parallel_size
    parallel_config = ParallelismConfig.from_args(margs, sequence_parallel=sequence_parallel, virtual_pipeline_model_parallel_size=vpp_size)
    
    if is_moe:
        moe_opt_config = MoeOptConfig.from_args(margs, moe_grouped_gemm=True)
    else:
        moe_opt_config = None

    breakpoint()
    qwen_config = Qwen3MCoreConfig.from_hf(
        hf_config,
        parallelism_config=parallel_config,
        moe_opt_config=moe_opt_config,
        perform_initialization=margs.perform_initialization,
        use_cpu_initialization=margs.use_cpu_initialization,
        init_model_with_meta_device=margs.init_model_with_meta_device,
    )

    # Update MCore global args with model specific config
    margs = qwen_config.update_mcore_args(margs)

    # Set up Megatron arguments
    # margs = configure_megatron_args(margs, config, dtype)

    # Validate args
    # See parallel_state -- needed for MoE parallel folding, which reuses TP / CP group for EP / ETP
    margs.world_size = max(
        margs.tensor_model_parallel_size
        * margs.pipeline_model_parallel_size
        * margs.context_parallel_size,
        margs.expert_model_parallel_size
        * margs.expert_tensor_parallel_size
        * margs.pipeline_model_parallel_size,
    )
    breakpoint()
    margs = validate_args(margs)

    if False:
        # Initialize Megatron environment
        module.MegatronModule.embedding_warning_printed = True
        set_global_variables(margs, build_tokenizer=False)
        mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
        mpu.set_virtual_pipeline_model_parallel_world_size(
            margs.virtual_pipeline_model_parallel_size
        )
        mpu.set_expert_model_parallel_world_size(margs.expert_model_parallel_size)
        mpu.set_expert_tensor_parallel_world_size(margs.expert_tensor_model_parallel_size)

        # Fake process groups for conversion
        fake_tp_group = _ConverterFakeProcessGroup(size=margs.tensor_model_parallel_size)
        fake_ep_group = _ConverterFakeProcessGroup(size=margs.expert_model_parallel_size)
        fake_etp_group = _ConverterFakeProcessGroup(size=margs.expert_tensor_model_parallel_size)

        mpu._TENSOR_MODEL_PARALLEL_GROUP = fake_tp_group
        mpu._EXPERT_MODEL_PARALLEL_GROUP = fake_ep_group
        mpu._EXPERT_TENSOR_PARALLEL_GROUP = fake_etp_group
        fused_kernels.load(margs)

        # Send metadata
        md = build_metadata(config, dtype)
        md.checkpoint_args = margs
        queue.put(md)

        # Send embeddings
        embeddings_msg = {
            "name": "embeddings",
            "word embeddings": hf_state_dict["model.embed_tokens.weight"],
        }
        queue.put(embeddings_msg)

        # Send transformer layers
        for layer_idx in tqdm(range(config.num_hidden_layers), desc="Processing layers"):
            layer_msg = {"name": f"transformer layer {layer_idx}"}

            # Add layer components
            add_layer_norms(layer_msg, hf_state_dict, layer_idx)
            add_attention_weights(layer_msg, hf_state_dict, config, layer_idx)

            # MLP
            if is_moe:
                add_moe_weights(layer_msg, hf_state_dict, config, layer_idx)
            else:
                add_dense_mlp_weights(layer_msg, hf_state_dict, config, layer_idx)

            queue.put(layer_msg)

        # Send final layer norm
        final_norm_msg = {"name": "final norm", "weight": hf_state_dict["model.norm.weight"]}
        if "model.norm.bias" in hf_state_dict:
            final_norm_msg["bias"] = hf_state_dict["model.norm.bias"]
        queue.put(final_norm_msg)

        # Send LM head (if needed)
        if md.output_layer:
            lm_head_msg = {"name": "lm head", "dense weight": hf_state_dict["lm_head.weight"]}
            if "lm_head.bias" in hf_state_dict:
                lm_head_msg["dense bias"] = hf_state_dict["lm_head.bias"]
            queue.put(lm_head_msg)

        # Send done signal
        queue.put("done")
        print("✓ Qwen-3 model loading complete!")

    # except Exception as e:
    #     print(f"Error during checkpoint loading: {e}")
    #     return
    #     queue.put("exit")
    #     exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Megatron Checkpoint Converter Arguments",
        allow_abbrev=False,
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--model-type", type=str, required=True, choices=["GPT", "BERT"], help="Type of the model"
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="megatron",
        help="Module name to load checkpoint, should be on python path",
    )
    parser.add_argument(
        "--saver",
        type=str,
        default="megatron",
        help="Module name to save checkpoint, should be on python path",
    )
    parser.add_argument(
        "--load-dir", type=str, required=True, help="Directory to load model checkpoint from"
    )
    parser.add_argument(
        "--save-dir", type=str, required=True, help="Directory to save model checkpoint to"
    )
    parser.add_argument(
        "--max-queue-size", type=int, default=50, help="Maximum number of tensors in the queue"
    )
    parser.add_argument(
        "--no-checking",
        action="store_false",
        help="Do not perform checking on the name and ordering of weights",
        dest="checking",
    )

    add_arguments(parser)
    args = parser.parse_args()
    load_checkpoint(queue=None, args=args)
