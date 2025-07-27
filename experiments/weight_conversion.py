import re

from megatron.core import mpu
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.training.utils import unwrap_model

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



def _extract_layer_number(name: str):
    match = LAYER_NUMBER_REGEX.search(name)

    if not match:
        raise ValueError(f"Could not identify layer number in {name}")

    layer_number = int(match.group(1))

    return layer_number

def remove_te_keys(keys: list[str]):
    return list(filter(lambda k: k.find(TE_STATE_PAT) < 0, keys))

# ---- Param Name Mappings ---- #


# Create map of local param names to their global param names for EP and PP
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

# ---- Param Name Mappings ---- #

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