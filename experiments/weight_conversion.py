import functools
import json
import re
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from megatron.training.utils import unwrap_model
from qwen3_configuration import Qwen3ConfigT
from safetensors import safe_open

# ---- Weight Conversion ---- #

LAYER_NUMBER_REGEX = re.compile(r"decoder\.layers\.(\d+)\.")
EXPERT_IDX_REGEX = re.compile(r"(?<=\.weight)(\d+)$")

MCORE_ATTN_PAT = "self_attention"
MCORE_QKV_PAT = "linear_qkv"
MCORE_ATTN_QKV_PAT = f"{MCORE_ATTN_PAT}.{MCORE_QKV_PAT}"

MCORE_MLP_PAT = "mlp"
MCORE_MLP_FC_PAT = "linear_fc"
MCORE_MLP_FC1_PAT = f"{MCORE_MLP_FC_PAT}1.weight"
MCORE_MLP_FC2_PAT = f"{MCORE_MLP_FC_PAT}2.weight"
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



class ShardLoader:
    def __init__(self, checkpoint_dir: str, max_open: int = 32, device="cpu"):
        self.ckpt_dir = Path(checkpoint_dir)
        self.device = device
        self.weight_to_file: dict[str, str] = self._build_weight_map()

        @functools.lru_cache(maxsize=max_open)
        def _open(fname: str):
            return safe_open(self.ckpt_dir / fname, framework="pt", device=self.device)

        self._open = _open

    def get_tensor(self, name: str):
        shard = self.weight_to_file[name]
        return self._open(shard).get_tensor(name)

    def close_all(self):
        for h in list(self._open.cache.values()):
            h.close()
        self._open.cache_clear()

    def _build_weight_map(self) -> dict[str, str]:
        idx_path = self.ckpt_dir / "model.safetensors.index.json"
        if idx_path.exists():
            with open(idx_path) as f:
                return json.load(f)["weight_map"]

        shards: list[Path] = sorted(self.ckpt_dir.glob("*.safetensors"))

        if not shards:
            raise FileNotFoundError(f"No *.safetensors files found in {self.ckpt_dir}")

        if len(shards) == 1:
            one = shards[0].name
            with safe_open(shards[0], framework="pt") as f:
                return {k: one for k in f.keys()}

        weight_map: dict[str, str] = {}
        for shard_path in shards:
            with safe_open(shard_path, framework="pt") as f:
                # f.keys() is cheap – reads the ~1 KB header only
                for k in f.keys():
                    weight_map[k] = shard_path.name
        return weight_map

    def _load_hf_weights(
        self,
        model: GPTModel,
        local_to_hf_map: dict[str, str],
        device: str = "cuda",
    ):
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        etp_rank = mpu.get_expert_tensor_parallel_rank()
        etp_size = mpu.get_expert_tensor_parallel_world_size()
        
        num_attn_heads = model.config.num_attention_heads
        num_kv_heads = model.config.num_query_groups
        head_dim = model.config.kv_channels
        hidden_size = model.config.hidden_size

        new_sd = {}

        def _hf_to_mcore_weights_format(
            mcore_name: str, hf_weights: list[torch.Tensor]
        ) -> torch.Tensor:
            if len(hf_weights) == 1:
                return hf_weights[0]
            elif MCORE_ATTN_QKV_PAT in mcore_name and "layer_norm" not in mcore_name:
                return _interleave_and_merge_qkv(
                    num_attn_heads=num_attn_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    hidden_size=hidden_size,
                    hf_weights=hf_weights,
                )
            elif MCORE_MLP_FC1_PAT in mcore_name:
                assert len(hf_weights) == 2
                gate, up = hf_weights
                return torch.cat([gate, up], dim=0)
            else:
                raise ValueError(f"{mcore_name} not recognized")

        def _shard_across_tp(
            name: str,
            src_param_mcore: torch.Tensor,
            param_to_load: torch.Tensor,
            tp_size: int,
        ) -> list[torch.Tensor]:
            if tp_size == 1:
                return [src_param_mcore]

            is_qkv = MCORE_ATTN_QKV_PAT in name and "layer_norm" not in name
            is_fc1 = MCORE_MLP_FC1_PAT in name
            is_fc2 = MCORE_MLP_FC2_PAT in name

            if is_qkv:
                sharded_weights = src_param_mcore.chunk(tp_size)
            elif is_fc1:
                gate, up = src_param_mcore.chunk(2)
                gates = gate.chunk(tp_size)
                ups = up.chunk(tp_size)
                sharded_weights = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
            elif is_fc2:
                sharded_weights = src_param_mcore.chunk(tp_size, dim=1)
            else:
                # Remaining non-attn and non-mlp cases

                # Replicated params
                if param_to_load.shape == src_param_mcore.shape:
                    sharded_weights = [src_param_mcore for _ in range(tp_size)]
                else:
                    # Misc
                    # TODO: more robust checking for this case
                    assert len(param_to_load.shape) == len(src_param_mcore.shape)
                    partition_dim = _find_partition_dim(param_to_load.shape, src_param_mcore.shape)
                    sharded_weights = src_param_mcore.chunk(tp_size, dim=partition_dim)

            return sharded_weights

        for local_name, hf_names in local_to_hf_map.items():
            param_to_load = model.state_dict()[local_name]

            src_params_hf = [self.get_tensor(n) for n in hf_names]
            src_param_mcore = _hf_to_mcore_weights_format(local_name, src_params_hf)

            if MCORE_EXPERTS_FC_PAT in local_name:
                _tp_size = etp_size
                _tp_rank = etp_rank
            else:
                _tp_size = tp_size
                _tp_rank = tp_rank

            sharded_weights = _shard_across_tp(local_name, src_param_mcore, param_to_load, _tp_size)
            from mcore_utils import dist_print

            dist_print(f"{local_name} {_tp_size}: {param_to_load.shape=} {sharded_weights[0].shape=}", rank0_only=True)
            sharded_weights = [w.to(device) for w in sharded_weights]
            new_sd[local_name] = sharded_weights[_tp_rank]

        # strict must be false because of empty TE states, assign must be True when using init on meta
        model.load_state_dict(new_sd, strict=False, assign=True)

    def load_hf_weights(self, mcore_model_parts: list[GPTModel], mcore_to_hf_maps: list[dict[str, str]], device: str = "cpu"):
        for model, map in zip(mcore_model_parts, mcore_to_hf_maps):
            self._load_hf_weights(model, map, device)
    
        return mcore_model_parts
    
def _interleave_and_merge_qkv(
    num_attn_heads, num_kv_heads, hidden_size, head_dim: int, hf_weights: list[torch.Tensor]
) -> torch.Tensor:
    assert len(hf_weights) == 3
    assert num_attn_heads % num_kv_heads == 0

    query_group_ratio = num_attn_heads // num_kv_heads
    qdim_per_kv_head = head_dim * query_group_ratio

    q, k, v = hf_weights
    q_proj_size = head_dim * num_attn_heads
    k_proj_size = head_dim * num_kv_heads

    assert q.shape[0] == q_proj_size
    assert k.shape[0] == k_proj_size
    assert q.shape[0] // qdim_per_kv_head == num_kv_heads

    q = q.view(
        [
            num_kv_heads,
            qdim_per_kv_head,
            -1,
        ]
    )
    k = k.view([num_kv_heads, head_dim, -1])
    v = v.view([num_kv_heads, head_dim, -1])

    qkv = torch.cat([q, k, v], dim=1).view(-1, hidden_size).contiguous()

    return qkv


def _merge_mlp(src_weights: list[torch.Tensor]) -> torch.Tensor:
    assert len(src_weights) == 2
    gate, up = src_weights
    return torch.cat([gate, up], dim=0)


def _find_partition_dim(src_shape: torch.Size, dst_shape: torch.Size):
    for partition_dim, (s1, s2) in enumerate(zip(src_shape, dst_shape)):
        if s1 != s2:
            break

    return partition_dim

def load_hf_weights(
    weights_loader: ShardLoader,
    hf_config: Qwen3ConfigT,
    model: GPTModel,
    local_to_hf_map: dict[str, str],
    device: str = "cuda",
):
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    etp_rank = mpu.get_expert_tensor_parallel_rank()
    etp_size = mpu.get_expert_tensor_parallel_world_size()

    num_attn_heads = hf_config.num_attention_heads
    num_kv_heads = hf_config.num_key_value_heads
    head_dim = hf_config.head_dim
    hidden_size = hf_config.hidden_size

    new_sd = {}

    def _hf_to_mcore_weights_format(
        mcore_name: str, hf_weights: list[torch.Tensor]
    ) -> torch.Tensor:
        if len(hf_weights) == 1:
            return hf_weights[0]
        elif MCORE_ATTN_QKV_PAT in mcore_name and "layer_norm" not in mcore_name:
            return _interleave_and_merge_qkv(
                num_attn_heads=num_attn_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                hf_weights=hf_weights,
            )
        elif MCORE_MLP_FC1_PAT in mcore_name:
            assert len(hf_weights) == 2
            gate, up = hf_weights
            return torch.cat([gate, up], dim=0)
        else:
            raise ValueError(f"{mcore_name} not recognized")

    def _shard_across_tp(
        name: str,
        src_param_mcore: torch.Tensor,
        param_to_load: torch.Tensor,
        tp_size: int,
    ) -> list[torch.Tensor]:
        if tp_size == 1:
            return [src_param_mcore]

        is_qkv = MCORE_ATTN_QKV_PAT in name and "layer_norm" not in name
        is_fc1 = MCORE_MLP_FC1_PAT in name
        is_fc2 = MCORE_MLP_FC2_PAT in name

        if is_qkv:
            sharded_weights = src_param_mcore.chunk(tp_size)
        elif is_fc1:
            gate, up = src_param_mcore.chunk(2)
            gates = gate.chunk(tp_size)
            ups = up.chunk(tp_size)
            sharded_weights = [torch.cat([g, u], dim=0) for g, u in zip(gates, ups)]
        elif is_fc2:
            sharded_weights = src_param_mcore.chunk(tp_size, dim=1)
        else:
            # Remaining non-attn and non-mlp cases

            # Replicated params
            if param_to_load.shape == src_param_mcore.shape:
                sharded_weights = [src_param_mcore for _ in range(tp_size)]
            else:
                # Misc
                # TODO: more robust checking for this case
                assert len(param_to_load.shape) == len(src_param_mcore.shape)
                partition_dim = _find_partition_dim(param_to_load.shape, src_param_mcore.shape)
                sharded_weights = src_param_mcore.chunk(tp_size, dim=partition_dim)

        return sharded_weights

    for local_name, hf_names in local_to_hf_map.items():
        param_to_load = model.state_dict()[local_name]

        src_params_hf = [weights_loader.get_tensor(n) for n in hf_names]
        src_param_mcore = _hf_to_mcore_weights_format(local_name, src_params_hf)

        if MCORE_EXPERTS_FC_PAT in local_name:
            _tp_size = etp_size
            _tp_rank = etp_rank
        else:
            _tp_size = tp_size
            _tp_rank = tp_rank

        sharded_weights = _shard_across_tp(local_name, src_param_mcore, param_to_load, _tp_size)
        sharded_weights = [w.to(device) for w in sharded_weights]
        new_sd[local_name] = sharded_weights[_tp_rank]

    # strict must be false because of empty TE states, assign must be True when using init on meta
    model.load_state_dict(new_sd, strict=False, assign=True)


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    chkpt_dir = snapshot_download(args.model_path)
    print(list(Path(chkpt_dir).iterdir()))
    loader = ShardLoader(chkpt_dir)
    pprint(loader.weight_to_file)
