import re

import torch
from megatron.core.models.gpt import GPTModel
from megatron.training.utils import unwrap_model

LAYER_NUMBER_PAT = r'decoder\.layers\.(\d+)\.'
TE_STATE_PAT = "_extra_state"

def remove_te_keys(keys: list[str]):
    return list(filter(lambda k: k.find(TE_STATE_PAT) < 0, keys))

def remap_pp(model: torch.nn.Module | GPTModel):
    model = unwrap_model(model)
    breakpoint()
    
    # Remap from pp layer shard idx -> global layer idx
    # NOTE: "layer_number" starts at 1
    local_to_global = {}
    assert hasattr(model, "decoder")
    
    for idx, layer in enumerate(model.decoder.layers):
        local_to_global[idx] = layer.layer_number - 1

    _all_param_names = [
        k for k in model.state_dict().keys() if "_extra_state" not in k
    ]
    all_param_names = remove_te_keys(model.state_dict().keys())
    assert _all_param_names == all_param_names
    
    def _rename_decoder_layers(param_names: list[str]):
        name_map = {}
        for name in param_names:
            match = re.search(LAYER_NUMBER_PAT, name)
            if match:
                local_layer_idx = int(match.group(1))
                global_layer_idx = local_to_global[local_layer_idx]
                new_name = name.replace(f"layers.{local_layer_idx}", f"layers.{global_layer_idx}")
            else:
                new_name = name
            name_map[name] = new_name
    
        return name_map
    
    name_map = _rename_decoder_layers(all_param_names)
    
    ret = {}

    for param_name in all_param_names:
        keyword = "decoder.layers."
        if keyword in param_name:
            layer_idx = int(param_name.split(keyword)[1].split(".")[0])
            
            global_layer_idx = local_to_global[layer_idx]
            ret[param_name] = param_name.replace(
                f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
            )
        else:
            ret[param_name] = param_name
    breakpoint()
    assert len(ret) == len(name_map)
    assert ret == name_map
    
    return ret, name_map

def _weight_name_mapping_mcore_local_to_global(
        self, model: torch.nn.Module, consider_ep: bool = True
    ) -> dict[str, str]:
        """
        Map local weight names to global weight names, supporting VPP and EP.

        Args:
            model: The model instance

        Returns:
            dict: Mapping from local weight names to global weight names
        """
        # vpp
        local_layer_to_global_layer = {}
        model = unwrap_model(model)
        if hasattr(model, "decoder"):
            for idx, layer in enumerate(model.decoder.layers):
                local_layer_to_global_layer[idx] = layer.layer_number - 1
        all_param_names = [
            k for k in model.state_dict().keys() if "_extra_state" not in k
        ]
        ret = {}
        for param_name in all_param_names:
            keyword = "decoder.layers."
            if keyword in param_name:
                layer_idx = int(param_name.split(keyword)[1].split(".")[0])
                global_layer_idx = local_layer_to_global_layer[layer_idx]
                ret[param_name] = param_name.replace(
                    f"layers.{layer_idx}.", f"layers.{global_layer_idx}."
                )
            else:
                ret[param_name] = param_name

        # ep
        if self.mpu.ep_size > 1 and consider_ep:
            num_experts = self.config.num_moe_experts
            num_experts_per_rank = num_experts // self.mpu.ep_size
            local_expert_to_global_expert = {
                i: i + num_experts_per_rank * self.mpu.ep_rank
                for i in range(num_experts_per_rank)
            }
            for k in ret.keys():
                v = ret[k]
                if ".mlp.experts.linear_fc" in v:
                    name_prefix, local_expert_id = v.split(".weight")
                    global_expert_idx = local_expert_to_global_expert[
                        int(local_expert_id)
                    ]
                    ret[k] = f"{name_prefix}.weight{global_expert_idx}"

        return ret
