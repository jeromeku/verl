from collections import Counter

import torch


def get_model_param_devices(model: torch.nn.Module):
    param_devices = Counter(p.device.type for p in model.parameters())
    return param_devices


def get_total_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())


def get_module_param_count(model: torch.nn.Module):
    return {n: get_total_params(m) for n, m in model.named_children()}

