import torch
import torch.nn as nn
import re


def build_vision_projector(config, delay_load=False, **kwargs):
    try:
        projector_type = getattr(config, "mm_projector_type", "linear")
        mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    except Exception as e:
        raise ValueError(f"Unknown projector type: {projector_type}")
