import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    try:
        vision_tower = getattr(vision_tower_cfg, "mm_vision_tower", getattr(vision_tower_cfg, "vision_tower", None))
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    except Exception as e:

        raise ValueError(f"Unknown vision tower: {vision_tower}")
