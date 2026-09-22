# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Data-only VLDA settings; dimensions and state conditioning live in ModelConfig."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VLDADataConfig:
    camera_keys: list[str] = field(default_factory=lambda: ["image", "wrist_image", ""])
    state_key: str = "state"
    action_key: str = "actions"
    image_size: int = 224
    max_token_len: int = 200
    norm_stats_path: str = ""  # Empty uses dataset/meta/stats.json; resume uses the checkpoint snapshot.
    normalization_mode: str = "q99"
    adapt_to_pi: bool = False
    delta_joint_actions: bool = False
    vjepa_image_cache_dir: str = ""
