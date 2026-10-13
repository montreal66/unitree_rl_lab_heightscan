"""Utilities for OmniRetarget HOI (Humanoid-Object Interaction) motion datasets."""

from unitree_rl_lab.utils.hoi.motion_loader import HOIMotionData, load_hoi_npz
from unitree_rl_lab.utils.hoi.path_resolver import AssetPaths, resolve_assets

__all__ = ["HOIMotionData", "load_hoi_npz", "AssetPaths", "resolve_assets"]
