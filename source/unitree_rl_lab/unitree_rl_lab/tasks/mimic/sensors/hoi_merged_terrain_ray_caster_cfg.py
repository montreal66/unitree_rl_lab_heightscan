from __future__ import annotations

from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils import configclass

from .hoi_merged_terrain_ray_caster import HoiMergedTerrainRayCaster


@configclass
class HoiMergedTerrainRayCasterCfg(RayCasterCfg):
    """RayCaster config that bakes all HOI terrain meshes into one static warp mesh."""

    class_type: type = HoiMergedTerrainRayCaster
    # Use global regex directly; ``{ENV_REGEX_NS}`` is also supported and resolved at bake time.
    terrain_prim_path: str = "/World/envs/env_.*/HOI_Terrain"
    # HOI terrain metadata (``mjcf_boxes``); empty string falls back to ``HOI_MIMIC_TERRAIN_META_FILE``.
    metadata_file: str = ""
    # Build warp mesh from ``mjcf_boxes`` (matches ``mdp.hoi_height_scan``). If False, bake USD Mesh prims.
    use_mjcf_boxes_mesh: bool = True
    # Bake a small ground quad (centered at world origin) into the warp mesh for ray hits outside boxes.
    include_ground_plane: bool = True
    # Must match ``InteractiveSceneCfg.env_spacing`` (synced in RobotEnvCfg.__post_init__ when possible).
    env_spacing: float = 4.5
    # Extra margin (m) beyond scan half-extent for each env ground patch.
    ground_plane_margin: float = 1.0
    # Max expected root/torso travel (m) from env origin during a motion clip (scan follows torso).
    ground_plane_motion_radius: float = 6.0
    # One ground quad per env at HOI_Terrain world XY (recommended). If False, single plane at world origin.
    ground_plane_per_env: bool = True
    # If ``ground_plane_per_env`` is False: derive global plane size from num_envs + env_spacing.
    auto_ground_plane_size: bool = True
    ground_plane_size: tuple[float, float] = (8.0, 8.0)
    ground_z: float = 0.0
    # Deprecated: use ``include_ground_plane`` + ``auto_ground_plane_size`` instead.
    fill_ground_on_miss: bool = False
    # Rebake warp mesh after each env reset. Keep False when terrain pose is fixed (default HOI setup).
    rebake_on_reset: bool = False
    # Placeholder key consumed by base RayCaster update path.
    mesh_prim_paths: list[str] = ["/World/ground"]
