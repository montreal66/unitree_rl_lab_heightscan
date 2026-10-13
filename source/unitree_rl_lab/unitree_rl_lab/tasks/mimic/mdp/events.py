from __future__ import annotations

import json
from pathlib import Path
import torch
from typing import TYPE_CHECKING, Any, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.log
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _sample_material_buckets(
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    make_consistent: bool,
) -> torch.Tensor:
    range_list = [static_friction_range, dynamic_friction_range, restitution_range]
    ranges = torch.tensor(range_list, device="cpu")
    material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")
    if make_consistent:
        material_buckets[:, 1] = torch.min(material_buckets[:, 0], material_buckets[:, 1])
    return material_buckets


def _env_ids_for_physx_instance_count(
    view_count: int, env_ids: torch.Tensor | None
) -> torch.Tensor:
    """Map to PhysX instance rows (shared assets often have count 1 while num_envs is large)."""
    if view_count == 1:
        return torch.tensor([0], dtype=torch.long, device="cpu")
    if env_ids is None:
        return torch.arange(view_count, device="cpu")
    env_ids = env_ids.cpu()
    if len(env_ids) == view_count:
        return env_ids
    return torch.arange(view_count, device="cpu")


def _randomize_materials_on_physx_view(
    view: Any,
    env_ids_eff: torch.Tensor,
    material_buckets: torch.Tensor,
    num_buckets: int,
    num_shapes_per_body: list[int] | None,
    asset_cfg: SceneEntityCfg | None,
) -> None:
    """Same material assignment as :class:`isaaclab.envs.mdp.events.randomize_rigid_body_material` ``__call__``."""
    total_num_shapes = view.max_shapes
    bucket_ids = torch.randint(0, num_buckets, (len(env_ids_eff), total_num_shapes), device="cpu")
    material_samples = material_buckets[bucket_ids]
    materials = view.get_material_properties()
    if num_shapes_per_body is not None and asset_cfg is not None:
        for body_id in asset_cfg.body_ids:
            start_idx = sum(num_shapes_per_body[:body_id])
            end_idx = start_idx + num_shapes_per_body[body_id]
            materials[env_ids_eff, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
    else:
        materials[env_ids_eff] = material_samples[:]
    view.set_material_properties(materials, env_ids_eff)


def _create_physx_view_for_prim_path(physics_sim_view: Any, prim_path: str) -> Any | None:
    """Resolve a rigid-body or articulation PhysX view from a prim path expression."""
    path_glob = prim_path.replace(".*", "*") if ".*" in prim_path else prim_path
    candidates = [path_glob, f"{path_glob}/*"]
    for cand in candidates:
        prim = sim_utils.find_first_matching_prim(cand)
        if prim is None:
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            view = physics_sim_view.create_articulation_view(cand)
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            view = physics_sim_view.create_rigid_body_view(cand)
        else:
            continue
        if view is not None and getattr(view, "_backend", None) is not None:
            return view
    return None


def randomize_terrain_importer_rigid_body_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    asset_cfg: SceneEntityCfg,
    make_consistent: bool = False,
) -> None:
    """Randomize terrain material for ``TerrainImporter`` entities (e.g., ground plane)."""
    asset = env.scene[asset_cfg.name]
    if not isinstance(asset, TerrainImporter):
        raise TypeError(
            f"randomize_terrain_importer_rigid_body_material expects TerrainImporter, got {type(asset).__name__!r}"
        )

    material_buckets = _sample_material_buckets(
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        make_consistent,
    )
    physics_sim_view = SimulationManager.get_physics_sim_view()
    if physics_sim_view is None:
        raise RuntimeError("Physics simulation view is unavailable for terrain material randomization.")

    for prim_path in asset.terrain_prim_paths:
        view = _create_physx_view_for_prim_path(physics_sim_view, prim_path)
        if view is None:
            omni.log.warn(
                f"[randomize_terrain_importer_rigid_body_material] No PhysX view for terrain prim {prim_path!r}."
            )
            continue
        env_ids_eff = _env_ids_for_physx_instance_count(int(view.count), env_ids)
        _randomize_materials_on_physx_view(view, env_ids_eff, material_buckets, num_buckets, None, None)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("JointPositionAction")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range (index env_ids only; ``coms[:, ...]`` breaks partial env resample)
    coms[env_ids[:, None], body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def reset_fixed_box_from_hoi_metadata(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    metadata_file: str,
    pose_range: dict[str, tuple[float, float]] | None = None,
):
    """Reset a fixed HOI box around its trajectory initial pose.

    The metadata file is emitted by ``scripts/mimic/hoi_to_mimic_npz.py`` and stores:
    ``object_init_pos`` and ``object_init_quat`` (wxyz) from the source HOI trajectory.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if len(env_ids) == 0:
        return

    cache_key = str(Path(metadata_file).resolve())
    if not hasattr(reset_fixed_box_from_hoi_metadata, "_metadata_cache"):
        reset_fixed_box_from_hoi_metadata._metadata_cache = {}
    cache: dict[str, dict] = reset_fixed_box_from_hoi_metadata._metadata_cache
    if cache_key not in cache:
        with Path(metadata_file).open("r", encoding="utf-8") as f:
            cache[cache_key] = json.load(f)
    meta = cache[cache_key]

    base_pos = torch.tensor(meta["object_init_pos"], dtype=torch.float32, device=asset.device)
    base_quat = torch.tensor(meta["object_init_quat"], dtype=torch.float32, device=asset.device)

    ranges = pose_range or {}
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [ranges.get(key, (0.0, 0.0)) for key in keys]
    range_tensor = torch.tensor(range_list, dtype=torch.float32, device=asset.device)
    rand = math_utils.sample_uniform(range_tensor[:, 0], range_tensor[:, 1], (len(env_ids), 6), device=asset.device)

    pos = base_pos.unsqueeze(0).repeat(len(env_ids), 1)
    pos += env.scene.env_origins[env_ids].to(device=asset.device, dtype=pos.dtype)
    pos += rand[:, :3]

    delta_quat = math_utils.quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
    quat = math_utils.quat_mul(delta_quat, base_quat.unsqueeze(0).repeat(len(env_ids), 1))

    root_state = asset.data.default_root_state[env_ids].clone()
    root_state[:, :3] = pos
    root_state[:, 3:7] = quat
    root_state[:, 7:13] = 0.0

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)
    asset.data.default_root_state[env_ids] = root_state


def reset_fixed_terrain_from_hoi_metadata(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    metadata_file: str,
    pose_range: dict[str, tuple[float, float]] | None = None,
):
    """Reset HOI terrain pose from metadata, offset by each env's ``scene.env_origins`` (world frame)."""
    asset = env.scene[asset_cfg.name]
    device = getattr(asset, "device", getattr(env, "device", "cpu"))

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    if len(env_ids) == 0:
        return

    cache_key = str(Path(metadata_file).resolve())
    if not hasattr(reset_fixed_terrain_from_hoi_metadata, "_metadata_cache"):
        reset_fixed_terrain_from_hoi_metadata._metadata_cache = {}
    cache: dict[str, dict] = reset_fixed_terrain_from_hoi_metadata._metadata_cache
    if cache_key not in cache:
        with Path(metadata_file).open("r", encoding="utf-8") as f:
            cache[cache_key] = json.load(f)
    meta = cache[cache_key]

    base_pos = torch.tensor(meta["terrain_init_pos"], dtype=torch.float32, device=device)
    base_quat = torch.tensor(meta["terrain_init_quat"], dtype=torch.float32, device=device)

    ranges = pose_range or {}
    keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list = [ranges.get(key, (0.0, 0.0)) for key in keys]
    range_tensor = torch.tensor(range_list, dtype=torch.float32, device=device)
    rand = math_utils.sample_uniform(range_tensor[:, 0], range_tensor[:, 1], (len(env_ids), 6), device=device)

    pos = base_pos.unsqueeze(0).repeat(len(env_ids), 1)
    pos += env.scene.env_origins[env_ids].to(device=device, dtype=pos.dtype)
    pos += rand[:, :3]

    delta_quat = math_utils.quat_from_euler_xyz(rand[:, 3], rand[:, 4], rand[:, 5])
    quat = math_utils.quat_mul(delta_quat, base_quat.unsqueeze(0).repeat(len(env_ids), 1))

    if hasattr(asset, "write_root_state_to_sim") and hasattr(asset, "data"):
        # Articulation / RigidObject style API
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] = pos
        root_state[:, 3:7] = quat
        root_state[:, 7:13] = 0.0
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        asset.data.default_root_state[env_ids] = root_state
        return

    # XFormPrim style API (used by fixed terrain spawned as AssetBaseCfg)
    if hasattr(asset, "set_world_poses"):
        # XFormPrim branch.
        prims = getattr(asset, "_prims", None)
        prim_count = len(prims) if prims is not None else 0
        if prim_count <= 0:
            return

        # If terrain is shared (single prim), randomize once and keep fixed afterwards.
        if prim_count == 1:
            if not hasattr(reset_fixed_terrain_from_hoi_metadata, "_shared_terrain_done"):
                reset_fixed_terrain_from_hoi_metadata._shared_terrain_done = {}
            shared_done: dict[str, bool] = reset_fixed_terrain_from_hoi_metadata._shared_terrain_done
            if shared_done.get(cache_key, False):
                return
            asset.set_world_poses(positions=pos[:1], orientations=quat[:1])
            shared_done[cache_key] = True
            return

        # Multi-prim terrain (one per env): update only valid indices.
        env_ids_long = env_ids.to(dtype=torch.long)
        valid_mask = (env_ids_long >= 0) & (env_ids_long < prim_count)
        if not torch.any(valid_mask):
            return

        valid_indices = env_ids_long[valid_mask]
        pos_valid = pos[valid_mask]
        quat_valid = quat[valid_mask]
        asset.set_world_poses(positions=pos_valid, orientations=quat_valid, indices=valid_indices)
        return

    raise TypeError(f"Unsupported terrain asset type for reset: {type(asset).__name__}")
