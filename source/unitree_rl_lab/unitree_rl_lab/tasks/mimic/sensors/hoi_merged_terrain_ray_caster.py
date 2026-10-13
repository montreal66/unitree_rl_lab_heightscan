from __future__ import annotations

import math
import os

import numpy as np
import omni.log
import omni.usd
import torch
import trimesh
from pxr import UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh

from unitree_rl_lab.tasks.mimic.mdp.hoi_height_scan import _load_boxes


def compute_ground_plane_size_for_grid(
    num_envs: int,
    env_spacing: float,
    scan_size_xy: tuple[float, float],
    margin: float = 1.0,
) -> tuple[float, float]:
    """Ground plane size (m) for a centered GridCloner env layout (same logic as Isaac Sim).

    The plane from ``make_plane(..., center_zero=True)`` is centered at the world origin, matching
    how ``GridCloner`` distributes ``/World/envs/env_*`` origins.
    """
    if num_envs <= 0:
        raise ValueError(f"num_envs must be > 0, got {num_envs}")

    num_per_row = int(math.sqrt(num_envs))
    num_rows = int(math.ceil(num_envs / num_per_row))
    num_cols = int(math.ceil(num_envs / num_per_row))

    grid_extent_x = max(0.0, float(num_rows - 1) * env_spacing)
    grid_extent_y = max(0.0, float(num_cols - 1) * env_spacing)

    scan_half_x = 0.5 * float(scan_size_xy[0])
    scan_half_y = 0.5 * float(scan_size_xy[1])

    half_x = 0.5 * grid_extent_x + scan_half_x + float(margin)
    half_y = 0.5 * grid_extent_y + scan_half_y + float(margin)
    return (2.0 * half_x, 2.0 * half_y)


def compute_ground_plane_size_per_env(
    scan_size_xy: tuple[float, float],
    margin: float = 1.0,
    motion_radius: float = 6.0,
) -> tuple[float, float]:
    """Ground patch size for one env, centered on that env's origin.

    The height-scan grid is anchored to ``torso_link``, which moves with the motion clip.
    ``motion_radius`` must cover max anchor/root travel relative to the env origin.
    """
    half_x = 0.5 * float(scan_size_xy[0]) + float(margin) + float(motion_radius)
    half_y = 0.5 * float(scan_size_xy[1]) + float(margin) + float(motion_radius)
    return (2.0 * half_x, 2.0 * half_y)


class HoiMergedTerrainRayCaster(RayCaster):
    """RayCaster that raycasts against a merged static HOI terrain mesh."""

    def _initialize_warp_meshes(self):
        # Defer warp-mesh baking until terrain is placed (first ``ensure_mesh_baked`` / sensor update).
        # ``RayCaster._update_buffers_impl`` reads this class-level cache directly.
        # Do not shadow it with an instance attribute or evict unrelated sensors' meshes.
        for mesh_key in self.cfg.mesh_prim_paths:
            RayCaster.meshes.pop(mesh_key, None)
        self._hoi_mesh_baked = False

    def invalidate_mesh(self) -> None:
        """Drop the baked mesh so the next update rebuilds from current terrain poses."""
        self._hoi_mesh_baked = False
        for mesh_key in self.cfg.mesh_prim_paths:
            RayCaster.meshes.pop(mesh_key, None)

    def ensure_mesh_baked(self) -> None:
        """Bake the merged terrain mesh once if not already baked."""
        if not getattr(self, "_hoi_mesh_baked", False):
            self._bake_hoi_merged_mesh()

    def _apply_ground_on_miss(self, env_ids) -> None:
        if not getattr(self.cfg, "fill_ground_on_miss", False):
            return
        ground_z = float(getattr(self.cfg, "ground_z", 0.0))
        hit_z = self._data.ray_hits_w[env_ids, :, 2]
        miss = ~torch.isfinite(hit_z)
        self._data.ray_hits_w[env_ids, :, 2] = torch.where(
            miss, torch.full_like(hit_z, ground_z), hit_z
        )

    def _update_buffers_impl(self, env_ids):
        self.ensure_mesh_baked()
        super()._update_buffers_impl(env_ids)
        self._apply_ground_on_miss(env_ids)

    def _resolve_scan_size_xy(self) -> tuple[float, float]:
        pattern = self.cfg.pattern_cfg
        size = getattr(pattern, "size", (1.6, 1.0))
        return (float(size[0]), float(size[1]))

    def _resolve_ground_plane_size(self) -> tuple[float, float]:
        if getattr(self.cfg, "auto_ground_plane_size", True):
            num_envs = int(self._view.count)
            env_spacing = float(getattr(self.cfg, "env_spacing", 4.5))
            margin = float(getattr(self.cfg, "ground_plane_margin", 1.0))
            return compute_ground_plane_size_for_grid(
                num_envs=num_envs,
                env_spacing=env_spacing,
                scan_size_xy=self._resolve_scan_size_xy(),
                margin=margin,
            )
        return tuple(getattr(self.cfg, "ground_plane_size", (8.0, 8.0)))

    def _append_ground_plane_mesh(
        self,
        vertices_parts: list[np.ndarray],
        indices_parts: list[np.ndarray],
        vertex_offset: int,
        *,
        plane_size: tuple[float, float],
        center_xy: tuple[float, float],
    ) -> int:
        ground_z = float(getattr(self.cfg, "ground_z", 0.0))
        ground_mesh = make_plane(size=plane_size, height=ground_z, center_zero=True)
        verts = np.asarray(ground_mesh.vertices, dtype=np.float32).copy()
        verts[:, 0] += float(center_xy[0])
        verts[:, 1] += float(center_xy[1])
        faces = np.asarray(ground_mesh.faces, dtype=np.int32)
        vertices_parts.append(verts)
        indices_parts.append(faces + vertex_offset)
        return vertex_offset + verts.shape[0]

    def _collect_ground_plane_meshes(
        self,
        terrain_paths: list[str],
        stage,
        vertices_parts: list[np.ndarray],
        indices_parts: list[np.ndarray],
        vertex_offset: int,
    ) -> tuple[int, tuple[float, float], int]:
        per_env = bool(getattr(self.cfg, "ground_plane_per_env", True))
        scan_size = self._resolve_scan_size_xy()
        margin = float(getattr(self.cfg, "ground_plane_margin", 1.0))
        motion_radius = float(getattr(self.cfg, "ground_plane_motion_radius", 6.0))

        if per_env:
            plane_size = compute_ground_plane_size_per_env(
                scan_size_xy=scan_size, margin=margin, motion_radius=motion_radius
            )
            num_planes = 0
            for terrain_path in terrain_paths:
                root_prim = stage.GetPrimAtPath(terrain_path)
                if not root_prim.IsValid():
                    omni.log.warn(f"HOI RayCaster: skipping invalid terrain prim {terrain_path}")
                    continue
                root_matrix = np.array(omni.usd.get_world_transform_matrix(root_prim), dtype=np.float64).T
                center_xy = (float(root_matrix[0, 3]), float(root_matrix[1, 3]))
                vertex_offset = self._append_ground_plane_mesh(
                    vertices_parts,
                    indices_parts,
                    vertex_offset,
                    plane_size=plane_size,
                    center_xy=center_xy,
                )
                num_planes += 1
            return vertex_offset, plane_size, num_planes

        plane_size = self._resolve_ground_plane_size()
        vertex_offset = self._append_ground_plane_mesh(
            vertices_parts,
            indices_parts,
            vertex_offset,
            plane_size=plane_size,
            center_xy=(0.0, 0.0),
        )
        return vertex_offset, plane_size, 1

    def _resolve_metadata_file(self) -> str:
        path = getattr(self.cfg, "metadata_file", "") or os.getenv("HOI_MIMIC_TERRAIN_META_FILE", "")
        if not path:
            raise RuntimeError(
                "HoiMergedTerrainRayCaster requires metadata_file in cfg or HOI_MIMIC_TERRAIN_META_FILE."
            )
        return path

    def _resolve_terrain_prim_path_expr(self) -> str:
        """Resolve ``{ENV_REGEX_NS}`` to a global USD regex (e.g. ``/World/envs/env_.*``)."""
        expr = self.cfg.terrain_prim_path
        if "{ENV_REGEX_NS}" not in expr:
            if not expr.startswith("/"):
                raise ValueError(f"terrain_prim_path must be global, got: {expr}")
            return expr

        if self.cfg.prim_path.startswith("/"):
            env_regex_ns = self.cfg.prim_path.rsplit("/", 2)[0]
        else:
            env_regex_ns = "/World/envs/env_.*"
        resolved = expr.format(ENV_REGEX_NS=env_regex_ns)
        if not resolved.startswith("/"):
            raise ValueError(f"Resolved terrain_prim_path is not global: {resolved}")
        return resolved

    @staticmethod
    def _pose_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
        rot = math_utils.matrix_from_quat(torch.from_numpy(quat_wxyz.astype(np.float32)).unsqueeze(0))[0].numpy()
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = rot
        mat[:3, 3] = pos
        return mat

    def _append_box_mesh(
        self,
        vertices_parts: list[np.ndarray],
        indices_parts: list[np.ndarray],
        vertex_offset: int,
        half_size: np.ndarray,
        world_matrix: np.ndarray,
    ) -> int:
        extents = (2.0 * half_size).astype(np.float64)
        box = trimesh.creation.box(extents=extents)
        box.apply_transform(world_matrix)
        verts = np.asarray(box.vertices, dtype=np.float32)
        faces = np.asarray(box.faces, dtype=np.int32)
        vertices_parts.append(verts)
        indices_parts.append(faces + vertex_offset)
        return vertex_offset + verts.shape[0]

    def _collect_mjcf_box_meshes(
        self, metadata_file: str, terrain_paths: list[str], stage
    ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        box_pos, box_quat, box_half = _load_boxes(metadata_file, torch.device("cpu"))
        box_pos_np = box_pos.numpy()
        box_quat_np = box_quat.numpy()
        box_half_np = box_half.numpy()
        num_boxes = box_pos_np.shape[0]

        vertices_parts: list[np.ndarray] = []
        indices_parts: list[np.ndarray] = []
        vertex_offset = 0
        mesh_count = 0

        for terrain_path in terrain_paths:
            root_prim = stage.GetPrimAtPath(terrain_path)
            if not root_prim.IsValid():
                omni.log.warn(f"HOI RayCaster: skipping invalid terrain prim {terrain_path}")
                continue
            root_matrix = np.array(omni.usd.get_world_transform_matrix(root_prim), dtype=np.float64).T

            for box_idx in range(num_boxes):
                local_matrix = self._pose_to_matrix(box_pos_np[box_idx], box_quat_np[box_idx])
                world_matrix = root_matrix @ local_matrix
                vertex_offset = self._append_box_mesh(
                    vertices_parts, indices_parts, vertex_offset, box_half_np[box_idx], world_matrix
                )
                mesh_count += 1

        return vertices_parts, indices_parts, mesh_count

    def _collect_usd_mesh_meshes(
        self, terrain_paths: list[str], stage
    ) -> tuple[list[np.ndarray], list[np.ndarray], int]:
        vertices_parts: list[np.ndarray] = []
        indices_parts: list[np.ndarray] = []
        vertex_offset = 0
        mesh_count = 0

        for terrain_path in terrain_paths:
            mesh_prims = sim_utils.get_all_matching_child_prims(
                terrain_path, predicate=lambda prim: prim.GetTypeName() == "Mesh", stage=stage
            )
            for mesh_prim in mesh_prims:
                usd_mesh = UsdGeom.Mesh(mesh_prim)
                points = np.asarray(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
                indices = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                if points.size == 0 or indices.size == 0:
                    continue
                transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim), dtype=np.float32).T
                points = np.matmul(points, transform_matrix[:3, :3].T)
                points += transform_matrix[:3, 3]

                vertices_parts.append(points)
                indices_parts.append(indices + vertex_offset)
                vertex_offset += points.shape[0]
                mesh_count += 1

        return vertices_parts, indices_parts, mesh_count

    def _bake_hoi_merged_mesh(self) -> None:
        stage = sim_utils.get_current_stage()
        terrain_path_expr = self._resolve_terrain_prim_path_expr()
        terrain_paths = sim_utils.find_matching_prim_paths(terrain_path_expr, stage=stage)
        if len(terrain_paths) == 0:
            raise RuntimeError(
                f"No HOI terrain prim matched terrain_prim_path={terrain_path_expr} for RayCaster."
            )

        use_mjcf = bool(getattr(self.cfg, "use_mjcf_boxes_mesh", True))
        if use_mjcf:
            metadata_file = self._resolve_metadata_file()
            vertices_parts, indices_parts, mesh_count = self._collect_mjcf_box_meshes(
                metadata_file, terrain_paths, stage
            )
        else:
            vertices_parts, indices_parts, mesh_count = self._collect_usd_mesh_meshes(terrain_paths, stage)

        if mesh_count == 0:
            raise RuntimeError(
                "No HOI terrain geometry collected for RayCaster. "
                f"use_mjcf_boxes_mesh={use_mjcf}, terrain_prim_path={self.cfg.terrain_prim_path}"
            )

        vertex_offset = sum(v.shape[0] for v in vertices_parts)

        plane_size = None
        num_ground_planes = 0
        if self.cfg.include_ground_plane:
            vertex_offset, plane_size, num_ground_planes = self._collect_ground_plane_meshes(
                terrain_paths, stage, vertices_parts, indices_parts, vertex_offset
            )

        merged_vertices = np.concatenate(vertices_parts, axis=0)
        merged_indices = np.concatenate(indices_parts, axis=0)
        mesh_key = self.cfg.mesh_prim_paths[0]
        RayCaster.meshes[mesh_key] = convert_to_warp_mesh(merged_vertices, merged_indices, device=self.device)
        self._hoi_mesh_baked = True
        z_min = float(merged_vertices[:, 2].min())
        z_max = float(merged_vertices[:, 2].max())
        plane_info = ""
        if plane_size is not None:
            mode = "per_env" if getattr(self.cfg, "ground_plane_per_env", True) else "global"
            plane_info = (
                f", ground_plane={plane_size[0]:.2f}x{plane_size[1]:.2f} m x{num_ground_planes} ({mode})"
            )
        omni.log.info(
            "Baked merged HOI RayCaster mesh: "
            f"terrain_prims={len(terrain_paths)}, mesh_prims={mesh_count}, "
            f"vertices={len(merged_vertices)}, faces={len(merged_indices) // 3}, z=[{z_min:.3f}, {z_max:.3f}]"
            f"{plane_info}."
        )
