"""Load OmniRetarget HOI trajectories from ``.npz`` files.

Each file contains ``fps`` and ``qpos`` of shape ``(T, D)`` where ``D`` is 36 (robot only)
or 43 (robot + free object). Layout per row:

- **0:7** — floating base quaternion ``[qw, qx, qy, qz]`` then position ``[x, y, z]``
- **7:36** — 29 G1 joint positions (same order as ``joint_sdk_names`` in
  ``unitree_rl_lab.assets.robots.unitree`` / the dataset URDF)
- **36:43** (optional) — object pose ``[qw, qx, qy, qz, x, y, z]``

**Mimic integration (TODO):** Mimic's ``MotionLoader`` expects per-body world poses and
velocities (``body_pos_w``, ``body_quat_w``, …). To convert HOI ``qpos`` for mimic:

1. Spawn the same G1 URDF in Isaac Lab (or use FK offline).
2. For each timestep, set articulation joint positions + root, run forward kinematics /
   read link poses from the simulator.
3. Pack arrays in the mimic npz schema (see ``scripts/mimic/csv_to_npz.py``).

This module only parses ``qpos`` and finite-differences velocities for kinematic replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HOIMotionData:
    """Decoded HOI trajectory tensors (CPU float32 unless otherwise noted)."""

    fps: float
    #: ``(T, 3)`` root position (world).
    root_pos: torch.Tensor
    #: ``(T, 4)`` root quaternion ``wxyz`` (world).
    root_quat: torch.Tensor
    #: ``(T, 29)`` joint positions in URDF / ``joint_sdk_names`` order.
    joint_pos: torch.Tensor
    #: ``(T, 29)`` joint velocities (central difference in time).
    joint_vel: torch.Tensor
    #: ``(T, 3)`` root linear velocity (finite difference of position).
    root_lin_vel: torch.Tensor
    #: ``(T, 3)`` root angular velocity (minimal placeholder; see ``view_retargeted_npz``).
    root_ang_vel: torch.Tensor
    #: Whether ``qpos`` included object channels (43D).
    has_object: bool
    #: ``(T, 3)`` object position if ``has_object``, else empty tensor.
    object_pos: torch.Tensor
    #: ``(T, 4)`` object quaternion ``wxyz`` if ``has_object``, else empty tensor.
    object_quat: torch.Tensor
    #: ``(T, 3)`` object linear velocity if ``has_object``.
    object_lin_vel: torch.Tensor
    #: ``(T, 3)`` object angular velocity placeholder if ``has_object``.
    object_ang_vel: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])


def _finite_diff(values: np.ndarray, dt: float) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, dt, axis=0).astype(np.float32)


def load_hoi_npz(file_path: str, device: str | torch.device = "cpu") -> HOIMotionData:
    """Load ``fps`` and ``qpos`` from a HOI ``.npz`` and build ``HOIMotionData``."""
    data = np.load(file_path, allow_pickle=True)
    if "qpos" not in data.files or "fps" not in data.files:
        raise ValueError(f"HOI npz must contain 'qpos' and 'fps'. Got keys: {sorted(data.files)}")

    qpos = np.asarray(data["qpos"], dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] not in (36, 43):
        raise ValueError(f"Expected qpos shape (T, 36) or (T, 43); got {qpos.shape}")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    dt = 1.0 / max(fps, 1e-6)
    has_object = qpos.shape[1] == 43

    quat_wxyz = qpos[:, 0:4]
    root_pos = qpos[:, 4:7]
    joint_pos = qpos[:, 7:36]

    root_lin_vel = _finite_diff(root_pos, dt)
    joint_vel = _finite_diff(joint_pos, dt)
    root_ang_vel = np.zeros((qpos.shape[0], 3), dtype=np.float32)

    t = qpos.shape[0]
    if has_object:
        obj_quat = qpos[:, 36:40]
        obj_pos = qpos[:, 40:43]
        object_lin_vel = _finite_diff(obj_pos, dt)
        object_ang_vel = np.zeros((t, 3), dtype=np.float32)
    else:
        obj_quat = np.zeros((t, 4), dtype=np.float32)
        obj_quat[:, 0] = 1.0
        obj_pos = np.zeros((t, 3), dtype=np.float32)
        object_lin_vel = np.zeros((t, 3), dtype=np.float32)
        object_ang_vel = np.zeros((t, 3), dtype=np.float32)

    dev = torch.device(device)
    return HOIMotionData(
        fps=fps,
        root_pos=torch.from_numpy(root_pos).to(dev),
        root_quat=torch.from_numpy(quat_wxyz).to(dev),
        joint_pos=torch.from_numpy(joint_pos).to(dev),
        joint_vel=torch.from_numpy(joint_vel).to(dev),
        root_lin_vel=torch.from_numpy(root_lin_vel).to(dev),
        root_ang_vel=torch.from_numpy(root_ang_vel).to(dev),
        has_object=has_object,
        object_pos=torch.from_numpy(obj_pos).to(dev),
        object_quat=torch.from_numpy(obj_quat).to(dev),
        object_lin_vel=torch.from_numpy(object_lin_vel).to(dev),
        object_ang_vel=torch.from_numpy(object_ang_vel).to(dev),
    )
