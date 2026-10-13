"""Verify that a converted mimic NPZ matches the HOI training articulation order.

This is a deterministic FK check, not a policy rollout.  It uses the same G1
USD and the same ``body_names -> robot.find_bodies(..., preserve_order=True)``
lookup as the HOI terrain training command, writes selected NPZ frames into
PhysX, and compares the resulting tracked-body poses to the NPZ values.

Example:
    ./isaaclab.sh -p scripts/mimic/verify_training_motion_index.py \
      --file logs/hoi_mimic_data/climb_15_z_scale_1.0_mimic.npz --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

# This machine also has a different ``unitree_rl_lab`` checkout installed.
# The check must inspect the checkout that owns this script, exactly as the
# training command should do.
PROJECT_SOURCE = Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"
sys.path.insert(0, str(PROJECT_SOURCE))
# Isaac Sim may preload the other checkout through an extension.  Drop only
# those conflicting package modules before importing this checkout's robot cfg.
for module_name, module in list(sys.modules.items()):
    if module_name == "unitree_rl_lab" or module_name.startswith("unitree_rl_lab."):
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not str(module_file).startswith(str(PROJECT_SOURCE)):
            del sys.modules[module_name]


parser = argparse.ArgumentParser(description="Check HOI mimic NPZ joint/body indexing against the training robot.")
parser.add_argument("--file", required=True, help="Converted mimic NPZ to check.")
parser.add_argument(
    "--frames",
    type=int,
    nargs="*",
    default=None,
    help="Frame indexes to check (default: five frames spanning the clip).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG


TRACKED_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


@configclass
class CheckSceneCfg(InteractiveSceneCfg):
    """The HOI training robot asset, without terrain or controllers."""

    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def quat_angle_error(reference: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Sign-invariant quaternion angular distance in radians."""

    cosine_half_angle = torch.sum(reference * actual, dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(cosine_half_angle)


def main() -> int:
    data = np.load(args_cli.file)
    required = {"joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(f"Not a converted mimic NPZ; missing fields: {missing}")
    if data["joint_pos"].shape[1] != 29:
        raise ValueError(f"Expected 29 robot joints, got {data['joint_pos'].shape[1]}")

    fps = float(np.asarray(data["fps"]).item())
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / fps))
    scene = InteractiveScene(CheckSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    body_indexes, resolved_names = robot.find_bodies(TRACKED_BODY_NAMES, preserve_order=True)
    if resolved_names != TRACKED_BODY_NAMES:
        raise RuntimeError(f"Training tracked-body lookup changed: {resolved_names}")

    print("[INDEX] robot joint names:")
    for index, name in enumerate(robot.joint_names):
        print(f"  {index:2d}: {name}")
    print("[INDEX] HOI training body mapping (NPZ raw index == Isaac runtime index):")
    for name, index in zip(TRACKED_BODY_NAMES, body_indexes, strict=True):
        print(f"  {index:2d}: {name}")

    total_frames = int(data["joint_pos"].shape[0])
    frames = args_cli.frames or sorted({0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, total_frames - 1})
    invalid = [frame for frame in frames if frame < 0 or frame >= total_frames]
    if invalid:
        raise ValueError(f"Frame indexes out of range 0..{total_frames - 1}: {invalid}")

    device = sim.device
    joint_pos = torch.as_tensor(data["joint_pos"], dtype=torch.float32, device=device)
    joint_vel = torch.as_tensor(data["joint_vel"], dtype=torch.float32, device=device)
    body_pos = torch.as_tensor(data["body_pos_w"], dtype=torch.float32, device=device)
    body_quat = torch.as_tensor(data["body_quat_w"], dtype=torch.float32, device=device)
    body_lin_vel = torch.as_tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
    body_ang_vel = torch.as_tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
    body_index_tensor = torch.tensor(body_indexes, dtype=torch.long, device=device)

    all_position_errors: list[torch.Tensor] = []
    all_rotation_errors: list[torch.Tensor] = []
    for frame in frames:
        # Exactly the training command's reset convention: its first tracked
        # body is pelvis, selected through the runtime body index lookup.
        root_state = robot.data.default_root_state.clone()
        pelvis_index = body_indexes[0]
        root_state[:, :3] = body_pos[frame, pelvis_index]
        root_state[:, 3:7] = body_quat[frame, pelvis_index]
        root_state[:, 7:10] = body_lin_vel[frame, pelvis_index]
        root_state[:, 10:13] = body_ang_vel[frame, pelvis_index]
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_pos[frame : frame + 1], joint_vel[frame : frame + 1])

        # This matches the converter which produced valid mimic NPZ files:
        # update the articulation transforms without advancing physics.
        sim.render()
        scene.update(sim.get_physics_dt())

        expected_pos = body_pos[frame, body_index_tensor]
        expected_quat = body_quat[frame, body_index_tensor]
        actual_pos = robot.data.body_pos_w[0, body_index_tensor]
        actual_quat = robot.data.body_quat_w[0, body_index_tensor]
        position_error = torch.linalg.vector_norm(expected_pos - actual_pos, dim=-1)
        rotation_error = quat_angle_error(expected_quat, actual_quat)
        all_position_errors.append(position_error)
        all_rotation_errors.append(rotation_error)

        print(
            f"[FRAME {frame:3d}] max position={position_error.max().item():.6g} m, "
            f"max rotation={rotation_error.max().item():.6g} rad"
        )

    position_errors = torch.stack(all_position_errors)
    rotation_errors = torch.stack(all_rotation_errors)
    max_position_error = position_errors.max().item()
    max_rotation_error = rotation_errors.max().item()
    print(
        "[RESULT] "
        f"max position={max_position_error:.6g} m, max rotation={max_rotation_error:.6g} rad "
        f"over {len(frames)} frames × {len(TRACKED_BODY_NAMES)} training bodies."
    )
    # Converter/replay run the same USD, so millimetre accuracy is expected.
    # Float32 quaternion dot products around identity quantize to about
    # 1.5 mrad after acos; a permutation produces errors orders larger.
    if max_position_error > 1.0e-3 or max_rotation_error > 2.0e-3:
        print("[FAIL] NPZ poses do not match the HOI training articulation convention.")
        return 1
    print("[PASS] NPZ joint and tracked-body indexes match the HOI training articulation convention.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
