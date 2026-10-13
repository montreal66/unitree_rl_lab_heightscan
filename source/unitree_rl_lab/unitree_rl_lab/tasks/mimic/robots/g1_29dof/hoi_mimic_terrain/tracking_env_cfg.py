from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_ACTION_SCALE
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG as ROBOT_CFG
from unitree_rl_lab.utils.hoi.assets import make_terrain_cfg
from . import curriculum_HOI as hoi_curriculum

VELOCITY_RANGE = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (-0.0, 0.0),
    "roll": (-0.00, 0.00),
    "pitch": (-0.00, 0.00),
    "yaw": (-0.00, 0.00),
}
# Final push DR target (used by curriculum terminal level).
PUSH_ROOT_VELOCITY_RANGE = dict(hoi_curriculum.PUSH_FINAL_VELOCITY_RANGE)


def _hoi_mimic_data_dir() -> Path:
    """Canonical directory for converted mimic npz + sidecar JSON.

    Override with env ``HOI_MIMIC_DATA_DIR``. Default: ``<unitree_project>/logs/hoi_mimic_data``
    when this file lives under ``unitree_project/source/...`` (parents[8]).
    See the repository-root ``HOI_MIMIC_RUN_GUIDE.md`` for the supported setup.
    """
    override = os.getenv("HOI_MIMIC_DATA_DIR")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    return (here.parents[8] / "logs" / "hoi_mimic_data").resolve()


_HOI_MIMIC_DATA_DIR = _hoi_mimic_data_dir()
_DEFAULT_MOTION = str(_HOI_MIMIC_DATA_DIR / "climb_15_z_scale_1.0_mimic.npz")
MOTION_FILE = os.getenv("HOI_MIMIC_TERRAIN_MOTION_FILE", _DEFAULT_MOTION)
_DEFAULT_META = str(_HOI_MIMIC_DATA_DIR / "climb_15_z_scale_1.0_mimic.terrain.json")
TERRAIN_META_FILE = os.getenv("HOI_MIMIC_TERRAIN_META_FILE", _DEFAULT_META)
MOTION_DEBUG_VIS = os.getenv("HOI_MIMIC_TERRAIN_DEBUG_VIS", "0").lower() in ("1", "true", "yes", "on")
_DEFAULT_HOI_ROOT = (Path(__file__).resolve().parents[8] / "datasets").resolve()
HOI_ROOT = Path(os.getenv("HOI_ROOT", str(_DEFAULT_HOI_ROOT))).resolve()
DEFAULT_TERRAIN_URDF = str((HOI_ROOT / "models" / "terrain" / "climb_15" / "multi_boxes_z_scale_1.0.urdf").resolve())
TERRAIN_URDF = os.getenv("HOI_MIMIC_TERRAIN_URDF", DEFAULT_TERRAIN_URDF)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Ground + robot + fixed HOI terrain scene for terrain mimic."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    hoi_terrain: AssetBaseCfg = make_terrain_cfg(TERRAIN_URDF, prim_path="{ENV_REGEX_NS}/HOI_Terrain")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
        # adaptive_uniform_ratio=0.6,
        # adaptive_kernel_size=3,
        # adaptive_lambda=0.9,
        # adaptive_alpha=0.001,

@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        # Generate from HOI terrain data (see the repository-root HOI_MIMIC_RUN_GUIDE.md):
        # python scripts/mimic/hoi_to_mimic_npz.py -f motion_dataset/HOI/robot-terrain/climb_15_z_scale_1.0.npz --output-dir logs/hoi_mimic_data
        motion_file=MOTION_FILE,
        anchor_body_name="torso_link",
        resampling_time_range=(1.0e9, 1.0e9),
        # Visualize current-vs-reference body/anchor frames when enabled.
        debug_vis=MOTION_DEBUG_VIS,
        pose_range={
            # Disable robot-root reset perturbation for terrain clips.
            # Terrain trajectories are tightly coupled to exact footholds.
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    # adaptive_kernel_size: int = 1
    # adaptive_lambda: float = 0.8
    # adaptive_uniform_ratio: float = 0.1
    # adaptive_alpha: float = 0.001
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.01, 0.01),
        adaptive_uniform_ratio=0.07,
        adaptive_lambda=0.8,
        adaptive_kernel_size=2,
        adaptive_alpha=0.002,
        body_names=[
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
        ],
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=UNITREE_G1_29DOF_MIMIC_ACTION_SCALE, use_default_offset=True
    )


PROPRIO_HISTORY_LENGTH = 8


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=PROPRIO_HISTORY_LENGTH
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=PROPRIO_HISTORY_LENGTH
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=PROPRIO_HISTORY_LENGTH
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=PROPRIO_HISTORY_LENGTH
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=PROPRIO_HISTORY_LENGTH)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=PROPRIO_HISTORY_LENGTH)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=PROPRIO_HISTORY_LENGTH)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=PROPRIO_HISTORY_LENGTH)
        actions = ObsTerm(func=mdp.last_action, history_length=PROPRIO_HISTORY_LENGTH)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            **hoi_curriculum.dr_level_physics_material_params(0),
        },
    )

    hoi_terrain_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("hoi_terrain", body_names=".*"),
            **hoi_curriculum.dr_level_hoi_terrain_physics_material_params(0),
        },
    )

    add_joint_default_pos_non_ankle = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["^(?!.*ankle).*$"]),
            "pos_distribution_params": hoi_curriculum.dr_level_joint_non_ankle_params(0),
            "operation": "add",
        },
    )
    add_joint_default_pos_ankle = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*ankle.*"]),
            "pos_distribution_params": hoi_curriculum.dr_level_joint_ankle_params(0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": hoi_curriculum.dr_level_base_com_range(0),
        },
    )

    reset_terrain_pose = EventTerm(
        func=mdp.reset_fixed_terrain_from_hoi_metadata,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("hoi_terrain"),
            "metadata_file": TERRAIN_META_FILE,
            "pose_range": {
                # Keep perturbation tiny to avoid robot/terrain interpenetration at reset.
                "x": (-0.00, 0.00),
                "y": (-0.00, 0.00),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.00, 0.00),
            },
        },
    )

    push_root_velocity = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=hoi_curriculum.INITIAL_PUSH_INTERVAL_S,
        params={"velocity_range": dict(hoi_curriculum.INITIAL_PUSH_VELOCITY_RANGE)},
    )


@configclass
class RewardsCfg:
    # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits_log1p,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    applied_torque_limits_by_ratio = RewTerm(
        func=mdp.applied_torque_limits_by_ratio_log1p,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*ankle.*",
                    ".*wrist.*",
                ],
            ),
            "limit_ratio": 0.9,
        },
    )

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5, # might need to be larger
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5, # might need to be larger
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_robot_state = DoneTerm(
        func=mdp.non_finite_robot_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Paper-style: |e_{p,z}| > 0.25 m on anchor (world z).
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    # Paper-style: ||log(R^d R^T)|| approximated by quat geodesic error > 0.8 rad (see bad_anchor_quat_error).
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_quat_error,
        params={"command_name": "motion", "threshold": 0.8},
    )
    # Paper-style: |e_{p,z}| > 0.25 m on EE bodies; uses commanded targets body_pos_relative_w (same as tracking rewards).

    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class HoiCurriculumCfg:
    """HOI terrain mimic curriculum knobs (level-up gating + DR application).

    Environment variables still override these fields when set (same names as before), so
    existing launch scripts keep working; use this block for defaults you want checked in
    version control.

    **Level-up criterion** (``level_up_criterion``):

    - ``metric_gate`` (default): streak advances only when both conditions hold on the last
      ``rolling_episode_length_window`` completed episodes:
      (1) mean episode length >= ``mean_episode_length_ratio_threshold * max_episode_length``
      (and optional absolute floor ``min_mean_episode_length``),
      (2) timeout ratio >= ``time_out_ratio_threshold``.
      This keeps the signal physically meaningful and aligns with RSL-RL-style completed-episode windows.

    Domain randomization terms updated by the curriculum can be toggled individually;
    ``enable_domain_randomization`` is a master switch (False disables every curriculum DR term).
    """

    # --- Level-up / gating ---
    enable: bool = True
    eval_steps: int | None = None
    """If None, defaults to ``max_episode_length // 2`` (computed when the env starts)."""

    level_up_criterion: str = "metric_gate"
    """Only ``metric_gate`` is supported."""

    rolling_episode_length_window: int = 100
    """Last N completed episodes for rolling mean (RSL-RL ``lenbuffer`` default is 100)."""

    required_windows: int = 3
    mean_episode_length_ratio_threshold: float = 0.85
    """Metric-gate: required fraction of ``max_episode_length`` for rolling mean length."""

    time_out_ratio_threshold: float = 0.80
    """Metric-gate: required timeout share over the same rolling completed-episode window."""

    min_mean_episode_length: int = 1000
    """Optional absolute floor on mean episode length (steps), in addition to the ratio threshold."""

    min_completed_samples: int = 32
    """Minimum finished episodes in the rolling window before trusting the mean/ratio gate."""

    # --- Curriculum-driven DR (physics / joints / CoM / HOI terrain) ---
    enable_domain_randomization: bool = True
    dr_physics_material: bool = True
    dr_joint_non_ankle: bool = True
    dr_joint_ankle: bool = True
    dr_base_com: bool = True
    dr_hoi_terrain_physics: bool = True
    """Randomize rigid-body material on ``hoi_terrain`` (see ``hoi_terrain_physics_material`` event)."""

    hoi_terrain_dr_scale_min: float = 0.15
    """Minimum curriculum scale for terrain rigid-body material (level 0); multiplied into ``curriculum_HOI._DR_HOI_TERRAIN_PHYSICS_BASE`` via ``_sym_interval``."""

    hoi_terrain_dr_scale_max: float = 1.35
    """Maximum terrain material scale (level N-1). Linear interpolation in between; same semantics as ``hoi_terrain_dr_scale_min``."""


def _hoi_resolve_startup_dr_flags(hc: HoiCurriculumCfg) -> tuple[bool, bool, bool, bool, bool, bool]:
    """(master, physics, joint_non_ankle, joint_ankle, base_com, hoi_terrain_physics). Environment variables override ``hc``."""

    raw_m = os.getenv("HOI_CURRICULUM_DR_ENABLE")
    if raw_m is not None:
        master = raw_m.lower() in ("1", "true", "yes", "on")
    else:
        master = bool(hc.enable_domain_randomization)

    def term(attr: str, env_key: str) -> bool:
        if not master:
            return False
        raw = os.getenv(env_key)
        if raw is not None:
            return raw.lower() in ("1", "true", "yes", "on")
        return bool(getattr(hc, attr))

    return (
        master,
        term("dr_physics_material", "HOI_CURRICULUM_DR_PHYSICS"),
        term("dr_joint_non_ankle", "HOI_CURRICULUM_DR_JOINT_NON_ANKLE"),
        term("dr_joint_ankle", "HOI_CURRICULUM_DR_JOINT_ANKLE"),
        term("dr_base_com", "HOI_CURRICULUM_DR_BASE_COM"),
        term("dr_hoi_terrain_physics", "HOI_CURRICULUM_DR_HOI_TERRAIN"),
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for HOI terrain mimic."""

    progress = CurrTerm(func=hoi_curriculum.hoi_curriculum_progress)
    push_level = CurrTerm(func=hoi_curriculum.hoi_curriculum_push_level)
    ee_threshold_level = CurrTerm(func=hoi_curriculum.hoi_curriculum_ee_threshold_level)
    ee_threshold_value = CurrTerm(func=hoi_curriculum.hoi_curriculum_ee_threshold_value)
    level_ready_score = CurrTerm(func=hoi_curriculum.hoi_curriculum_level_ready_score)
    agg_episode_length = CurrTerm(func=hoi_curriculum.hoi_curriculum_agg_episode_length)
    completed_sample_count = CurrTerm(func=hoi_curriculum.hoi_curriculum_completed_sample_count)
    time_out_ratio = CurrTerm(func=hoi_curriculum.hoi_curriculum_time_out_ratio)
    length_ratio_to_max = CurrTerm(func=hoi_curriculum.hoi_curriculum_length_ratio_to_max)
    dr_physics_static_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_physics_static_low)
    dr_physics_static_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_physics_static_high)
    dr_physics_dynamic_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_physics_dynamic_low)
    dr_physics_dynamic_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_physics_dynamic_high)
    dr_physics_restitution_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_physics_restitution_low)
    dr_physics_restitution_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_physics_restitution_high)
    dr_joint_non_ankle_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_joint_non_ankle_low)
    dr_joint_non_ankle_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_joint_non_ankle_high)
    dr_joint_ankle_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_joint_ankle_low)
    dr_joint_ankle_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_joint_ankle_high)
    dr_com_x_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_com_x_low)
    dr_com_x_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_com_x_high)
    dr_com_y_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_com_y_low)
    dr_com_y_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_com_y_high)
    dr_com_z_low = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_com_z_low)
    dr_com_z_high = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_com_z_high)
    # Kept for backward compatibility with existing dashboards.
    dr_scale = CurrTerm(func=hoi_curriculum.hoi_curriculum_dr_scale)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    hoi_curriculum: HoiCurriculumCfg = HoiCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_max_rigid_contact_count = 2**27
        self.sim.physx.gpu_collision_stack_size = 2**27
        hoi_curriculum.apply_hoi_terrain_dr_scales_from_min_max(
            float(self.hoi_curriculum.hoi_terrain_dr_scale_min),
            float(self.hoi_curriculum.hoi_terrain_dr_scale_max),
        )
        _, use_phys, use_jna, use_ja, use_com, use_hoi_terrain = _hoi_resolve_startup_dr_flags(self.hoi_curriculum)
        if not use_phys:
            self.events.physics_material.params.update(hoi_curriculum.frozen_physics_material_event_params())
        if not use_hoi_terrain:
            self.events.hoi_terrain_physics_material.params.update(hoi_curriculum.frozen_physics_material_event_params())
        else:
            self.events.hoi_terrain_physics_material.params.update(hoi_curriculum.dr_level_hoi_terrain_physics_material_params(0))
        if not use_jna:
            self.events.add_joint_default_pos_non_ankle.params["pos_distribution_params"] = (
                hoi_curriculum.frozen_joint_pos_distribution_params()
            )
        if not use_ja:
            self.events.add_joint_default_pos_ankle.params["pos_distribution_params"] = hoi_curriculum.frozen_joint_pos_distribution_params()
        if not use_com:
            self.events.base_com.params["com_range"] = hoi_curriculum.frozen_base_com_range()


class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        # Isaac Lab's spawn_ground_plane binds physics material to the first child prim typed "Plane".
        # On some Isaac Sim / USD builds that lookup returns None, and bind_physics_material crashes with
        # GetPrimAtPath(Stage, NoneType). Skip material binding on the ground prim for play only; keep the
        # same friction settings on SimulationCfg (set in RobotEnvCfg.__post_init__) for default contacts.
        terrain_mat = self.scene.terrain.physics_material
        self.scene.terrain = self.scene.terrain.replace(physics_material=None)
        super().__post_init__()
        self.sim.physics_material = terrain_mat
        self.scene.num_envs = 1
        # Training uses 2**27 contact/stack buffers for 4096+ envs (~10 GB prealloc). Play keeps num_envs=1
        # but inherited limits still OOM during sim.reset() (especially with headless video rendering).
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**26
        self.episode_length_s = 1e9
        self.commands.motion.motion_start_time_step = 0
        # Replay-only root translation for controlled terrain-offset tests.
        # MotionCommand applies this at reset and subsequently aligns the
        # reference body targets to the reset anchor, so the NPZ stays intact
        # while the complete robot/reference trajectory moves relative to the
        # fixed HOI terrain.  +Y is the climb_15 travel direction.
        play_root_y_offset = float(os.getenv("HOI_MIMIC_PLAY_ROOT_Y_OFFSET", "0.0"))
        self.commands.motion.pose_range["y"] = (play_root_y_offset, play_root_y_offset)
