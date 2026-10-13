from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import unitree_rl_lab.tasks.mimic.mdp as mdp
from unitree_rl_lab.tasks.mimic.robots.g1_29dof.hoi_mimic_terrain import tracking_env_cfg as blind_cfg
from unitree_rl_lab.tasks.mimic.sensors import HoiMergedTerrainRayCasterCfg


@configclass
class RobotSceneCfg(blind_cfg.RobotSceneCfg):
    height_scanner = HoiMergedTerrainRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.6]),
        terrain_prim_path="{ENV_REGEX_NS}/HOI_Terrain",
        metadata_file=blind_cfg.TERRAIN_META_FILE,
        use_mjcf_boxes_mesh=True,
        include_ground_plane=True,
        rebake_on_reset=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=blind_cfg.PROPRIO_HISTORY_LENGTH
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=blind_cfg.PROPRIO_HISTORY_LENGTH
        )
        last_action = ObsTerm(func=mdp.last_action, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)

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
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 5.0),
            history_length=blind_cfg.PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)
        actions = ObsTerm(func=mdp.last_action, history_length=blind_cfg.PROPRIO_HISTORY_LENGTH)

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


def _check_perceptive_config(cfg) -> None:
    """Fail early if the perceptive-task configuration is incomplete."""
    if cfg.scene.height_scanner is None:
        raise ValueError(
            "RobotSceneCfg.height_scanner must be configured for the perceptive RayCaster task."
        )
    if cfg.observations.policy.height_scanner is None:
        raise ValueError(
            "PolicyCfg.height_scanner must be configured for the perceptive RayCaster task."
        )
    if cfg.observations.critic.height_scanner is None:
        raise ValueError(
            "PrivilegedCfg.height_scanner must be configured for the perceptive RayCaster task."
        )


@configclass
class RobotEnvCfg(blind_cfg.RobotEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _check_perceptive_config(self)
        self.scene.height_scanner.env_spacing = float(self.scene.env_spacing)


@configclass
class RobotPlayEnvCfg(blind_cfg.RobotPlayEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=4.5)
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        _check_perceptive_config(self)
        self.scene.height_scanner.env_spacing = float(self.scene.env_spacing)
