import gymnasium as gym

_HOI_TERRAIN_ENV_ENTRY = "unitree_rl_lab.tasks.mimic.robots.g1_29dof.hoi_mimic_terrain.hoi_terrain_rl_env:HoiTerrainMimicRLEnv"

gym.register(
    id="Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast",
    entry_point=_HOI_TERRAIN_ENV_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.tracking_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg:HoiTerrainPerceptiveRaycastPpoRunnerCfg",
    },
)
