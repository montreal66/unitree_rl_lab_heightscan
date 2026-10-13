"""RL env subclass for HOI terrain mimic: reset ordering avoids terrain–robot interpenetration."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from isaaclab.envs import ManagerBasedRLEnv


class HoiTerrainMimicRLEnv(ManagerBasedRLEnv):
    """Runs motion ``command`` reset before ``mode='reset'`` events (terrain pose).

    The default :class:`ManagerBasedRLEnv` applies reset events before
    :meth:`CommandManager.reset`, so the terrain jumps to the metadata pose while the
    robot is still at the previous episode state for the rest of the reset pipeline.
    That one-frame mismatch can cause visible penetration and unstable contact impulses.
    Here the robot is teleported to the resampled motion frame first, then the HOI
    terrain is placed to match the same clip layout.

    After writing robot/terrain root states, we call :meth:`scene.write_data_to_sim` and
    :meth:`sim.forward` (same as :meth:`ManagerBasedEnv.reset`). The default
    :meth:`ManagerBasedRLEnv.step` path does **not** do this after mid-rollout resets, so
    the viewer/fabric can briefly show stale poses until the next physics substep.
    """

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ):
        obs, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        # Guarantee warp mesh exists after the first full reset + observation pass.
        self._ensure_hoi_raycaster_mesh_baked()
        return obs, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        self.curriculum_manager.compute(env_ids=env_ids)
        self.scene.reset(env_ids)

        self.extras["log"] = dict()
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)

        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        self.episode_length_buf[env_ids] = 0

        # Keep USD / fabric in sync with PhysX after teleport (step() skips this; gym reset repeats it).
        self.scene.write_data_to_sim()
        self.sim.forward()
        if self._hoi_raycaster_rebake_on_reset():
            self._invalidate_hoi_raycaster_mesh()

    def _hoi_raycaster_rebake_on_reset(self) -> bool:
        sensor = self.scene.sensors.get("height_scanner")
        if sensor is None:
            return False
        return bool(getattr(sensor.cfg, "rebake_on_reset", False))

    def _invalidate_hoi_raycaster_mesh(self) -> None:
        """Drop baked mesh so the next sensor update rebuilds (only when ``rebake_on_reset``)."""
        sensor = self.scene.sensors.get("height_scanner")
        if sensor is not None and hasattr(sensor, "invalidate_mesh"):
            sensor.invalidate_mesh()

    def _ensure_hoi_raycaster_mesh_baked(self) -> None:
        sensor = self.scene.sensors.get("height_scanner")
        if sensor is not None and hasattr(sensor, "ensure_mesh_baked"):
            sensor.ensure_mesh_baked()

    def _maybe_enable_hoi_height_scan_vis(self) -> None:
        if hasattr(self, "_hoi_height_scan_vis_checked"):
            return
        self._hoi_height_scan_vis_checked = True
        raw = os.getenv("HOI_HEIGHT_SCAN_DEBUG_VIS", "0").strip().lower()
        enabled = raw in ("1", "true", "yes", "on")
        if not enabled:
            self._hoi_height_scan_vis = None
            return

        env_id_raw = os.getenv("HOI_HEIGHT_SCAN_DEBUG_ENV_ID", "0").strip()
        try:
            env_id = int(env_id_raw)
        except ValueError:
            env_id = 0

        from unitree_rl_lab.tasks.mimic.mdp import HoiHeightScanDebugVis

        self._hoi_height_scan_vis = HoiHeightScanDebugVis(env_id=env_id)
        self._hoi_height_scan_vis.enable(self)
        print(f"[INFO] HOI height scan debug visualization enabled (env_id={env_id}).")

    def step(self, action):
        self._maybe_enable_hoi_height_scan_vis()
        out = super().step(action)
        if "invalid_robot_state" in self.termination_manager.active_terms:
            invalid = self.termination_manager.get_term("invalid_robot_state")
            if invalid.any():
                out[1][invalid] = 0.0
        vis = getattr(self, "_hoi_height_scan_vis", None)
        if vis is not None:
            vis.update(self)
        return out
