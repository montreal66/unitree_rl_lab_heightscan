from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .hoi_height_scan import HoiHeightScanResult, compute_hoi_height_scan


def _build_params_key(params: dict) -> tuple:
    """Build a stable cache key from height-scan params."""
    return (
        str(Path(str(params["metadata_file"])).resolve()),
        str(params["terrain_asset_name"]),
        str(params["robot_asset_name"]),
        str(params["body_name"]),
        float(params["size_xy"][0]),
        float(params["size_xy"][1]),
        float(params["resolution"]),
        float(params["offset"]),
    )


def resolve_hoi_height_scan_params(env) -> dict:
    """Read height-scan parameters from env.cfg.observations.policy.height_scanner.

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Find `env.cfg.observations.policy.height_scanner`.
    2. Read its `params` dictionary.
    3. Check that `metadata_file` exists in params.
    4. Fill default values for terrain_asset_name, robot_asset_name, body_name,
       size_xy, resolution, and offset.
    5. Return a plain dictionary.
    """
    raise NotImplementedError("Legacy analytical height-scan visualization is not implemented.")


def get_hoi_height_scan_result(env, params: dict | None = None) -> HoiHeightScanResult:
    """Return cached scan result when possible; otherwise recompute it.

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Resolve params if params is None.
    2. Read `env._hoi_height_scan_debug`.
    3. If the cache has the same params_key and contains heights/hits_w, reuse it.
    4. Otherwise call `compute_hoi_height_scan(env=env, cache_on_env=True, **params)`.
    """
    raise NotImplementedError("Legacy analytical height-scan visualization is not implemented.")


class HoiHeightScanDebugVis:
    """Small helper that renders height-scan hit points in the Isaac viewer."""

    def __init__(
        self,
        *,
        env_id: int = 0,
        z_offset: float = 0.001,
        sphere_radius: float = 0.018,
    ):
        self.env_id = int(env_id)
        self.z_offset = float(z_offset)
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/HOI_HeightScan/hits",
            markers={
                "hit": sim_utils.SphereCfg(
                    radius=float(sphere_radius),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        self.visualizer = VisualizationMarkers(cfg)
        self.visualizer.set_visibility(False)
        self._enabled = False

    def enable(self, env=None) -> None:
        del env
        self._enabled = True
        self.visualizer.set_visibility(True)

    def disable(self) -> None:
        self._enabled = False
        self.visualizer.set_visibility(False)

    def update(self, env, env_id: int | None = None) -> None:
        """Draw hit points for one environment.

        Legacy analytical-scanner outline (not part of the RayCaster assignment):
        1. Return immediately if visualization is disabled.
        2. Get the latest HoiHeightScanResult.
        3. Clamp env_id to a valid range.
        4. Select `result.hits_w[env_id]`.
        5. Add z_offset to the selected hit points to reduce z-fighting.
        6. Call `self.visualizer.visualize(translations=hits)`.
        """
        raise NotImplementedError("Legacy analytical height-scan visualization is not implemented.")
