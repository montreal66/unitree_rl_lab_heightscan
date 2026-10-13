from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@dataclass
class HoiHeightScanResult:
    """Container returned by the analytical HOI terrain height scanner."""

    heights: torch.Tensor
    hits_w: torch.Tensor
    world_rays_xy: torch.Tensor


def _build_grid_xy(size_xy: tuple[float, float], resolution: float, device: torch.device) -> torch.Tensor:
    """Build a local XY scan grid centered at the robot torso.

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Read size_x and size_y from size_xy.
    2. Validate that resolution is positive.
    3. Compute the number of samples along x/y as round(size / resolution) + 1.
       The default parameters should produce 17 * 11 = 187 points.
    4. Create x values from -size_x / 2 to +size_x / 2.
    5. Create y values from -size_y / 2 to +size_y / 2.
    6. Use torch.meshgrid and return a tensor with shape [num_points, 2].
    """
    raise NotImplementedError("The legacy analytical HOI height scanner is not implemented.")


def _normalize_box_record(entry: dict, idx: int) -> tuple[list[float], list[float], list[float]]:
    """Validate one metadata box and return pos, quat, half_size.

    ``full_size`` is accepted as a compatibility fallback and converted to
    half extents.  All errors identify the malformed box by index.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"mjcf_boxes[{idx}] must be an object, got {type(entry).__name__}.")

    pos = entry.get("pos")
    quat = entry.get("quat")
    half_size = entry.get("half_size")
    if half_size is None:
        full_size = entry.get("full_size")
        if full_size is None:
            raise ValueError(f"mjcf_boxes[{idx}] must contain half_size or full_size.")
        half_size = _as_finite_vector(full_size, 3, "full_size", idx)
        half_size = [value * 0.5 for value in half_size]

    pos = _as_finite_vector(pos, 3, "pos", idx)
    quat = _as_finite_vector(quat, 4, "quat", idx)
    half_size = _as_finite_vector(half_size, 3, "half_size", idx)

    if sum(value * value for value in quat) == 0.0:
        raise ValueError(f"mjcf_boxes[{idx}].quat must not be the zero quaternion.")
    if any(value <= 0.0 for value in half_size):
        raise ValueError(f"mjcf_boxes[{idx}].half_size must contain only positive values.")
    return pos, quat, half_size


def _as_finite_vector(value: object, length: int, name: str, idx: int) -> list[float]:
    """Convert one JSON vector to finite floats with a box-specific error."""
    if not isinstance(value, (list, tuple)) or len(value) != length:
        actual_length = len(value) if isinstance(value, (list, tuple)) else "non-sequence"
        raise ValueError(f"mjcf_boxes[{idx}].{name} must have length {length}, got {actual_length}.")
    try:
        result = [float(component) for component in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"mjcf_boxes[{idx}].{name} must contain numeric values.") from error
    if not all(torch.isfinite(torch.tensor(result, dtype=torch.float32)).tolist()):
        raise ValueError(f"mjcf_boxes[{idx}].{name} must contain finite values.")
    return result


def _load_boxes(metadata_file: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load terrain boxes from a `*.terrain.json` file.

    Returns:
        box_pos_local: [num_boxes, 3], terrain-body local box centers.
        box_quat_local: [num_boxes, 4], terrain-body local box orientation in wxyz.
        box_half: [num_boxes, 3], box half extents.

    This is called while the merged terrain mesh is baked, rather than from an
    observation term on every simulation step.
    """
    path = Path(metadata_file).expanduser()
    try:
        with path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except OSError as error:
        raise ValueError(f"Could not read terrain metadata file '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Terrain metadata file '{path}' is not valid JSON: {error}") from error

    if not isinstance(metadata, dict):
        raise ValueError(f"Terrain metadata file '{path}' must contain a JSON object.")
    boxes = metadata.get("mjcf_boxes")
    if not isinstance(boxes, list) or not boxes:
        raise ValueError(f"Terrain metadata file '{path}' must contain a non-empty mjcf_boxes list.")

    normalized = [_normalize_box_record(entry, idx) for idx, entry in enumerate(boxes)]
    box_pos_local, box_quat_local, box_half = zip(*normalized)
    return (
        torch.tensor(box_pos_local, dtype=torch.float32, device=device),
        torch.tensor(box_quat_local, dtype=torch.float32, device=device),
        torch.tensor(box_half, dtype=torch.float32, device=device),
    )


def _store_scan_cache(
    env: ManagerBasedEnv,
    *,
    result: HoiHeightScanResult,
    params_key: tuple,
) -> None:
    """Store latest scan result on env for visualization/debug tools."""
    env._hoi_height_scan_debug = {
        "heights": result.heights,
        "hits_w": result.hits_w,
        "world_rays_xy": result.world_rays_xy,
        "params_key": params_key,
    }


def compute_hoi_height_scan(
    env: ManagerBasedEnv,
    *,
    command_name: str | None = None,
    metadata_file: str,
    terrain_asset_name: str = "hoi_terrain",
    robot_asset_name: str = "robot",
    body_name: str = "torso_link",
    size_xy: tuple[float, float] = (1.6, 1.0),
    resolution: float = 0.1,
    offset: float = 0.5,
    cache_on_env: bool = True,
) -> HoiHeightScanResult:
    """Compute analytical HOI height-scan hits and height observations.

    The output observation should be:

        torso_z - hit_z - offset

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Get robot and terrain assets from `env.scene`.
    2. Resolve `body_name` to a body id on the robot.
    3. Read torso position and quaternion from `robot.data.body_pos_w/body_quat_w`.
    4. Build a local XY scan grid and rotate it with torso yaw only.
    5. Translate the grid to the torso world position to get world sample XY points.
    6. Initialize hit_z with the ground plane height z=0.
    7. Read terrain pose from `terrain.data`.
    8. Load metadata boxes with `_load_boxes`.
    9. Transform each box from terrain local frame to world frame.
    10. For each box, test whether each sample point is inside the box top projection.
    11. For covered points, update hit_z with the highest box top z.
    12. Build and return HoiHeightScanResult.
    13. If cache_on_env is true, call `_store_scan_cache`.

    Legacy scope:
    - Validation boxes have horizontal top faces (world translation/yaw are allowed).
    - Keep env and scan-point operations batched; a loop over the small box list is OK.
    """
    del command_name
    raise NotImplementedError("The legacy analytical HOI height scanner is not implemented.")


def hoi_height_scan(
    env: ManagerBasedEnv,
    *,
    command_name: str | None = None,
    metadata_file: str,
    terrain_asset_name: str = "hoi_terrain",
    robot_asset_name: str = "robot",
    body_name: str = "torso_link",
    size_xy: tuple[float, float] = (1.6, 1.0),
    resolution: float = 0.1,
    offset: float = 0.5,
) -> torch.Tensor:
    """Observation term used by the perceptive task.

    Legacy analytical-scanner outline (not part of the RayCaster assignment):
    1. Call `compute_hoi_height_scan`.
    2. Return only `result.heights`.
    """
    raise NotImplementedError("The legacy analytical HOI height scanner is not implemented.")
