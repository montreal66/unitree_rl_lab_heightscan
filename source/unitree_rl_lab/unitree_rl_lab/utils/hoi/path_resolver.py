"""Resolve HOI dataset URDF paths from ``.npz`` location and filename.

Logic mirrors ``motion_dataset/HOI/visualize.py`` (Drake viewer).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AssetPaths:
    """Resolved asset paths under ``hoi_root``."""

    robot_urdf: str
    #: URDF for the manipulated object (largebox / chair), if applicable.
    object_urdf: str | None
    #: Static terrain ``multi_boxes`` URDF, if applicable.
    terrain_urdf: str | None
    #: Subset folder name: ``robot-object``, ``robot-terrain``, or ``robot-object-terrain``.
    subset: str


def _natural_sort_key(text: str) -> tuple:
    parts = re.findall(r"\d+|\D+", text)
    key: list = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part.lower()))
    return tuple(key)


def list_npz_for_task(hoi_root: Path, task: str, filter_substr: str = "") -> list[Path]:
    """List ``.npz`` files for a subset, optionally filtered, naturally sorted."""
    subset_dir = {"object": "robot-object", "terrain": "robot-terrain", "object-terrain": "robot-object-terrain"}[
        task
    ]
    base = hoi_root / subset_dir
    if not base.is_dir():
        raise FileNotFoundError(f"HOI subset directory not found: {base}")
    paths = sorted(base.glob("*.npz"), key=lambda p: _natural_sort_key(p.name))
    if filter_substr:
        paths = [p for p in paths if filter_substr in p.name]
    return paths


def resolve_assets(npz_path: str | Path, hoi_root: str | Path | None = None) -> AssetPaths:
    """Resolve robot / object / terrain URDF paths for one trajectory file.

    Args:
        npz_path: Absolute or relative path to a ``.npz`` trajectory.
        hoi_root: Root folder containing ``models/`` and subset dirs. If ``None``,
            inferred as ``<parent of subset>/..`` when ``npz_path`` is under
            ``.../HOI/<subset>/file.npz`` (i.e. parent of ``robot-object`` etc.).
    """
    npz_path = Path(npz_path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    subset = npz_path.parent.name
    if subset not in ("robot-object", "robot-terrain", "robot-object-terrain"):
        raise ValueError(
            f"npz must live under robot-object, robot-terrain, or robot-object-terrain; got parent={subset!r}"
        )

    if hoi_root is None:
        hoi_root = npz_path.parent.parent
    hoi_root = Path(hoi_root).resolve()
    models = hoi_root / "models"

    file_name = npz_path.stem  # without .npz
    # Segmented clips may append suffixes after a double underscore:
    #   <base>__00_segment, <base>__03_climb, ...
    # Asset resolution should map back to the original base name.
    base_file_name = file_name
    if "__" in file_name:
        candidate = file_name.split("__", 1)[0]
        if len(candidate) > 8:
            base_file_name = candidate

    if subset == "robot-object":
        robot_urdf = str(models / "g1" / "g1_29dof.urdf")
        object_urdf = str(models / "largebox" / "largebox.urdf")
        terrain_urdf = None

    elif subset == "robot-terrain":
        robot_urdf = str(models / "g1" / "g1_29dof_spherehand.urdf")
        object_urdf = None
        terrain_folder = models / "terrain" / base_file_name[:8]
        z_suffix = base_file_name[8:]  # e.g. "_z_scale_1.0"
        terrain_urdf = str(terrain_folder / f"multi_boxes{z_suffix}.urdf")

    elif subset == "robot-object-terrain":
        robot_urdf = str(models / "g1" / "g1_29dof_spherehand.urdf")
        terrain_folder = models / "terrain" / base_file_name[:8]
        if "z_scale" in base_file_name:
            z_scale = base_file_name[-12:]  # e.g. "_z_scale_1.0"
        else:
            z_scale = "_z_scale_1.0"
        terrain_urdf = str(terrain_folder / f"multi_boxes{z_scale}.urdf")

        if "original" in base_file_name:
            object_urdf = str(models / "chair" / "chair.urdf")
        else:
            chair_scale = base_file_name[9:25]
            object_urdf = str(models / "chair" / f"{chair_scale}.urdf")
    else:
        raise AssertionError(subset)

    if not os.path.isfile(robot_urdf):
        raise FileNotFoundError(f"Robot URDF missing: {robot_urdf}")
    if object_urdf is not None and not os.path.isfile(object_urdf):
        raise FileNotFoundError(f"Object URDF missing: {object_urdf}")
    if terrain_urdf is not None and not os.path.isfile(terrain_urdf):
        raise FileNotFoundError(f"Terrain URDF missing: {terrain_urdf}")

    return AssetPaths(
        robot_urdf=robot_urdf,
        object_urdf=object_urdf,
        terrain_urdf=terrain_urdf,
        subset=subset,
    )
