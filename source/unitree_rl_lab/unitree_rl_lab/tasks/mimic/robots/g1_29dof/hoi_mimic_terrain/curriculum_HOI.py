from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from unitree_rl_lab.tasks.mimic.mdp.events import randomize_joint_default_pos
from unitree_rl_lab.tasks.mimic.mdp.events import randomize_rigid_body_com

_DEPRECATION_WARNED: set[str] = set()

LEVEL_UP_METRIC_GATE = "metric_gate"


def _warn_deprecated_once(old_key: str, new_key: str) -> None:
    if old_key in _DEPRECATION_WARNED:
        return
    print(f"[HOI curriculum] '{old_key}' is deprecated, use '{new_key}'.")
    _DEPRECATION_WARNED.add(old_key)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be int, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be float, got {raw!r}") from exc


def _env_min_episode_length_floor() -> int:
    old_raw = os.getenv("HOI_CURRICULUM_MIN_EPISODE_LENGTH")
    if old_raw is not None:
        _warn_deprecated_once("HOI_CURRICULUM_MIN_EPISODE_LENGTH", "HOI_CURRICULUM_MIN_MEAN_EPISODE_LENGTH")
        try:
            return max(int(old_raw), 0)
        except ValueError as exc:
            raise ValueError(f"HOI_CURRICULUM_MIN_EPISODE_LENGTH must be int, got {old_raw!r}") from exc
    raw = os.getenv("HOI_CURRICULUM_MIN_MEAN_EPISODE_LENGTH")
    if raw is None:
        return 0
    try:
        return max(int(raw), 0)
    except ValueError as exc:
        raise ValueError(f"HOI_CURRICULUM_MIN_MEAN_EPISODE_LENGTH must be int, got {raw!r}") from exc


def _env_min_completed_samples() -> int:
    return max(_env_int("HOI_CURRICULUM_MIN_COMPLETED_SAMPLES", 32), 0)


def _hoi_curriculum_cfg(env: ManagerBasedRLEnv) -> object | None:
    root = getattr(env, "cfg", None)
    if root is None:
        return None
    return getattr(root, "hoi_curriculum", None)


def _merged_min_episode_length_floor(cfg: object | None) -> int:
    old_raw = os.getenv("HOI_CURRICULUM_MIN_EPISODE_LENGTH")
    if old_raw is not None:
        _warn_deprecated_once("HOI_CURRICULUM_MIN_EPISODE_LENGTH", "HOI_CURRICULUM_MIN_MEAN_EPISODE_LENGTH")
        try:
            return max(int(old_raw), 0)
        except ValueError as exc:
            raise ValueError(f"HOI_CURRICULUM_MIN_EPISODE_LENGTH must be int, got {old_raw!r}") from exc
    raw = os.getenv("HOI_CURRICULUM_MIN_MEAN_EPISODE_LENGTH")
    if raw is not None:
        try:
            return max(int(raw), 0)
        except ValueError as exc:
            raise ValueError(f"HOI_CURRICULUM_MIN_MEAN_EPISODE_LENGTH must be int, got {raw!r}") from exc
    if cfg is not None:
        v = getattr(cfg, "min_mean_episode_length", None)
        if v is not None:
            return max(int(v), 0)
    return 0


def _merged_min_completed_samples(cfg: object | None) -> int:
    if os.getenv("HOI_CURRICULUM_MIN_COMPLETED_SAMPLES") is not None:
        return max(_env_int("HOI_CURRICULUM_MIN_COMPLETED_SAMPLES", 32), 0)
    if cfg is not None:
        return max(int(getattr(cfg, "min_completed_samples", 32)), 0)
    return 32


def _merged_curriculum_enable(cfg: object | None) -> bool:
    raw = os.getenv("HOI_CURRICULUM_ENABLE")
    if raw is not None:
        return raw.lower() in ("1", "true", "yes", "on")
    if cfg is not None:
        return bool(getattr(cfg, "enable", True))
    return True


def _merged_eval_steps(cfg: object | None, eval_steps_default: int) -> int:
    if os.getenv("HOI_CURRICULUM_EVAL_STEPS") is not None:
        return max(_env_int("HOI_CURRICULUM_EVAL_STEPS", eval_steps_default), 1)
    if cfg is not None:
        v = getattr(cfg, "eval_steps", None)
        if v is not None:
            return max(int(v), 1)
    return max(eval_steps_default, 1)


def _merged_required_windows(cfg: object | None) -> int:
    if os.getenv("HOI_CURRICULUM_REQUIRED_WINDOWS") is not None:
        return max(_env_int("HOI_CURRICULUM_REQUIRED_WINDOWS", 3), 1)
    if cfg is not None:
        return max(int(getattr(cfg, "required_windows", 3)), 1)
    return 3


def _normalize_level_up_criterion(raw: str) -> str:
    s = raw.strip().lower().replace("-", "_")
    if s == LEVEL_UP_METRIC_GATE:
        return LEVEL_UP_METRIC_GATE
    raise ValueError(
        f"Unsupported HOI curriculum level_up_criterion={raw!r}. Only {LEVEL_UP_METRIC_GATE!r} is supported."
    )


def _merged_level_up_criterion(cfg: object | None) -> str:
    raw = os.getenv("HOI_CURRICULUM_LEVEL_UP_CRITERION")
    if raw is not None:
        return _normalize_level_up_criterion(raw)
    if cfg is not None:
        v = getattr(cfg, "level_up_criterion", None)
        if isinstance(v, str) and v.strip():
            return _normalize_level_up_criterion(v)
    return LEVEL_UP_METRIC_GATE


def _merged_rolling_episode_length_window(cfg: object | None) -> int:
    if os.getenv("HOI_CURRICULUM_ROLLING_LEN_WINDOW") is not None:
        return max(_env_int("HOI_CURRICULUM_ROLLING_LEN_WINDOW", 100), 1)
    if cfg is not None:
        return max(int(getattr(cfg, "rolling_episode_length_window", 100)), 1)
    return 100


def _merged_mean_episode_length_ratio_threshold(cfg: object | None) -> float:
    raw = os.getenv("HOI_CURRICULUM_MEAN_EPISODE_LENGTH_RATIO_THRESHOLD")
    if raw is not None:
        try:
            return min(max(float(raw), 0.0), 1.0)
        except ValueError as exc:
            raise ValueError(
                f"HOI_CURRICULUM_MEAN_EPISODE_LENGTH_RATIO_THRESHOLD must be float, got {raw!r}"
            ) from exc
    if cfg is not None:
        return min(max(float(getattr(cfg, "mean_episode_length_ratio_threshold", 0.85)), 0.0), 1.0)
    return 0.85


def _merged_time_out_ratio_threshold(cfg: object | None) -> float:
    raw = os.getenv("HOI_CURRICULUM_TIME_OUT_RATIO_THRESHOLD")
    if raw is not None:
        try:
            return min(max(float(raw), 0.0), 1.0)
        except ValueError as exc:
            raise ValueError(f"HOI_CURRICULUM_TIME_OUT_RATIO_THRESHOLD must be float, got {raw!r}") from exc
    if cfg is not None:
        return min(max(float(getattr(cfg, "time_out_ratio_threshold", 0.80)), 0.0), 1.0)
    return 0.80


def _merged_dr_curriculum_master(cfg: object | None) -> bool:
    raw = os.getenv("HOI_CURRICULUM_DR_ENABLE")
    if raw is not None:
        return raw.lower() in ("1", "true", "yes", "on")
    if cfg is not None:
        return bool(getattr(cfg, "enable_domain_randomization", True))
    return True


def _merged_dr_term(cfg: object | None, *, attr: str, env_key: str, dr_master: bool, default: bool = True) -> bool:
    if not dr_master:
        return False
    raw = os.getenv(env_key)
    if raw is not None:
        return raw.lower() in ("1", "true", "yes", "on")
    if cfg is not None:
        return bool(getattr(cfg, attr, default))
    return default


def _sym_interval(lo: float, hi: float, scale: float) -> tuple[float, float]:
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) * max(scale, 0.0)
    return (center - half, center + half)


PUSH_FINAL_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def _scaled_push_velocity_range(scale: float) -> dict[str, tuple[float, float]]:
    return {k: (lo * scale, hi * scale) for k, (lo, hi) in PUSH_FINAL_VELOCITY_RANGE.items()}


_HOI_CURRICULUM_NUM_LEVELS = 8

# Four anchor curriculum rows (legacy levels 0..3). Push / EE interpolate to `_HOI_CURRICULUM_NUM_LEVELS`
# so endpoints match anchors[0] and anchors[-1].
_ANCHOR_PUSH_LEVELS: list[dict[str, object]] = [
    {"velocity_range": _scaled_push_velocity_range(0.0), "interval_range_s": (1.0, 5.0)},
    {"velocity_range": _scaled_push_velocity_range(0.35), "interval_range_s": (2.0, 4.0)},
    {"velocity_range": _scaled_push_velocity_range(0.75), "interval_range_s": (1.0, 4.0)},
    {"velocity_range": _scaled_push_velocity_range(1.25), "interval_range_s": (1.0, 3.0)},
]
_ANCHOR_EE_THRESHOLDS = [0.55, 0.45, 0.35, 0.3]
_ANCHOR_BODY_POS_PASS_Y = [0.15, 0.08, 0.06]


def _lerp_f(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _continuous_curriculum_index(k: int, num_levels: int, num_anchors: int) -> float:
    return k * float(num_anchors - 1) / float(max(num_levels - 1, 1))


def _lerp_push_level_row(a: dict[str, object], b: dict[str, object], t: float) -> dict[str, object]:
    vr_a = a["velocity_range"]  # type: ignore[assignment]
    vr_b = b["velocity_range"]  # type: ignore[assignment]
    keys = vr_a.keys()
    vr: dict[str, tuple[float, float]] = {
        key: (_lerp_f(float(vr_a[key][0]), float(vr_b[key][0]), t), _lerp_f(float(vr_a[key][1]), float(vr_b[key][1]), t))
        for key in keys
    }
    ia = tuple(a["interval_range_s"])  # type: ignore[assignment]
    ib = tuple(b["interval_range_s"])  # type: ignore[assignment]
    interval = (_lerp_f(float(ia[0]), float(ib[0]), t), _lerp_f(float(ia[1]), float(ib[1]), t))
    return {"velocity_range": vr, "interval_range_s": interval}


def _interpolate_push_level(k: int) -> dict[str, object]:
    anchors = _ANCHOR_PUSH_LEVELS
    n = len(anchors)
    u = _continuous_curriculum_index(k, _HOI_CURRICULUM_NUM_LEVELS, n)
    if u >= n - 1 - 1e-15:
        row = anchors[-1]
        return {"velocity_range": dict(row["velocity_range"]), "interval_range_s": tuple(row["interval_range_s"])}  # type: ignore[arg-type]
    i = int(math.floor(u))
    f = u - float(i)
    return _lerp_push_level_row(anchors[i], anchors[i + 1], f)


def _interpolate_scalar_anchors(anchors: Sequence[float], k: int) -> float:
    n = len(anchors)
    u = _continuous_curriculum_index(k, _HOI_CURRICULUM_NUM_LEVELS, n)
    if u >= n - 1 - 1e-15:
        return float(anchors[-1])
    i = int(math.floor(u))
    f = u - float(i)
    return _lerp_f(float(anchors[i]), float(anchors[i + 1]), f)


def _build_body_pos_pass_thresholds(num_levels: int) -> list[float]:
    """Piecewise-linear on transition index u in [0, 2] through legacy (0.15)->(0.08)->(0.06)."""
    ys = _ANCHOR_BODY_POS_PASS_Y
    m = num_levels - 1
    denom = max(m - 1, 1)
    out: list[float] = []
    for j in range(m):
        u = j * 2.0 / float(denom)
        if u <= 0.0:
            out.append(float(ys[0]))
        elif u >= 2.0:
            out.append(float(ys[-1]))
        else:
            seg = int(math.floor(u))
            f = u - float(seg)
            out.append(_lerp_f(float(ys[seg]), float(ys[seg + 1]), f))
    return out


_DEFAULT_PUSH_LEVELS = [_interpolate_push_level(k) for k in range(_HOI_CURRICULUM_NUM_LEVELS)]
_DEFAULT_EE_THRESHOLDS = [_interpolate_scalar_anchors(_ANCHOR_EE_THRESHOLDS, k) for k in range(_HOI_CURRICULUM_NUM_LEVELS)]
_DEFAULT_BODY_POS_PASS_THRESHOLDS = _build_body_pos_pass_thresholds(_HOI_CURRICULUM_NUM_LEVELS)

INITIAL_PUSH_VELOCITY_RANGE = dict(_DEFAULT_PUSH_LEVELS[0]["velocity_range"])  # type: ignore[arg-type]
INITIAL_PUSH_INTERVAL_S = tuple(_DEFAULT_PUSH_LEVELS[0]["interval_range_s"])  # type: ignore[arg-type]

_DR_PHYSICS_BASE = {
    "static_friction_range": (0.2, 1.8),
    "dynamic_friction_range": (0.2, 1.5),
    "restitution_range": (0.0, 0.6),
    "num_buckets": 64,
}
# HOI articulated terrain contact material: separate symmetric-interval base from the robot (edit here only for terrain).
_DR_HOI_TERRAIN_PHYSICS_BASE = {
    "static_friction_range": (0.9, 1.1),
    "dynamic_friction_range": (0.9, 1.1),
    "restitution_range": (0.0, 0.1),
    "num_buckets": 64,
}
_DR_JOINT_NON_ANKLE_BASE = (-0.01, 0.01)
_DR_JOINT_ANKLE_BASE = (-0.1, 0.1)
_DR_BASE_COM_BASE = {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}

# ---------------------------------------------------------------------------
# Domain-randomization curriculum scales (one float per curriculum level 0..N-1).
# Edit these lists in place, then call :func:`sync_hoi_dr_level_dicts_from_public_scales`, or use
# ``set_hoi_dr_*_scales`` below. Defaults match the former 4-anchor linear interpolation.
# ---------------------------------------------------------------------------
def _scales_to_dr_level_entries(scales: Sequence[float]) -> list[dict[str, object]]:
    return [{"mode": "scale", "scale": float(x)} for x in scales]


def _validate_hoi_dr_scale_sequence(scales: Sequence[float], *, name: str) -> list[float]:
    s = [float(x) for x in scales]
    n = _HOI_CURRICULUM_NUM_LEVELS
    if len(s) != n:
        raise ValueError(f"{name}: expected {n} floats, got {len(s)}")
    return s


# Explicit per-curriculum-level DR scales (level 0 .. _HOI_CURRICULUM_NUM_LEVELS-1). Edit here or use set_hoi_dr_*_scales().
HOI_DR_PHYSICS_MATERIAL_SCALES: list[float] = [
    0.1,
    0.25,
    0.4,
    0.55,
    0.7,
    0.85,
    0.95,
    1.0,
]
HOI_DR_JOINT_NON_ANKLE_SCALES: list[float] = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    0.95,
    1.05,
    1.2,
]
HOI_DR_JOINT_ANKLE_SCALES: list[float] = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    0.95,
    1.05,
    1.2,
]
HOI_DR_BASE_COM_SCALES: list[float] = [
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    0.95,
    1.05,
    1.2,
]


def sync_hoi_dr_level_dicts_from_public_scales() -> None:
    """Rebuild ``DR_*_LEVELS`` from ``HOI_DR_*_SCALES`` after manual edits."""
    global DR_PHYSICS_MATERIAL_LEVELS, DR_JOINT_NON_ANKLE_LEVELS, DR_JOINT_ANKLE_LEVELS, DR_BASE_COM_LEVELS
    _validate_hoi_dr_scale_sequence(HOI_DR_PHYSICS_MATERIAL_SCALES, name="HOI_DR_PHYSICS_MATERIAL_SCALES")
    _validate_hoi_dr_scale_sequence(HOI_DR_JOINT_NON_ANKLE_SCALES, name="HOI_DR_JOINT_NON_ANKLE_SCALES")
    _validate_hoi_dr_scale_sequence(HOI_DR_JOINT_ANKLE_SCALES, name="HOI_DR_JOINT_ANKLE_SCALES")
    _validate_hoi_dr_scale_sequence(HOI_DR_BASE_COM_SCALES, name="HOI_DR_BASE_COM_SCALES")
    DR_PHYSICS_MATERIAL_LEVELS = _scales_to_dr_level_entries(HOI_DR_PHYSICS_MATERIAL_SCALES)
    DR_JOINT_NON_ANKLE_LEVELS = _scales_to_dr_level_entries(HOI_DR_JOINT_NON_ANKLE_SCALES)
    DR_JOINT_ANKLE_LEVELS = _scales_to_dr_level_entries(HOI_DR_JOINT_ANKLE_SCALES)
    DR_BASE_COM_LEVELS = _scales_to_dr_level_entries(HOI_DR_BASE_COM_SCALES)


def set_hoi_dr_physics_material_scales(scales: Sequence[float]) -> None:
    """Replace physics-material DR scales for levels ``0 .. _HOI_CURRICULUM_NUM_LEVELS-1``."""
    s = _validate_hoi_dr_scale_sequence(scales, name="physics_material")
    HOI_DR_PHYSICS_MATERIAL_SCALES.clear()
    HOI_DR_PHYSICS_MATERIAL_SCALES.extend(s)
    sync_hoi_dr_level_dicts_from_public_scales()


def set_hoi_dr_joint_non_ankle_scales(scales: Sequence[float]) -> None:
    """Replace default-joint (non-ankle) DR scales."""
    s = _validate_hoi_dr_scale_sequence(scales, name="joint_non_ankle")
    HOI_DR_JOINT_NON_ANKLE_SCALES.clear()
    HOI_DR_JOINT_NON_ANKLE_SCALES.extend(s)
    sync_hoi_dr_level_dicts_from_public_scales()


def set_hoi_dr_joint_ankle_scales(scales: Sequence[float]) -> None:
    """Replace ankle-joint DR scales."""
    s = _validate_hoi_dr_scale_sequence(scales, name="joint_ankle")
    HOI_DR_JOINT_ANKLE_SCALES.clear()
    HOI_DR_JOINT_ANKLE_SCALES.extend(s)
    sync_hoi_dr_level_dicts_from_public_scales()


def set_hoi_dr_base_com_scales(scales: Sequence[float]) -> None:
    """Replace torso CoM DR scales."""
    s = _validate_hoi_dr_scale_sequence(scales, name="base_com")
    HOI_DR_BASE_COM_SCALES.clear()
    HOI_DR_BASE_COM_SCALES.extend(s)
    sync_hoi_dr_level_dicts_from_public_scales()


DR_PHYSICS_MATERIAL_LEVELS: list[dict[str, object]] = _scales_to_dr_level_entries(HOI_DR_PHYSICS_MATERIAL_SCALES)
DR_JOINT_NON_ANKLE_LEVELS: list[dict[str, object] | tuple[float, float]] = _scales_to_dr_level_entries(
    HOI_DR_JOINT_NON_ANKLE_SCALES
)
DR_JOINT_ANKLE_LEVELS: list[dict[str, object] | tuple[float, float]] = _scales_to_dr_level_entries(HOI_DR_JOINT_ANKLE_SCALES)
DR_BASE_COM_LEVELS: list[dict[str, object]] = _scales_to_dr_level_entries(HOI_DR_BASE_COM_SCALES)

# HOI terrain rigid-body material DR: scales × :data:`_DR_HOI_TERRAIN_PHYSICS_BASE` via :func:`_sym_interval` (robot uses ``_DR_PHYSICS_BASE``).
HOI_DR_HOI_TERRAIN_SCALES: list[float] = []
DR_HOI_TERRAIN_MATERIAL_LEVELS: list[dict[str, object]] = []


def apply_hoi_terrain_dr_scales_from_min_max(lo: float, hi: float) -> None:
    """Linear scales on curriculum levels 0..N-1 between ``lo`` and ``hi``; rebuilds ``DR_HOI_TERRAIN_MATERIAL_LEVELS``."""
    global DR_HOI_TERRAIN_MATERIAL_LEVELS
    lo_f, hi_f = float(lo), float(hi)
    if lo_f < 0.0 or hi_f < 0.0:
        raise ValueError("HOI terrain DR scale endpoints must be nonnegative.")
    if lo_f > hi_f:
        raise ValueError(f"hoi_terrain_dr_scale_min ({lo_f}) must be <= hoi_terrain_dr_scale_max ({hi_f}).")
    n = _HOI_CURRICULUM_NUM_LEVELS
    denom = max(n - 1, 1)
    scales = [_lerp_f(lo_f, hi_f, float(k) / float(denom)) for k in range(n)]
    HOI_DR_HOI_TERRAIN_SCALES.clear()
    HOI_DR_HOI_TERRAIN_SCALES.extend(scales)
    DR_HOI_TERRAIN_MATERIAL_LEVELS = _scales_to_dr_level_entries(HOI_DR_HOI_TERRAIN_SCALES)


apply_hoi_terrain_dr_scales_from_min_max(0.15, 1.35)


def _resolve_range_entry(entry: dict[str, object] | tuple[float, float], base: tuple[float, float]) -> tuple[float, float]:
    if isinstance(entry, tuple):
        return (float(entry[0]), float(entry[1]))
    if str(entry.get("mode", "absolute")).lower() == "scale":
        return _sym_interval(base[0], base[1], float(entry.get("scale", 1.0)))
    raw = entry.get("range", None)
    if not isinstance(raw, tuple):
        raise ValueError(f"Absolute range entry must define tuple 'range', got {entry!r}")
    return (float(raw[0]), float(raw[1]))


def dr_level_physics_material_params(level: int) -> dict[str, object]:
    entry = DR_PHYSICS_MATERIAL_LEVELS[level]
    if str(entry.get("mode", "absolute")).lower() == "scale":
        s = float(entry.get("scale", 1.0))
        return {
            "static_friction_range": _sym_interval(*_DR_PHYSICS_BASE["static_friction_range"], s),
            "dynamic_friction_range": _sym_interval(*_DR_PHYSICS_BASE["dynamic_friction_range"], s),
            "restitution_range": _sym_interval(*_DR_PHYSICS_BASE["restitution_range"], s),
            "num_buckets": int(_DR_PHYSICS_BASE["num_buckets"]),
        }
    return {
        "static_friction_range": tuple(entry["static_friction_range"]),  # type: ignore[arg-type]
        "dynamic_friction_range": tuple(entry["dynamic_friction_range"]),  # type: ignore[arg-type]
        "restitution_range": tuple(entry["restitution_range"]),  # type: ignore[arg-type]
        "num_buckets": int(entry.get("num_buckets", _DR_PHYSICS_BASE["num_buckets"])),
    }


def dr_level_hoi_terrain_physics_material_params(level: int) -> dict[str, object]:
    """Rigid-body material sampling ranges for ``hoi_terrain`` (uses :data:`_DR_HOI_TERRAIN_PHYSICS_BASE`, not robot ``_DR_PHYSICS_BASE``)."""
    entry = DR_HOI_TERRAIN_MATERIAL_LEVELS[level]
    if str(entry.get("mode", "absolute")).lower() == "scale":
        s = float(entry.get("scale", 1.0))
        return {
            "static_friction_range": _sym_interval(*_DR_HOI_TERRAIN_PHYSICS_BASE["static_friction_range"], s),
            "dynamic_friction_range": _sym_interval(*_DR_HOI_TERRAIN_PHYSICS_BASE["dynamic_friction_range"], s),
            "restitution_range": _sym_interval(*_DR_HOI_TERRAIN_PHYSICS_BASE["restitution_range"], s),
            "num_buckets": int(_DR_HOI_TERRAIN_PHYSICS_BASE["num_buckets"]),
        }
    return {
        "static_friction_range": tuple(entry["static_friction_range"]),  # type: ignore[arg-type]
        "dynamic_friction_range": tuple(entry["dynamic_friction_range"]),  # type: ignore[arg-type]
        "restitution_range": tuple(entry["restitution_range"]),  # type: ignore[arg-type]
        "num_buckets": int(entry.get("num_buckets", _DR_HOI_TERRAIN_PHYSICS_BASE["num_buckets"])),
    }


def dr_level_joint_non_ankle_params(level: int) -> tuple[float, float]:
    return _resolve_range_entry(DR_JOINT_NON_ANKLE_LEVELS[level], _DR_JOINT_NON_ANKLE_BASE)


def dr_level_joint_ankle_params(level: int) -> tuple[float, float]:
    return _resolve_range_entry(DR_JOINT_ANKLE_LEVELS[level], _DR_JOINT_ANKLE_BASE)


def dr_level_base_com_range(level: int) -> dict[str, tuple[float, float]]:
    entry = DR_BASE_COM_LEVELS[level]
    if str(entry.get("mode", "absolute")).lower() == "scale":
        s = float(entry.get("scale", 1.0))
        return {"x": _sym_interval(*_DR_BASE_COM_BASE["x"], s), "y": _sym_interval(*_DR_BASE_COM_BASE["y"], s), "z": _sym_interval(*_DR_BASE_COM_BASE["z"], s)}
    return {"x": tuple(entry["x"]), "y": tuple(entry["y"]), "z": tuple(entry["z"])}  # type: ignore[arg-type]


def dr_physics_material_event_params(dr_scale: float) -> dict[str, object]:
    return {
        "static_friction_range": _sym_interval(*_DR_PHYSICS_BASE["static_friction_range"], dr_scale),
        "dynamic_friction_range": _sym_interval(*_DR_PHYSICS_BASE["dynamic_friction_range"], dr_scale),
        "restitution_range": _sym_interval(*_DR_PHYSICS_BASE["restitution_range"], dr_scale),
        "num_buckets": int(_DR_PHYSICS_BASE["num_buckets"]),
    }


def dr_joint_non_ankle_params(dr_scale: float) -> tuple[float, float]:
    return _sym_interval(*_DR_JOINT_NON_ANKLE_BASE, dr_scale)


def dr_joint_ankle_params(dr_scale: float) -> tuple[float, float]:
    return _sym_interval(*_DR_JOINT_ANKLE_BASE, dr_scale)


def dr_base_com_range(dr_scale: float) -> dict[str, tuple[float, float]]:
    return {"x": _sym_interval(*_DR_BASE_COM_BASE["x"], dr_scale), "y": _sym_interval(*_DR_BASE_COM_BASE["y"], dr_scale), "z": _sym_interval(*_DR_BASE_COM_BASE["z"], dr_scale)}


def frozen_physics_material_event_params() -> dict[str, object]:
    """No randomization: fixed contact numbers (startup / disabled DR)."""

    return {"static_friction_range": (1.0, 1.0), "dynamic_friction_range": (1.0, 1.0), "restitution_range": (0.0, 0.0), "num_buckets": 64}


def frozen_joint_pos_distribution_params() -> tuple[float, float]:
    return (0.0, 0.0)


def frozen_base_com_range() -> dict[str, tuple[float, float]]:
    return {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}


def _validate_level_lists(num_levels: int) -> None:
    if len(_DEFAULT_PUSH_LEVELS) != num_levels or len(_DEFAULT_EE_THRESHOLDS) != num_levels:
        raise RuntimeError("push/ee list length mismatch")
    if len(_DEFAULT_BODY_POS_PASS_THRESHOLDS) != num_levels - 1:
        raise RuntimeError("body-pos threshold list length mismatch")
    if (
        len(DR_PHYSICS_MATERIAL_LEVELS) != num_levels
        or len(DR_JOINT_NON_ANKLE_LEVELS) != num_levels
        or len(DR_JOINT_ANKLE_LEVELS) != num_levels
        or len(DR_BASE_COM_LEVELS) != num_levels
        or len(DR_HOI_TERRAIN_MATERIAL_LEVELS) != num_levels
    ):
        raise RuntimeError("DR level list length mismatch")


def _get_term_cfg_if_available(manager: object, term_name: str) -> object | None:
    get_term_cfg = getattr(manager, "get_term_cfg", None)
    if callable(get_term_cfg):
        try:
            return get_term_cfg(term_name)
        except Exception:
            return None
    return None


def _invoke_hoi_terrain_physics_material_resample(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Run the registered ``hoi_terrain_physics_material`` term (Isaac ``randomize_rigid_body_material`` *instance*).

    Do **not** call ``isaaclab.envs.mdp.events.randomize_rigid_body_material`` as a class from here: after sim
    starts, :class:`~isaaclab.managers.event_manager.EventManager` replaces class terms with instances that hold
    PhysX views and ``material_buckets``. Calling the class from curriculum corrupts sim state and can yield NaN
    observations, especially on curriculum level-up when this path runs.
    """
    term_cfg = _get_term_cfg_if_available(env.event_manager, "hoi_terrain_physics_material")
    if term_cfg is None:
        return
    fn = term_cfg.func
    if not callable(fn):
        return
    p = term_cfg.params
    fn(
        env,
        env_ids,
        static_friction_range=p["static_friction_range"],  # type: ignore[arg-type]
        dynamic_friction_range=p["dynamic_friction_range"],  # type: ignore[arg-type]
        restitution_range=p["restitution_range"],  # type: ignore[arg-type]
        num_buckets=int(p["num_buckets"]),
        asset_cfg=p["asset_cfg"],  # type: ignore[arg-type]
        make_consistent=bool(p.get("make_consistent", False)),
    )


def _update_event_term_params(env: ManagerBasedRLEnv, term_name: str, updates: dict[str, object]) -> None:
    term_cfg = _get_term_cfg_if_available(env.event_manager, term_name)
    if term_cfg is not None:
        for k, v in updates.items():
            term_cfg.params[k] = v
    if hasattr(env.cfg, "events"):
        cfg_term = getattr(env.cfg.events, term_name, None)
        if cfg_term is not None:
            for k, v in updates.items():
                cfg_term.params[k] = v


def _apply_push_level(env: ManagerBasedRLEnv, level: int) -> None:
    cfg = _DEFAULT_PUSH_LEVELS[level]
    vr = dict(cfg["velocity_range"])  # type: ignore[arg-type]
    ir = tuple(cfg["interval_range_s"])  # type: ignore[arg-type]
    term_cfg = _get_term_cfg_if_available(env.event_manager, "push_root_velocity")
    if term_cfg is not None:
        term_cfg.params["velocity_range"] = vr
        term_cfg.interval_range_s = ir
    if hasattr(env.cfg, "events") and hasattr(env.cfg.events, "push_root_velocity"):
        env.cfg.events.push_root_velocity.params["velocity_range"] = vr
        env.cfg.events.push_root_velocity.interval_range_s = ir


def _curriculum_env_ids_nonempty(env_ids: object) -> bool:
    """True if curriculum ``compute`` passed any env indices (Isaac uses ``slice(None)`` when env_ids is None)."""
    if isinstance(env_ids, slice):
        # CurriculumManager maps None -> slice(None) meaning "all envs".
        return env_ids == slice(None)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.numel() > 0
    if env_ids is None:
        return True
    return len(env_ids) > 0  # type: ignore[arg-type]


def _tensorize_env_ids_for_dr_resample(env: ManagerBasedRLEnv, env_ids: object) -> torch.Tensor:
    """1D long tensor on ``env.device`` for DR resampling; ``slice(None)`` → all envs."""
    if isinstance(env_ids, slice):
        if env_ids != slice(None):
            raise ValueError(f"Unsupported env_ids slice for HOI curriculum DR resample: {env_ids!r}")
        return torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        out = env_ids.to(device=env.device, dtype=torch.long).flatten()
        if out.numel() == 0:
            raise ValueError("empty env_ids tensor for HOI curriculum DR resample")
        return out
    return torch.tensor(list(env_ids), dtype=torch.long, device=env.device)  # type: ignore[arg-type]


def _apply_ee_threshold(env: ManagerBasedRLEnv, threshold: float) -> None:
    term_cfg = _get_term_cfg_if_available(env.termination_manager, "ee_body_pos")
    if term_cfg is not None:
        term_cfg.params["threshold"] = threshold
    if hasattr(env.cfg, "terminations") and hasattr(env.cfg.terminations, "ee_body_pos"):
        env.cfg.terminations.ee_body_pos.params["threshold"] = threshold


def _current_dr_params(level: int) -> dict[str, object]:
    return {
        "physics_material": dr_level_physics_material_params(level),
        "hoi_terrain": dr_level_hoi_terrain_physics_material_params(level),
        "joint_non_ankle": dr_level_joint_non_ankle_params(level),
        "joint_ankle": dr_level_joint_ankle_params(level),
        "base_com": dr_level_base_com_range(level),
    }


def _apply_dr_level(env: ManagerBasedRLEnv, level: int) -> dict[str, object]:
    dr = _current_dr_params(level)
    state = getattr(env, "_hoi_curriculum_state", None)
    phys = dr["physics_material"]  # type: ignore[assignment]
    ht = dr["hoi_terrain"]  # type: ignore[assignment]
    if state is None:
        _update_event_term_params(env, "physics_material", {"static_friction_range": phys["static_friction_range"], "dynamic_friction_range": phys["dynamic_friction_range"], "restitution_range": phys["restitution_range"], "num_buckets": phys["num_buckets"]})  # type: ignore[index]
        _update_event_term_params(
            env,
            "hoi_terrain_physics_material",
            {
                "static_friction_range": ht["static_friction_range"],
                "dynamic_friction_range": ht["dynamic_friction_range"],
                "restitution_range": ht["restitution_range"],
                "num_buckets": ht["num_buckets"],
            },
        )  # type: ignore[index]
        _update_event_term_params(env, "add_joint_default_pos_non_ankle", {"pos_distribution_params": dr["joint_non_ankle"]})
        _update_event_term_params(env, "add_joint_default_pos_ankle", {"pos_distribution_params": dr["joint_ankle"]})
        _update_event_term_params(env, "base_com", {"com_range": dr["base_com"]})
        return dr
    if bool(state["dr_physics_material_enabled"]):
        _update_event_term_params(env, "physics_material", {"static_friction_range": phys["static_friction_range"], "dynamic_friction_range": phys["dynamic_friction_range"], "restitution_range": phys["restitution_range"], "num_buckets": phys["num_buckets"]})  # type: ignore[index]
    if bool(state.get("dr_hoi_terrain_physics_enabled", True)):
        _update_event_term_params(
            env,
            "hoi_terrain_physics_material",
            {
                "static_friction_range": ht["static_friction_range"],
                "dynamic_friction_range": ht["dynamic_friction_range"],
                "restitution_range": ht["restitution_range"],
                "num_buckets": ht["num_buckets"],
            },
        )  # type: ignore[index]
    if bool(state["dr_joint_non_ankle_enabled"]):
        _update_event_term_params(env, "add_joint_default_pos_non_ankle", {"pos_distribution_params": dr["joint_non_ankle"]})
    if bool(state["dr_joint_ankle_enabled"]):
        _update_event_term_params(env, "add_joint_default_pos_ankle", {"pos_distribution_params": dr["joint_ankle"]})
    if bool(state["dr_base_com_enabled"]):
        _update_event_term_params(env, "base_com", {"com_range": dr["base_com"]})
    return dr


def _resample_dr_for_all_envs(
    env: ManagerBasedRLEnv, dr: dict[str, object], *, env_ids: torch.Tensor | None = None
) -> None:
    """Apply DR randomization for rigid material / joint defaults / CoM.

    When called from curriculum level-up, ``env_ids`` must be the **same** indices Isaac Lab passed to
    :meth:`CurriculumManager.compute` (subset of envs being reset). Resampling **all** envs mid-episode
    while only a subset is resetting corrupts physics and can produce NaN policy observations.
    """
    state = getattr(env, "_hoi_curriculum_state", None)
    if env_ids is None:
        ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        ids = env_ids
    if state is None or bool(state.get("dr_hoi_terrain_physics_enabled", True)):
        _invoke_hoi_terrain_physics_material_resample(env, ids)
    if state is None or bool(state["dr_joint_non_ankle_enabled"]):
        randomize_joint_default_pos(env, ids, asset_cfg=SceneEntityCfg("robot", joint_names=["^(?!.*ankle).*$"]), pos_distribution_params=dr["joint_non_ankle"], operation="add")  # type: ignore[arg-type]
    if state is None or bool(state["dr_joint_ankle_enabled"]):
        randomize_joint_default_pos(env, ids, asset_cfg=SceneEntityCfg("robot", joint_names=[".*ankle.*"]), pos_distribution_params=dr["joint_ankle"], operation="add")  # type: ignore[arg-type]
    if state is None or bool(state["dr_base_com_enabled"]):
        randomize_rigid_body_com(env, ids, com_range=dr["base_com"], asset_cfg=SceneEntityCfg("robot", body_names="torso_link"))  # type: ignore[arg-type]


def _apply_level(
    env: ManagerBasedRLEnv, level: int, *, resample_dr: bool, resample_env_ids: torch.Tensor | None = None
) -> dict[str, object]:
    _apply_push_level(env, level)
    _apply_ee_threshold(env, float(_DEFAULT_EE_THRESHOLDS[level]))
    dr = _apply_dr_level(env, level)
    if resample_dr:
        _resample_dr_for_all_envs(env, dr, env_ids=resample_env_ids)
    return dr


def _ensure_rolling_episode_deque(env: ManagerBasedRLEnv, maxlen: int) -> deque[float]:
    maxlen = max(int(maxlen), 1)
    if not hasattr(env, "_hoi_rolling_episode_length_deque"):
        env._hoi_rolling_episode_length_deque = deque(maxlen=maxlen)  # type: ignore[attr-defined]
    dq: deque[float] = env._hoi_rolling_episode_length_deque  # type: ignore[attr-defined]
    if dq.maxlen != maxlen:
        env._hoi_rolling_episode_length_deque = deque(dq, maxlen=maxlen)  # type: ignore[attr-defined]
        dq = env._hoi_rolling_episode_length_deque  # type: ignore[attr-defined]
    return dq


def _ensure_timeout_deque(env: ManagerBasedRLEnv, maxlen: int) -> deque[float]:
    maxlen = max(int(maxlen), 1)
    if not hasattr(env, "_hoi_timeout_flag_deque"):
        env._hoi_timeout_flag_deque = deque(maxlen=maxlen)  # type: ignore[attr-defined]
    dq: deque[float] = env._hoi_timeout_flag_deque  # type: ignore[attr-defined]
    if dq.maxlen != maxlen:
        env._hoi_timeout_flag_deque = deque(dq, maxlen=maxlen)  # type: ignore[attr-defined]
        dq = env._hoi_timeout_flag_deque  # type: ignore[attr-defined]
    return dq


def _compute_rolling_mean_episode_length(env: ManagerBasedRLEnv) -> tuple[float, int]:
    dq = getattr(env, "_hoi_rolling_episode_length_deque", None)
    if not isinstance(dq, deque) or len(dq) == 0:
        return float("nan"), 0
    n = len(dq)
    return float(sum(dq) / n), n


def _compute_timeout_ratio(env: ManagerBasedRLEnv) -> tuple[float, int]:
    dq = getattr(env, "_hoi_timeout_flag_deque", None)
    if not isinstance(dq, deque) or len(dq) == 0:
        return float("nan"), 0
    n = len(dq)
    return float(sum(dq) / n), n


def _sync_completed_episode_ring_from_delta(env: ManagerBasedRLEnv) -> None:
    buf = getattr(env, "episode_length_buf", None)
    if not isinstance(buf, torch.Tensor) or buf.numel() == 0:
        return
    prev = getattr(env, "_hoi_prev_episode_length_buf", None)
    if not isinstance(prev, torch.Tensor) or prev.shape != buf.shape or prev.device != buf.device:
        env._hoi_prev_episode_length_buf = buf.clone()  # type: ignore[attr-defined]
        return
    # Episode end: buffer reset to 0 then incremented on the same env step, so we often see
    # (prev=L, buf=1) instead of (prev=L, buf=0). Treat any drop prev>buf as a completed episode.
    ended = (prev > 0) & ((buf == 0) | (buf < prev))
    if ended.any():
        state = _get_state(env)
        max_ep_len = int(getattr(env, "max_episode_length", 0))
        timeout_cutoff = max(max_ep_len - 1, 1)
        for idx in torch.where(ended)[0].tolist():
            L = float(prev[idx].item())
            rolling_w = int(state["rolling_episode_length_window"])
            _ensure_rolling_episode_deque(env, rolling_w).append(L)
            # Approximate timeout label from completed episode length.
            _ensure_timeout_deque(env, rolling_w).append(1.0 if L >= float(timeout_cutoff) else 0.0)
    env._hoi_prev_episode_length_buf = buf.clone()  # type: ignore[attr-defined]


def _get_state(env: ManagerBasedRLEnv) -> dict[str, object]:
    state = getattr(env, "_hoi_curriculum_state", None)
    if state is not None:
        return state
    levels = len(_DEFAULT_PUSH_LEVELS)
    _validate_level_lists(levels)
    eval_steps_default = max(int(getattr(env, "max_episode_length", 1)) // 2, 1)
    cfg_obj = _hoi_curriculum_cfg(env)
    dr_master = _merged_dr_curriculum_master(cfg_obj)
    level_up_crit = _merged_level_up_criterion(cfg_obj)
    floor_i = _merged_min_episode_length_floor(cfg_obj)
    rolling_w = _merged_rolling_episode_length_window(cfg_obj)
    mean_len_ratio_thr = _merged_mean_episode_length_ratio_threshold(cfg_obj)
    timeout_ratio_thr = _merged_time_out_ratio_threshold(cfg_obj)
    state = {
        "enabled": _merged_curriculum_enable(cfg_obj),
        "num_levels": levels,
        "eval_steps": _merged_eval_steps(cfg_obj, eval_steps_default),
        "required_windows": _merged_required_windows(cfg_obj),
        "level_up_criterion": level_up_crit,
        "rolling_episode_length_window": rolling_w,
        "mean_episode_length_ratio_threshold": mean_len_ratio_thr,
        "time_out_ratio_threshold": timeout_ratio_thr,
        "min_episode_length_floor": floor_i,
        "min_completed_samples": _merged_min_completed_samples(cfg_obj),
        "dr_curriculum": dr_master,
        "dr_physics_material_enabled": _merged_dr_term(
            cfg_obj, attr="dr_physics_material", env_key="HOI_CURRICULUM_DR_PHYSICS", dr_master=dr_master
        ),
        "dr_joint_non_ankle_enabled": _merged_dr_term(
            cfg_obj, attr="dr_joint_non_ankle", env_key="HOI_CURRICULUM_DR_JOINT_NON_ANKLE", dr_master=dr_master
        ),
        "dr_joint_ankle_enabled": _merged_dr_term(
            cfg_obj, attr="dr_joint_ankle", env_key="HOI_CURRICULUM_DR_JOINT_ANKLE", dr_master=dr_master
        ),
        "dr_base_com_enabled": _merged_dr_term(cfg_obj, attr="dr_base_com", env_key="HOI_CURRICULUM_DR_BASE_COM", dr_master=dr_master),
        "dr_hoi_terrain_physics_enabled": _merged_dr_term(
            cfg_obj, attr="dr_hoi_terrain_physics", env_key="HOI_CURRICULUM_DR_HOI_TERRAIN", dr_master=dr_master
        ),
        "level": 0,
        "streak": 0,
        "last_eval_step": -1,
        "last_update_step": -1,
        "last_agg_episode_length": float("nan"),
        "last_completed_sample_count": 0,
        "last_time_out_ratio": float("nan"),
        "last_length_ratio_to_max": float("nan"),
        "ready_score": 0.0,
        "current_dr_params": _current_dr_params(0),
    }
    setattr(env, "_hoi_curriculum_state", state)
    _ensure_rolling_episode_deque(env, rolling_w)
    _ensure_timeout_deque(env, rolling_w)
    state["current_dr_params"] = _apply_level(env, 0, resample_dr=False)
    return state


def _maybe_update(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict[str, object]:
    state = _get_state(env)
    step = int(getattr(env, "common_step_counter", 0))
    if int(state["last_update_step"]) != step:
        _sync_completed_episode_ring_from_delta(env)
    if int(state["last_update_step"]) == step:
        return state
    state["last_update_step"] = step
    if not bool(state["enabled"]):
        return state
    eval_steps = int(state["eval_steps"])
    if step < eval_steps:
        return state
    if int(state["last_eval_step"]) >= 0 and (step - int(state["last_eval_step"])) < eval_steps:
        return state
    if not _curriculum_env_ids_nonempty(env_ids):
        return state
    state["last_eval_step"] = step
    level = int(state["level"])
    max_level = int(state["num_levels"]) - 1
    min_samples = int(state["min_completed_samples"])
    floor = int(state["min_episode_length_floor"])
    agg_len, n_roll = _compute_rolling_mean_episode_length(env)
    timeout_ratio, n_timeout = _compute_timeout_ratio(env)
    max_ep_len = max(int(getattr(env, "max_episode_length", 1)), 1)
    len_ratio_to_max = float(agg_len) / float(max_ep_len) if agg_len == agg_len else float("nan")
    ratio_thr = float(state["mean_episode_length_ratio_threshold"])
    timeout_thr = float(state["time_out_ratio_threshold"])
    ratio_target = ratio_thr * float(max_ep_len)
    # Keep optional absolute floor as an additional lower bound.
    length_target = max(ratio_target, float(max(floor, 0)))
    state["last_agg_episode_length"] = agg_len
    state["last_completed_sample_count"] = float(n_roll)
    state["last_time_out_ratio"] = timeout_ratio
    state["last_length_ratio_to_max"] = len_ratio_to_max
    if level >= max_level:
        state["ready_score"] = 1.0
        return state
    enough = n_roll >= min_samples and n_timeout >= min_samples
    length_ok = enough and agg_len == agg_len and float(agg_len) >= length_target
    timeout_ok = enough and timeout_ratio == timeout_ratio and float(timeout_ratio) >= timeout_thr
    if enough and agg_len == agg_len and timeout_ratio == timeout_ratio:
        length_prog = min(max(float(agg_len), 0.0) / max(length_target, 1e-6), 1.0)
        timeout_prog = min(max(float(timeout_ratio), 0.0) / max(timeout_thr, 1e-6), 1.0)
        state["ready_score"] = min(length_prog, timeout_prog)
    else:
        state["ready_score"] = 0.0
    state["streak"] = int(state["streak"]) + 1 if (length_ok and timeout_ok) else 0

    if int(state["streak"]) < int(state["required_windows"]):
        return state
    next_level = min(level + 1, max_level)
    state["level"] = next_level
    state["streak"] = 0
    dr_subset = _tensorize_env_ids_for_dr_resample(env, env_ids)
    state["current_dr_params"] = _apply_level(env, next_level, resample_dr=True, resample_env_ids=dr_subset)
    return state


def _dr_value(state: dict[str, object], key: str) -> float:
    dr = state.get("current_dr_params", _current_dr_params(int(state["level"])))
    nan = float("nan")
    if key == "physics_static_low":
        if not bool(state.get("dr_physics_material_enabled", True)):
            return nan
        return float(dr["physics_material"]["static_friction_range"][0])  # type: ignore[index]
    if key == "physics_static_high":
        if not bool(state.get("dr_physics_material_enabled", True)):
            return nan
        return float(dr["physics_material"]["static_friction_range"][1])  # type: ignore[index]
    if key == "physics_dynamic_low":
        if not bool(state.get("dr_physics_material_enabled", True)):
            return nan
        return float(dr["physics_material"]["dynamic_friction_range"][0])  # type: ignore[index]
    if key == "physics_dynamic_high":
        if not bool(state.get("dr_physics_material_enabled", True)):
            return nan
        return float(dr["physics_material"]["dynamic_friction_range"][1])  # type: ignore[index]
    if key == "physics_restitution_low":
        if not bool(state.get("dr_physics_material_enabled", True)):
            return nan
        return float(dr["physics_material"]["restitution_range"][0])  # type: ignore[index]
    if key == "physics_restitution_high":
        if not bool(state.get("dr_physics_material_enabled", True)):
            return nan
        return float(dr["physics_material"]["restitution_range"][1])  # type: ignore[index]
    if key == "joint_non_ankle_low":
        if not bool(state.get("dr_joint_non_ankle_enabled", True)):
            return nan
        return float(dr["joint_non_ankle"][0])  # type: ignore[index]
    if key == "joint_non_ankle_high":
        if not bool(state.get("dr_joint_non_ankle_enabled", True)):
            return nan
        return float(dr["joint_non_ankle"][1])  # type: ignore[index]
    if key == "joint_ankle_low":
        if not bool(state.get("dr_joint_ankle_enabled", True)):
            return nan
        return float(dr["joint_ankle"][0])  # type: ignore[index]
    if key == "joint_ankle_high":
        if not bool(state.get("dr_joint_ankle_enabled", True)):
            return nan
        return float(dr["joint_ankle"][1])  # type: ignore[index]
    if key == "com_x_low":
        if not bool(state.get("dr_base_com_enabled", True)):
            return nan
        return float(dr["base_com"]["x"][0])  # type: ignore[index]
    if key == "com_x_high":
        if not bool(state.get("dr_base_com_enabled", True)):
            return nan
        return float(dr["base_com"]["x"][1])  # type: ignore[index]
    if key == "com_y_low":
        if not bool(state.get("dr_base_com_enabled", True)):
            return nan
        return float(dr["base_com"]["y"][0])  # type: ignore[index]
    if key == "com_y_high":
        if not bool(state.get("dr_base_com_enabled", True)):
            return nan
        return float(dr["base_com"]["y"][1])  # type: ignore[index]
    if key == "com_z_low":
        if not bool(state.get("dr_base_com_enabled", True)):
            return nan
        return float(dr["base_com"]["z"][0])  # type: ignore[index]
    if key == "com_z_high":
        if not bool(state.get("dr_base_com_enabled", True)):
            return nan
        return float(dr["base_com"]["z"][1])  # type: ignore[index]
    raise KeyError(key)


def hoi_curriculum_progress(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    state = _maybe_update(env, env_ids)
    max_level = max(int(state["num_levels"]) - 1, 1)
    return torch.tensor(float(state["level"]) / float(max_level), device=env.device)


def hoi_curriculum_push_level(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(float(_maybe_update(env, env_ids)["level"]), device=env.device)


def hoi_curriculum_ee_threshold_level(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(float(_maybe_update(env, env_ids)["level"]), device=env.device)


def hoi_curriculum_ee_threshold_value(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    state = _maybe_update(env, env_ids)
    return torch.tensor(float(_DEFAULT_EE_THRESHOLDS[int(state["level"])]), device=env.device)


def hoi_curriculum_level_ready_score(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(float(_maybe_update(env, env_ids)["ready_score"]), device=env.device)


def hoi_curriculum_agg_episode_length(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(float(_maybe_update(env, env_ids)["last_agg_episode_length"]), device=env.device)


def hoi_curriculum_completed_sample_count(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(float(_maybe_update(env, env_ids)["last_completed_sample_count"]), device=env.device)


def hoi_curriculum_time_out_ratio(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(float(_maybe_update(env, env_ids)["last_time_out_ratio"]), device=env.device)


def hoi_curriculum_length_ratio_to_max(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(float(_maybe_update(env, env_ids)["last_length_ratio_to_max"]), device=env.device)


def hoi_curriculum_dr_physics_static_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "physics_static_low"), device=env.device)


def hoi_curriculum_dr_physics_static_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "physics_static_high"), device=env.device)


def hoi_curriculum_dr_physics_dynamic_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "physics_dynamic_low"), device=env.device)


def hoi_curriculum_dr_physics_dynamic_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "physics_dynamic_high"), device=env.device)


def hoi_curriculum_dr_physics_restitution_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "physics_restitution_low"), device=env.device)


def hoi_curriculum_dr_physics_restitution_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "physics_restitution_high"), device=env.device)


def hoi_curriculum_dr_joint_non_ankle_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "joint_non_ankle_low"), device=env.device)


def hoi_curriculum_dr_joint_non_ankle_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "joint_non_ankle_high"), device=env.device)


def hoi_curriculum_dr_joint_ankle_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "joint_ankle_low"), device=env.device)


def hoi_curriculum_dr_joint_ankle_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "joint_ankle_high"), device=env.device)


def hoi_curriculum_dr_com_x_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "com_x_low"), device=env.device)


def hoi_curriculum_dr_com_x_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "com_x_high"), device=env.device)


def hoi_curriculum_dr_com_y_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "com_y_low"), device=env.device)


def hoi_curriculum_dr_com_y_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "com_y_high"), device=env.device)


def hoi_curriculum_dr_com_z_low(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "com_z_low"), device=env.device)


def hoi_curriculum_dr_com_z_high(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(_dr_value(_maybe_update(env, env_ids), "com_z_high"), device=env.device)


def hoi_curriculum_dr_scale(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    state = _maybe_update(env, env_ids)
    dr = state["current_dr_params"]  # type: ignore[assignment]
    p = dr["physics_material"]  # type: ignore[index]
    phys_full = _DR_PHYSICS_BASE["static_friction_range"][1] - _DR_PHYSICS_BASE["static_friction_range"][0]
    phys_cur = p["static_friction_range"][1] - p["static_friction_range"][0]  # type: ignore[index]
    jn_full = _DR_JOINT_NON_ANKLE_BASE[1] - _DR_JOINT_NON_ANKLE_BASE[0]
    jn_cur = dr["joint_non_ankle"][1] - dr["joint_non_ankle"][0]  # type: ignore[index]
    ja_full = _DR_JOINT_ANKLE_BASE[1] - _DR_JOINT_ANKLE_BASE[0]
    ja_cur = dr["joint_ankle"][1] - dr["joint_ankle"][0]  # type: ignore[index]
    cx_full = _DR_BASE_COM_BASE["x"][1] - _DR_BASE_COM_BASE["x"][0]
    cx_cur = dr["base_com"]["x"][1] - dr["base_com"]["x"][0]  # type: ignore[index]
    parts: list[float] = []
    if bool(state.get("dr_physics_material_enabled", True)):
        parts.append(phys_cur / max(phys_full, 1e-6))
    if bool(state.get("dr_joint_non_ankle_enabled", True)):
        parts.append(jn_cur / max(jn_full, 1e-6))
    if bool(state.get("dr_joint_ankle_enabled", True)):
        parts.append(ja_cur / max(ja_full, 1e-6))
    if bool(state.get("dr_base_com_enabled", True)):
        parts.append(cx_cur / max(cx_full, 1e-6))
    if bool(state.get("dr_hoi_terrain_physics_enabled", True)):
        ht = dr["hoi_terrain"]  # type: ignore[assignment]
        ht_cur = float(ht["static_friction_range"][1]) - float(ht["static_friction_range"][0])  # type: ignore[index]
        ht_full = _DR_HOI_TERRAIN_PHYSICS_BASE["static_friction_range"][1] - _DR_HOI_TERRAIN_PHYSICS_BASE["static_friction_range"][0]
        parts.append(ht_cur / max(ht_full, 1e-6))
    if not parts:
        return torch.tensor(float("nan"), device=env.device)
    r = sum(parts) / float(len(parts))
    return torch.tensor(float(max(min(r, 1.0), 0.0)), device=env.device)


def _assert_hoi_curriculum_eight_level_invariants() -> None:
    """Module-load checks: list lengths, push/EE endpoints vs anchors, HOI DR scales vs DR_* tables."""
    n = _HOI_CURRICULUM_NUM_LEVELS
    _validate_level_lists(n)
    eps = 1e-9

    def _push_rows_close(x: dict[str, object], y: dict[str, object]) -> None:
        vx = x["velocity_range"]  # type: ignore[assignment]
        vy = y["velocity_range"]  # type: ignore[assignment]
        for key in vx:
            assert abs(float(vx[key][0]) - float(vy[key][0])) <= eps  # type: ignore[index]
            assert abs(float(vx[key][1]) - float(vy[key][1])) <= eps  # type: ignore[index]
        ix = tuple(x["interval_range_s"])  # type: ignore[arg-type]
        iy = tuple(y["interval_range_s"])  # type: ignore[arg-type]
        assert abs(float(ix[0]) - float(iy[0])) <= eps
        assert abs(float(ix[1]) - float(iy[1])) <= eps

    _push_rows_close(_DEFAULT_PUSH_LEVELS[0], _ANCHOR_PUSH_LEVELS[0])
    _push_rows_close(_DEFAULT_PUSH_LEVELS[-1], _ANCHOR_PUSH_LEVELS[-1])

    assert abs(_DEFAULT_EE_THRESHOLDS[0] - _ANCHOR_EE_THRESHOLDS[0]) <= eps
    assert abs(_DEFAULT_EE_THRESHOLDS[-1] - _ANCHOR_EE_THRESHOLDS[-1]) <= eps
    for k in range(n - 1):
        assert _DEFAULT_EE_THRESHOLDS[k] + eps >= _DEFAULT_EE_THRESHOLDS[k + 1]

    assert len(HOI_DR_PHYSICS_MATERIAL_SCALES) == n
    for scales, levels in (
        (HOI_DR_PHYSICS_MATERIAL_SCALES, DR_PHYSICS_MATERIAL_LEVELS),
        (HOI_DR_JOINT_NON_ANKLE_SCALES, DR_JOINT_NON_ANKLE_LEVELS),
        (HOI_DR_JOINT_ANKLE_SCALES, DR_JOINT_ANKLE_LEVELS),
        (HOI_DR_BASE_COM_SCALES, DR_BASE_COM_LEVELS),
    ):
        for k in range(n):
            assert abs(float(levels[k]["scale"]) - float(scales[k])) <= eps  # type: ignore[index]

    assert len(HOI_DR_HOI_TERRAIN_SCALES) == n
    for k in range(n):
        assert abs(float(DR_HOI_TERRAIN_MATERIAL_LEVELS[k]["scale"]) - float(HOI_DR_HOI_TERRAIN_SCALES[k])) <= eps  # type: ignore[index]

    assert len(_DEFAULT_BODY_POS_PASS_THRESHOLDS) == n - 1
    assert abs(_DEFAULT_BODY_POS_PASS_THRESHOLDS[0] - _ANCHOR_BODY_POS_PASS_Y[0]) <= eps
    assert abs(_DEFAULT_BODY_POS_PASS_THRESHOLDS[-1] - _ANCHOR_BODY_POS_PASS_Y[-1]) <= eps
    for k in range(n - 2):
        assert _DEFAULT_BODY_POS_PASS_THRESHOLDS[k] + eps >= _DEFAULT_BODY_POS_PASS_THRESHOLDS[k + 1]


_assert_hoi_curriculum_eight_level_invariants()
