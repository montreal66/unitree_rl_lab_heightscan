"""Isaac Lab spawn configs for HOI dataset URDFs."""

from __future__ import annotations

import os
import re
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg

from unitree_rl_lab.assets.robots.unitree import (
    UNITREE_G1_29DOF_CFG,
    UnitreeArticulationCfg,
    UnitreeUrdfFileCfg,
)


def ensure_world_link_urdf(urdf_path: str) -> str:
    """Return a path to a URDF usable by Isaac Lab (terrain URDFs may omit ``world`` link).

    If the file already defines ``<link name="world" .../>``, returns ``urdf_path`` unchanged.
    Otherwise writes a temp copy with ``<link name="world"/>`` inserted after the opening
    ``<robot>`` tag.
    """
    path = Path(urdf_path).resolve()
    text = path.read_text(encoding="utf-8")
    if re.search(r'<link\s+name\s*=\s*["\']world["\']', text):
        return str(path)

    robot_open = re.search(r"<robot\b[^>]*>", text)
    if not robot_open:
        raise ValueError(f"No <robot> tag in {urdf_path}")

    insert_at = robot_open.end()
    patched = text[:insert_at] + '\n  <link name="world"/>' + text[insert_at:]

    # Keep patched file beside original so relative mesh paths
    # (e.g. ``box_models/box1.obj``) continue to resolve.
    patched_path = path.with_name(f"{path.stem}_isaac_world{path.suffix}")
    patched_path.write_text(patched, encoding="utf-8")
    return str(patched_path)


def make_robot_cfg(
    urdf_path: str,
    *,
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    use_usd: bool = False,
) -> UnitreeArticulationCfg:
    """Articulation config for G1 replay (dataset URDF by default).

    Args:
        urdf_path: Absolute path to ``g1_29dof*.urdf`` when ``use_usd`` is False.
        prim_path: USD/articulation prim path template.
        use_usd: If True, ignore ``urdf_path`` and use ``UNITREE_G1_29DOF_CFG`` spawn (USD).
    """
    if use_usd:
        return UNITREE_G1_29DOF_CFG.replace(
            prim_path=prim_path,
            actuators={
                "replay": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=0.0,
                    damping=0.0,
                ),
            },
        )

    abs_urdf = os.path.abspath(urdf_path)
    return UnitreeArticulationCfg(
        prim_path=prim_path,
        spawn=UnitreeUrdfFileCfg(asset_path=abs_urdf),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "replay": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
        joint_sdk_names=UNITREE_G1_29DOF_CFG.joint_sdk_names.copy(),
    )


def make_object_cfg(
    urdf_path: str,
    *,
    prim_path: str = "{ENV_REGEX_NS}/HOI_Object",
) -> AssetBaseCfg:
    """Object config for largebox / chair URDF.

    Chair-like assets are multi-link and require ``ArticulationCfg``. Single-link assets
    (e.g. ``largebox``) are spawned as ``RigidObjectCfg``.
    """
    abs_urdf = os.path.abspath(urdf_path)
    urdf_text = Path(abs_urdf).read_text(encoding="utf-8")
    has_joint = re.search(r"<joint\b", urdf_text) is not None

    if has_joint:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UrdfFileCfg(
                asset_path=abs_urdf,
                fix_base=False,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
        )

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=abs_urdf,
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
    )


def make_terrain_cfg(urdf_path: str, *, prim_path: str = "{ENV_REGEX_NS}/HOI_Terrain") -> ArticulationCfg:
    """Fixed terrain from ``multi_boxes`` URDF as articulation for manager compatibility."""
    fixed_urdf = ensure_world_link_urdf(urdf_path)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UrdfFileCfg(
            asset_path=os.path.abspath(fixed_urdf),
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
