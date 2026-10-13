"""Batch-evaluate a HOI checkpoint at a fixed world-Y trajectory offset.

The terrain stays fixed.  The entire reference target is aligned to the robot
reset anchor through MotionCommand, so ``--root-y-offset`` changes only the
robot/terrain relative placement and does not edit the Mimic NPZ.

One vectorized rollout evaluates one full reference clip per environment.
Success means no termination before the final reference frame.  The JSON
result includes the first termination term for every failed environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate a HOI checkpoint at a fixed terrain-relative offset.")
parser.add_argument("--task", required=True, type=str)
parser.add_argument("--checkpoint", required=True, type=Path)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--root-y-offset", type=float, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Output JSON path (default: logs/offset_eval/<task>_<checkpoint>_<offset>.json).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs <= 0:
    raise ValueError("--num_envs must be positive")
if not args_cli.checkpoint.is_file():
    raise FileNotFoundError(args_cli.checkpoint)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def default_output_path() -> Path:
    safe_task = args_cli.task.lower().replace("-", "_")
    safe_offset = f"{args_cli.root_y_offset:+.2f}".replace("+", "plus").replace("-", "minus").replace(".", "p")
    return Path("logs/offset_eval") / f"{safe_task}_{args_cli.checkpoint.stem}_y{safe_offset}_seed{args_cli.seed}.json"


def main() -> None:
    # Use the training env entry point, rather than the play entry point: the
    # latter forcibly reduces the scene to one environment.  Frame 0 is fixed
    # so every vector element evaluates the same reference clip under its own
    # startup domain-randomization sample.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="env_cfg_entry_point",
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.motion.motion_start_time_step = 0
    env_cfg.commands.motion.pose_range["y"] = (args_cli.root_y_offset, args_cli.root_y_offset)

    # Keep level-0 startup domain randomization, but do not let a short
    # evaluation mutate the curriculum state while it is being measured.
    env_cfg.hoi_curriculum.enable = False

    print(f"[INFO] Creating {args_cli.num_envs} evaluation environments...", flush=True)
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=None)
    print("[INFO] Environment reset complete.", flush=True)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}", flush=True)

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    motion = env.unwrapped.command_manager.get_term("motion")
    # On this task, MotionCommand writes root/joint states during reset but
    # computes ``body_pos_relative_w`` only at the end of the first env.step.
    # The training loop consequently treats that first step as reset warm-up;
    # scoring it would classify every fresh environment as ee_body_pos failed.
    # Exclude it, then score reference frames 1..T-1.
    horizon_steps = int(motion.motion.time_step_total) - 2
    if horizon_steps <= 0:
        raise RuntimeError("Motion must contain at least two frames")

    termination_manager = env.unwrapped.termination_manager
    term_names = list(termination_manager.active_terms)
    device = env.unwrapped.device
    alive = torch.ones(args_cli.num_envs, dtype=torch.bool, device=device)
    first_failure_step = torch.full((args_cli.num_envs,), -1, dtype=torch.int32, device=device)
    first_failure_term = ["" for _ in range(args_cli.num_envs)]
    print("[INFO] Running unscored reset warm-up step...", flush=True)
    with torch.inference_mode():
        warmup_actions = policy(obs)
        obs, _, _, _ = env.step(warmup_actions)

    print(f"[INFO] Evaluating {horizon_steps} scored policy steps at y offset {args_cli.root_y_offset:+.2f} m...", flush=True)

    with torch.inference_mode():
        for step in range(1, horizon_steps + 1):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            done = dones.to(dtype=torch.bool)
            newly_failed = alive & done
            if torch.any(newly_failed):
                for term_name in term_names:
                    term_fired = termination_manager.get_term(term_name)
                    rows = torch.nonzero(newly_failed & term_fired, as_tuple=False).flatten().cpu().tolist()
                    for row in rows:
                        if not first_failure_term[row]:
                            first_failure_term[row] = term_name
                # A done must always map to a term; retain a visible fallback
                # instead of silently classifying it as success.
                rows = torch.nonzero(newly_failed, as_tuple=False).flatten().cpu().tolist()
                for row in rows:
                    if not first_failure_term[row]:
                        first_failure_term[row] = "unknown_done"
                first_failure_step[newly_failed] = step
                alive[newly_failed] = False
            if step % 25 == 0 or step == horizon_steps:
                print(f"[INFO] Step {step}/{horizon_steps}; alive={int(alive.sum().item())}", flush=True)

    first_failure_step_cpu = first_failure_step.cpu().tolist()
    success_count = int(alive.sum().item())
    term_counts = {name: 0 for name in term_names}
    term_counts["unknown_done"] = 0
    failures = []
    for env_id, term_name in enumerate(first_failure_term):
        if term_name:
            term_counts[term_name] = term_counts.get(term_name, 0) + 1
            failures.append({"env_id": env_id, "step": first_failure_step_cpu[env_id], "term": term_name})

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": args_cli.task,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "seed": args_cli.seed,
        "num_envs": args_cli.num_envs,
        "root_y_offset_m": args_cli.root_y_offset,
        "horizon_steps": horizon_steps,
        "success_definition": "After one unscored reset warm-up step, no termination through reference frame T-1.",
        "success_count": success_count,
        "success_rate": success_count / args_cli.num_envs,
        "first_termination_counts": term_counts,
        "failures": failures,
    }
    output = args_cli.output or default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[INFO] Wrote {output}")


if __name__ == "__main__":
    main()
    # Isaac Sim 5.1 can keep Kit worker threads alive after a normal close in
    # short headless batch jobs.  Results are already flushed before this
    # point; exit directly so each condition in a shell loop releases its GPU.
    sys.stdout.flush()
    os._exit(0)
