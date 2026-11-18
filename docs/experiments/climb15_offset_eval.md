# climb_15 terrain-relative offset evaluation

This record compares the fixed-terrain **blind** policy against the
height-scan **perceptive raycast** policy under a fixed offset between the
robot/reference trajectory and terrain.

## Protocol

- Checkpoints: blind `model_3999.pt`; raycast `model_4000.pt`.
- Seed 42, 128 vectorized environments, one full 184-step reference clip.
- The evaluator applies a fixed `root_y_offset_m` before reset.
- The first reset step is warm-up only.  A rollout is successful if it has no
  termination from reference frames 1 through T-1.
- Curriculum mutation is disabled during evaluation, while level-0 startup
  domain randomization remains active.

## Results

| Root-y offset (m) | Blind success | Perceptive raycast success |
| ---: | ---: | ---: |
| -0.20 | 121 / 128 (94.5%) | 128 / 128 (100.0%) |
| 0.00 | 128 / 128 (100.0%) | 128 / 128 (100.0%) |
| +0.20 | 23 / 128 (18.0%) | 99 / 128 (77.3%) |

At the challenging `+0.20 m` offset, terrain-height perception improves the
single-seed success rate by 59.8 percentage points.  Raw evaluator outputs are
stored beside this record in [`climb15_offset_eval/`](climb15_offset_eval/).

The evaluator is [`scripts/rsl_rl/eval_hoi_offset.py`](../../scripts/rsl_rl/eval_hoi_offset.py).
These are single-seed measurements, not a multi-seed confidence interval.
