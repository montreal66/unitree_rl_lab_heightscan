# climb_15 HOI terrain task

This package registers the fixed-terrain blind baseline:

`Unitree-G1-29dof-Mimic-HOI_terrain`

The sibling `hoi_mimic_terrain_perceptive_raycast` package registers the height-scan policy:

`Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast`

Both tasks require generated, non-versioned assets below the Isaac stack root:

```text
logs/hoi_mimic_data/climb_15_z_scale_1.0_mimic.npz
logs/hoi_mimic_data/climb_15_z_scale_1.0_mimic.terrain.json
datasets/models/terrain/climb_15/multi_boxes_z_scale_1.0.urdf
```

The helper `scripts/rsl_rl/eval_hoi_offset.py` evaluates a checkpoint at a fixed reference frame with a controlled root/terrain relative offset.  It writes JSON results under `logs/offset_eval/` and does not version checkpoints or generated data.
