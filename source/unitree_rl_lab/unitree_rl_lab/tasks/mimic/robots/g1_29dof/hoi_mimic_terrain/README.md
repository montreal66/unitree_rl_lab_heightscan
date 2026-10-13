# climb_15 HOI terrain task

This package registers the fixed-terrain blind baseline:

`Unitree-G1-29dof-Mimic-HOI_terrain`

The sibling `hoi_mimic_terrain_perceptive_raycast` package registers the height-scan policy:

`Unitree-G1-29dof-Mimic-HOI_terrain-Perceptive-Raycast`

The required fixed assets are versioned with this branch:

```text
logs/hoi_mimic_data/climb_15_z_scale_1.0_mimic.npz
logs/hoi_mimic_data/climb_15_z_scale_1.0_mimic.terrain.json
datasets/models/terrain/climb_15/multi_boxes_z_scale_1.0.urdf
```

The G1 29-DoF USD used by the task is also included below `assets/unitree_model/G1/29dof/`. The default model path is repository-relative; set `UNITREE_MODEL_DIR` only when using a different Unitree model installation.

The helper `scripts/rsl_rl/eval_hoi_offset.py` evaluates a checkpoint at a fixed reference frame with a controlled root/terrain relative offset. It writes JSON results under `logs/offset_eval/`; checkpoints and generated evaluation results are not versioned.
