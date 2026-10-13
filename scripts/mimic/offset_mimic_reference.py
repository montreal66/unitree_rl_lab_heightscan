"""Create a translated copy of a converted Mimic motion archive.

The translation is applied to every tracked body in every frame.  It therefore
keeps reset poses and motion-command targets consistent while leaving the
terrain fixed -- useful for controlled terrain-offset tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser(description="Translate a converted Mimic NPZ in world coordinates.")
parser.add_argument("--input", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--x", type=float, default=0.0)
parser.add_argument("--y", type=float, default=0.0)
parser.add_argument("--z", type=float, default=0.0)
args = parser.parse_args()

if not args.input.is_file():
    raise FileNotFoundError(args.input)
if args.output.resolve() == args.input.resolve():
    raise ValueError("--output must differ from --input")

with np.load(args.input, allow_pickle=False) as archive:
    data = {name: archive[name] for name in archive.files}

if "body_pos_w" not in data or data["body_pos_w"].ndim != 3 or data["body_pos_w"].shape[-1] != 3:
    raise ValueError("Expected converted Mimic field body_pos_w with shape (frames, bodies, 3)")

offset = np.asarray((args.x, args.y, args.z), dtype=data["body_pos_w"].dtype)
data["body_pos_w"] = data["body_pos_w"].copy()
data["body_pos_w"] += offset
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez(args.output, **data)
print(f"Wrote {args.output} with world offset {offset.tolist()} applied to all bodies.")
