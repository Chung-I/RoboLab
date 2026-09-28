"""Write one repeat-study pairs file per object (8 θ × all grasps × 8 repeats). C as the v5 driver computes it."""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robolab.tasks.test_lift.repeat_layout import repeat_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cands-dir", default="output/test_lift/corpus/cands")
ap.add_argument("--n-cand", type=int, default=24)
ap.add_argument("--out", required=True)
ap.add_argument("objects", nargs="+")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
for obj in args.objects:
    cf = np.load(os.path.join(args.cands_dir, f"{obj}.npz"), allow_pickle=True)
    C = int(min(args.n_cand, len(cf["confs"])))
    np.savez(os.path.join(args.out, f"{obj}.npz"), **repeat_pairs(C))
    print(obj, C, 8 * C * 8)
