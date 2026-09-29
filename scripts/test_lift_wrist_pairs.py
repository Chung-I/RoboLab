"""Pairs files for the wrist-value study (daily-logs SPEC_wrist_value.md) from its frozen selection JSON."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robolab.tasks.test_lift.repeat_layout import rep_blocks, wrist_pairs  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--selection", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
for o in json.load(open(args.selection))["objects"]:
    for k, reps in enumerate(rep_blocks(o["C"])):
        p = wrist_pairs(o["C"], o["thetas"], reps)
        np.savez(os.path.join(args.out, f"{o['name']}_b{k}.npz"), **p)
        print(o["name"], k, len(p["theta_idx"]))
