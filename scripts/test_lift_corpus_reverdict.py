#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-verdict the corpus qualification sweep by lift outcome only (user ruling 2026-09-27).

A grasp succeeds only when the object is lifted and held: ``final_ok`` (held through the
15 cm clear lift). Reach and close-on-air are NOT criteria. They stay as diagnostics of why a
grasp failed. An object passes when at least ``--min-lifts`` of its checked grasps succeed.

Reads ``<out>/results.jsonl`` (the sweep's rows) and ``<out>/labels/<key>/theta_*/cand_*.npz``.
Writes ``<out>/results_lift.csv``. Does not touch the running sweep or its modules.

    python3 scripts/test_lift_corpus_reverdict.py --out output/test_lift/corpus --min-lifts 3
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--min-lifts", type=int, default=3)
args = ap.parse_args()

rows = {}
with open(os.path.join(args.out, "results.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        rows[r["key"]] = r                 # a resumed object overwrites its older row

out_rows = []
for key, r in rows.items():
    files = sorted(glob.glob(os.path.join(args.out, "labels", key, "theta_*", "cand_*.npz")))
    ok, n = 0, 0
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            if bool(z["pad"]):
                continue
            n += 1
            ok += bool(z["final_ok"])
    if n == 0:
        verdict, reason = "FAIL", f"no lift episodes ({r.get('reason', '')})"
    elif ok >= args.min_lifts:
        verdict, reason = "PASS", f"{ok}/{n} lifted"
    else:
        verdict, reason = "FAIL", f"{ok}/{n} lifted < {args.min_lifts}"
    out_rows.append(dict(key=key, dataset=r.get("dataset"), n_checked=n, n_lifted=ok,
                         lift_rate=round(ok / n, 3) if n else None, verdict=verdict, reason=reason,
                         old_verdict=r.get("verdict"), reach=r.get("reach"),
                         close_on_air=r.get("close_on_air")))

out_rows.sort(key=lambda d: (d["verdict"] != "PASS", -(d["lift_rate"] or 0)))
path = os.path.join(args.out, "results_lift.csv")
with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
npass = sum(d["verdict"] == "PASS" for d in out_rows)
print(f"{npass}/{len(out_rows)} PASS (min lifts {args.min_lifts}) -> {path}")
for d in out_rows:
    print(f"  {d['verdict']:4s} {d['key']:28s} {d['reason']:32s} (old {d['old_verdict']})")
