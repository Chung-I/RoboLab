# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarise a corpus qualification sweep as a markdown table (docs/studies/2026-09-27-test-lift-corpus.md).

Reads ``<out>/results.jsonl`` (written by ``scripts/test_lift_corpus_sweep.py``) and each
object's label log. Adds one column the sweep does not store: how many reach misses are
along-axis blocks (``d_along > 7 mm`` and ``d_lat < 6 mm`` in the ``[reach]`` line), which is
the signature of a hand stopped by contact before the 1 cm ``GRASP_DEPTH_OFFSET`` target.

Usage::

    python3 scripts/test_lift_corpus_report.py --out /home/chungyili/Codes/RoboLab/output/test_lift/corpus
"""
import argparse
import json
import os
import re
from collections import Counter

_REACH = re.compile(r"^\[reach\] g1 .*ik_err=([-\d.]+) d_along=([-+\d.]+) d_lat=([-\d.]+)")


def blocked_misses(log_path):
    n_miss = n_block = 0
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _REACH.match(line)
            if m and float(m.group(1)) >= 0.01:
                n_miss += 1
                n_block += float(m.group(2)) > 0.007 and float(m.group(3)) < 0.006
    return n_miss, n_block


def fmt(v, pct=False, nd=4):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{100 * v:.0f}" if pct else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    with open(os.path.join(a.out, "results.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["key"]] = r
    print("| object | dataset | verdict | z_table (m) | n cone | n both | n final | off-object in both % "
          "| reach % | close-on-air % | held % | blocked misses | reason |")
    print("|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for k, r in rows.items():
        nm, nb = blocked_misses(os.path.join(a.out, "logs", f"{k}.label.log"))
        print(f"| `{k}` | {r.get('dataset')} | {'**PASS**' if r.get('verdict') == 'PASS' else 'FAIL'} "
              f"| {fmt(r.get('z_table'))} | {fmt(r.get('n_cone'))} | {fmt(r.get('n_both'))} | {fmt(r.get('n_final'))} "
              f"| {fmt(r.get('off_object_both_frac'), pct=True)} | {fmt(r.get('reach'), pct=True)} "
              f"| {fmt(r.get('close_on_air'), pct=True)} | {fmt(r.get('held'), pct=True)} "
              f"| {'-' if nm is None else f'{nb}/{nm}'} | {r.get('reason', '')[:60]} |")
    v = Counter(r.get("verdict") for r in rows.values())
    reasons = Counter(r.get("reason", "").split(":")[0].split(" ")[0] for r in rows.values() if r.get("verdict") != "PASS")
    print(f"\n{len(rows)} objects: {v.get('PASS', 0)} PASS, {v.get('FAIL', 0)} FAIL. "
          f"First failed criterion: {dict(reasons)}")


if __name__ == "__main__":
    main()
