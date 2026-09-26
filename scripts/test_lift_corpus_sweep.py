# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-lift corpus qualification sweep (docs/studies/2026-09-27-test-lift-corpus.md).

For each catalog object that passes the cheap filter (``analysis.test_lift.corpus``), in the
order handal, ycb, hope, then the other datasets:

1. **Dump** (one Isaac process): ``test_lift_batch.py --task-file generic_test_lift_task.py
   --dump-candidates`` with ``--candidate-filter cone --n-candidates 1000``. The process
   settles the object on the table and writes the cone-filtered GraspGenX candidates, the
   rest pose and ``z_table``.
2. **Funnel** (this process, numpy): the ``both`` filter of the driver is recomputed on that
   set with the driver's own ``candidate_keep_mask`` (the generic scene has no neighbours, so
   ``both`` = cone AND table-free, exactly what ``--candidate-filter both`` keeps), then the
   pad-occupancy filter (``corpus.on_object_mask``). The final set goes to
   ``cands/<key>.npz`` in the driver's candidates-file format, order preserved.
3. **Frame check**: ``Z_TABLE_MIN_M <= z_table <= Z_TABLE_MAX_M``. A failure stops here.
4. **Grasp check** (one Isaac process): ``--candidates-file cands/<key>.npz --label-all
   --theta-id 0 --cand-range 0 min(32, n)``: every env executes one candidate at the default
   theta (catalog mass, CoM offset 0).
5. **Verdict**: ``corpus.verdict``; one JSON line per object in ``results.jsonl``, and the
   whole table rewritten to ``results.csv``.

Resumable: an object with a line in ``results.jsonl`` is skipped (``--redo`` repeats it), and a
finished dump or label set is reused. Every Isaac process runs under
``systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=2G`` and starts only when
``MemAvailable`` is at least ``--min-avail-gb``. Each stage has a wall-clock deadline
(``--stage-timeout-s``); on expiry the orchestrator kills that stage's process group.

The GraspGenX server must already answer on 127.0.0.1:5556 (the dump stage needs it).

Usage::

    /home/chungyili/Codes/RoboLab/.venv/bin/python3 -u scripts/test_lift_corpus_sweep.py \\
        --out /home/chungyili/Codes/RoboLab/output/test_lift/corpus
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from analysis.test_lift import corpus as C  # noqa: E402
from analysis.test_lift.batch import APPROACH_Z_MAX, GRASP_DEPTH_OFFSET, candidate_keep_mask  # noqa: E402
from analysis.test_lift.episode_log import read_episode  # noqa: E402

TASK_FILE = "generic_test_lift_task.py"
CSV_COLS = ["key", "dataset", "name", "mass", "verdict", "reason", "z_table", "n_raw", "n_cone", "n_both",
            "n_final", "tip_dist_both_median", "off_object_both_frac", "n_checked", "n_reached", "reach",
            "close_on_air", "held", "tip_z_median", "override", "wall_s"]


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def run_stage(cmd, log_path, env, timeout_s, min_avail_gb) -> tuple[int, float]:
    """Run one Isaac stage in its own process group under the memory-capped scope."""
    while mem_available_gb() < min_avail_gb:
        print(f"  [wait] MemAvailable {mem_available_gb():.1f} GB < {min_avail_gb} GB", flush=True)
        time.sleep(60)
    scoped = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=12G",
              "-p", "MemorySwapMax=2G", "--"] + cmd
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(scoped, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=env,
                             start_new_session=True)
        print(f"  [stage] pid={p.pid} log={log_path}", flush=True)
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = -999
    return rc, time.time() - t0


def log_tail(path, n=3) -> str:
    try:
        with open(path, errors="replace") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        err = [ln for ln in lines if "Error" in ln or "error" in ln or "Traceback" in ln]
        return " | ".join((err[-n:] if err else lines[-n:]))[:400]
    except OSError:
        return ""


def write_csv(out_dir, rows):
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def load_rows(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["key"]] = r
    return rows


def qualify(key, entry, a, env) -> dict:
    t0 = time.time()
    mass = C.default_mass(entry)
    row = dict(key=key, dataset=entry["dataset"], name=entry["name"], mass=mass)
    usd = os.path.join(REPO, entry["usd_path"])
    if not os.path.exists(usd):
        row["error"] = f"asset file missing: {entry['usd_path']}"
        return row
    for d in ("cands_cone", "cands", "logs", "labels", "scratch"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    base = [a.python, "-u", "scripts/test_lift_batch.py", "--task-file", TASK_FILE, "--object", key,
            "--mass", f"{mass}", "--com-offset", "0", "0", "0", "--seeds", "0", "--headless"]

    # ---- 1. dump -----------------------------------------------------------------------
    cone_path = os.path.join(a.out, "cands_cone", f"{key}.npz")
    if not os.path.exists(cone_path):
        cmd = base + ["--arms", "top1", "--candidate-filter", "cone", "--n-candidates", str(a.n_candidates),
                      "--dump-candidates", cone_path, "--out", os.path.join(a.out, "scratch")]
        log = os.path.join(a.out, "logs", f"{key}.dump.log")
        rc, dt = run_stage(cmd, log, env, a.stage_timeout_s, a.min_avail_gb)
        print(f"  [dump] rc={rc} {dt:.0f}s", flush=True)
        if not os.path.exists(cone_path):
            row["error"] = f"dump failed rc={rc}: {log_tail(log)}"
            row["wall_s"] = time.time() - t0
            return row

    # ---- 2. funnel ---------------------------------------------------------------------
    c = np.load(cone_path, allow_pickle=False)
    G, confs, P, T_rest = c["grasps_o"], c["confs"], c["points_o"], c["T_obj_rest"]
    z_table = float(c["z_table"])
    both = candidate_keep_mask(G, T_rest, APPROACH_Z_MAX, "both", z_table, np.zeros((0, 3)),
                               GRASP_DEPTH_OFFSET, None)
    onobj = C.on_object_mask(G, P, GRASP_DEPTH_OFFSET)
    final = both & onobj
    td = C.tip_distance(G[both], P) if both.any() else np.array([np.nan])
    row.update(z_table=z_table, n_raw=int(c["n_raw"]), n_cone=int(len(G)), n_both=int(both.sum()),
               n_final=int(final.sum()), tip_dist_both_median=float(np.nanmedian(td)),
               off_object_both_frac=float(np.mean(td > 0.015)) if both.any() else float("nan"),
               override=os.path.exists(os.path.join(a.usd_dir, f"{key}.override.usda")))
    cand_path = os.path.join(a.out, "cands", f"{key}.npz")
    if final.any():
        np.savez_compressed(cand_path, grasps_o=G[final], confs=confs[final], points_o=P, T_obj_rest=T_rest,
                            z_table=z_table, object=key, candidate_filter="both+on_object",
                            n_raw=int(c["n_raw"]))

    # ---- 3. frame check / 4. grasp check -----------------------------------------------
    ok_frame = C.Z_TABLE_MIN_M <= z_table <= C.Z_TABLE_MAX_M
    if ok_frame and final.any():
        n_chk = min(C.N_CHECK, int(final.sum()))
        lab_dir = os.path.join(a.out, "labels", key, "theta_00")
        have = sorted(glob.glob(os.path.join(lab_dir, "cand_*.npz")))
        if len(have) < n_chk:
            cmd = base + ["--candidates-file", cand_path, "--label-all", "--theta-id", "0",
                          "--cand-range", "0", str(n_chk), "--out", os.path.join(a.out, "labels")]
            log = os.path.join(a.out, "logs", f"{key}.label.log")
            rc, dt = run_stage(cmd, log, env, a.stage_timeout_s, a.min_avail_gb)
            print(f"  [label] rc={rc} {dt:.0f}s", flush=True)
            have = sorted(glob.glob(os.path.join(lab_dir, "cand_*.npz")))
            if len(have) < n_chk:
                row["error"] = f"label stage wrote {len(have)}/{n_chk} rc={rc}: {log_tail(log)}"
        eps = [read_episode(f) for f in have[:n_chk]]
        row.update(C.grasp_check_stats(eps))
    row["wall_s"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "output", "test_lift", "corpus"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only", nargs="*", default=None, help="corpus keys to run (default: the whole filtered list)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--n-candidates", type=int, default=1000)
    ap.add_argument("--stage-timeout-s", type=float, default=1200)
    ap.add_argument("--min-avail-gb", type=float, default=10.0)
    a = ap.parse_args()
    a.out = os.path.abspath(a.out)
    a.usd_dir = os.path.join(a.out, "usd")
    os.makedirs(a.out, exist_ok=True)

    cat = C.load_catalog()
    keys = C.corpus_keys(cat)
    order = a.only if a.only else C.sweep_order(cat)
    if a.limit:
        order = order[:a.limit]
    res_path = os.path.join(a.out, "results.jsonl")
    rows = load_rows(res_path)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", ROBOLAB_TEST_LIFT_CORPUS_USD_DIR=a.usd_dir,
               PYTHONUNBUFFERED="1")
    print(f"[corpus] {len(order)} objects, {sum(k in rows for k in order)} already done, out={a.out}", flush=True)
    t_start, n_run = time.time(), 0
    for i, key in enumerate(order):
        if key in rows and not a.redo:
            continue
        print(f"[corpus] {i + 1}/{len(order)} {key} ({keys[key]['dataset']})", flush=True)
        try:
            row = qualify(key, keys[key], a, env)
        except Exception as e:  # one bad asset must not stop the sweep
            row = dict(key=key, dataset=keys[key]["dataset"], name=keys[key]["name"], error=repr(e)[:400])
        ok, why = C.verdict(row)
        row.update(verdict="PASS" if ok else "FAIL", reason=why)
        rows[key] = row
        with open(res_path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        write_csv(a.out, [rows[k] for k in order if k in rows])
        n_run += 1
        el = time.time() - t_start
        left = sum(k not in rows for k in order)
        print(f"[corpus] {key}: {row['verdict']} ({why}) | {n_run} run in {el / 60:.1f} min, "
              f"{el / n_run / 60:.2f} min/object, {left} left, ETA {left * el / n_run / 60:.0f} min", flush=True)
    n_pass = sum(rows[k].get("verdict") == "PASS" for k in order if k in rows)
    print(f"[corpus] DONE: {n_pass} PASS of {sum(k in rows for k in order)}", flush=True)


if __name__ == "__main__":
    main()
