# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral metrics from the v2 event log (spec §3.5, §5.4, §8).

success@30s is the task predicate firing inside the step budget; grasp/lift
rates are cap-insensitive: they fire the moment the stage is reached, so a
timeout artifact (heavy trials running out of clock) shows as success_rate
falling while lift_rate holds.

Matching is case-insensitive: event ``name`` values are upper-cased
StatusCode enum members (e.g. ``OBJECT_GRABBED_SUCCESS``) while ``info``
values carry the lower-case predicate function name (e.g.
``object_picked_up(...)``), and STAGE_SUBSTRINGS is written in the lower-case
form, so every comparison below lower-cases both sides.

An empirical dump (Task 8, step 1 amendment) of get_all_env_events() from one
real OJCartonInCrateTask carton lift produced a single combined event at
step 157: ``name='OBJECT_GRABBED_SUCCESS'``, ``info='success:
object_picked_up(object=orange_juice_carton, surface=table). advanced 2
step(s)...'``. With case-insensitive matching this event correctly satisfies
*both* the grasp and lift substrings, so ``t_grasp_s == t_lift_s == 157/15``
for that episode: the state machine's subtask tracker advanced two
conditions (object_grabbed, object_picked_up) in the same check and folded
them into one event, so grasp and lift are recorded as simultaneous — not,
as an earlier version of this docstring claimed, as an unrecoverable None.
The subtlety that remains open for Phase 1 is entirely about *when* that
folded event fires relative to the true moment of grasp: object_picked_up's
`_and(object_grabbed(...), ...)` re-evaluates object_grabbed's contact-based
signal at check time, the same flaky in_contact reporting carried as a risk
from Task 5, so a *genuinely earlier* standalone t_grasp (before the object
is already lifted) requires fixing that contact-sensing lag at the
RoboLab/env level — this module's case-insensitive substring matching cannot
recover a transition that was never logged as its own event. See
task-8-report.md for the full dump and root-cause trace.
"""

import argparse
import csv
import json
from pathlib import Path

STAGE_SUBSTRINGS = {"grasp": "object_grabbed", "lift": "object_picked_up",
                    "place": "object_in_container", "drop": "object_dropped"}


def _blob(e: dict) -> str:
    """Lower-cased "name info" blob for substring matching, name and info
    joined with a separator so a substring can't span the boundary between
    them."""
    return ((e.get("name") or "") + " " + (e.get("info") or "")).lower()


def _first_step(events, key):
    sub = STAGE_SUBSTRINGS[key]
    for e in events:
        if sub in _blob(e):
            return e["step"]
    return None


def episode_metrics(events: list[dict], num_steps: int, control_hz: float = 15.0) -> dict:
    t_grasp = _first_step(events, "grasp")
    t_lift = _first_step(events, "lift")
    success = any(
        (STAGE_SUBSTRINGS["place"] in _blob(e))
        and e.get("score", 0.0) >= 1.0
        for e in events
    )
    # a regrasp = grasp after a drop that itself followed a grasp
    n_regrasp, seen_grasp, dropped = 0, False, False
    for e in events:
        blob = _blob(e)
        if STAGE_SUBSTRINGS["grasp"] in blob:
            if seen_grasp and dropped:
                n_regrasp += 1
            seen_grasp, dropped = True, False
        elif STAGE_SUBSTRINGS["drop"] in blob and seen_grasp:
            dropped = True
    return {
        "success": success,
        "grasped": t_grasp is not None,
        "lifted": t_lift is not None,
        "t_grasp_s": (t_grasp / control_hz) if t_grasp is not None else None,
        "t_lift_s": (t_lift / control_hz) if t_lift is not None else None,
        "n_regrasp": n_regrasp,
    }


def aggregate_cell(episodes: list[dict]) -> dict:
    n = len(episodes)
    def rate(k): return sum(1 for e in episodes if e[k]) / n if n else 0.0
    def mean(k):
        vals = [e[k] for e in episodes if e[k] is not None]
        return (sum(vals) / len(vals)) if vals else None
    return {
        "n": n,
        "success_rate": rate("success"),
        "grasp_rate": rate("grasped"),
        "lift_rate": rate("lifted"),
        "mean_t_grasp_s": mean("t_grasp_s"),
        "mean_t_lift_s": mean("t_lift_s"),
        "mean_n_regrasp": (sum(e["n_regrasp"] for e in episodes) / n) if n else 0.0,
    }


def load_episode_events(output_folder: str) -> dict[str, list[list[dict]]]:
    """Map env/cell name -> list of per-episode v2 event lists, from the
    results files an eval run writes under output/<folder>/<ENV_NAME>/."""
    cells = {}
    for results_file in sorted(Path(output_folder).glob("*/**/*.json")):
        try:
            data = json.loads(results_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        # v2 layout: per-episode dicts carrying an "events" list (see
        # robolab/eval/summarize.py + robolab/core/logging/results.py).
        episodes = data.get("episode_results") or data.get("episodes") or []
        ev_lists = [ep.get("events_list") or ep.get("events") or [] for ep in episodes
                    if isinstance(ep, dict)]
        if ev_lists:
            cells.setdefault(results_file.parent.name, []).extend(
                e for e in ev_lists if isinstance(e, list))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_folder")
    parser.add_argument("--num-steps", type=int, default=450)
    parser.add_argument("--csv", default="metrics.csv")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mass-com-vla-probing")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cells = load_episode_events(args.output_folder)
    rows = []
    for cell, ev_lists in sorted(cells.items()):
        eps = [episode_metrics(ev, args.num_steps) for ev in ev_lists]
        rows.append({"cell": cell, **aggregate_cell(eps)})
        print(rows[-1])
    out = Path(args.output_folder) / args.csv
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")

    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name,
                         config={"output_folder": args.output_folder})
        run.log({"metrics": wandb.Table(
            columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])})
        run.finish()


if __name__ == "__main__":
    main()
