# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral metrics from the v2 event log (spec §3.5, §5.4, §8).

success@30s is the task predicate firing inside the step budget; grasp/lift
rates are cap-insensitive: they fire the moment the stage is reached, so a
timeout artifact (heavy trials running out of clock) shows as success_rate
falling while lift_rate holds.

Stage matching is by numeric StatusCode, NOT by name/info text. Text matching
produced a real false positive: ``WRONG_OBJECT_GRABBED_FAILURE`` (code 250)
lower-cases to a string containing ``object_grabbed`` and so scored as a
successful grasp, and the state-machine regression event
``OBJECT_GRABBED_FAILURE`` (code 248) that follows every slip did the same.
Codes verified against ``robolab/core/task/status.py``:

  * line 77  ``OBJECT_GRABBED_SUCCESS = 139``       -- grasp
  * line 61  ``OBJECT_IN_CONTAINER_SUCCESS = 125``  -- place (task success)
  * line 155 ``TARGET_OBJECT_DROPPED = 263``        -- drop mid-transport

The lift stage has no StatusCode of its own: ``object_picked_up`` is a
predicate name, and ``StatusCode.subtask_to_success`` (status.py:170) falls
back to ``UNKNOWN_SUCCESS = 100`` for predicates with no matching member, so
100 cannot identify it. Lift therefore stays text-matched on the predicate
text in ``info``, with an explicit failure-word guard. Text matching is used
for the coded stages only when an event carries no ``code`` at all.

An empirical dump (Task 8, step 1 amendment) of get_all_env_events() from one
real OJCartonInCrateTask carton lift produced a single combined event at
step 157: ``code=139``, ``name='OBJECT_GRABBED_SUCCESS'``, ``info='success:
object_picked_up(object=orange_juice_carton, surface=table). advanced 2
step(s)...'``. That one event correctly satisfies *both* grasp (by code) and
lift (by info text), so ``t_grasp_s == t_lift_s == 157/15`` for that episode:
the state machine's subtask tracker advanced two conditions (object_grabbed,
object_picked_up) in the same check and folded them into one event. The
subtlety that remains open for Phase 1 is entirely about *when* that folded
event fires relative to the true moment of grasp: object_picked_up's
`_and(object_grabbed(...), ...)` re-evaluates object_grabbed's contact-based
signal at check time, the same flaky in_contact reporting carried as a risk
from Task 5, so a *genuinely earlier* standalone t_grasp (before the object
is already lifted) requires fixing that contact-sensing lag at the
RoboLab/env level — no log-side matching can recover a transition that was
never logged as its own event. See task-8-report.md for the full dump.
"""

import argparse
import csv
import json
import math
from pathlib import Path

# StatusCode values (robolab/core/task/status.py, lines cited in the module
# docstring). None = no dedicated code exists for that stage.
STAGE_CODES = {"grasp": 139, "lift": None, "place": 125, "drop": 263}

# Text fallbacks, used only for `lift` and for events that carry no `code`.
STAGE_SUBSTRINGS = {"grasp": "object_grabbed", "lift": "object_picked_up",
                    "place": "object_in_container", "drop": "object_dropped"}

# Words that mark an event as a failure/regression rather than a stage reached.
# "wrong_" catches WRONG_OBJECT_GRABBED_*; "failure"/"failed:"/"regress"
# catch OBJECT_GRABBED_FAILURE and the state machine's regression text.
FAILURE_MARKERS = ("wrong_", "failure", "failed:", "regress")


def _blob(e: dict) -> str:
    """Lower-cased "name info" blob for substring matching, name and info
    joined with a separator so a substring can't span the boundary between
    them."""
    return ((e.get("name") or "") + " " + (e.get("info") or "")).lower()


def _code(e: dict) -> int | None:
    c = e.get("code")
    return c if isinstance(c, int) else None


def _is_stage(e: dict, key: str) -> bool:
    """True if event `e` marks stage `key` being reached (never a failure)."""
    blob = _blob(e)
    want = STAGE_CODES[key]
    if want is not None:
        code = _code(e)
        if code is not None:
            return code == want
        # No code recorded (older logs / synthetic fixtures): fall back to the
        # predicate text, but never accept a failure or regression event.
        return STAGE_SUBSTRINGS[key] in blob and not _is_failure(blob)
    # Stages with no StatusCode of their own (lift) are text-only.
    return STAGE_SUBSTRINGS[key] in blob and not _is_failure(blob)


def _is_failure(blob: str) -> bool:
    return any(m in blob for m in FAILURE_MARKERS)


def _first_step(events, key, max_step=None):
    for e in events:
        if _is_stage(e, key) and (max_step is None or e["step"] <= max_step):
            return e["step"]
    return None


def _p95(vals: list[float]) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    return xs[max(0, math.ceil(0.95 * len(xs)) - 1)]


def episode_metrics(
    events: list[dict],
    num_steps: int | None,
    control_hz: float = 15.0,
    success: bool | None = None,
    final_step: int | None = None,
) -> dict:
    """Per-episode behavioral metrics.

    Args:
        events: the episode's v2 event list.
        num_steps: step budget. A stage counts only if its event fired at
            ``step <= num_steps``, which is what lets one re-score a longer
            recorded episode against a shorter cap (spec §3.5 success@30s).
            None (or a None `final_step` fallback) means "no cap".
        control_hz: control rate used to convert steps to seconds.
        success: the per-episode log file's own ``success`` bool. Preferred
            over the events-derived value when given, because it is the env's
            own termination verdict; it is still subject to the `num_steps`
            re-cap when a qualifying place event is present and lands beyond
            the budget.
        final_step: the log file's ``final_step``; used as the cap when
            `num_steps` is None.
    """
    cap = num_steps if num_steps is not None else final_step
    t_grasp = _first_step(events, "grasp", cap)
    t_lift = _first_step(events, "lift", cap)
    t_place = _first_step(events, "place", cap)
    # Events-derived success: a place event inside the budget. Some logs score
    # the place event 1.0; an uncoded log needs that score to disambiguate.
    ev_success = any(
        _is_stage(e, "place") and (cap is None or e["step"] <= cap)
        and (_code(e) == STAGE_CODES["place"] or e.get("score", 0.0) >= 1.0)
        for e in events
    )
    if success is None:
        is_success = ev_success
    else:
        # The file's own verdict wins, except when we can see that the place
        # event that earned it landed outside a tighter re-scoring budget.
        is_success = bool(success)
        if is_success and t_place is None and _first_step(events, "place") is not None:
            is_success = False
    # a regrasp = grasp (139) after a drop (263) that itself followed a grasp
    n_regrasp, seen_grasp, dropped = 0, False, False
    for e in events:
        if _is_stage(e, "grasp"):
            if seen_grasp and dropped:
                n_regrasp += 1
            seen_grasp, dropped = True, False
        elif _is_stage(e, "drop") and seen_grasp:
            dropped = True
    return {
        "success": is_success,
        "grasped": t_grasp is not None,
        "lifted": t_lift is not None,
        "t_grasp_s": (t_grasp / control_hz) if t_grasp is not None else None,
        "t_lift_s": (t_lift / control_hz) if t_lift is not None else None,
        "t_success_s": (t_place / control_hz) if t_place is not None else None,
        "n_regrasp": n_regrasp,
    }


def aggregate_cell(episodes: list[dict]) -> dict:
    n = len(episodes)
    def rate(k): return sum(1 for e in episodes if e[k]) / n if n else 0.0
    def vals(k): return [e[k] for e in episodes if e[k] is not None]
    def mean(k):
        v = vals(k)
        return (sum(v) / len(v)) if v else None
    return {
        "n": n,
        "success_rate": rate("success"),
        "grasp_rate": rate("grasped"),
        "lift_rate": rate("lifted"),
        "mean_t_grasp_s": mean("t_grasp_s"),
        "mean_t_lift_s": mean("t_lift_s"),
        "mean_t_success_s": mean("t_success_s"),
        "p95_t_success_s": _p95(vals("t_success_s")),
        "mean_n_regrasp": (sum(e["n_regrasp"] for e in episodes) / n) if n else 0.0,
    }


def load_episode_events(output_folder: str) -> dict[str, list[dict]]:
    """Map cell (env) name -> list of episode records.

    Each record is ``{"events": [...], "success": bool|None, "final_step":
    int|None}``.

    Real layout written by ``robolab/core/logging/results.py``: one file per
    episode at ``output/<folder>/<ENV_NAME>/log_{run}_env{eid}.json``, holding
    a SINGLE dict ``{"schema_version": 2, "dt", "task", "env_id", "run",
    "success", "final_step", "events": [...]}``. The cell is the parent
    directory name (the registered env name). ``episode_results.jsonl`` at the
    folder root is *not* read here: its per-episode ``events`` field is a dict
    of tallies, not the event list.

    The ``episode_results`` / ``episodes`` container layouts are kept as
    fallbacks for any writer that batches episodes into one file.
    """
    cells: dict[str, list[dict]] = {}
    for results_file in sorted(Path(output_folder).glob("*/**/*.json")):
        try:
            data = json.loads(results_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        cell = results_file.parent.name
        if isinstance(data.get("events"), list):
            # one file == one episode (the real v2 layout)
            cells.setdefault(cell, []).append({
                "events": data["events"],
                "success": data.get("success"),
                "final_step": data.get("final_step"),
            })
            continue
        # fallback: a container file holding many episodes
        episodes = data.get("episode_results") or data.get("episodes") or []
        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            ev = ep.get("events_list")
            if not isinstance(ev, list):
                ev = ep.get("events")
            if not isinstance(ev, list):
                continue
            cells.setdefault(cell, []).append({
                "events": ev,
                "success": ep.get("success"),
                "final_step": ep.get("final_step") or ep.get("episode_step"),
            })
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_folder")
    parser.add_argument("--num-steps", type=int, default=450,
                        help=("Step budget a stage must be reached within "
                              "(success@cap). Use 0 for no cap."))
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--csv", default="metrics.csv")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mass-com-vla-probing")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cells = load_episode_events(args.output_folder)
    cap = args.num_steps if args.num_steps > 0 else None
    rows = []
    for cell, records in sorted(cells.items()):
        eps = [episode_metrics(r["events"], cap, args.control_hz,
                               success=r.get("success"),
                               final_step=r.get("final_step"))
               for r in records]
        rows.append({"cell": cell, **aggregate_cell(eps)})
        print(rows[-1])
    if not rows:
        raise SystemExit(
            f"No episodes found under '{args.output_folder}'. Expected per-episode "
            "logs at <folder>/<ENV_NAME>/log_{run}_env{eid}.json, each a single JSON "
            'dict with a top-level "events" list (schema_version 2). Check that the '
            "folder is the eval run's output directory (the one containing the "
            "per-env subdirectories), not a parent of it."
        )
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
