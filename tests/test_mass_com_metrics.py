# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral metrics: stage matching by StatusCode, and the real log schema."""

import json

from analysis.mass_com.metrics import (
    aggregate_cell,
    episode_metrics,
    load_episode_events,
)

# Real StatusCodes (robolab/core/task/status.py) — fixtures use the same
# code/name pairs a live run writes, so a matcher regression shows up here.
GRASP = (139, "OBJECT_GRABBED_SUCCESS", "success: object_grabbed(object=orange_juice_carton). advanced 1 step(s) to step 1 for orange_juice_carton.")
LIFT = (139, "OBJECT_GRABBED_SUCCESS", "success: object_picked_up(object=orange_juice_carton, surface=table). advanced 2 step(s) to step 2 for orange_juice_carton.")
PLACE = (125, "OBJECT_IN_CONTAINER_SUCCESS", "Completed subtask 'pick_and_place' 1/1")
DROP = (263, "TARGET_OBJECT_DROPPED", "Target object dropped during transport")
GRASP_REGRESSION = (248, "OBJECT_GRABBED_FAILURE", "failed: object_grabbed(object=orange_juice_carton). regressing to step 0 for orange_juice_carton.")
WRONG_GRAB = (250, "WRONG_OBJECT_GRABBED_FAILURE", "Wrong object grabbed")
LIFT_ONLY = (100, "UNKNOWN_SUCCESS", "success: object_picked_up(object=orange_juice_carton, surface=table).")


def _ev(step, stage, score=0.0):
    code, name, info = stage
    return {"step": step, "code": code, "name": name, "info": info, "score": score}


def test_full_success_episode():
    events = [_ev(40, GRASP), _ev(70, LIFT_ONLY),
              _ev(200, PLACE, score=1.0)]
    m = episode_metrics(events, num_steps=450, control_hz=15.0)
    assert m == {"success": True, "grasped": True, "lifted": True,
                 "t_grasp_s": 40 / 15.0, "t_lift_s": 70 / 15.0,
                 "t_success_s": 200 / 15.0, "n_regrasp": 0}


def test_slip_and_regrasp_counted():
    # A real slip logs TARGET_OBJECT_DROPPED (263) and then the state-machine
    # regression OBJECT_GRABBED_FAILURE (248) — the latter is NOT a grasp.
    events = [_ev(40, GRASP), _ev(60, DROP), _ev(60, GRASP_REGRESSION),
              _ev(90, GRASP), _ev(120, LIFT_ONLY)]
    m = episode_metrics(events, num_steps=450)
    assert m["n_regrasp"] == 1 and m["lifted"] and not m["success"]
    assert m["t_success_s"] is None


def test_wrong_object_grabbed_is_not_a_grasp():
    # code 250 lower-cases to a string containing "object_grabbed"; the
    # substring matcher scored it as a successful grasp (finding I1).
    m = episode_metrics([_ev(30, WRONG_GRAB)], num_steps=450)
    assert m["grasped"] is False and m["n_regrasp"] == 0


def test_drop_without_regrasp_counts_zero():
    events = [_ev(40, GRASP), _ev(60, DROP)]
    assert episode_metrics(events, num_steps=450)["n_regrasp"] == 0


def test_release_into_container_is_not_a_regrasp():
    # On a successful place the log carries 263 at the same step as 125.
    events = [_ev(40, GRASP), _ev(138, DROP, score=1.0), _ev(138, PLACE, score=1.0)]
    m = episode_metrics(events, num_steps=450)
    assert m["n_regrasp"] == 0 and m["success"] is True


def test_timeout_without_grasp():
    m = episode_metrics([], num_steps=450)
    assert m == {"success": False, "grasped": False, "lifted": False,
                 "t_grasp_s": None, "t_lift_s": None, "t_success_s": None,
                 "n_regrasp": 0}


def test_num_steps_caps_every_stage():
    events = [_ev(40, GRASP), _ev(70, LIFT_ONLY), _ev(200, PLACE, score=1.0)]
    m = episode_metrics(events, num_steps=100)
    assert m["grasped"] and m["lifted"]          # both inside the budget
    assert m["success"] is False                 # place is not
    assert m["t_success_s"] is None


def test_file_success_bool_is_preferred():
    # No place event logged, but the env said the episode succeeded.
    m = episode_metrics([_ev(40, GRASP)], num_steps=450, success=True)
    assert m["success"] is True
    # ...and a False verdict wins over a stray in-container event too.
    m2 = episode_metrics([_ev(40, PLACE)], num_steps=450, success=False)
    assert m2["success"] is False


def test_file_success_is_recapped_by_num_steps():
    events = [_ev(40, GRASP), _ev(400, PLACE, score=1.0)]
    assert episode_metrics(events, num_steps=100, success=True)["success"] is False


def test_aggregate_cell_rates_and_means():
    eps = [episode_metrics([_ev(30, GRASP), _ev(60, LIFT_ONLY),
                            _ev(100, PLACE, score=1.0)], 450),
           episode_metrics([], 450)]
    agg = aggregate_cell(eps)
    assert agg["success_rate"] == 0.5
    assert agg["grasp_rate"] == 0.5 and agg["lift_rate"] == 0.5
    assert agg["mean_t_grasp_s"] == 2.0
    assert agg["mean_t_success_s"] == 100 / 15.0
    assert agg["p95_t_success_s"] == 100 / 15.0


def test_real_folded_event_case_insensitive():
    # Actual get_all_env_events() dump from one real OJCartonInCrateTask
    # carton lift (task-8-report.md): the subtask state machine advanced two
    # conditions (object_grabbed, object_picked_up) in a single check, so
    # name/code carry the FIRST condition while info carries the lower-case
    # predicate text for the LAST condition. Code matching must recover the
    # grasp and text matching the lift from this one event.
    events = [{
        "step": 157, "code": 139, "name": "OBJECT_GRABBED_SUCCESS",
        "info": ("success: object_picked_up(object=orange_juice_carton, "
                  "surface=table). advanced 2 step(s) to step 2 for "
                  "orange_juice_carton."),
        "score": 0.0,
    }]
    m = episode_metrics(events, num_steps=450)
    assert m["grasped"] is True and m["lifted"] is True
    assert m["t_grasp_s"] == m["t_lift_s"] == 157 / 15.0


def test_loader_reads_real_per_episode_schema(tmp_path):
    # Real layout: output/<folder>/<ENV_NAME>/log_{run}_env{eid}.json, each a
    # SINGLE dict with a top-level "events" list.
    env_dir = tmp_path / "OJCartonInCrateTask_MassMedium_CoMUp"
    env_dir.mkdir()
    (env_dir / "env_cfg.json").write_text(json.dumps({"scene": {}}))  # must be skipped
    for eid, (success, evs) in enumerate([
        (True, [_ev(40, GRASP), _ev(100, PLACE, score=1.0)]),
        (False, []),
    ]):
        (env_dir / f"log_0_env{eid}.json").write_text(json.dumps({
            "schema_version": 2, "dt": 1 / 15.0, "task": "OJCartonInCrateTask",
            "env_id": eid, "run": 0, "success": success, "final_step": 450,
            "events": evs,
        }))
    # the JSONL summary at the folder root carries tallies, not event lists
    (tmp_path / "episode_results.jsonl").write_text(
        json.dumps({"env_name": "x", "success": True, "events": {}}) + "\n")

    cells = load_episode_events(str(tmp_path))
    assert list(cells) == ["OJCartonInCrateTask_MassMedium_CoMUp"]
    records = cells["OJCartonInCrateTask_MassMedium_CoMUp"]
    assert len(records) == 2
    assert [r["success"] for r in records] == [True, False]
    assert records[0]["final_step"] == 450
    agg = aggregate_cell([episode_metrics(r["events"], 450, success=r["success"])
                          for r in records])
    assert agg["n"] == 2 and agg["success_rate"] == 0.5


def test_loader_container_fallback(tmp_path):
    d = tmp_path / "SomeEnv"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"episode_results": [
        {"success": True, "events_list": [_ev(10, GRASP)]},
        {"success": False, "events": [{"step": 1, "code": 255,
                                        "name": "GRIPPER_HIT_TABLE",
                                        "info": "Gripper hit table"}]},
    ]}))
    cells = load_episode_events(str(tmp_path))
    assert len(cells["SomeEnv"]) == 2
    assert cells["SomeEnv"][0]["events"][0]["code"] == 139
