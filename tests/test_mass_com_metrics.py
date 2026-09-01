# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from analysis.mass_com.metrics import aggregate_cell, episode_metrics


def _ev(step, name, score=0.0):
    return {"step": step, "code": 1, "name": name, "info": name, "score": score}


def test_full_success_episode():
    events = [_ev(40, "object_grabbed"), _ev(70, "object_picked_up"),
              _ev(200, "object_in_container", score=1.0)]
    m = episode_metrics(events, num_steps=200, control_hz=15.0)
    assert m == {"success": True, "grasped": True, "lifted": True,
                 "t_grasp_s": 40 / 15.0, "t_lift_s": 70 / 15.0, "n_regrasp": 0}


def test_slip_and_regrasp_counted():
    events = [_ev(40, "object_grabbed"), _ev(60, "object_dropped"),
              _ev(90, "object_grabbed"), _ev(120, "object_picked_up")]
    m = episode_metrics(events, num_steps=450)
    assert m["n_regrasp"] == 1 and m["lifted"] and not m["success"]


def test_timeout_without_grasp():
    m = episode_metrics([], num_steps=450)
    assert m == {"success": False, "grasped": False, "lifted": False,
                 "t_grasp_s": None, "t_lift_s": None, "n_regrasp": 0}


def test_aggregate_cell_rates_and_means():
    eps = [episode_metrics([_ev(30, "object_grabbed"), _ev(60, "object_picked_up"),
                            _ev(100, "object_in_container", score=1.0)], 450),
           episode_metrics([], 450)]
    agg = aggregate_cell(eps)
    assert agg["success_rate"] == 0.5
    assert agg["grasp_rate"] == 0.5 and agg["lift_rate"] == 0.5
    assert agg["mean_t_grasp_s"] == 2.0
