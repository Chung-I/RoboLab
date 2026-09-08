# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.test_lift.episode_log import EPISODE_KEYS, read_episode, validate_episode, write_episode


def _dummy():
    d = {k: np.zeros(1) for k in EPISODE_KEYS}
    d.update(object="banana", arm="belief", grasps_o=np.zeros((3, 4, 4)), confs=np.zeros(3), yaw_fix="none",
              wrench_trace_h=np.zeros((15, 6), np.float32), wrench_bias_trace_h=np.zeros((15, 6), np.float32))
    return d


def test_roundtrip(tmp_path):
    p = tmp_path / "e.npz"
    write_episode(p, **_dummy())
    d = read_episode(p)
    assert str(d["object"]) == "banana" and d["grasps_o"].shape == (3, 4, 4)
    validate_episode(d)


def test_validate_missing_key():
    d = _dummy(); d.pop("m_post")
    with pytest.raises(KeyError):
        validate_episode(d)


def test_trace_keys_roundtrip(tmp_path):
    p = tmp_path / "e.npz"
    d = _dummy(); d["wrench_trace_h"] = np.arange(90, dtype=np.float32).reshape(15, 6)
    write_episode(p, **d)
    out = read_episode(p)
    assert out["wrench_trace_h"].shape == (15, 6) and out["wrench_trace_h"][3, 4] == 22.0


@pytest.mark.parametrize("key", ["wrench_trace_h", "wrench_bias_trace_h"])
def test_validate_missing_trace_keys(key):
    d = _dummy(); d.pop(key)
    with pytest.raises(KeyError):
        validate_episode(d)
