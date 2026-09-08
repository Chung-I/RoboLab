# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from analysis.test_lift.episode_log import EPISODE_KEYS, read_episode, validate_episode, write_episode


def _dummy():
    d = {k: np.zeros(1) for k in EPISODE_KEYS}
    d.update(object="banana", arm="belief", grasps_o=np.zeros((3, 4, 4)), confs=np.zeros(3), yaw_fix="none")
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
