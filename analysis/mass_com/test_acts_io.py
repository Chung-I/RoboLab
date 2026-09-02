# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for acts_io.load_acts (Plan-2 Task 5 loader helper).

Runs with the openpi venv: ~/Codes/openpi/.venv/bin/python -m pytest ... -v
"""

import json

import numpy as np
import pytest

import acts_io


def _make_meta():
    return {
        "positions": [
            {"index": 0, "name": "last_prefix_token", "stream": "paligemma",
             "valid_dims": [0, 2048]},
            {"index": 1, "name": "image_tokens_mean", "stream": "paligemma",
             "valid_dims": [0, 2048]},
            {"index": 2, "name": "first_suffix_token", "stream": "expert",
             "valid_dims": [0, 1024]},
        ],
    }


@pytest.fixture
def cond_dir(tmp_path):
    T, L, P, D = 4, 3, 3, 2048
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(T, L, P, D)).astype(np.float16)
    acts[:, :, 2, 1024:] = 0.0  # expert zero-pad region
    np.savez(
        tmp_path / "acts.npz",
        acts=acts,
        actions_out=rng.normal(size=(T, 15, 8)).astype(np.float32),
        n_lang_valid=np.full(T, 48, dtype=np.int32),
        lang_text_ranges=np.tile([768, 783], (T, 1)).astype(np.int32),
        lang_state_ranges=np.tile([783, 811], (T, 1)).astype(np.int32),
        lang_tail_ranges=np.tile([811, 816], (T, 1)).astype(np.int32),
    )
    (tmp_path / "raw.json").write_text(json.dumps({}))
    return tmp_path, acts


def test_load_acts_slices_each_position_to_valid_dims(cond_dir):
    tmp_path, acts = cond_dir
    out = acts_io.load_acts(tmp_path, _make_meta())
    assert out["last_prefix_token"].shape == (4, 3, 2048)
    assert out["image_tokens_mean"].shape == (4, 3, 2048)
    assert out["first_suffix_token"].shape == (4, 3, 1024)  # zero-pad sliced away
    np.testing.assert_array_equal(out["last_prefix_token"], acts[:, :, 0, :])
    np.testing.assert_array_equal(out["image_tokens_mean"], acts[:, :, 1, :])
    np.testing.assert_array_equal(out["first_suffix_token"], acts[:, :, 2, :1024])


def test_load_acts_carries_stream_labels_and_extras(cond_dir):
    tmp_path, _ = cond_dir
    out = acts_io.load_acts(tmp_path, _make_meta())
    assert out["streams"] == {
        "last_prefix_token": "paligemma",
        "image_tokens_mean": "paligemma",
        "first_suffix_token": "expert",
    }
    assert out["actions_out"].shape == (4, 15, 8)
    assert out["n_lang_valid"].shape == (4,)
    assert out["lang_state_ranges"].shape == (4, 2)


def test_load_acts_rejects_meta_without_structured_positions(cond_dir):
    tmp_path, _ = cond_dir
    bad_meta = {"positions": [{"index": 0, "name": "last_prefix_token"}]}
    with pytest.raises(KeyError):
        acts_io.load_acts(tmp_path, bad_meta)
