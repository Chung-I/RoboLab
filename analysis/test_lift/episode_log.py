# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-episode .npz written by scripts/test_lift_episode.py and read by results.py."""
from __future__ import annotations

import numpy as np

EPISODE_KEYS = (
    "object", "arm", "mass_true", "com_true_o", "com_offset_xyz", "grasps_o", "confs",
    "idx_first", "idx_second", "m_prior", "c_prior_o", "c_prior_cov", "m_post", "c_post_o", "c_post_cov",
    "wrench_bias_h", "wrench_hold_h", "wrench_trace_h", "wrench_bias_trace_h",
    "hold_prob_first", "first_lift_ok", "second_lift_ok", "final_ok",
    "n_grasps", "wall_s", "yaw_fix",
)


def write_episode(path, **arrays) -> None:
    validate_episode(arrays)
    np.savez_compressed(path, **arrays)


def read_episode(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def validate_episode(d) -> None:
    missing = [k for k in EPISODE_KEYS if k not in d]
    if missing:
        raise KeyError(f"episode log missing keys: {missing}")
