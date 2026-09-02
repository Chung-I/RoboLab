# analysis/mass_com/replay_lib.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for the Phase-2 replay corpus (spec §4 Phase 2, §5.3).

The pre-contact boundary deliberately avoids the contact-lagged
object_grabbed event: it is min(commanded gripper close, first measured
contact) — strictly conservative for the negative-control window.
"""

import numpy as np


def jointpos_actions_from_states(joint_pos: np.ndarray, gripper_actions: np.ndarray) -> np.ndarray:
    grip = np.asarray(gripper_actions, np.float32).reshape(-1)
    T = min(len(joint_pos), len(grip))
    return np.concatenate(
        [np.asarray(joint_pos[:T, :7], np.float32), grip[:T, None]], axis=1)


def drift_curve(src: np.ndarray, replay: np.ndarray) -> np.ndarray:
    T = min(len(src), len(replay))
    return np.linalg.norm(
        np.asarray(replay[:T, :7], np.float32) - np.asarray(src[:T, :7], np.float32), axis=1)


def matched_window(drift: np.ndarray, anchor_step: int, threshold: float) -> int:
    n = 0
    for v in drift[anchor_step:]:
        if v >= threshold:
            break
        n += 1
    return n


def gripper_close_step(actions: np.ndarray, closed: float = 0.5):
    idx = np.nonzero(np.asarray(actions)[:, -1] >= closed)[0]
    return int(idx[0]) if len(idx) else None


def first_contact_step(contact_norm: np.ndarray, threshold: float = 0.1):
    idx = np.nonzero(np.asarray(contact_norm) >= threshold)[0]
    return int(idx[0]) if len(idx) else None


def precontact_boundary(actions: np.ndarray, contact_norm: np.ndarray) -> int:
    cands = [s for s in (gripper_close_step(actions), first_contact_step(contact_norm))
             if s is not None]
    if not cands:
        raise ValueError("neither gripper close nor contact onset found")
    return min(cands)
