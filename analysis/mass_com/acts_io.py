# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader helper for Plan-2 Task 5 activation captures.

The saved ``acts`` array is (T, L, P, D=2048), but position 2
(first_suffix_token) comes from the action-expert stream (gemma_300m,
D=1024) and its dims 1024:2048 are constant zero padding — dim k at P2 is
NOT the same feature as dim k at P0/P1 (gemma_2b). ``load_acts`` consumes
the structured ``positions`` entries in meta.json (``stream``,
``valid_dims``) and returns each position already sliced to its valid
dims, so downstream code cannot silently mix the two streams.

Usage:
    meta = json.loads((acts_root / "meta.json").read_text())
    data = acts_io.load_acts(acts_root / "orange_juice_carton/MassMedium_CoMCenter", meta)
    data["last_prefix_token"]   # (T, L, 2048) float16, paligemma stream
    data["first_suffix_token"]  # (T, L, 1024) float16, expert stream
"""

from pathlib import Path

import numpy as np

EXTRA_KEYS = (
    "actions_out",
    "n_lang_valid",
    "lang_text_ranges",
    "lang_state_ranges",
    "lang_tail_ranges",
)


def load_acts(condition_dir: str | Path, meta: dict) -> dict:
    """Load one condition's acts.npz, slicing each position to its valid dims.

    Args:
        condition_dir: directory containing ``acts.npz``.
        meta: the capture ``meta.json`` dict; every entry of
            ``meta["positions"]`` must carry ``name``, ``stream`` and
            ``valid_dims`` (raises KeyError otherwise).

    Returns a dict with one (T, L, n_valid_dims) float16 array per position
    name, a ``streams`` map {position name: stream}, and the auxiliary arrays
    (actions_out, n_lang_valid, lang_*_ranges) passed through.
    """
    with np.load(Path(condition_dir) / "acts.npz") as z:
        acts = z["acts"]
        out = {k: z[k] for k in EXTRA_KEYS if k in z.files}
    streams = {}
    for pos in meta["positions"]:
        name, stream = pos["name"], pos["stream"]
        lo, hi = pos["valid_dims"]
        out[name] = acts[:, :, pos["index"], lo:hi]
        streams[name] = stream
    out["streams"] = streams
    return out
