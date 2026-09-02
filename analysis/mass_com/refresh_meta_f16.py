# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot meta.json refresh for an existing Task-5 capture (no model rerun).

Recomputes, from the shipped acts.npz files:
  - per-condition f16 clip tables (per_condition.<cond>.f16_clip_table)
  - the corpus-wide aggregate clip table (f16_clip.aggregate)
and injects the structured position fields (stream, valid_dims) and the
token_blocks text/state stub keys that newer capture_pi05.build_meta versions
emit natively. Idempotent; rewrites <out>/meta.json in place.

Usage:
    ~/Codes/openpi/.venv/bin/python analysis/mass_com/refresh_meta_f16.py \\
        --out output/activations/pi05
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_pi05 import (  # noqa: E402
    ACTION_HORIZON,
    EXPERT_D,
    F16_MAX,
    LANG_OFFSET,
    PREFIX_D,
    PREFIX_LEN,
    aggregate_clip_tables,
    compute_clip_table,
)

POSITION_STREAMS = {
    "last_prefix_token": ("paligemma", [0, PREFIX_D]),
    "image_tokens_mean": ("paligemma", [0, PREFIX_D]),
    "first_suffix_token": ("expert", [0, EXPERT_D]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="output/activations/pi05")
    args = ap.parse_args()
    out_root = Path(args.out)
    meta_path = out_root / "meta.json"
    meta = json.loads(meta_path.read_text())

    acts_paths = []
    for cond, stats in meta["per_condition"].items():
        path = out_root / cond / "acts.npz"
        with np.load(path) as z:
            acts = z["acts"]
        table = compute_clip_table(acts)
        assert table["total"] == stats["n_f16_clipped_elems"], (
            cond, table["total"], stats["n_f16_clipped_elems"])
        stats["f16_clip_table"] = table
        acts_paths.append(path)

    meta["f16_clip"] = {
        "clip_value": F16_MAX,
        "note": ("activations are clipped to the finite float16 range before "
                 "saving. Per-condition breakdowns: per_condition.<cond>."
                 "f16_clip_table {total, per_layer_position, top_dims}; totals "
                 "also in per_condition.<cond>.n_f16_clipped_elems; corpus-wide "
                 "aggregate under f16_clip.aggregate."),
        "aggregate": aggregate_clip_tables(acts_paths),
    }

    for pos in meta["positions"]:
        stream, valid_dims = POSITION_STREAMS[pos["name"]]
        pos["stream"] = stream
        pos["valid_dims"] = valid_dims
    # Make the expert-stream footgun explicit in the definition text too.
    meta["positions"][2]["definition"] = (
        "expert stream (gemma_300m, D=1024; dims 1024:2048 are constant zero "
        "padding — dim k here is NOT the same feature as dim k at positions "
        "0/1), action token 0, captured at the first denoise step (t=1.0, "
        "x_t = fixed noise). Load via analysis/mass_com/acts_io.load_acts to "
        "get pre-sliced arrays.")

    tb = meta["token_blocks"]
    tb["text"] = ("per-step: lang_text_ranges (T,2) in each acts.npz; "
                  "step-0: per_condition.<cond>.lang_blocks_step0.text")
    tb["state"] = ("per-step: lang_state_ranges (T,2) in each acts.npz; "
                   "step-0: per_condition.<cond>.lang_blocks_step0.state")
    assert tb["lang_slots"] == [LANG_OFFSET, PREFIX_LEN]
    assert tb["suffix"] == [0, ACTION_HORIZON]

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    agg = meta["f16_clip"]["aggregate"]
    print(f"Refreshed {meta_path}: aggregate clipped={agg['total']} over "
          f"{agg['total_steps']} steps; "
          f"cells={[(e['layer'], e['position'], e['count']) for e in agg['per_layer_position'][:5]]}; "
          f"top dim={agg['top_dims'][0] if agg['top_dims'] else None}")


if __name__ == "__main__":
    main()
