# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure parts of capture_pi05 (Plan-2 Task 5).

Runs with the openpi venv: ~/Codes/openpi/.venv/bin/python -m pytest ... -v
No model, no GPU: request building, token-block-range extraction, position
indexing, and LayerTap hook plumbing on toy nn.Modules.
"""

import numpy as np
import pytest
import torch
from torch import nn

import capture_pi05 as cap


# ---------------------------------------------------------------- request building


def test_build_step_request_keys_shapes_dtypes():
    cam1 = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    wrist = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    joint = np.arange(7, dtype=np.float32) * 0.1
    grip = np.array([0.5], dtype=np.float32)

    req = cap.build_step_request(cam1, wrist, joint, grip, "Put the carton in the bin")

    assert set(req) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    for k in ("observation/exterior_image_1_left", "observation/wrist_image_left"):
        assert req[k].shape == (224, 224, 3)
        assert req[k].dtype == np.uint8
    np.testing.assert_array_equal(req["observation/joint_position"], joint)
    np.testing.assert_array_equal(req["observation/gripper_position"], grip)
    assert req["prompt"] == "Put the carton in the bin"


def test_build_step_request_pads_vertically_for_wide_image():
    # 720x1280 -> resize_with_pad(224, 224): scaled to 126x224, ~49 rows of
    # zero padding top and bottom.
    cam = np.full((720, 1280, 3), 255, dtype=np.uint8)
    req = cap.build_step_request(cam, cam, np.zeros(7, np.float32), np.zeros(1, np.float32), "x")
    img = req["observation/exterior_image_1_left"]
    assert img[:40].max() == 0 and img[-40:].max() == 0  # padded rows
    assert img[112].min() > 0  # image content in the middle


# ---------------------------------------------------------------- token block ranges


def test_lang_block_ranges_basic():
    pieces = [
        "<bos>", "▁Task", ":", "▁Put", "▁the", "▁carton", ",",
        "▁State", ":", "▁12", "3", "▁45", "▁0",
        ";", "▁Action", ":", "▁",
    ]
    r = cap.lang_block_ranges(pieces)
    assert r["text"] == (0, 9)      # everything before the first state digit
    assert r["state"] == (9, 13)    # ▁12 3 ▁45 ▁0
    assert r["tail"] == (13, 17)    # ;\nAction:_


def test_lang_block_ranges_digits_only_inside_state():
    # A digit-like piece in the instruction text must not start the state
    # block: state starts only after the "State" marker piece.
    pieces = ["<bos>", "▁Task", ":", "▁pick", "▁2", "▁cans", ",",
              "▁State", ":", "▁7", "▁255", ";", "▁Action", ":"]
    r = cap.lang_block_ranges(pieces)
    assert r["state"] == (9, 11)
    assert r["text"] == (0, 9)
    assert r["tail"] == (11, 14)


def test_lang_block_ranges_raises_without_state_marker():
    with pytest.raises(ValueError):
        cap.lang_block_ranges(["<bos>", "▁hello", "▁world"])


# ---------------------------------------------------------------- position indexing


def test_prefix_constants():
    assert cap.NUM_IMG_TOKENS == 256
    assert cap.IMG_BLOCKS["img_cam1"] == (0, 256)
    assert cap.IMG_BLOCKS["img_cam2"] == (256, 512)
    assert cap.IMG_BLOCKS["img_pad_right_wrist"] == (512, 768)
    assert cap.LANG_OFFSET == 768
    assert cap.PREFIX_LEN == 968  # 3*256 + max_token_len(200)


def test_capture_positions():
    pos = cap.capture_positions(n_lang_valid=57)
    assert pos["last_prefix_token"] == 768 + 57 - 1
    assert pos["image_token_slice"] == (0, 512)  # valid cams only, pad cam excluded
    assert pos["first_suffix_token"] == 0


def test_capture_positions_rejects_bad_lang_len():
    with pytest.raises(ValueError):
        cap.capture_positions(n_lang_valid=0)
    with pytest.raises(ValueError):
        cap.capture_positions(n_lang_valid=201)


# ---------------------------------------------------------------- f16 clipping


def test_to_f16_clipped_passthrough_and_clip():
    v = np.array([0.5, -61696.0, 70000.0, -70000.0], dtype=np.float32)  # -61696 is f16-exact
    out, n_clipped = cap.to_f16_clipped(v)
    assert out.dtype == np.float16
    assert np.isfinite(out).all()
    f16max = float(np.finfo(np.float16).max)
    np.testing.assert_array_equal(
        out.astype(np.float32), np.array([0.5, -61696.0, f16max, -f16max], np.float32))
    assert n_clipped == 2


def test_to_f16_clipped_no_clip():
    v = np.linspace(-1, 1, 8, dtype=np.float32)
    out, n_clipped = cap.to_f16_clipped(v)
    assert n_clipped == 0
    np.testing.assert_allclose(out.astype(np.float32), v, atol=1e-3)


# ---------------------------------------------------------------- LayerTap


class _TupleBlock(nn.Module):
    """Mimics GemmaDecoderLayer: returns a 1-tuple of hidden states."""

    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return (self.lin(x),)


class _TensorBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x)


def _toy_stack(block_cls, n_layers=3, dim=8):
    torch.manual_seed(0)
    return nn.ModuleList([block_cls(dim) for _ in range(n_layers)])


def test_layertap_records_tuple_and_tensor_outputs():
    for block_cls in (_TupleBlock, _TensorBlock):
        blocks = _toy_stack(block_cls)
        tap = cap.LayerTap()
        tap.register({f"layer.{i}": m for i, m in enumerate(blocks)})
        x = torch.randn(2, 5, 8)
        for m in blocks:
            out = m(x)
            x = out[0] if isinstance(out, tuple) else out
        recs = tap.records
        assert list(recs) == ["layer.0", "layer.1", "layer.2"]
        for name in recs:
            assert len(recs[name]) == 1
            assert recs[name][0].shape == (2, 5, 8)
        tap.remove()


def test_layertap_multiple_calls_and_clear():
    blocks = _toy_stack(_TupleBlock, n_layers=2)
    tap = cap.LayerTap()
    tap.register({f"l{i}": m for i, m in enumerate(blocks)})
    x = torch.randn(1, 4, 8)
    for _ in range(3):  # e.g. denoise steps
        y = x
        for m in blocks:
            y = m(y)[0]
    assert all(len(v) == 3 for v in tap.records.values())
    stacked = tap.stacked(call_index=0)
    assert stacked.shape == (2, 1, 4, 8)
    assert isinstance(stacked, np.ndarray)
    # call_index selects distinct captures
    assert not np.array_equal(tap.stacked(call_index=0), tap.stacked(call_index=1) * np.nan)
    tap.clear()
    assert all(len(v) == 0 for v in tap.records.values())
    tap.remove()
    for m in blocks:
        m(torch.randn(1, 4, 8))
    assert all(len(v) == 0 for v in tap.records.values())  # hooks removed


def test_layertap_records_are_detached_cpu_copies():
    blocks = _toy_stack(_TupleBlock, n_layers=1)
    tap = cap.LayerTap()
    tap.register({"l0": blocks[0]})
    x = torch.randn(1, 2, 8, requires_grad=True)
    blocks[0](x)
    rec = tap.records["l0"][0]
    assert not rec.requires_grad
    assert rec.device.type == "cpu"
    tap.remove()
