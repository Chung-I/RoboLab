# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump GraspGenX discriminator embeddings for a fixed candidate set (v1 spec §4.3).

Run with ~/Codes/GraspGenX/.venv/bin/python (never `uv run` there):
  cd ~/Codes/GraspGenX && .venv/bin/python ~/Codes/RoboLab/scripts/graspgenx_dump_embeddings.py \
      --candidates ~/Codes/RoboLab/output/test_lift/v1/candidates/banana.npz \
      --out ~/Codes/RoboLab/output/test_lift/v1/embeddings/banana.npz

Also saves the frozen discriminator's prediction-head weights once per run, beside
--out, as prediction_head.pt (Task-6 controller Ruling 2) -- a later task warm-starts
a new head from it.

The object point cloud must be preprocessed EXACTLY as the serving path preprocesses it,
or the embeddings belong to a different object encoding than the stored confidences do.
``GraspGenXSampler.run_inference`` calls ``point_cloud_outlier_removal`` (20-NN mean
distance < 1.4 cm) and only then centres on the SURVIVING points' mean
(graspgenx/grasp_server.py, ``remove_outliers=True`` by default). Skipping that step is
invisible on a compact object -- banana and rubiks_cube keep 2048/2048 points -- but the
mug loses 137 points off its rim and handle, which moves the centre and biases every
rescored confidence down by 0.073 (max gap 0.288, far past GAP_TOL). With the removal
applied the mug's gap is 0.024. Task 9 found this; do not drop the call.
"""
import argparse, os, sys
import numpy as np, torch
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers.graspmoe import _score_grasps_with_discriminator
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.point_cloud import point_cloud_outlier_removal

GAP_TOL = 0.05
MIN_POINTS_AFTER_OUTLIER_REMOVAL = 10   # grasp_server.py's own fallback threshold

ap = argparse.ArgumentParser()
ap.add_argument("--candidates", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--config", default=os.path.expanduser("~/Codes/GraspGenX/ext/graspgenx_checkpoints/release"))
ap.add_argument("--assets_dir", default=os.path.expanduser("~/Codes/GraspGenX/assets"))
ap.add_argument("--gripper", default="franka_panda")
a = ap.parse_args()

cfg = load_model_cfg(os.path.join(a.config, "gen"), os.path.join(a.config, "dis"))
sampler = GraspGenXSampler(cfg, a.gripper, assets_dir=a.assets_dir)
z = np.load(a.candidates, allow_pickle=False)
pts = z["points_o"].astype(np.float32); grasps = z["grasps_o"].astype(np.float32)
device = next(sampler.model.parameters()).device

# Mirror grasp_server.run_inference: drop outliers first, then centre on what is left.
pc_raw = torch.from_numpy(pts).to(device)
pc_kept, _ = point_cloud_outlier_removal(pc_raw)
if len(pc_kept) < MIN_POINTS_AFTER_OUTLIER_REMOVAL:
    print(f"[dump] outlier removal left {len(pc_kept)}/{len(pts)} points; using the raw cloud",
          file=sys.stderr)
    pc_kept = pc_raw
center = pc_kept.mean(0).cpu().numpy().astype(np.float64)
pc_centered = pc_kept - torch.as_tensor(center, dtype=pc_kept.dtype, device=device)[None]
print(f"[dump] outlier removal kept {len(pc_kept)}/{len(pts)} points")

captured = {}
def hook(mod, inp, out): captured["e"] = inp[0].detach().cpu().numpy()
h = sampler.model.grasp_discriminator.prediction_head.register_forward_hook(hook)
conf = _score_grasps_with_discriminator(grasps, pc_centered, center, sampler)
h.remove()
e = captured["e"].reshape(len(grasps), -1).astype(np.float32)
gap = float(np.abs(conf - z["confs"]).max())
if gap > GAP_TOL:
    print(f"[dump] FAIL frame/centring mismatch: gap={gap:.4f}", file=sys.stderr)
    sys.exit(1)
np.savez_compressed(a.out, e_g=e, conf_rescored=conf.astype(np.float32), D=int(e.shape[1]),
                    conf_ref=z["confs"].astype(np.float32), object=str(z["object"]))
print(f"[dump] {len(grasps)} grasps, D={e.shape[1]}, max|conf_rescored-conf_ref|={gap:.4f} -> {a.out}")

head_path = os.path.join(os.path.dirname(a.out), "prediction_head.pt")
torch.save({k: v.cpu() for k, v in sampler.model.grasp_discriminator.prediction_head.state_dict().items()}, head_path)
print(f"[dump] prediction_head weights -> {head_path}")
