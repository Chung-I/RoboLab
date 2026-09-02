# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan-2 Task 2: single-condition replay driver with per-step F/T logging.

Replays one recorded demo (source condition) open-loop into a freshly
registered mass/CoM condition env, restoring the exact recorded initial
scene state (`robolab.core.replay.scene_state.restore_recorded_initial_state`)
so the replay starts from what the recording env saw. Two use cases:

- Self-replay (source condition == target condition): sanity check that the
  driver reproduces the recording (small drift throughout).
- Cross-condition replay (e.g. medium -> heavy): the commanded joint-space
  trajectory is replayed against a *different* physics condition, producing
  divergence after the anchor step (gripper close) that is the whole point
  of the Phase-2 replay corpus (spec Phase 2).

Per run writes, under ``<out>/<object>/<condition>/``:
  - RoboLab's own episode recording (``replay.hdf5``, images ON).
  - ``ft.npz`` with keys: wrench (T,6), contact_force (T,3),
    applied_torque (T,7), joint_pos_achieved (T,7), object_root_pose (T,7),
    actions (T,8), drift (T,), and scalars mass_kg, com_axis, com_offset_m,
    anchor_step, precontact_boundary, matched_window_N.

Usage:
  uv run --no-sync python -u scripts/build_replay_corpus.py \
      --source-h5 output/phase1a_pi05/OJCartonInCrateTask_MassMedium_CoMCenter/run_0.hdf5 \
      --demo 2 --task-file oj_carton_in_crate_task.py --object orange_juice_carton \
      --condition MassMedium_CoMCenter --out output/replay_corpus/ --headless
"""

import argparse
import os
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Single-condition replay driver with F/T logging.")
parser.add_argument("--source-h5", "--source_h5", type=str, required=True,
                     help="Path to the source run_<i>.hdf5 recording to replay actions/states from.")
parser.add_argument("--demo", type=int, required=True, help="Demo index within --source-h5 (demo_<i>).")
parser.add_argument("--task-file", "--task_file", type=str, required=True,
                     help="Task filename under robolab/tasks/ (e.g. oj_carton_in_crate_task.py).")
parser.add_argument("--object", type=str, required=True,
                     help="Scene entity name of the target rigid object (e.g. orange_juice_carton).")
parser.add_argument("--condition", type=str, required=True,
                     help="Target mass/CoM condition name (one of CONDITIONS in "
                          "robolab/registrations/droid/auto_env_registrations_mass_variations.py).")
parser.add_argument("--out", type=str, default="output/replay_corpus/",
                     help="Output root; writes under <out>/<object>/<condition>/.")
parser.add_argument("--source-mode", "--source_mode", choices=["actions", "states"], default="actions",
                     help="'actions': replay the recorded action stream as-is. 'states': re-derive a "
                          "joint-position action stream from the recorded joint states + gripper actions "
                          "(analysis.mass_com.replay_lib.jointpos_actions_from_states).")
parser.add_argument("--calibration-path", "--calibration_path", type=str, default=None,
                     help="Explicit mass_levels.json for load_mass_levels(); omit to use the default "
                          "calibration file / pre-calibration defaults.")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.constants import PACKAGE_DIR, TASK_DIR, set_output_dir  # noqa: E402
from robolab.core.environments.factory import auto_discover_and_create_cfgs  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.replay import restore_recorded_initial_state  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_mass_variations import (  # noqa: E402
    COM_OFFSET_AXIS, COM_OFFSET_BY_OBJECT, COM_OFFSET_M, CONDITIONS, load_mass_levels,
)
from robolab.variations.physics import make_object_physics_events_cfg  # noqa: E402

# `analysis` is a repo-root package (not installed into site-packages, unlike
# robolab/policies/dashboard -- see pyproject.toml [tool.setuptools.packages.find]),
# so it is only importable when the repo root is on sys.path. Running this
# script directly (`python scripts/build_replay_corpus.py`) puts scripts/ on
# sys.path[0], not the repo root, so add it explicitly before importing.
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

from analysis.mass_com.replay_lib import (  # noqa: E402
    drift_curve, gripper_close_step, jointpos_actions_from_states, matched_window, precontact_boundary,
)

# Images ON for the replay corpus (spec Phase 2): the recording is consumed
# downstream for visual inspection / VLA probing, not just F/T signals.
robolab.constants.RECORD_IMAGE_DATA = True


def _register_condition(mass_kg: float, com_offset_m: float, com_offset_axis: str) -> str:
    """Register a single env for --task-file / --object at the target condition.

    Mirrors scripts/pilot_uncapped.py's `_register_single_medium_cell`, but
    parameterized by the CLI args instead of hardcoded to carton/medium.
    """
    from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg, DroidJointPositionActionCfg, ProprioceptionObservationCfg, WristCameraCfg, contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    cameras = WRIST_LEFT
    scene_cameras = [c for c in cameras if c is not WristCameraCfg]
    ImageObsCfg = generate_image_obs_from_cameras(cameras)
    ViewportCameraCfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    ObservationCfg = generate_obs_cfg({
        "image_obs": ImageObsCfg(),
        "proprio_obs": ProprioceptionObservationCfg(),
        "viewport_cam": ViewportCameraCfg()})

    result = auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        tasks=args_cli.task_file,
        env_postfix=f"_Replay_{args_cli.condition}",
        events_cfg=(lambda o=args_cli.object, m=mass_kg, d=com_offset_m, a=com_offset_axis:
                    make_object_physics_events_cfg(o, mass_kg=m, com_offset_m=d, com_offset_axis=a)),
        observations_cfg=ObservationCfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=SphereLightCfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=1,
    )
    # `result` is keyed by the raw task identifier passed to `tasks=`, not the
    # registered gym env name (see auto_env_registrations_mass_variations.py).
    cfg_cls = next(iter(result.values()))
    return cfg_cls.__name__.removesuffix("EnvCfg")


def main() -> None:
    cond_map = {name: (level_key, sign) for name, level_key, sign in CONDITIONS}
    if args_cli.condition not in cond_map:
        raise ValueError(f"Unknown --condition {args_cli.condition!r}; expected one of {sorted(cond_map)}")
    level_key, com_sign = cond_map[args_cli.condition]

    levels = load_mass_levels(args_cli.calibration_path)
    if args_cli.object not in levels:
        raise ValueError(f"Unknown --object {args_cli.object!r}; expected one of {sorted(levels)}")
    mass_kg = levels[args_cli.object][level_key]

    com_axis, com_mag = COM_OFFSET_BY_OBJECT.get(args_cli.object, (COM_OFFSET_AXIS, COM_OFFSET_M))
    com_offset_m = com_sign * com_mag

    print(f"[build_replay_corpus] condition={args_cli.condition} object={args_cli.object} "
          f"mass_kg={mass_kg} com_axis={com_axis} com_offset_m={com_offset_m:+.4f}")

    env_name = _register_condition(mass_kg, com_offset_m, com_axis)
    print(f"[build_replay_corpus] registered env {env_name}")

    out_dir = os.path.join(args_cli.out, args_cli.object, args_cli.condition)
    set_output_dir(out_dir)

    env, env_cfg = create_env(env_name, device=args_cli.device, num_envs=1, use_fabric=True)

    try:
        obs, _ = env.reset()

        if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
            env.recorder_manager.set_hdf5_file("replay.hdf5")
            env.recorder_manager.set_episode_index(0, env_ids=[0])

        # Restore the recorded initial scene state so the open-loop replay
        # starts from exactly what the source recording saw (Step-1 finding:
        # env.reset_to(state, env_ids=None, is_relative=True) via
        # robolab.core.replay.scene_state.restore_recorded_initial_state).
        restore_recorded_initial_state(env, args_cli.source_h5, args_cli.demo)

        with h5py.File(args_cli.source_h5, "r") as f:
            src = f[f"data/demo_{args_cli.demo}"]
            src_joint_pos = src["states/articulation/robot/joint_position"][:]
            src_gripper_actions = src["actions"][:, -1]
            src_actions = src["actions"][:].astype(np.float32)

        if args_cli.source_mode == "states":
            actions_np = jointpos_actions_from_states(src_joint_pos, src_gripper_actions)
        else:
            actions_np = src_actions

        robot = env.scene["robot"]
        body_names = list(robot.data.body_names)
        if "base_link" in body_names:
            body_idx = body_names.index("base_link")
        else:
            print(f"WARNING: 'base_link' not in robot.data.body_names {body_names}; using index 0.")
            body_idx = 0

        # robolab.core.sensors.contact_sensor_utils.create_contact_sensors names the
        # gripper-vs-all-objects batch sensor "{gripper_name}__all_objs" (confirmed
        # at runtime: env.scene.sensors has no "gripper" key, but does have
        # "gripper__all_objs" plus per-object pairwise "gripper__<object>" sensors).
        sensors = dict(getattr(env.scene, "sensors", {}) or {})
        contact = sensors.get("gripper__all_objs")
        if contact is None:
            print(f"WARNING: 'gripper__all_objs' sensor not found; available sensors: {list(sensors.keys())}. "
                  "contact_force will be logged as zeros.")

        logs = {k: [] for k in ("wrench", "contact_force", "applied_torque",
                                 "joint_pos_achieved", "object_root_pose")}

        for t in range(len(actions_np)):
            action = torch.as_tensor(actions_np[t:t + 1], device=env.device)
            env.step(action)
            logs["wrench"].append(robot.data.body_incoming_joint_wrench_b[0, body_idx].cpu().numpy())
            logs["applied_torque"].append(robot.data.applied_torque[0, :7].cpu().numpy())
            logs["joint_pos_achieved"].append(robot.data.joint_pos[0, :7].cpu().numpy())
            logs["object_root_pose"].append(env.scene[args_cli.object].data.root_pose_w[0].cpu().numpy())
            logs["contact_force"].append(
                contact.data.net_forces_w[0, 0].cpu().numpy() if contact is not None
                else np.zeros(3, np.float32))
            if env.all_terminated:
                break

        # Guarantee export even if the replay never hit a termination/truncation
        # condition (export_episodes() is a no-op for envs already auto-exported
        # by RobolabEnv._reset_idx on freeze).
        end_episode(env)

        actions_arr = np.asarray(actions_np, dtype=np.float32)
        joint_pos_achieved = np.stack(logs["joint_pos_achieved"]).astype(np.float32)
        wrench = np.stack(logs["wrench"]).astype(np.float32)
        applied_torque = np.stack(logs["applied_torque"]).astype(np.float32)
        object_root_pose = np.stack(logs["object_root_pose"]).astype(np.float32)
        contact_force = np.stack(logs["contact_force"]).astype(np.float32)

        drift = drift_curve(src_joint_pos[:, :7], joint_pos_achieved)

        anchor_step = gripper_close_step(actions_arr)
        if anchor_step is None:
            print("WARNING: gripper never closes in the commanded action stream; anchor_step defaults to 0.")
            anchor_step = 0

        contact_norm = np.linalg.norm(contact_force, axis=1)
        boundary = precontact_boundary(actions_arr, contact_norm)
        matched_n = matched_window(drift, anchor_step, threshold=0.05)

        ft_path = os.path.join(out_dir, "ft.npz")
        np.savez(
            ft_path,
            wrench=wrench,
            contact_force=contact_force,
            applied_torque=applied_torque,
            joint_pos_achieved=joint_pos_achieved,
            object_root_pose=object_root_pose,
            actions=actions_arr,
            drift=drift,
            mass_kg=np.float32(mass_kg),
            com_axis=com_axis,
            com_offset_m=np.float32(com_offset_m),
            anchor_step=np.int64(anchor_step),
            precontact_boundary=np.int64(boundary),
            matched_window_N=np.int64(matched_n),
        )

        T = len(actions_arr)
        max_drift = float(np.max(drift)) if len(drift) else float("nan")
        pre_anchor_wrench_norm = (
            float(np.mean(np.linalg.norm(wrench[:anchor_step], axis=1))) if anchor_step > 0 else float("nan")
        )
        post_anchor_wrench_norm = (
            float(np.mean(np.linalg.norm(wrench[anchor_step:], axis=1))) if anchor_step < len(wrench) else float("nan")
        )
        print(f"[build_replay_corpus] wrote {ft_path}")
        print(f"[build_replay_corpus] T={T} anchor_step={anchor_step} precontact_boundary={boundary} "
              f"matched_window_N={matched_n} max_drift={max_drift:.5f} "
              f"mean|wrench|_pre={pre_anchor_wrench_norm:.4f} mean|wrench|_post={post_anchor_wrench_norm:.4f}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[build_replay_corpus] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
    simulation_app.close()
