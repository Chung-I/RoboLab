# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared scene/physics for the rolling-ball task pair.

``RollingBallInBowlTask``  -- the ball is given an initial velocity and rolls.
``StaticBallInBowlTask``   -- byte-identical spawn, zero initial velocity.

Both use RoboLab's banana_bowl scene, robot and camera rig, so everything the
policy sees is the in-distribution setup DROID-trained policies already handle.

Design notes, all measured rather than assumed:

* The ball is SPAWNED (``sim_utils.SphereCfg``) rather than referenced from the
  scene USD. Objects that come from ``import_scene`` have ``spawn = None``,
  which means mass/friction/damping are baked in by the artist and cannot be set
  from a task -- and every native candidate tested (orange, canned tuna, sauce
  bottle) stops dead within one control step when pushed. Spawning is the only
  way to get an object that rolls.

* Motion comes from ``init_state.lin_vel``, NOT from an event that writes pose
  or velocity each step. Two reasons. Writing the root pose every step makes the
  body kinematic, so contact forces are discarded and the gripper can never lift
  it -- that flaw silently invalidated an entire 4-method matrix. And writing
  velocity to a resting body is ignored, because PhysX puts it to sleep. A body
  spawned WITH velocity starts awake and moving, stays fully dynamic, and
  ``reset_scene_to_default`` restores the default root state (which includes
  velocities) on every episode, so the impulse recurs per trial for free.

* ``angular_damping`` is what stops the roll. This is a simulation fudge, not
  physics: PhysX models no rolling resistance, and sliding friction cannot slow
  a rolling sphere because the contact point does no sliding work -- verified,
  a ball with default 0.5/0.5 friction and near-zero damping rolled 2.12 m
  without decelerating. Real fruit assets were tried as a more natural
  alternative (their ellipsoid geometry should stop them); lemon, pomegranate
  and orange all refused to roll at all, so the damped sphere it is.
  Measured travel from a 0.25 m/s start: damping 2 -> 0.31 m over 3.8 s;
  6 -> 0.11 m; 15 -> 0.04 m. 2.0 is chosen: about one workspace width.

* The table is level (a free-rolling ball creeps 0.0002 m over 8 s), so the
  static control genuinely stays put and needs no correction.
"""

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import os
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.constants import OBJECT_DIR, SCENE_DIR
from robolab.core.task.conditionals import object_in_container, pick_and_place
from robolab.core.task.task import Task

BALL_RADIUS = 0.035          # r8 config (the only one with recorded successes)
# 0.3 kg (dense rubber ball), up from DOM's 0.05 kg -- the user-chosen fix for
# grasp-attempt knock-away. With zero rolling resistance the ball never stops
# once touched, so the only free variable that softens a knock WITHOUT damping
# is inertia: v_imparted = impulse / m, so 6x mass means ~6x slower knock-away.
# The r11 sweep measured the problem: EDGE 0/4 at EVERY speed 0.002-0.015 m/s
# (even near-static), i.e. the failure was contact sensitivity, not target speed.
# Env-var overridable for future sweeps (never edit mid-run; drivers re-import
# this module per block).
#
# The default was 0.05 until 2026-08-03, contradicting everything written above:
# the 0.3 kg decision was documented here but never applied, and no driver ever
# set ARENA_BALL_MASS, so EVERY ball-task result recorded before this date ran at
# DOM's 0.05 kg -- i.e. with the knock-away problem the r11 sweep diagnosed still
# fully present. Treat pre-2026-08-03 ball numbers as measuring the unfixed
# benchmark; they are not comparable to runs after it.
BALL_MASS = float(os.environ.get("ARENA_BALL_MASS", "0.3"))
# angular_damping = 0: no simulation fudge at all. PhysX models no rolling
# resistance and sliding friction cannot slow a rolling sphere, so the ball
# rolls at CONSTANT speed. That is both more honest physics and a cleaner
# stimulus -- the staleness penalty stays constant instead of decaying as a
# damped ball slows, which would make late grasps artificially easy.
BALL_ANG_DAMPING = 0.0
BALL_START = (0.55, 0.26, 0.040)
# RETUNED after the first acceptance test. 0.03 m/s x a 25 s episode = 0.6 m of
# travel, which took the ball clear out of the arm's reach (and off the table)
# mid-episode: the video showed the gripper reaching it at 14 s and 18 s, then
# nothing to grasp. The error was verifying the roll over an 8 s probe window
# while episodes ran 25 s.
# Budget instead from the two measured constraints: the arm's useful reach is
# ~0.30 m of travel, and the one successful static grasp took 17.7 s. So
# 0.015 m/s x 20 s = 0.30 m -- the ball stays catchable for the whole episode,
# while still drifting 1.4 cm per BASE cycle (0.93 s) vs 0.2 cm per EDGE cycle,
# i.e. inside the ~2 cm capture tolerance where staleness still costs a grasp.
# Speed is env-var-overridable for the velocity sweep (ARENA_BALL_SPEED, m/s).
# Read once at import; each evaluation block is a fresh process, so a sweep
# driver sets it per block -- never edit this file mid-run (a driver re-imports
# the module per block, so an edit would silently change speed mid-experiment).
BALL_SPEED = float(os.environ.get("ARENA_BALL_SPEED", "0.015"))
BALL_VELOCITY = (0.0, -BALL_SPEED, 0.0)
EPISODE_S = 25   # back to 25 s: a successful static grasp took 17.7 s
BANANA_PARK = (5.0, 5.0, 0.05)   # out of the workspace and out of frame

# NO play-area walls, deliberately. The task is: someone nudges the ball, it
# rolls, and the robot must chase it down, arrest it and bowl it. Under that
# definition a ball that rolls off the table is a REAL failure -- the robot was
# too slow or too clumsy -- and walling the table in would hide exactly the skill
# being measured. With angular_damping = 0 nothing stops a rolling sphere, so a
# clumsy approach is punished permanently; that is the intended difficulty.
# Note the ball never leaves on its own: at 0.015 m/s it covers 0.375 m over a
# 25 s episode, well inside the table, so any loss is caused by the robot.
BOWL_POSE = (0.33, -0.10, 0.11)
BOWL_ROT = (0.67, -0.74, 0.0, 0.0)


def ball_spawn_cfg():
    """Shared by both variants so their physics cannot drift apart -- an earlier
    pair duplicated these props and a damping fix reached only one half of it."""
    return sim_utils.UsdFileCfg(
        # A MESH sphere, authored for this task (assets/objects/basic/ball.usd,
        # 24x48 UV sphere, r=0.035). It cannot be a procedural sim_utils.SphereCfg:
        # RoboLab's WorldState reads local MESH points to build bounding boxes
        # (_read_local_mesh_points), so an analytic UsdGeomSphere dies with
        # "no mesh geometry found under prim 'ball'". It also cannot be one of the
        # shipped fruit assets: their baked collision approximation is a faceted
        # convex hull, so they tip onto a facet instead of rolling (measured -- a
        # kicked lemon, pomegranate and orange all travelled < 1 mm). A finely
        # tessellated sphere satisfies both: it is a mesh, and its convex hull is
        # essentially a sphere.
        usd_path=os.path.join(OBJECT_DIR, "basic", "ball.usd"),
        # REQUIRED: contact_object_list names "ball", so RoboLab builds a
        # gripper__ball contact sensor; without this, env construction dies with
        # "could not find any bodies with contact reporter API".
        activate_contact_sensors=True,
        mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            angular_damping=BALL_ANG_DAMPING,
            max_depenetration_velocity=0.5,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        # NOTE: UsdFileCfg has no physics_material field (RigidObjectSpawnerCfg
        # exposes only mass/rigid/collision props + activate_contact_sensors), so
        # friction comes from the asset. That is fine here: sliding friction cannot
        # slow a rolling sphere anyway.
    )


def _scene_cfg(lin_vel):
    """Scene builder shared by both variants; only ``lin_vel`` differs."""

    @configclass
    class _Scene:
        scene = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/scene",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(SCENE_DIR, "banana_bowl.usda"),
                activate_contact_sensors=True,
            ),
        )
        # The banana is baked into banana_bowl.usda and cannot be deleted from a
        # task, so it is PARKED far outside the workspace and the camera frusta.
        # It has to go: the instruction names the ball, but the acceptance-test
        # video showed the arm visiting the banana, and a DROID-trained policy has
        # seen far more bananas than red balls -- leaving it in makes any failure
        # ambiguous between "cannot track a moving target" and "grasped the wrong
        # object".
        banana = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/scene/banana",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=BANANA_PARK, rot=(1.0, 0.0, 0.0, 0.0)),
        )
        bowl = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/scene/bowl",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(pos=BOWL_POSE, rot=BOWL_ROT),
        )
        ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ball",
            spawn=ball_spawn_cfg(),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=BALL_START, rot=(1.0, 0.0, 0.0, 0.0), lin_vel=lin_vel),
        )

    return _Scene


@configclass
class BallInBowlTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_in_container,
        params={"object": "ball", "container": "bowl", "gripper_name": "gripper",
                "tolerance": 0.05, "require_contact_with": True,
                "require_gripper_detached": True},
    )


