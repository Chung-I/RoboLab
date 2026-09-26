# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated one-object test-lift scenes for the corpus study (docs/studies/2026-09-27-test-lift-corpus.md).

``build(key)`` writes two files under ``output/test_lift/corpus/usd/`` and never edits an
asset in place:

* ``<key>.scene.usda`` -- the table, robot mount and ground plane of
  ``assets/scenes/bin_mug_mustard_marker_bowl.usda`` (the mustard scene, which v3 qualified),
  with one object prim ``/world/<key>`` and nothing else. The object sits at
  ``OBJECT_XY`` with its bottom ``DROP_M`` above the nominal table top.
* ``<key>.override.usda`` -- only when the asset needs it (the frame fix, below). It
  references the original asset and edits it in a layer of its own.

THE FRAME FIX
-------------
The test-lift drivers read the object's mesh points in the frame of the prim the task
declares (``predicate_logic._read_local_mesh_points``) and the object's pose from the rigid
body IsaacLab finds under that prim. The two frames agree only when the rigid body IS the
declared prim. Four objaverse assets carry ``PhysicsRigidBodyAPI`` on a child prim
(``apple``, ``gregorys_coffee_cup``, ``lunchbag``, ``snickers_bar``; ``snickers_bar`` has two).
For those, the override removes the API from every descendant, applies it (with
``PhysicsMassAPI``) to the root, and deactivates any joint under the root.

The v3 "mesh frame 5 cm off the root" diagnosis of ``cordless_drill`` and ``measuring_cup`` is
not a mesh offset: both assets are centred on their root, and their v3 scene
(``mugs4_measuringcup_drill_bowl.usda``) places its table at z = 0.050 instead of about 0.005.
A generated scene uses one table for every object, so ``z_table`` is comparable across objects.

The root's own authored rotation and scale are kept: the prim's ops are rewritten as
``translate, orient, scale`` with the asset root's rotation and scale, and the task's
``init_state.rot`` is that same rotation, so a reset does not rotate the object.
"""
from __future__ import annotations

import os

import numpy as np

from analysis.test_lift.corpus import REPO_ROOT, default_mass, resolve

#: Where generated USDs go. ``ROBOLAB_TEST_LIFT_CORPUS_USD_DIR`` overrides it (the sweep sets it).
OUT_DIR = os.environ.get("ROBOLAB_TEST_LIFT_CORPUS_USD_DIR",
                         os.path.join(REPO_ROOT, "output", "test_lift", "corpus", "usd"))
TABLE_USD = os.path.join(REPO_ROOT, "assets", "fixtures", "table_oak.usd")
FRANKA_TABLE_USD = os.path.join(REPO_ROOT, "assets", "fixtures", "franka_table.usd")
TABLE_TRANSLATE = (0.5471368432044983, 0.0, -0.3449999988079071)   # mustard scene
TABLE_TOP_Z = 0.005            # nominal; the settled mustard measures z_table = 0.0028
OBJECT_XY = (0.55, 0.0)        # mustard passed v3 at (0.59, 0.00); table x spans [0.197, 0.897]
DROP_M = 0.005                 # spawn gap above the table top; the driver settles it

_SCENE = """#usda 1.0
(
    defaultPrim = "world"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "world"
{{
    def Material "PhysicsMaterial" (
        prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]
    )
    {{
        float physics:dynamicFriction = 2
        float physics:staticFriction = 2
        uniform token physxMaterial:frictionCombineMode = "max"
    }}

    def "table" (
        prepend payload = @{table}@
    )
    {{
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = ({tx}, {ty}, {tz})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def "{key}" (
        prepend payload = @{obj}@
    )
    {{
        quatf xformOp:orient = ({qw!r}, {qx!r}, {qy!r}, {qz!r})
        float3 xformOp:scale = ({sx!r}, {sy!r}, {sz!r})
        double3 xformOp:translate = ({px!r}, {py!r}, {pz!r})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    }}

    def "franka_table" (
        prepend payload = @{franka}@
    )
    {{
        quatd xformOp:orient = (6.123233995736766e-17, 0, 0, 1)
        double3 xformOp:translate = (-0.087, 0, 0)
    }}

    def Xform "GroundPlane"
    {{
        token visibility = "invisible"
        quatf xformOp:orient = (1, 0, 0, 0)
        float3 xformOp:scale = (1, 1, 1)
        double3 xformOp:translate = (0, 0, -0.697)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

        def Mesh "CollisionMesh"
        {{
            uniform bool doubleSided = 0
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0, 1, 2, 3]
            normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)]
            point3f[] points = [(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)]
        }}

        def Plane "CollisionPlane" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            uniform token axis = "Z"
            uniform token purpose = "guide"
        }}
    }}
}}
"""


def _open(usd_path):
    from pxr import Usd
    st = Usd.Stage.Open(usd_path)
    if st is None:
        raise RuntimeError(f"cannot open {usd_path}")
    return st, st.GetDefaultPrim()


def rigid_body_prims(usd_path) -> tuple[str, list[str]]:
    """(default prim path, paths of every prim under it with PhysicsRigidBodyAPI)."""
    from pxr import Usd, UsdPhysics
    st, root = _open(usd_path)
    rb = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
          if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    return str(root.GetPath()), rb


def write_override(key: str, usd_path: str) -> str:
    """Write ``<key>.override.usda``: the rigid body moved to the default prim (see module doc)."""
    from pxr import Sdf, Usd, UsdPhysics
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{key}.override.usda")
    _, src_root = _open(usd_path)
    name = src_root.GetName()
    layer = Sdf.Layer.CreateNew(out) if not os.path.exists(out) else Sdf.Layer.FindOrOpen(out)
    layer.Clear()
    st = Usd.Stage.Open(layer)
    root = st.DefinePrim(f"/{name}", "Xform")
    root.GetReferences().AddReference(os.path.abspath(usd_path))
    st.SetDefaultPrim(root)
    # Instance proxies are read-only: make every instance under the root editable first.
    changed = True
    while changed:
        changed = False
        for p in Usd.PrimRange(root):
            if p.IsInstance():
                p.SetInstanceable(False)
                changed = True
    for p in Usd.PrimRange(root):
        if p == root:
            continue
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if p.IsA(UsdPhysics.Joint):
            p.SetActive(False)
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root)
    layer.Save()
    return out


def root_pose_and_points(usd_path):
    """The asset root's rotation (w, x, y, z) and scale, and its mesh points in the root's
    rotated frame at world scale (the same convention as ``_read_local_mesh_points``)."""
    from pxr import Gf, Usd, UsdGeom
    st, root = _open(usd_path)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    M = xc.GetLocalToWorldTransform(root)
    tf = Gf.Transform(M)
    q = tf.GetRotation().GetQuat()
    s = tf.GetScale()
    inv = M.RemoveScaleShear().GetInverse()
    pts = []
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(p)
        if mesh.GetPurposeAttr().Get() not in (None, UsdGeom.Tokens.default_):
            continue
        v = mesh.GetPointsAttr().Get()
        if not v:
            continue
        m2w = xc.GetLocalToWorldTransform(p)
        pts.extend([list(inv.Transform(m2w.Transform(Gf.Vec3d(x)))) for x in v])
    if not pts:
        raise RuntimeError(f"no mesh geometry under {root.GetPath()} in {usd_path}")
    rot = (q.GetReal(), *q.GetImaginary())
    return rot, tuple(float(x) for x in s), np.asarray(pts, dtype=float)


def _quat_rotate(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return (R @ np.asarray(v).T).T


def build(key: str) -> dict:
    """Generate (or reuse) the scene for corpus key ``key``; return what the task needs."""
    entry = resolve(key)
    usd = os.path.join(REPO_ROOT, entry["usd_path"])
    root_path, rbs = rigid_body_prims(usd)
    override = None
    if rbs != [root_path]:
        override = write_override(key, usd)
        usd_used = override
    else:
        usd_used = usd
    rot, scale, pts = root_pose_and_points(usd_used)
    z_bottom = float(_quat_rotate(rot, pts)[:, 2].min())        # bottom below the root, world frame
    pos = (OBJECT_XY[0], OBJECT_XY[1], TABLE_TOP_Z + DROP_M - z_bottom)
    os.makedirs(OUT_DIR, exist_ok=True)
    scene_path = os.path.join(OUT_DIR, f"{key}.scene.usda")
    text = _SCENE.format(table=TABLE_USD, tx=TABLE_TRANSLATE[0], ty=TABLE_TRANSLATE[1], tz=TABLE_TRANSLATE[2],
                         key=key, obj=os.path.abspath(usd_used), franka=FRANKA_TABLE_USD,
                         qw=float(rot[0]), qx=float(rot[1]), qy=float(rot[2]), qz=float(rot[3]),
                         sx=scale[0], sy=scale[1], sz=scale[2], px=pos[0], py=pos[1], pz=pos[2])
    tmp = f"{scene_path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, scene_path)
    return dict(key=key, entry=entry, scene_usd=scene_path, override_usd=override, rigid_bodies=rbs,
                pos=pos, rot=tuple(float(x) for x in rot), scale=scale, mass=default_mass(entry),
                extent=(pts.max(0) - pts.min(0)).tolist())
