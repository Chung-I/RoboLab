"""v5 env helpers that need the Isaac app (pxr); run with the repo conftest, on cml7."""
import numpy as np


def test_object_mesh_frame():
    from robolab.tasks.test_lift.generic_scene import build, root_pose_and_points
    from robolab.tasks.test_lift.v5_env import object_mesh, object_usd
    usd = object_usd(build("sugar_box"))
    v, f, scale = object_mesh(usd)
    _, _, pts = root_pose_and_points(usd)
    assert f.shape[1] == 3 and len(f) > 0 and scale.shape == (3,)
    assert np.allclose(v.min(0), pts.min(0), atol=1e-4) and np.allclose(v.max(0), pts.max(0), atol=1e-4)
