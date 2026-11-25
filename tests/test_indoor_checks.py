"""Orientation and alignment checks in a large covered space.

A station hall: the vault above the walk reconstructs more densely than the
floor, and the splat holds plenty the mesh does not (lamps, foliage, what is
seen through openings). Neither may fail an upright, aligned scene, and both
must still catch a scene that is upside down or offset.
"""

import numpy as np
import open3d as o3d

from simkit.gate import layer_residual
from simkit.geometry.navmesh import surface_samples
from simkit.geometry.up_check import camera_up_verdict, splat_up_verdict
from simkit.io.splat_io import Splat

RNG = np.random.default_rng(0)
_CAMERAS = np.c_[np.linspace(2, 28, 60), np.full(60, 15.0), np.full(60, 2.0)]


def _splat(points: np.ndarray) -> Splat:
    n = len(points)
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    return Splat(means=points, opacities=np.ones(n), scales=np.full((n, 3), 0.02), quats=quats,
                 sh_dc=np.zeros((n, 3)))


def _views(turned_over=False, n=600):
    """Five views per centre as the dataset cuts them: four around, one down."""
    views = []
    for k in range(n):
        yaw = RNG.uniform(0, 2 * np.pi)
        for extra in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
            a = yaw + extra
            optical = np.array([np.cos(a), np.sin(a), RNG.normal(0, 0.05)])
            optical /= np.linalg.norm(optical)
            down = np.array([0.0, 0.0, -1.0])
            right = np.cross(down, optical)
            right /= np.linalg.norm(right)
            down = np.cross(optical, right)
            views.append(np.column_stack([right, down, optical]))
        views.append(np.column_stack([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]]))
    views = np.array(views)
    if turned_over:  # the frame upside down: rotate everything 180 deg about x
        flip = np.diag([1.0, -1.0, -1.0])
        views = np.einsum("ij,njk->nik", flip, views)
    return views


def test_a_vault_denser_than_the_floor_fools_the_slab_but_not_the_cameras():
    xy = lambda n: RNG.uniform(0, 30, (n, 2))  # noqa: E731
    hall = np.vstack([np.c_[xy(20_000), RNG.normal(0.0, 0.01, 20_000)],
                      np.c_[xy(40_000), RNG.normal(10.0, 0.05, 40_000)]])
    assert splat_up_verdict(_splat(hall))["inverted"]
    verdict = camera_up_verdict(_views())
    assert verdict["decidable"] and not verdict["inverted"]
    assert verdict["image_down_along_gravity_median"] > 0.95


def test_a_frame_turned_over_is_caught():
    verdict = camera_up_verdict(_views(turned_over=True))
    assert verdict["decidable"] and verdict["inverted"]


def test_too_few_horizontal_views_is_undecidable():
    assert not camera_up_verdict(_views(n=10))["decidable"]


def _box_surface():
    box = o3d.geometry.TriangleMesh.create_box(30.0, 30.0, 8.0).subdivide_midpoint(3)
    return np.asarray(box.vertices), np.asarray(box.triangles)


def _on_surface(vertices, faces, n):
    points, _ = surface_samples(vertices, faces, 0.5)
    return points[RNG.choice(len(points), min(n, len(points)), replace=False)]


def test_what_the_mesh_lacks_is_not_an_offset():
    vertices, faces = _box_surface()
    points = np.vstack([_on_surface(vertices, faces, 30_000), np.c_[RNG.uniform(3, 27, (30_000, 2)),
                                                                   RNG.uniform(0.5, 6.0, 30_000)]])
    result = layer_residual(_splat(points), vertices, faces, _CAMERAS, corridor_m=40.0)
    assert result["median_m"] > 0.25            # the plain distance fails an aligned scene
    assert result["displacement_p90_m"] < 0.05  # the fitted offset does not


def test_an_offset_is_measured_as_such():
    vertices, faces = _box_surface()
    points = _on_surface(vertices, faces, 40_000) + np.array([0.2, -0.2, 0.0])
    result = layer_residual(_splat(points), vertices, faces, _CAMERAS, corridor_m=40.0)
    assert abs(result["displacement_p90_m"] - np.hypot(0.2, 0.2)) < 0.04
