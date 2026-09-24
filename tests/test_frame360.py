"""The frame keeps ARKit gravity and places the walked floor at z = 0."""

import numpy as np

from simkit.frame360 import SIM_FROM_SOURCE_ROTATION, scene_frame
from tests.conftest import grid_floor


def _walk(count=60, floor_up=0.0, slope_deg=0.0):
    t = np.linspace(-7, 7, count)
    return [(x, 0.3 * np.sin(x), floor_up + np.tan(np.radians(slope_deg)) * x) for x in t]


def test_rotation_is_the_fixed_axis_change(cameras_file):
    vertices, faces = grid_floor(height_up=1.3)
    frame, _ = scene_frame(vertices, faces, cameras_file(_walk(floor_up=1.3)))
    np.testing.assert_allclose(frame.transform[:3, :3], SIM_FROM_SOURCE_ROTATION)
    assert np.linalg.det(SIM_FROM_SOURCE_ROTATION) == 1.0


def test_floor_is_measured_not_assumed(cameras_file):
    vertices, faces = grid_floor(height_up=-2.4)
    frame, walk = scene_frame(vertices, faces, cameras_file(_walk(floor_up=-2.4)))
    assert abs(frame.floor_up_m + 2.4) < 1e-6
    moved = vertices @ frame.transform[:3, :3].T + frame.transform[:3, 3]
    assert abs(np.median(moved[:, 2])) < 1e-6
    assert abs(walk["camera_height_m"] - 2.1) < 1e-6


def test_pole_height_is_recorded_not_gated(cameras_file):
    vertices, faces = grid_floor()
    _, low = scene_frame(vertices, faces, cameras_file(_walk(), height=0.4))
    _, high = scene_frame(vertices, faces, cameras_file(_walk(), height=2.8))
    assert abs(low["camera_height_m"] - 0.4) < 1e-6 and abs(high["camera_height_m"] - 2.8) < 1e-6
    assert low["grounded_fraction"] == high["grounded_fraction"] == 1.0


def test_a_slope_is_kept_and_both_witnesses_see_it(cameras_file):
    vertices, faces = grid_floor(slope_deg=5.0)
    frame, walk = scene_frame(vertices, faces, cameras_file(_walk(slope_deg=5.0)))
    np.testing.assert_allclose(frame.transform[:3, :3], SIM_FROM_SOURCE_ROTATION)
    assert abs(walk["floor_slope_deg"] - 5.0) < 0.1
    assert walk["mesh_agreement_deg"] < 0.1


def test_a_walk_off_the_floor_disagrees(cameras_file):
    vertices, faces = grid_floor()
    _, walk = scene_frame(vertices, faces, cameras_file(_walk(slope_deg=3.0)))
    assert walk["mesh_agreement_deg"] > 2.5
