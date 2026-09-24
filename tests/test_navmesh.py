"""Tests for walkable-space extraction.

Overstating walkable area is the easiest way to make a bundle look more valuable
than it is, so the cases that matter here are the ones that should be *rejected*.
"""

import numpy as np
import pytest

from simkit.geometry.navmesh import Navmesh, build_navmesh, largest_connected_region


class FakeMesh:
    """Minimal stand-in: build_navmesh only reads vertices and vertex normals."""

    def __init__(self, vertices, normals):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.vertex_normals = np.asarray(normals, dtype=np.float64)


def _floor_patch(x_range, y_range, step=0.05, z=0.0):
    xs = np.arange(*x_range, step)
    ys = np.arange(*y_range, step)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z)])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    return points, normals


def test_flat_floor_is_walkable():
    points, normals = _floor_patch((0, 2), (0, 2))
    navmesh = build_navmesh(FakeMesh(points, normals), cell_size=0.2)

    assert navmesh.grid.any()
    assert navmesh.area_m2 == pytest.approx(4.0, abs=0.5)


def test_sloped_outdoor_ground_stays_walkable():
    """Outdoor captures routinely span metres of elevation; a flat-floor
    definition would discard most of the genuinely walkable ground."""
    xs = np.arange(0, 6, 0.05)
    ys = np.arange(0, 2, 0.05)
    grid_x, grid_y = np.meshgrid(xs, ys)
    slope = grid_x * 0.10  # a gentle 10% ramp, 60 cm of rise
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), slope.ravel()])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))

    navmesh = build_navmesh(FakeMesh(points, normals), cell_size=0.2)

    assert navmesh.area_m2 == pytest.approx(12.0, abs=1.5)
    assert navmesh.elevation_span_m > 0.4  # the ramp is represented, not flattened


def test_ground_height_is_per_cell_not_global():
    """Ground height must track the terrain, not collapse to one plane."""
    xs = np.arange(0, 6, 0.05)
    ys = np.arange(0, 2, 0.05)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ramp = grid_x * 0.08
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), ramp.ravel()])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))

    navmesh = build_navmesh(FakeMesh(points, normals), cell_size=0.2)
    heights = navmesh.ground_z[navmesh.grid]

    assert np.nanmin(heights) == pytest.approx(0.0, abs=0.05)
    assert np.nanmax(heights) == pytest.approx(0.47, abs=0.08)
    assert len(np.unique(np.round(heights, 2))) > 5  # a gradient, not a single plane


def test_a_sharp_kerb_creates_a_conservative_seam():
    """A cell straddling a step contains both ground levels, so it reads as
    obstructed. That is conservative and correct: better to under-report
    walkable area than to promise a foothold on the edge of a drop."""
    xs = np.arange(0, 4, 0.05)
    ys = np.arange(0, 2, 0.05)
    grid_x, grid_y = np.meshgrid(xs, ys)
    step = np.where(grid_x > 2, 0.15, 0.0)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel(), step.ravel()])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))

    navmesh = build_navmesh(FakeMesh(points, normals), cell_size=0.2)

    assert navmesh.grid.any()
    assert navmesh.area_m2 < 8.0  # the seam costs area, and that is the intent


def test_floor_under_an_obstacle_is_not_walkable():
    """A robot cannot stand where a table already is."""
    floor, floor_normals = _floor_patch((0, 2), (0, 2))
    # A block occupying 0.5 m of the floor, at knee height.
    block, block_normals = _floor_patch((0, 0.5), (0, 2), z=0.5)

    navmesh = build_navmesh(
        FakeMesh(np.vstack([floor, block]), np.vstack([floor_normals, block_normals])),
        cell_size=0.2,
    )
    covered = navmesh.grid[:, :2]  # first columns lie under the block

    assert not covered.any()
    assert navmesh.grid.any()  # the rest of the floor survives


def test_vertical_surfaces_are_never_walkable():
    points, _ = _floor_patch((0, 2), (0, 2))
    wall_normals = np.tile([1.0, 0.0, 0.0], (len(points), 1))

    navmesh = build_navmesh(FakeMesh(points, wall_normals), cell_size=0.2)

    assert not navmesh.grid.any()


def test_an_unreachable_ledge_is_rejected():
    """A rooftop is horizontal and clear, but not somewhere a robot can walk to."""
    floor, floor_normals = _floor_patch((0, 3), (0, 2))
    ledge, ledge_normals = _floor_patch((3, 4), (0, 2), z=2.5)

    navmesh = build_navmesh(
        FakeMesh(np.vstack([floor, ledge]), np.vstack([floor_normals, ledge_normals])),
        cell_size=0.2,
    )
    heights = navmesh.ground_z[navmesh.grid]

    assert navmesh.grid.any()
    assert np.nanmax(heights) < 1.0  # the 2.5 m ledge did not survive


def test_largest_connected_region_drops_unreachable_patches():
    grid = np.zeros((10, 20), dtype=bool)
    grid[2:8, 2:8] = True  # main area
    grid[4:6, 15:17] = True  # island behind a wall

    kept = largest_connected_region(Navmesh(grid, np.zeros(grid.shape), np.zeros(2), 0.1))

    assert kept.grid[2:8, 2:8].all()
    assert not kept.grid[4:6, 15:17].any()


def test_navmesh_area_uses_cell_size():
    grid = np.ones((4, 4), dtype=bool)
    navmesh = Navmesh(grid, np.zeros((4, 4)), np.zeros(2), 0.5)
    assert navmesh.area_m2 == pytest.approx(4.0)


def test_sample_points_carry_their_ground_height():
    grid = np.ones((4, 4), dtype=bool)
    ground = np.full((4, 4), 1.25)
    samples = Navmesh(grid, ground, np.zeros(2), 0.5).sample_points(5)

    assert samples.shape == (5, 3)
    np.testing.assert_allclose(samples[:, 2], 1.25)


def test_empty_mesh_yields_empty_navmesh():
    navmesh = build_navmesh(FakeMesh(np.zeros((0, 3)), np.zeros((0, 3))))
    assert navmesh.area_m2 == 0.0
