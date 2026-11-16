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


def test_seeds_keep_the_walked_region_not_the_biggest():
    """A kerb splits the scene; the operator walked on the small side."""
    grid = np.zeros((10, 30), dtype=bool)
    grid[2:8, 1:8] = True    # the walked yard, 42 cells
    grid[2:8, 12:29] = True  # a bigger yard past a kerb, 102 cells
    ground = np.zeros(grid.shape); ground[:, 12:] = 0.4
    navmesh = Navmesh(grid, ground, np.zeros(2), 0.1)

    kept = largest_connected_region(navmesh, seed_points=np.array([[0.45, 0.45]]))
    assert kept.grid[2:8, 1:8].all() and not kept.grid[:, 12:].any()

    default = largest_connected_region(navmesh)
    assert default.grid[:, 12:29].any() and not default.grid[:, 1:8].any()


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


class FakeTriangleMesh(FakeMesh):
    """A real surface: build_navmesh samples its triangles, not its vertices."""

    def __init__(self, vertices, triangles):
        super().__init__(vertices, np.zeros((len(vertices), 3)))
        self.triangles = np.asarray(triangles, dtype=np.int64)


def test_a_floor_made_of_two_large_triangles_is_walkable_everywhere():
    """Four vertices, 100 m2 of floor: vertex counting saw four cells of ground
    and 396 of void. The surface is what is walked on, so it is what is read."""
    quad = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
    navmesh = build_navmesh(FakeTriangleMesh(quad, [[0, 1, 2], [0, 2, 3]]), cell_size=0.5)

    assert navmesh.area_m2 == pytest.approx(100.0, rel=0.05)


def test_a_large_panel_facing_down_is_a_ceiling_not_a_floor():
    quad = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], dtype=float)
    navmesh = build_navmesh(FakeTriangleMesh(quad, [[0, 2, 1], [0, 3, 2]]), cell_size=0.5)

    assert not navmesh.grid.any()


def test_a_thin_post_still_blocks_its_cell():
    quad = np.array([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], dtype=float)
    post = np.array([[2.0, 2.0, 0.0], [2.05, 2.0, 0.0], [2.05, 2.0, 1.2], [2.0, 2.0, 1.2]])
    vertices = np.vstack([quad, post])
    faces = [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [6, 5, 4], [7, 6, 4]]
    navmesh = build_navmesh(FakeTriangleMesh(vertices, faces), cell_size=0.5)

    column = int((2.02 - navmesh.origin[0]) / navmesh.cell_size)
    row = int((2.0 - navmesh.origin[1]) / navmesh.cell_size)
    assert not navmesh.grid[row, column]
    assert navmesh.area_m2 == pytest.approx(16.0, abs=1.5)


def test_empty_mesh_yields_empty_navmesh():
    navmesh = build_navmesh(FakeMesh(np.zeros((0, 3)), np.zeros((0, 3))))
    assert navmesh.area_m2 == 0.0


# --- multi-level -------------------------------------------------------------

from simkit.geometry.navmesh import build_navmesh_levels  # noqa: E402


def _two_storeys():
    """Ground floor 0..6 m, first floor 0..4 m at 3 m, joined by nothing walkable."""
    ground, ground_n = _floor_patch((0, 6), (0, 3))
    upper, upper_n = _floor_patch((0, 4), (0, 3), z=3.0)
    slab_under = upper.copy()
    slab_under[:, 2] = 2.8  # the slab's underside, facing down
    under_n = np.tile([0.0, 0.0, -1.0], (len(slab_under), 1))
    points = np.vstack([ground, upper, slab_under])
    normals = np.vstack([ground_n, upper_n, under_n])
    return FakeMesh(points, normals)


def test_single_level_navmesh_loses_the_storey_above():
    """The defect the multi-level navmesh exists for: lowest surface wins."""
    navmesh = build_navmesh(_two_storeys(), cell_size=0.25)
    assert np.nanmax(navmesh.ground_z[navmesh.grid]) < 1.0


def test_both_storeys_are_walkable_where_they_overlap():
    seeds = np.array([[1.0, 1.5, 0.0], [1.0, 1.5, 3.0]])
    levels = build_navmesh_levels(_two_storeys(), cell_size=0.25, seed_points=seeds)

    assert levels.levels == 2
    stacked = levels.grid.sum(axis=0) >= 2
    assert stacked.sum() * 0.25**2 > 6.0  # most of the 4 x 3 m overlap
    assert np.nanmin(levels.ground_z[1][levels.grid[1]]) == pytest.approx(3.0, abs=0.05)
    # the part of the ground floor with nothing above it is a single level
    assert levels.grid[0][:, -4:].any() and not levels.grid[1][:, -4:].any()


def test_a_storey_nobody_walked_on_is_not_kept():
    seeds = np.array([[1.0, 1.5, 0.0]])  # the walk stayed downstairs
    levels = build_navmesh_levels(_two_storeys(), cell_size=0.25, seed_points=seeds)

    heights = levels.ground_z[levels.grid]
    assert heights.size and np.nanmax(heights) < 1.0


def test_a_table_top_is_not_a_storey():
    """Clear and horizontal, but nobody stands on a desk."""
    floor, floor_n = _floor_patch((0, 4), (0, 3))
    top, top_n = _floor_patch((1, 2), (1, 2), z=0.75)
    seeds = np.array([[3.5, 0.5, 0.0]])
    levels = build_navmesh_levels(
        FakeMesh(np.vstack([floor, top]), np.vstack([floor_n, top_n])), cell_size=0.25, seed_points=seeds)

    heights = levels.ground_z[levels.grid]
    assert np.nanmax(heights) < 0.1
    rows, cols = np.nonzero(levels.grid.any(axis=0))
    xy = levels.origin + np.stack([cols, rows], axis=1) * 0.25 + 0.125
    under = (xy[:, 0] > 1.1) & (xy[:, 0] < 1.9) & (xy[:, 1] > 1.1) & (xy[:, 1] < 1.9)
    assert not under.any()  # nor is the floor under it


def test_lowest_channel_matches_single_level_on_one_storey():
    floor, floor_n = _floor_patch((0, 3), (0, 2))
    mesh = FakeMesh(floor, floor_n)
    single = build_navmesh(mesh, cell_size=0.2)
    levels = build_navmesh_levels(mesh, cell_size=0.2, stance_step_m=1.0)
    assert levels.levels == 1
    assert np.array_equal(levels.grid[0], single.grid)


def test_a_desk_seeded_by_a_camera_passing_over_it_is_dropped():
    """The ground ray of a camera carried over a desk lands on the desk top."""
    floor, floor_n = _floor_patch((0, 4), (0, 3))
    top, top_n = _floor_patch((1, 2.2), (1, 2.2), z=0.85)  # 1.4 m2 desk, its interior holds a stance
    seeds = np.array([[3.5, 0.5, 0.0], [1.7, 1.7, 0.85]])
    levels = build_navmesh_levels(
        FakeMesh(np.vstack([floor, top]), np.vstack([floor_n, top_n])), cell_size=0.25, seed_points=seeds)

    assert np.nanmax(levels.ground_z[levels.grid]) < 0.1
