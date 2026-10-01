"""Coverage walls: as tall as what the capture saw there, full height where it saw nothing.

The scene: walkable floor on x < 2 m, and past it a strip the navmesh never
kept (ground unknown), which therefore gets a wall two cells deep (x 2.0-2.5 m).
What stands on that strip changes per test; the assertions are on the
collision boxes that come out, measured from the ground under each cell.
"""

import numpy as np
import pytest

from simkit.geometry.coverage_bounds import coverage_walls, mesh_witness, uncovered_cells, wall_boxes
from simkit.geometry.splat_obstacles import obstacle_boxes
from simkit.geometry.navmesh import Navmesh

CELL = 0.25
FULL = 1.6
STEP = 0.06  # wall_boxes rounds wall tops up by at most this
WALL_X = (2.125, 2.375)  # centres of the two wall columns
NONE = np.zeros((0, 3))


def _ground(y):
    return 0.0 * y


def _navmesh(ground=_ground):
    rows, cols = 16, 16  # 4 m x 4 m
    grid = np.zeros((rows, cols), dtype=bool)
    grid[:, :8] = True  # walkable for x < 2 m
    y = (np.arange(rows)[:, None] + 0.5) * CELL * np.ones((1, cols))
    return Navmesh(grid, np.where(grid, ground(y), np.nan), np.zeros(2), CELL)


def _block(x0, x1, y0, y1, top, step=0.02, ground=_ground):
    """Points filling an object from the ground to ``top`` above it."""
    xs, ys, zs = np.arange(x0, x1, step), np.arange(y0, y1, step), np.arange(0.0, top + 1e-9, step)
    gx, gy, gz = np.meshgrid(xs, ys, zs)
    # The ground of a cell is read at the cell centre, as the navmesh stores it.
    cell_y = (np.floor(gy / CELL) + 0.5) * CELL
    return np.column_stack([gx.ravel(), gy.ravel(), (gz + ground(cell_y)).ravel()])


def _side(points):
    """Normals of a vertical face, for mesh points."""
    return np.tile([1.0, 0.0, 0.0], (len(points), 1))


def _walls(mesh=NONE, splat=NONE, normals=None, ground=_ground):
    navmesh = _navmesh(ground)
    fence = uncovered_cells(navmesh)
    assert fence[:, 8:10].all() and not fence[:, 10:].any()
    normals = _side(mesh) if normals is None else normals
    cells, heights = coverage_walls(navmesh, mesh, normals, splat)
    return wall_boxes(navmesh, cells, heights)


def _boxes_over(boxes, x, y):
    return [b for b in boxes
            if abs(x - b["centre"][0]) < b["half_extents"][0] and abs(y - b["centre"][1]) < b["half_extents"][1]]


def _height_at(boxes, x, y, ground=_ground):
    """Wall top over (x, y), above the ground of that cell."""
    tops = [b["centre"][2] + b["half_extents"][2] for b in boxes
            if abs(x - b["centre"][0]) < b["half_extents"][0] and abs(y - b["centre"][1]) < b["half_extents"][1]]
    assert len(tops) == 1, f"expected one wall box over ({x}, {y}), found {len(tops)}"
    return tops[0] - ground((np.floor(y / CELL) + 0.5) * CELL)


def _assert_full(boxes, x, y, ground=_ground):
    assert FULL <= _height_at(boxes, x, y, ground) <= FULL + STEP


def test_unobserved_strip_keeps_a_full_wall():
    boxes = _walls()
    for x in WALL_X:
        for y in np.arange(0, 4, CELL) + CELL / 2:
            _assert_full(boxes, x, y)


def test_nothing_seen_past_the_fence_gets_no_box():
    boxes = _walls(_block(2.0, 4.0, 0.0, 4.0, top=0.0))  # floor only
    assert not _boxes_over(boxes, 2.875, 1.0) and not _boxes_over(boxes, 3.625, 1.0)


def test_the_whole_sofa_is_in_collision_seat_and_backrest_past_the_fence():
    """A sofa 1 m deep (seat 0.42 m) with a 0.86 m backrest at its far end:
    only its front half lies in the fence ring."""
    seat = _block(2.0, 3.0, 0.0, 2.0, top=0.42)
    back = _block(3.0, 3.25, 0.0, 2.0, top=0.86)
    boxes = _walls(np.vstack([seat, back]))
    for x in (2.125, 2.625, 2.875):  # seat, inside the ring and past it
        assert 0.42 <= _height_at(boxes, x, 1.0) <= 0.42 + STEP
    assert 0.86 <= _height_at(boxes, 3.125, 1.0) <= 0.86 + STEP  # backrest
    assert not _boxes_over(boxes, 3.625, 1.0)  # floor behind the sofa


def test_low_furniture_gets_its_own_height_and_the_wall_beside_it_stays_full():
    sofa = _block(2.0, 2.5, 0.0, 2.0, top=0.42)  # seat 0.42 m, along y < 2 m
    wall = _block(2.0, 2.5, 2.0, 4.0, top=2.6)  # a real wall to the ceiling, y > 2 m
    boxes = _walls(np.vstack([sofa, wall]))
    for x in WALL_X:
        for y in (0.125, 0.875, 1.875):
            assert 0.42 <= _height_at(boxes, x, y) <= 0.42 + STEP
        for y in (2.125, 3.125, 3.875):
            _assert_full(boxes, x, y)


def test_furniture_on_a_slope_is_measured_from_its_own_ground():
    """Ground rising 10% along y: 40 cm over the strip."""
    def ramp(y):
        return 0.5 + 0.1 * y
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42, ground=ramp)
    boxes = _walls(sofa, ground=ramp)
    for x in WALL_X:
        for y in np.arange(0, 4, CELL) + CELL / 2:
            assert 0.42 <= _height_at(boxes, x, y, ramp) <= 0.42 + STEP


def test_a_pane_the_mesh_lost_is_closed_by_the_splat():
    """Mesh: only the bottom of the pane, 40 cm. Splat: the frosted pane up to 1.5 m."""
    rail = _block(2.0, 2.1, 0.0, 4.0, top=0.40)
    pane = _block(2.0, 2.1, 0.0, 4.0, top=1.50, step=0.04)
    boxes = _walls(rail, pane)
    for y in (0.125, 1.0, 3.875):
        assert _height_at(boxes, 2.125, y) >= 1.5


def test_a_wall_with_its_middle_lost_keeps_full_height():
    """Mesh: a 30 cm skirting and the wall again from 1.8 m up, nothing between."""
    skirting = _block(2.0, 2.1, 0.0, 4.0, top=0.30)
    upper = _block(2.0, 2.1, 0.0, 4.0, top=2.6)
    upper = upper[upper[:, 2] > 1.8]
    _assert_full(_walls(np.vstack([skirting, upper])), 2.125, 1.0)


def test_a_ceiling_over_low_furniture_does_not_raise_it():
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42)
    ceiling = _block(2.0, 2.5, 0.0, 4.0, top=0.0) + [0.0, 0.0, 2.7]
    mesh = np.vstack([sofa, ceiling])
    normals = np.vstack([_side(sofa), np.tile([0.0, 0.0, -1.0], (len(ceiling), 1))])
    assert 0.42 <= _height_at(_walls(mesh, normals=normals), 2.125, 1.0) <= 0.42 + STEP


def test_a_few_splat_floaters_do_not_raise_a_seen_cell():
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42)
    floaters = np.array([[2.1, 1.0, 1.3], [2.2, 1.05, 1.35], [2.15, 1.1, 1.5]])
    assert 0.42 <= _height_at(_walls(sofa, floaters), 2.125, 1.0) <= 0.42 + STEP


def test_mesh_debris_does_not_size_a_wall_down_but_furniture_does():
    """A sofa made of a few hundred triangles, and one stray triangle at 30 cm."""
    import open3d as o3d

    sofa = o3d.geometry.TriangleMesh.create_box(0.5, 1.0, 0.42).translate((2.0, 0.5, 0.0))
    sofa = sofa.subdivide_midpoint(number_of_iterations=2)
    stray = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[2.05, 3.05, 0.3], [2.2, 3.05, 0.3], [2.05, 3.2, 0.3]]),
        o3d.utility.Vector3iVector([[0, 1, 2]]))
    points, normals = mesh_witness(sofa + stray, CELL)
    boxes = _walls(points, normals=normals)
    assert 0.42 <= _height_at(boxes, 2.125, 1.0) <= 0.42 + STEP
    _assert_full(boxes, 2.125, 3.125)


def test_tilted_facets_in_a_ceiling_do_not_raise_low_furniture():
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42)
    facets = np.array([[2.1, 1.0, 2.7], [2.2, 1.05, 2.72], [2.15, 1.1, 2.69]])
    mesh = np.vstack([sofa, facets])
    assert 0.42 <= _height_at(_walls(mesh), 2.125, 1.0) <= 0.42 + STEP


def test_floor_or_a_climbable_kerb_keeps_the_wall_full():
    """Past the wall lies filled, unobserved terrain: a wall the robot can step
    over (the navmesh climbs 0.25 m) would not contain it."""
    floor = _block(2.0, 4.0, 0.0, 2.0, top=0.0)
    kerb = _block(2.0, 4.0, 2.0, 4.0, top=0.20)
    boxes = _walls(np.vstack([floor, kerb]))
    _assert_full(boxes, 2.125, 1.0)
    _assert_full(boxes, 2.125, 3.0)


def test_every_wall_cell_gets_a_box_of_its_own_height_however_many_boxes_that_takes():
    """Alternating heights make one box per cell; none may be dropped."""
    size = 420
    grid = np.zeros((size, size), dtype=bool)
    grid[2:-2, 2:-2] = True
    navmesh = Navmesh(grid, np.where(grid, 0.0, np.nan), np.zeros(2), CELL)
    cells = uncovered_cells(navmesh)
    rows, cols = np.indices(cells.shape)
    heights = np.where((rows + cols) % 2 == 0, 0.42, FULL) * cells
    boxes = wall_boxes(navmesh, cells, heights)
    assert len(boxes) > 1500
    covered = np.zeros(cells.shape, dtype=int)
    for box in boxes:
        c0 = int(round((box["centre"][0] - box["half_extents"][0]) / CELL))
        c1 = int(round((box["centre"][0] + box["half_extents"][0]) / CELL))
        r0 = int(round((box["centre"][1] - box["half_extents"][1]) / CELL))
        r1 = int(round((box["centre"][1] + box["half_extents"][1]) / CELL))
        covered[r0:r1, c0:c1] += 1
        top = box["centre"][2] + box["half_extents"][2]
        assert heights[r0:r1, c0:c1].max() <= top <= heights[r0:r1, c0:c1].min() + STEP
    assert (covered[cells] == 1).all() and not covered[~cells].any()


def test_the_splat_alone_never_lowers_a_wall():
    """A hole in the mesh where the splat holds something low stays a full wall."""
    blob = _block(2.0, 2.5, 0.0, 4.0, top=0.6, step=0.04)
    _assert_full(_walls(splat=blob), 2.125, 1.0)


def test_every_splat_obstacle_cluster_gets_a_box():
    """More than the 150 clusters the old cap kept."""
    size = 100
    grid = np.ones((size, size), dtype=bool)
    navmesh = Navmesh(grid, np.zeros((size, size)), np.zeros(2), CELL)
    blocked = np.zeros((size, size), dtype=bool)
    blocked[::4, ::4] = True  # 625 single-cell clusters
    assert len(obstacle_boxes(navmesh, blocked)) == int(blocked.sum())


def test_walkable_ground_never_gets_a_box():
    """A wall face hanging above walkable floor (a soffit from 1.8 m up)."""
    soffit = _block(0.9, 1.1, 0.0, 4.0, top=2.6)
    soffit = soffit[soffit[:, 2] > 1.8]
    assert not _boxes_over(_walls(soffit), 1.0, 1.0)
