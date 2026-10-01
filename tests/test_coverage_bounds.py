"""Coverage walls: as tall as what the capture saw there, full height where it saw nothing.

The scene: walkable floor on x < 2 m, and past it a strip the navmesh never
kept (ground unknown), which therefore gets a wall. What stands on that strip
changes per test; the assertions are on the collision boxes that come out.
"""

import numpy as np
import pytest

from simkit.geometry.coverage_bounds import uncovered_cells, wall_boxes, wall_heights
from simkit.geometry.navmesh import Navmesh

CELL = 0.25
FULL = 1.6


def _navmesh():
    rows, cols = 16, 16  # 4 m x 4 m
    grid = np.zeros((rows, cols), dtype=bool)
    grid[:, :8] = True  # walkable for x < 2 m
    ground = np.where(grid, 0.0, np.nan)
    return Navmesh(grid, ground, np.zeros(2), CELL)


def _block(x0, x1, y0, y1, top, step=0.02):
    """Points filling the volume of an object from the floor to ``top``."""
    xs, ys, zs = np.arange(x0, x1, step), np.arange(y0, y1, step), np.arange(0.0, top + 1e-9, step)
    gx, gy, gz = np.meshgrid(xs, ys, zs)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _walls(mesh, splat=np.zeros((0, 3))):
    witnesses = [(mesh, 1), (splat, 12)]
    navmesh = _navmesh()
    cells = uncovered_cells(navmesh)
    assert cells.any()
    boxes = wall_boxes(navmesh, cells, wall_heights(navmesh, cells, witnesses))
    return navmesh, cells, boxes


def _top_at(boxes, x, y):
    tops = [b["centre"][2] + b["half_extents"][2] for b in boxes
            if abs(x - b["centre"][0]) < b["half_extents"][0] and abs(y - b["centre"][1]) < b["half_extents"][1]]
    assert len(tops) == 1, f"expected one wall box over ({x}, {y}), found {len(tops)}"
    return tops[0]


def test_unobserved_strip_keeps_a_full_wall():
    _, cells, boxes = _walls(np.zeros((0, 3)))
    assert {round(_top_at(boxes, 2.125, y + CELL / 2), 3) for y in np.arange(0, 4, CELL)} == {FULL}


def test_low_furniture_gets_its_own_height_and_the_wall_beside_it_stays_full():
    sofa = _block(2.0, 2.5, 0.0, 2.0, top=0.42)  # seat 0.42 m, along y < 2 m
    wall = _block(2.0, 2.5, 2.0, 4.0, top=2.6)  # a real wall to the ceiling, y > 2 m
    _, _, boxes = _walls(np.vstack([sofa, wall]))

    for y in (0.125, 0.875, 1.875):
        assert _top_at(boxes, 2.125, y) == pytest.approx(0.42, abs=0.06)
        assert _top_at(boxes, 2.125, y) >= 0.42  # rounded up, never shorter than seen
    for y in (2.125, 3.125, 3.875):
        assert _top_at(boxes, 2.125, y) == pytest.approx(FULL)


def test_a_pane_the_mesh_lost_is_closed_by_the_splat():
    """Mesh: only the bottom of the pane, 40 cm. Splat: the frosted pane up to 1.5 m."""
    rail = _block(2.0, 2.1, 0.0, 4.0, top=0.40)
    pane = _block(2.0, 2.1, 0.0, 4.0, top=1.50, step=0.04)
    _, _, mesh_only = _walls(rail)
    _, _, both = _walls(rail, pane)

    assert _top_at(mesh_only, 2.125, 1.0) < 0.5  # the hole the robot walked through
    for y in (0.125, 1.0, 3.875):
        assert _top_at(both, 2.125, y) >= 1.5


def test_a_few_floaters_do_not_size_a_wall_down():
    """Splat floaters over a cell the mesh did not see."""
    floaters = np.array([[2.1, 1.0, 0.3], [2.2, 1.05, 0.35], [2.15, 1.1, 0.5]])
    _, _, boxes = _walls(np.zeros((0, 3)), floaters)
    assert _top_at(boxes, 2.125, 1.0) == pytest.approx(FULL)


def test_a_sparse_mesh_wall_is_not_outvoted_by_a_lower_splat():
    """A few mesh samples on a wall face reaching 1.5 m, a dense splat blob at 0.8 m:
    the wall follows the mesh, not the splat."""
    face = np.array([[2.1, 0.9, z] for z in (0.4, 0.9, 1.5)])
    blob = _block(2.0, 2.25, 0.75, 1.0, top=0.8, step=0.05)
    _, _, boxes = _walls(face, blob)
    assert _top_at(boxes, 2.125, 0.875) >= 1.5


def test_floor_or_a_climbable_kerb_keeps_the_wall_full():
    """Past the wall lies filled, unobserved terrain: a wall the robot can step
    over (the navmesh climbs 0.25 m) would not contain it."""
    floor = _block(2.0, 4.0, 0.0, 2.0, top=0.0)
    kerb = _block(2.0, 4.0, 2.0, 4.0, top=0.20)
    _, _, boxes = _walls(np.vstack([floor, kerb]))
    assert _top_at(boxes, 2.125, 1.0) == pytest.approx(FULL)
    assert _top_at(boxes, 2.125, 3.0) == pytest.approx(FULL)


def test_every_wall_cell_gets_a_box_however_many_boxes_that_takes():
    """Alternating heights make one box per cell; none may be dropped."""
    size = 420
    grid = np.zeros((size, size), dtype=bool)
    grid[2:-2, 2:-2] = True
    navmesh = Navmesh(grid, np.where(grid, 0.0, np.nan), np.zeros(2), CELL)
    cells = uncovered_cells(navmesh)
    rows, cols = np.indices(cells.shape)
    heights = np.where((rows + cols) % 2 == 0, 0.4, FULL) * cells
    boxes = wall_boxes(navmesh, cells, heights)
    assert len(boxes) > 1500
    assert sum(b["cells"] for b in boxes) == int(cells.sum())
