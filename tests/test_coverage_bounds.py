"""Coverage walls: the fence stands along every coverage gap next to ground a robot can stand on."""

import numpy as np

from simkit.geometry.coverage_bounds import uncovered_cells, wall_boxes
from simkit.geometry.navmesh import Navmesh

CELL = 0.25


def _scene():
    """4 m x 4 m: walkable for x < 1.5 m; ground the step filter took out of
    the grid for 1.5-2.0 m (it keeps its ground); never observed past 2.0 m."""
    grid = np.zeros((16, 16), dtype=bool)
    grid[:, :6] = True
    ground = np.full(grid.shape, np.nan)
    ground[:, :8] = 0.0
    return Navmesh(grid, ground, np.zeros(2), CELL)


def test_the_gap_behind_ground_taken_out_of_the_grid_is_fenced():
    cells = uncovered_cells(_scene())
    assert cells[:, 8].all()  # x 2.0-2.25 m: the first unobserved column
    boxes = wall_boxes(_scene(), cells)
    covered = np.zeros(cells.shape, dtype=bool)
    for b in boxes:
        c0 = int(round((b["centre"][0] - b["half_extents"][0]) / CELL))
        c1 = int(round((b["centre"][0] + b["half_extents"][0]) / CELL))
        r0 = int(round((b["centre"][1] - b["half_extents"][1]) / CELL))
        r1 = int(round((b["centre"][1] + b["half_extents"][1]) / CELL))
        covered[r0:r1, c0:c1] = True
        assert b["half_extents"][2] * 2 == 1.6
    assert covered[:, 8].all()


def test_ground_with_ground_beside_it_gets_no_wall():
    cells = uncovered_cells(_scene())
    assert not cells[:, :8].any()  # walkable or with ground: never a wall


def test_a_small_pocket_inside_the_ground_stays_open():
    navmesh = _scene()
    navmesh.ground_z[6:8, 2:4] = np.nan  # 0.5 m x 0.5 m hole in the walkable floor
    navmesh.grid[6:8, 2:4] = False
    assert not uncovered_cells(navmesh)[6:8, 2:4].any()


def test_on_a_slope_no_wall_is_lower_than_its_height_over_its_own_ground():
    """Ground rising 10% along y: 40 cm over the strip the boxes merge along."""
    navmesh = _scene()
    rows = np.arange(16)[:, None] * np.ones((1, 16))
    navmesh.ground_z[:, :8] = 0.1 * (rows[:, :8] + 0.5) * CELL
    cells = uncovered_cells(navmesh)
    for b in wall_boxes(navmesh, cells):
        r0 = int(round((b["centre"][1] - b["half_extents"][1]) / CELL))
        r1 = int(round((b["centre"][1] + b["half_extents"][1]) / CELL))
        top = b["centre"][2] + b["half_extents"][2]
        assert top >= navmesh.ground_z[r0:r1, 7].max() + 1.6 - 1e-9  # the ground each wall cell is filled from
