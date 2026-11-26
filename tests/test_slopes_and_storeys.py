"""Navmesh and collision rules that must hold on hills and across storeys."""

import numpy as np

from simkit.export.heightfield import step_span
from simkit.gate import floor_registration
from simkit.geometry.structure import structure_boxes, walkable_points_inside
from simkit.geometry.navmesh import Navmesh, NavmeshLevels

CELL = 0.25
X = np.arange(40) * CELL


def test_a_slope_is_not_a_step():
    for degrees in (4, 8, 15):
        street = np.tile(np.tan(np.radians(degrees)) * X, (40, 1))
        assert step_span(street).max() < 1e-9
        diagonal = np.add.outer(X, X) * np.tan(np.radians(degrees))
        assert step_span(diagonal).max() < 1e-9


def test_a_kerb_is_a_step_on_level_ground_and_on_a_hill():
    kerb = np.tile(np.where(X > 4.9, 0.08, 0.0), (40, 1))
    assert abs(step_span(kerb).max() - 0.08) < 1e-9
    hill = kerb + np.tile(np.tan(np.radians(10)) * X, (40, 1))
    assert abs(step_span(hill).max() - 0.08) < 1e-9


def test_level_ground_reads_as_before():
    from scipy import ndimage

    rng = np.random.default_rng(0)
    floor = rng.normal(0, 0.01, (40, 40))
    old = ndimage.maximum_filter(floor, 3) - ndimage.minimum_filter(floor, 3)
    assert np.median(step_span(floor)) > 0.8 * np.median(old)


def test_the_floor_is_compared_with_the_storey_the_walk_was_on():
    grid = np.ones((2, 4, 4), dtype=bool)
    ground = np.stack([np.full((4, 4), -3.0), np.full((4, 4), 0.01)])
    levels = NavmeshLevels(grid, ground, np.zeros(2), 1.0)
    cameras = np.array([[0.5, 0.5, 0.0], [2.5, 1.5, 0.0], [3.5, 3.5, 0.0]])
    assert abs(floor_registration(levels, cameras)["median_abs_m"] - 0.01) < 1e-9
    lowest = Navmesh(grid[0], ground[0], np.zeros(2), 1.0)
    assert floor_registration(lowest, cameras)["median_abs_m"] == 3.0


def test_structure_boxes_stand_on_walls_and_never_over_walkable_cells():
    grid = np.zeros((20, 20), dtype=bool)
    grid[5:15, 5:15] = True
    ground = np.zeros((20, 20))
    navmesh = Navmesh(grid, np.where(grid, 0.0, np.nan), np.zeros(2), CELL)
    # a room: four walls 2.5 m high just outside the walkable square, and a ceiling over it
    walls = []
    for a, b in (((1.1, 1.1), (1.1, 3.9)), ((3.9, 1.1), (3.9, 3.9)), ((1.1, 1.1), (3.9, 1.1)), ((1.1, 3.9), (3.9, 3.9))):
        walls.append([[a[0], a[1], 0], [b[0], b[1], 0], [b[0], b[1], 2.5], [a[0], a[1], 2.5]])
    walls.append([[1.1, 1.1, 2.5], [3.9, 1.1, 2.5], [3.9, 3.9, 2.5], [1.1, 3.9, 2.5]])
    vertices = np.array(walls, dtype=float).reshape(-1, 3)
    faces = np.concatenate([[[4 * k, 4 * k + 1, 4 * k + 2], [4 * k, 4 * k + 2, 4 * k + 3]] for k in range(5)])
    boxes, stats = structure_boxes(navmesh, ground, vertices, faces)
    assert stats["structure_cells"] > 40 and boxes
    assert walkable_points_inside(boxes, navmesh) == 0
    intruder = {"centre": [2.5, 2.5, 0.5], "half_extents": [0.3, 0.3, 0.5]}  # 2x2 cells, shin and waist
    assert walkable_points_inside([intruder], navmesh) == 8
    assert max(b["centre"][2] + b["half_extents"][2] for b in boxes) <= 2.0 + 1e-9  # the ceiling is not structure
