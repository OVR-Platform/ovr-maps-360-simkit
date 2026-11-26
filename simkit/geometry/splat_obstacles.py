"""Obstacles the splat can see and the mesh has lost.

A car reconstructs badly from a walk-through — reflective, sometimes moving —
and often arrives as a smear in the splat and almost nothing in the mesh. The
robot then ghosts through a visually present object: measured, 324 solid
gaussians inside the robot's body volume at the worst step of one run, with no
collision geometry anywhere near.

The splat is already this pipeline's witness for the frame; here it witnesses
occupancy. Solid gaussians standing in the stance-to-torso band above a cell's
ground mean *something* is there, whatever the mesh thinks. Those cells are cut
from the navmesh, and their clusters get box colliders so that even a robot that
strays is stopped by physics rather than passing through the scenery.
"""

from __future__ import annotations

import numpy as np


def obstacle_cells(
    navmesh,
    splat_points: np.ndarray,
    *,
    band_low_m: float = 0.25,
    band_high_m: float = 1.8,
    min_gaussians: int = 12,
) -> np.ndarray:
    """Boolean grid of walkable cells with solid splat mass at body height.

    ``min_gaussians`` guards against stray floaters: a dozen solid gaussians in
    a 0.6 m cell is structure, one or two are reconstruction noise.
    """
    grid = np.zeros_like(navmesh.grid, dtype=bool)
    if len(splat_points) == 0:
        return grid

    cols = ((splat_points[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(int)
    rows = ((splat_points[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(int)
    inside = (
        (rows >= 0) & (rows < navmesh.grid.shape[0])
        & (cols >= 0) & (cols < navmesh.grid.shape[1])
    )
    rows, cols, heights = rows[inside], cols[inside], splat_points[inside, 2]

    ground = navmesh.ground_z[rows, cols]
    relative = heights - ground
    in_band = np.isfinite(ground) & (relative > band_low_m) & (relative < band_high_m)

    counts = np.zeros_like(navmesh.grid, dtype=np.int32)
    np.add.at(counts, (rows[in_band], cols[in_band]), 1)
    grid = counts >= min_gaussians
    return grid


def obstacle_boxes(navmesh, blocked: np.ndarray, *, height_m: float = 1.6, max_boxes: int = 1500):
    """Collision boxes over the splat-witnessed obstacle cells.

    Full band height over every blocked cell, because their job is to stop a
    walking robot, not to model a car. They cover the blocked cells exactly
    (``coverage_bounds.wall_boxes``): they used to be the bounding box of each
    cluster, and the bounding box of an L-shaped cluster covers walkable cells.
    """
    from simkit.geometry.coverage_bounds import wall_boxes

    boxes = wall_boxes(navmesh, blocked, height_m=height_m, max_boxes=max_boxes)
    for box in boxes:
        box["kind"] = "splat_obstacle"
    return boxes
