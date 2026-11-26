"""S4 — structure above the ground, as columns of boxes on the navmesh's cells.

The ground is an exact height field; what is left for collision is whatever
stands on it within a walking robot's reach: walls, shelving, clutter. It used
to be convex parts from CoACD, one decomposition per 3-25 m tile with up to
four hulls each, and that cannot work for a walking robot: the convex hull of a
tile's walls is the room between them. Measured on Amazon 57f7e18d, 24 of 25
tiles had a hull holding walkable space (5,000-16,000 of the points a standing
robot occupies over walkable cells, per hull); MuJoCo pushed four of forty
probes out through a flat, exact height field (leak 10%), and a robot standing
there is inside a solid. The rule that dropped hulls dipping into the ground
removed most of them by accident and kept the rest.

Here every cell that is not walkable and holds surface between ``margin`` and
``body_m`` above its ground gets a column from the ground to the top of that
surface (capped at ``body_m``); columns of equal height are merged into boxes
by an exact greedy cover, so a box covers its cells and nothing else. A
walkable cell never holds a column by construction: it was declared walkable
because nothing stands in it below 1.6 m. 1.5-3.5 s on scenes of 6,000-13,000 m2,
against one to several minutes of CoACD; 2,000-11,000 boxes, most of them
clutter under a metre (median top 0.5 m above the ground on a hall), so merging
by height buys little. The probe run keeps only the boxes near its probes
(``simkit.physics.probes``).
"""

from __future__ import annotations

import numpy as np


def structure_cells(navmesh, ground: np.ndarray, vertices: np.ndarray, faces: np.ndarray, *,
                    margin: float = 0.10, body_m: float = 2.0, min_samples: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Cells holding structure within ``body_m`` of the ground, and the top of it.

    ``ground`` is the filled ground per cell (the height field's surface), so a
    wall standing where the mesh saw no floor is footed on the terrain beside it.
    """
    from simkit.geometry.navmesh import surface_samples

    height, width = navmesh.grid.shape
    occupied = np.zeros((height, width), dtype=bool)
    top = np.full((height, width), -np.inf)
    if len(faces) == 0:
        return occupied, top
    points, _ = surface_samples(vertices, faces, navmesh.cell_size, per_cell=4.0)
    cols = np.floor((points[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(int)
    rows = np.floor((points[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    points, rows, cols = points[inside], rows[inside], cols[inside]
    relative = points[:, 2] - ground[rows, cols]
    body = (relative > margin) & (relative < body_m)
    count = np.zeros((height, width), dtype=np.int64)
    np.add.at(count, (rows[body], cols[body]), 1)
    np.maximum.at(top, (rows[body], cols[body]), points[body, 2])
    occupied = (count >= min_samples) & ~navmesh.grid
    return occupied, top


def structure_boxes(navmesh, ground: np.ndarray, vertices: np.ndarray, faces: np.ndarray, *,
                    margin: float = 0.10, body_m: float = 2.0, step_m: float = 0.25,
                    max_boxes: int = 20_000) -> tuple[list[dict], dict]:
    """Boxes covering the structure cells exactly, tops rounded up to ``step_m``."""
    occupied, top = structure_cells(navmesh, ground, vertices, faces, margin=margin, body_m=body_m)
    ground = np.asarray(ground, dtype=np.float64)
    rise = np.where(occupied, np.ceil(np.clip(top - ground, step_m, body_m) / step_m), 0).astype(np.int64)
    level = np.where(occupied, np.floor(ground / step_m).astype(np.int64) + rise, 0)

    remaining = occupied.copy()
    rows_n, cols_n = remaining.shape
    boxes, dropped = [], 0
    for row in range(rows_n):
        col = 0
        while col < cols_n:
            if not remaining[row, col]:
                col += 1
                continue
            key = level[row, col]
            end = col
            while end + 1 < cols_n and remaining[row, end + 1] and level[row, end + 1] == key:
                end += 1
            bottom = row
            while (bottom + 1 < rows_n and remaining[bottom + 1, col:end + 1].all()
                   and (level[bottom + 1, col:end + 1] == key).all()):
                bottom += 1
            remaining[row:bottom + 1, col:end + 1] = False
            if len(boxes) >= max_boxes:
                dropped += 1
                col = end + 1
                continue
            base = float(ground[row:bottom + 1, col:end + 1].min()) - margin
            upper = float((ground[row:bottom + 1, col:end + 1] + rise[row:bottom + 1, col:end + 1] * step_m).max())
            centre = navmesh.origin + np.array([(col + end + 1) / 2, (row + bottom + 1) / 2]) * navmesh.cell_size
            boxes.append({
                "kind": "structure",
                "centre": [float(centre[0]), float(centre[1]), (base + upper) / 2],
                "half_extents": [(end - col + 1) / 2 * navmesh.cell_size,
                                 (bottom - row + 1) / 2 * navmesh.cell_size, (upper - base) / 2],
                "cells": int((end - col + 1) * (bottom - row + 1)),
            })
            col = end + 1
    if dropped:
        import sys
        print(f"structure: {dropped} boxes over max_boxes={max_boxes} dropped", file=sys.stderr)
    return boxes, {"representation": "columns of boxes on the navmesh cells",
                   "structure_cells": int(occupied.sum()), "boxes": len(boxes), "dropped_boxes": dropped,
                   "body_m": body_m, "step_m": step_m}


def walkable_points_inside(boxes: list[dict], navmesh, heights=(0.15, 0.8, 1.45), *,
                           tolerance: float = 0.02) -> int:
    """How many of the points a standing robot occupies (over every walkable
    cell's centre, at shin, waist and head height) lie inside a box.

    Reads each box's cells off the grid instead of testing every point against
    every box: 11,000 boxes against 280,000 points took nine minutes.
    """
    grid, ground = navmesh.grid, navmesh.ground_z
    heights = np.asarray(heights, dtype=np.float64)
    inside = np.zeros(grid.shape + (len(heights),), dtype=bool)
    cell = navmesh.cell_size
    for box in boxes:
        (cx, cy, cz), (hx, hy, hz) = box["centre"], box["half_extents"]
        hx, hy, hz = hx - tolerance, hy - tolerance, hz - tolerance
        # cells whose centre lies strictly inside the box's footprint
        c0 = max(0, int(np.ceil((cx - hx - navmesh.origin[0]) / cell - 0.5)))
        c1 = min(grid.shape[1], int(np.floor((cx + hx - navmesh.origin[0]) / cell - 0.5)) + 1)
        r0 = max(0, int(np.ceil((cy - hy - navmesh.origin[1]) / cell - 0.5)))
        r1 = min(grid.shape[0], int(np.floor((cy + hy - navmesh.origin[1]) / cell - 0.5)) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        block = grid[r0:r1, c0:c1]
        if not block.any():
            continue
        z = ground[r0:r1, c0:c1, None] + heights
        inside[r0:r1, c0:c1] |= block[..., None] & (np.abs(z - cz) < hz)
    return int(inside.sum())


def walkable_space_points(navmesh, heights=(0.15, 0.8, 1.45)) -> np.ndarray:
    """Points a standing robot occupies: over every walkable cell's centre, at
    shin, waist and head height (within the 1.6 m it was judged walkable with)."""
    rows, columns = np.nonzero(navmesh.grid)
    if len(rows) == 0:
        return np.zeros((0, 3))
    x = navmesh.origin[0] + (columns + 0.5) * navmesh.cell_size
    y = navmesh.origin[1] + (rows + 0.5) * navmesh.cell_size
    z = navmesh.ground_z[rows, columns]
    return np.concatenate([np.column_stack([x, y, z + h]) for h in heights])
