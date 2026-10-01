"""Containment walls at the edge of mesh coverage.

The height field fills unobserved cells with nearest-neighbour ground so the
terrain has no fall-through holes — but that fill is invented terrain: a flat
continuation of ground the capture never saw. Route planning already stays on
the navmesh; a pushed or drifting robot does not, and nothing physical stops
it from strolling onto invention. The requirement is strict: the environment's
movement space is the mesh-covered area. These walls make that physical —
uncovered regions that border the walkable area get box colliders, so leaving
the observed world means meeting a wall, not walking onto guesswork.

Small uncovered pockets (below ``min_hole_m2``) are left open on purpose: a
crack in the scan inside otherwise-covered floor is honest nearest-neighbour
fill, and walling every two-cell speckle would fence the robot inside its own
spawn cell.

A wall cell is not always unobserved. The navmesh drops every cell outside the
walked region, so a sofa (its seat is ground 0.42 m above the floor, out of
step reach) and a glass partition read as uncovered exactly like a gap in the
scan. Each wall cell therefore stands as tall as the tallest thing the mesh or
the splat saw in it, and full height only where neither saw anything: on the
office scene 4b14390a the sofa went from a 1.6 m wall to 0.42-0.50 m, against
0.35-0.45 m of mesh under it. The
splat is the second witness because the mesh loses glass (OpenMVS leaves holes
in frosted panes where the splat still carries them as solid gaussians), and a
fence sized from the mesh alone would open those holes.
"""

from __future__ import annotations

import numpy as np


def uncovered_cells(navmesh, *, margin_m: float = 0.45, min_hole_m2: float = 1.0) -> np.ndarray:
    """Boolean grid of uncovered cells that need a wall.

    A cell needs a wall when (a) the mesh never observed ground there, (b) it
    belongs to an uncovered region large enough to be a genuine coverage gap
    rather than scan noise, and (c) it lies within ``margin_m`` of somewhere
    the robot can actually walk. The world outside the grid counts as
    uncovered — the grid is padded before labelling — so walkable ground that
    approaches the terrain's own edge gets fenced too.
    """
    from scipy import ndimage

    covered = np.isfinite(np.asarray(navmesh.ground_z))
    if covered.size == 0:
        return np.zeros_like(covered, dtype=bool)

    # Pad with an uncovered ring: the outside of the height field is the
    # biggest uncovered region of all, and it must join the labelling so the
    # scene's outer boundary is walled wherever the walkable area reaches it.
    pad_uncovered = np.pad(~covered, 1, constant_values=True)
    labels, count = ndimage.label(pad_uncovered)
    if count == 0:
        return np.zeros_like(covered, dtype=bool)

    min_cells = max(1, int(np.ceil(min_hole_m2 / navmesh.cell_size**2)))
    sizes = np.bincount(labels.ravel())
    large = sizes >= min_cells
    large[0] = False
    big_uncovered = large[labels][1:-1, 1:-1]

    margin_cells = max(1, int(round(margin_m / navmesh.cell_size)))
    near_walkable = ndimage.binary_dilation(navmesh.grid, iterations=margin_cells)
    return big_uncovered & near_walkable


def _filled_ground(navmesh) -> np.ndarray:
    """Ground per cell, nearest observed ground where none was observed: the
    same fill the height field uses, so a wall stands on the terrain."""
    from scipy import ndimage

    ground = np.asarray(navmesh.ground_z, dtype=np.float64)
    known = np.isfinite(ground)
    if not known.any():
        return np.zeros_like(ground)
    indices = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    return ground[tuple(indices)]


def wall_heights(navmesh, wall_cells: np.ndarray, witnesses, *, height_m: float = 1.6,
                 climbable_m: float = 0.25) -> np.ndarray:
    """Height of the wall over each wall cell (0 elsewhere).

    ``witnesses`` are ``(points, min_points)`` pairs, points in the simulation
    frame. A witness sees a cell when at least ``min_points`` of its points
    stand between ``climbable_m`` and ``height_m`` above the cell's ground; the
    wall then reaches the highest of them. The cell takes the tallest witness,
    and ``height_m`` when no witness sees it.

    ``climbable_m`` is the step the navmesh lets a robot climb
    (``build_navmesh``, ``max_step_m``). Anything lower would not contain the
    robot: past the wall the height field is the nearest-neighbour fill, not
    observed ground, so a kerb-height wall would let it step onto guesswork.
    Such cells keep the full wall.

    ``min_points`` is 1 for mesh surface samples (``surface_samples`` puts at
    least one on every triangle, so a sample is surface the mesh has) and 12
    for solid splat means, the splat obstacles' floater guard
    (``obstacle_cells``): a dozen is structure, a couple are noise, and noise
    must not size a wall down.
    """
    base = _filled_ground(navmesh)
    rows_n, cols_n = wall_cells.shape
    top = np.full(wall_cells.shape, -np.inf)
    for points, min_points in witnesses:
        points = np.asarray(points, dtype=np.float64)
        if len(points) == 0:
            continue
        cols = np.floor((points[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(np.int64)
        rows = np.floor((points[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(np.int64)
        inside = (rows >= 0) & (rows < rows_n) & (cols >= 0) & (cols < cols_n)
        rows, cols, z = rows[inside], cols[inside], points[inside, 2]
        relative = z - base[rows, cols]
        band = wall_cells[rows, cols] & (relative > climbable_m) & (relative < height_m)
        rows, cols, relative = rows[band], cols[band], relative[band]
        counts = np.zeros(wall_cells.shape, dtype=np.int64)
        np.add.at(counts, (rows, cols), 1)
        highest = np.full(wall_cells.shape, -np.inf)
        np.maximum.at(highest, (rows, cols), relative)
        top = np.maximum(top, np.where(counts >= min_points, highest, -np.inf))
    heights = np.where(np.isfinite(top), top, height_m)
    return np.where(wall_cells, heights, 0.0)


def wall_boxes(navmesh, wall_cells: np.ndarray, heights: np.ndarray, *,
               height_m: float = 1.6, height_step_m: float = 0.06):
    """Collision boxes tiling the wall cells exactly.

    Not the cluster-AABB merge used for splat obstacles: the axis-aligned box
    of a perimeter ring is the whole scene. Instead a greedy exact cover —
    horizontal runs grown downward while the full run stays present — so the
    boxes cover the wall cells and nothing else, all of them: a wall box left
    out is a hole in the fence. (A cap of 1500 boxes used to drop the rest,
    and 40 of the 50 sample scenes hit it.)

    ``heights`` (per cell, from ``wall_heights``) are rounded up to ``height_step_m`` and a box only joins cells of
    one rounded height, so a wall is never shorter than what was seen and at
    most 6 cm taller (never above ``height_m``): the step that
    ``filter_navmesh_by_step`` treats as flat ground under a stance.

    Wall cells have no observed ground by definition, so bases come from the
    nearest observed ground: the same fill the height field itself uses, which
    keeps each wall footed on the terrain it stands on.
    """
    filled = _filled_ground(navmesh)
    level = np.ceil(np.round(heights / height_step_m, 6)).astype(np.int64)
    rounded = np.minimum(level * height_step_m, height_m)

    remaining = wall_cells.copy()
    rows_n, cols_n = remaining.shape
    boxes = []
    for row in range(rows_n):
        col = 0
        while col < cols_n:
            if not remaining[row, col]:
                col += 1
                continue
            here = level[row, col]
            end = col
            while end + 1 < cols_n and remaining[row, end + 1] and level[row, end + 1] == here:
                end += 1
            bottom = row
            while (bottom + 1 < rows_n and remaining[bottom + 1, col : end + 1].all()
                   and (level[bottom + 1, col : end + 1] == here).all()):
                bottom += 1
            remaining[row : bottom + 1, col : end + 1] = False
            base = float(filled[row : bottom + 1, col : end + 1].min())
            top = float((filled[row : bottom + 1, col : end + 1]
                         + rounded[row, col]).max())
            centre = navmesh.origin + np.array(
                [(col + end + 1) / 2, (row + bottom + 1) / 2]
            ) * navmesh.cell_size
            boxes.append({
                "kind": "coverage_wall",
                "centre": [float(centre[0]), float(centre[1]), (base + top) / 2],
                "half_extents": [
                    (end - col + 1) / 2 * navmesh.cell_size,
                    (bottom - row + 1) / 2 * navmesh.cell_size,
                    (top - base) / 2,
                ],
                "cells": int((end - col + 1) * (bottom - row + 1)),
            })
            col = end + 1
    return boxes
