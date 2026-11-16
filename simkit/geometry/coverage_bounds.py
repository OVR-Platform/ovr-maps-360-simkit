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
"""

from __future__ import annotations

import numpy as np


def uncovered_cells(navmesh, *, margin_m: float = 0.45, min_hole_m2: float = 1.0) -> np.ndarray:
    """Boolean grid of uncovered cells that need a wall.

    A cell needs a wall when (a) the mesh never observed ground there, (b) it
    belongs to an uncovered region large enough to be a genuine coverage gap
    rather than scan noise, and (c) it lies within ``margin_m`` of ground the
    robot can stand on. The world outside the grid counts as uncovered — the
    grid is padded before labelling — so walkable ground that approaches the
    terrain's own edge gets fenced too.

    "Ground the robot can stand on" is every cell with ground, not only the
    navmesh grid: the stance-step filter and the splat obstacles take cells
    out of the grid but leave their ground, and a robot still steps onto
    them. Measured from the grid alone, the margin stopped short of the
    coverage gap behind such cells and the fence had holes: on the office
    scene 4b14390a, a glass partition with holes in the mesh was left open.
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
    near_walkable = ndimage.binary_dilation(covered, iterations=margin_cells)
    return big_uncovered & near_walkable


def wall_boxes(navmesh, wall_cells: np.ndarray, *, height_m: float = 1.6, max_boxes: int = 1500):
    """Collision boxes tiling the wall cells exactly.

    Not the cluster-AABB merge used for splat obstacles: the axis-aligned box
    of a perimeter ring is the whole scene. Instead a greedy exact cover —
    horizontal runs grown downward while the full run stays present — so the
    boxes cover the wall cells and nothing else.

    Wall cells have no observed ground by definition, so bases come from the
    nearest observed ground: the same fill the height field itself uses, which
    keeps each wall footed on the terrain it stands on. The top is
    ``height_m`` over the highest ground under the box, so on a slope no cell's
    wall is lower than ``height_m``.
    """
    from scipy import ndimage

    ground = np.asarray(navmesh.ground_z, dtype=np.float64)
    known = np.isfinite(ground)
    if known.any():
        indices = ndimage.distance_transform_edt(
            ~known, return_distances=False, return_indices=True
        )
        filled = ground[tuple(indices)]
    else:
        filled = np.zeros_like(ground)

    remaining = wall_cells.copy()
    rows_n, cols_n = remaining.shape
    boxes = []
    dropped = 0
    for row in range(rows_n):
        col = 0
        while col < cols_n:
            if not remaining[row, col]:
                col += 1
                continue
            end = col
            while end + 1 < cols_n and remaining[row, end + 1]:
                end += 1
            bottom = row
            while bottom + 1 < rows_n and remaining[bottom + 1, col : end + 1].all():
                bottom += 1
            remaining[row : bottom + 1, col : end + 1] = False
            if len(boxes) >= max_boxes:
                dropped += 1
                col = end + 1
                continue
            base = float(filled[row : bottom + 1, col : end + 1].min())
            top = float(filled[row : bottom + 1, col : end + 1].max()) + height_m
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
    if dropped:
        # A truncated wall is a hole in the fence — surface it, never hide it.
        import sys
        print(
            f"coverage_bounds: {dropped} wall boxes over max_boxes={max_boxes} dropped",
            file=sys.stderr,
        )
    return boxes
