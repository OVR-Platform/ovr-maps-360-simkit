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
scan. Each wall cell therefore stands as tall as what the mesh saw in it, and
full height where the mesh saw nothing (``wall_heights``): on the office scene
4b14390a the sofa went from a 1.6 m wall to 0.31-0.49 m, against 0.30-0.46 m
of mesh under it. The splat raises walls where the mesh lost glass (OpenMVS
leaves holes in frosted panes the splat still carries as solid gaussians) and
never lowers one. Past the fence, the structure the mesh sees within reach
(the rest of the sofa, its backrest) gets boxes by the same rule
(``coverage_walls``).
"""

from __future__ import annotations

import numpy as np

# How far past walkable ground structure gets collision: the reach
# ``above_ground_mesh`` gives the convex structure (3 m).
REACH_M = 3.0


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


def _cell_counts(navmesh, wall_cells, base, points, low, high):
    """Per wall cell: how many of ``points`` stand between ``low`` and ``high``
    above the cell's ground, and the highest of them."""
    rows_n, cols_n = wall_cells.shape
    counts = np.zeros(wall_cells.shape, dtype=np.int64)
    highest = np.full(wall_cells.shape, -np.inf)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.floor((points[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(np.int64)
    rows = np.floor((points[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(np.int64)
    inside = (rows >= 0) & (rows < rows_n) & (cols >= 0) & (cols < cols_n)
    rows, cols, z = rows[inside], cols[inside], points[inside, 2]
    relative = z - base[rows, cols]
    chosen = wall_cells[rows, cols] & (relative > low) & (relative < high)
    np.add.at(counts, (rows[chosen], cols[chosen]), 1)
    np.maximum.at(highest, (rows[chosen], cols[chosen]), relative[chosen])
    return counts, highest


def mesh_witness(mesh, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Surface samples and normals of the mesh's structure, for ``wall_heights``.

    Connected components under ``MIN_COMPONENT_TRIANGLES`` (the LODs' debris
    threshold) are left out: a stray OpenMVS triangle at seat height must not
    size a wall down. What remains is sampled as the navmesh samples it, so a
    single sample in a cell is real surface there.
    """
    from simkit.geometry.navmesh import surface_samples
    from simkit.lod import MIN_COMPONENT_TRIANGLES

    clusters, sizes, _ = mesh.cluster_connected_triangles()
    keep = np.asarray(sizes)[np.asarray(clusters)] >= MIN_COMPONENT_TRIANGLES
    return surface_samples(np.asarray(mesh.vertices), np.asarray(mesh.triangles)[keep], cell_size)


def wall_heights(navmesh, cells: np.ndarray, mesh_points: np.ndarray, mesh_normals: np.ndarray,
                 splat_points: np.ndarray, *, height_m: float = 1.6, climbable_m: float = 0.25,
                 min_points: int = 12, flat_normal: float = 0.85) -> np.ndarray:
    """Height of the structure the mesh sees over each of ``cells``, NaN where
    it sees none (and outside ``cells``).

    Two witnesses, the mesh structure (``mesh_witness``) and the solid splat
    means. A witness sees a cell when its points stand between ``climbable_m``
    and ``height_m`` above the cell's ground (one mesh sample, or
    ``min_points`` gaussians), and its height there is the highest of them.
    Where the mesh sees the cell it takes the taller witness. The splat only
    ever raises a wall (a pane the mesh lost), it never makes one: a cell the
    mesh has no surface in is a hole in the mesh, whatever the splat holds.

    A cell is ``height_m`` tall also when the mesh has ``min_points`` samples
    of non-horizontal surface above ``height_m`` in it (``|n_z| <
    flat_normal``, the navmesh's own test for ground): a wall whose middle the
    mesh lost (a pane, a blank wall) still has its top, and a skirting under a
    missing pane must not size the wall down to the skirting. Horizontal
    surface up there is a ceiling and says nothing about the cell.

    ``climbable_m`` is the step the navmesh lets a robot climb
    (``build_navmesh``, ``max_step_m``): lower than that is floor to a robot,
    not an obstacle, and would not contain it. ``min_points`` is the splat
    obstacles' floater guard (``obstacle_cells``): a dozen points is
    structure, a couple are noise (a tilted facet in a ceiling, a floater).
    """
    base = _filled_ground(navmesh)
    _, mesh_top = _cell_counts(navmesh, cells, base, mesh_points, climbable_m, height_m)
    counts, splat_top = _cell_counts(navmesh, cells, base, splat_points, climbable_m, height_m)
    splat_top = np.where(counts >= min_points, splat_top, -np.inf)
    top = np.where(np.isfinite(mesh_top), np.maximum(mesh_top, splat_top), -np.inf)
    vertical = np.abs(np.asarray(mesh_normals).reshape(-1, 3)[:, 2]) < flat_normal
    above, _ = _cell_counts(navmesh, cells, base, np.asarray(mesh_points).reshape(-1, 3)[vertical],
                            height_m, np.inf)
    top[above >= min_points] = height_m
    return np.where(cells & np.isfinite(top), top, np.nan)


def coverage_walls(navmesh, mesh_points: np.ndarray, mesh_normals: np.ndarray, splat_points: np.ndarray,
                   *, reach_m: float = REACH_M, height_m: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """The cells that get a box, and the height of each.

    Two kinds, one rule for their height (``wall_heights``):

    - the fence (``uncovered_cells``): uncovered cells next to the walkable
      area. Where the mesh sees nothing it stands ``height_m`` tall, because
      past it the terrain is guesswork;
    - furniture and walls further in: every other uncovered cell within
      ``reach_m`` of the walkable area where the mesh sees structure. Without
      these the inside of a sofa (seat, backrest) had no collision at all: the
      height field fills it with the floor beside it, and the convex parts of
      anything standing on the floor are dropped (``build_collision_proxy``,
      ``_ground_clearance``).

    The seat stays out of the height field on purpose. MuJoCo interpolates a
    height field between neighbouring samples, so a 0.42 m seat sample beside
    a floor sample is a 0.42 m ramp over one cell, half of it on walkable
    ground; a box gives the sofa its vertical front and puts the seat at the
    top of the box.
    """
    from scipy import ndimage

    fence = uncovered_cells(navmesh)
    near = ndimage.binary_dilation(navmesh.grid, iterations=max(1, int(round(reach_m / navmesh.cell_size))))
    candidates = fence | (near & ~np.isfinite(np.asarray(navmesh.ground_z)))
    seen = wall_heights(navmesh, candidates, mesh_points, mesh_normals, splat_points, height_m=height_m)
    cells = fence | np.isfinite(seen)
    return cells, np.where(np.isfinite(seen), seen, np.where(cells, height_m, 0.0))


def wall_boxes(navmesh, wall_cells: np.ndarray, heights: np.ndarray, *, step_m: float = 0.06):
    """Collision boxes tiling the wall cells exactly.

    Not the cluster-AABB merge used for splat obstacles: the axis-aligned box
    of a perimeter ring is the whole scene. Instead a greedy exact cover —
    horizontal runs grown downward while the full run stays present — so the
    boxes cover the wall cells and nothing else, all of them: a wall box left
    out is a hole in the fence. (A cap of 1500 boxes used to drop the rest,
    and 40 of the 50 sample scenes hit it.)

    ``heights`` (per cell, from ``wall_heights``) are measured from each
    cell's ground. A box joins cells whose wall top, rounded up to ``step_m``,
    is the same, and stands from the lowest ground under it to that top: no
    cell's wall comes out shorter than measured, or more than ``step_m``
    taller, on a slope too. ``step_m`` trades box count against over-height;
    6 cm is the step a stance already treats as flat
    (``filter_navmesh_by_step``).

    Wall cells have no observed ground by definition, so bases come from the
    nearest observed ground: the same fill the height field itself uses, which
    keeps each wall footed on the terrain it stands on.
    """
    filled = _filled_ground(navmesh)
    level = np.ceil(np.round((filled + heights) / step_m, 6)).astype(np.int64)

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
            top = float(here * step_m)
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
