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
(the rest of the sofa, its backrest) gets a wall by the same rule
(``coverage_walls``). A wall stands from the ground, or from the lowest
structure in its cell when the Go2 fits under it (a table top).

Walls within the Go2's reach of walkable ground are boxes; the rest are one
height field (``boxed_cells``, ``wall_field``), because MuJoCo pays for every
geom on every step.
"""

from __future__ import annotations

import numpy as np

# How far past walkable ground structure gets collision: the reach
# ``above_ground_mesh`` gives the convex structure (3 m).
REACH_M = 3.0
# The reference robot, Unitree Go2 (unitree.com/go2, all models): standing
# 0.70 x 0.31 x 0.40 m, climbs steps of about 16 cm; foot spheres of 0.022 m
# (unitree_ros, go2_description.urdf).
GO2_HEIGHT_M = 0.40
GO2_HALF_LENGTH_M = 0.35
GO2_FOOT_RADIUS_M = 0.022
# How far from walkable ground the fence reaches.
FENCE_MARGIN_M = 0.45


def _large_uncovered(navmesh, min_hole_m2: float) -> np.ndarray:
    """Uncovered cells in regions of at least ``min_hole_m2``. The world
    outside the grid counts as uncovered: the grid is padded before labelling,
    so the outside joins the biggest region of all."""
    from scipy import ndimage

    covered = np.isfinite(np.asarray(navmesh.ground_z))
    if covered.size == 0:
        return np.zeros_like(covered, dtype=bool)
    labels, count = ndimage.label(np.pad(~covered, 1, constant_values=True))
    if count == 0:
        return np.zeros_like(covered, dtype=bool)
    min_cells = max(1, int(np.ceil(min_hole_m2 / navmesh.cell_size**2)))
    large = np.bincount(labels.ravel()) >= min_cells
    large[0] = False
    return large[labels][1:-1, 1:-1]


def uncovered_cells(navmesh, *, margin_m: float = FENCE_MARGIN_M, min_hole_m2: float = 1.0) -> np.ndarray:
    """Boolean grid of uncovered cells that need a wall.

    A cell needs a wall when (a) the mesh never observed ground there, (b) it
    belongs to an uncovered region large enough to be a genuine coverage gap
    rather than scan noise (``min_hole_m2``), and (c) it lies within
    ``margin_m`` of somewhere the robot can actually walk. Walkable ground
    that approaches the terrain's own edge gets fenced too.
    """
    from scipy import ndimage

    margin_cells = max(1, int(round(margin_m / navmesh.cell_size)))
    near_walkable = ndimage.binary_dilation(navmesh.grid, iterations=margin_cells)
    return _large_uncovered(navmesh, min_hole_m2) & near_walkable


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


def _wall_base(navmesh, margin_m: float = FENCE_MARGIN_M) -> np.ndarray:
    """The ground a wall cell is measured from: the highest walkable ground
    within ``margin_m``, where a robot comes from, else the filled ground.

    Measured from its own filled ground alone, a cell at the top of a
    stairwell whose nearest walkable ground is the floor below got the steps
    as its wall, with its top under the floor above: on 4b14390a a Go2 walked
    off the upper floor over it.
    """
    from scipy import ndimage

    ground = np.asarray(navmesh.ground_z, dtype=np.float64)
    size = 2 * max(1, int(round(margin_m / navmesh.cell_size))) + 1
    highest = ndimage.maximum_filter(np.where(np.isfinite(ground), ground, -np.inf), size=size, mode="constant",
                                     cval=-np.inf)
    return np.where(np.isfinite(highest), highest, _filled_ground(navmesh))


def _cell_counts(navmesh, wall_cells, base, points, low, high):
    """Per wall cell: how many of ``points`` stand between ``low`` and ``high``
    above the cell's ground, the highest and the lowest of them."""
    rows_n, cols_n = wall_cells.shape
    counts = np.zeros(wall_cells.shape, dtype=np.int64)
    highest = np.full(wall_cells.shape, -np.inf)
    lowest = np.full(wall_cells.shape, np.inf)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = np.floor((points[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(np.int64)
    rows = np.floor((points[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(np.int64)
    inside = (rows >= 0) & (rows < rows_n) & (cols >= 0) & (cols < cols_n)
    rows, cols, z = rows[inside], cols[inside], points[inside, 2]
    relative = z - base[rows, cols]
    chosen = wall_cells[rows, cols] & (relative > low) & (relative < high)
    np.add.at(counts, (rows[chosen], cols[chosen]), 1)
    np.maximum.at(highest, (rows[chosen], cols[chosen]), relative[chosen])
    np.minimum.at(lowest, (rows[chosen], cols[chosen]), relative[chosen])
    return counts, highest, lowest


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
                 splat_min_points: int = 12, wall_above_m: float = 0.4, floor_m: float = 0.12,
                 flat_normal: float = 0.85, per_cell: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Top and bottom, above each cell's ground, of the structure the mesh
    sees over ``cells``; NaN where it sees none (and outside ``cells``).

    The mesh sees a cell when its samples (``mesh_witness``) stand between
    ``climbable_m`` and ``height_m`` above the cell's ground; the top is the
    highest of them. ``climbable_m`` is the navmesh's step limit
    (``build_navmesh``, ``max_step_m``): lower than that is floor to a robot,
    not an obstacle. It is above what the Go2 climbs (16 cm), so anything the
    mesh sees over it stops the Go2.

    The splat only raises a cell the mesh sees (a pane the mesh lost), never
    makes one: a cell the mesh has no surface in is a hole in the mesh,
    whatever the splat holds. It counts from ``splat_min_points`` solid
    gaussians: over walkable cells, where the navmesh's clearance test says
    the space is empty, the most found in one 0.25 m cell between 0.25 and
    1.6 m was 11 (scenes 4b14390a, e5cc9f26, c8c675bc, 9d2fe9b1).

    A seen cell is ``height_m`` tall when the mesh also has non-horizontal
    surface (``|n_z| < flat_normal``, the navmesh's ground test) in the
    ``wall_above_m`` just above ``height_m``: a wall whose middle the mesh
    lost (a pane, a blank wall) still has its top, and a skirting under it
    must not size it down. The slab stops at 2.0 m, under the lowest ceiling
    over walkable ground in the office (2.31 m, 0.1st percentile), so vaults
    and ceilings do not count. It needs vertical surface of half a cell's
    width over the slab: ``per_cell`` samples per cell of area
    (``surface_samples``) times that area.

    The bottom is the lowest surface the mesh has in the cell that is not
    floor (up-facing, within ``floor_m``, the navmesh's obstacle threshold, of
    the ground). When it is at least ``GO2_HEIGHT_M`` up and the mesh saw the
    floor under it, the Go2 fits under: a table top, a sign, a branch. Below
    that, or over unobserved ground, the bottom is the ground: a gap the
    reference robot cannot use buys nothing and splits boxes.
    """
    base = _wall_base(navmesh)
    _, top, _ = _cell_counts(navmesh, cells, base, mesh_points, climbable_m, height_m)
    count, splat_top, _ = _cell_counts(navmesh, cells, base, splat_points, climbable_m, height_m)
    seen = np.isfinite(top)
    top = np.maximum(top, np.where(count >= splat_min_points, splat_top, -np.inf))

    mesh_points = np.asarray(mesh_points).reshape(-1, 3)
    nz = np.asarray(mesh_normals).reshape(-1, 3)[:, 2]
    vertical, _, _ = _cell_counts(navmesh, cells, base, mesh_points[np.abs(nz) < flat_normal],
                                  height_m, height_m + wall_above_m)
    min_vertical = per_cell * (navmesh.cell_size / 2 * wall_above_m) / navmesh.cell_size**2
    top[vertical >= min_vertical] = height_m

    up = nz > flat_normal
    floor, _, _ = _cell_counts(navmesh, cells, base, mesh_points[up], -floor_m, floor_m)
    lowest = np.minimum(_cell_counts(navmesh, cells, base, mesh_points[~up], -floor_m, height_m)[2],
                        _cell_counts(navmesh, cells, base, mesh_points[up], floor_m, height_m)[2])
    bottom = np.where((lowest >= GO2_HEIGHT_M) & (floor > 0), lowest, 0.0)
    # Only cells the mesh sees: the splat and the wall above only raise them.
    return np.where(seen, top, np.nan), np.where(seen, bottom, np.nan)


def coverage_walls(navmesh, mesh_points: np.ndarray, mesh_normals: np.ndarray, splat_points: np.ndarray,
                   *, reach_m: float = REACH_M, height_m: float = 1.6,
                   min_hole_m2: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The cells that get a box, and the top and bottom of each above its ground.

    Two kinds, one rule for their extent (``wall_heights``):

    - the fence (``uncovered_cells``): uncovered cells next to the walkable
      area. Where the mesh sees nothing it stands from the ground to
      ``height_m``, because past it the terrain is guesswork;
    - furniture and walls further in: every other cell of a large uncovered
      region within ``reach_m`` of the walkable area where the mesh sees
      structure. Without these the inside of a sofa (seat, backrest) had no
      collision at all: the height field fills it with the floor beside it,
      and the convex parts of anything standing on the floor are dropped
      (``build_collision_proxy``, ``_ground_clearance``). Pockets under
      ``min_hole_m2`` inside the walkable area stay open, as for the fence.

    The seat stays out of the height field on purpose. MuJoCo interpolates a
    height field between neighbouring samples, so a 0.42 m seat sample beside
    a floor sample is a 0.42 m ramp over one cell, half of it on walkable
    ground; a box gives the sofa its vertical front and puts the seat at the
    top of the box.
    """
    from scipy import ndimage

    fence = uncovered_cells(navmesh, min_hole_m2=min_hole_m2)
    near = ndimage.binary_dilation(navmesh.grid, iterations=max(1, int(round(reach_m / navmesh.cell_size))))
    candidates = fence | (near & _large_uncovered(navmesh, min_hole_m2))
    top, bottom = wall_heights(navmesh, candidates, mesh_points, mesh_normals, splat_points, height_m=height_m)
    seen = np.isfinite(top)
    cells = fence | seen
    return (cells, np.where(seen, top, np.where(cells, height_m, np.nan)),
            np.where(seen, bottom, np.where(cells, 0.0, np.nan)))


def boxed_cells(navmesh, wall_cells: np.ndarray, bottoms: np.ndarray, *,
                reach_m: float = GO2_HALF_LENGTH_M) -> np.ndarray:
    """The wall cells that are boxes; the rest go into ``wall_field``.

    Boxes are what a robot meets: contact with a box is a handful of contact
    points, contact with a fine height field one per sample under the body
    (measured with a walking Go2 against a field-only bundle: 61-102 contacts
    against 3-11, and those steps 3-5x slower). So the cells within ``reach_m`` of a walkable cell
    centre are boxes, all eight neighbours of the walkable area at the Go2's
    half length, and so is every cell with a raised bottom, which a height
    field cannot represent. What lies further in is reached only by a robot
    already over those boxes.
    """
    from scipy import ndimage

    rings = max(1, int(np.ceil((reach_m - navmesh.cell_size / 2) / navmesh.cell_size)))
    near = ndimage.binary_dilation(navmesh.grid, structure=np.ones((3, 3), bool), iterations=rings)
    return wall_cells & (near | (np.where(wall_cells, bottoms, 0.0) > 0))


def wall_field(navmesh, wall_cells: np.ndarray, tops: np.ndarray, *, foot_m: float = GO2_FOOT_RADIUS_M):
    """The walls past the robot's reach (``boxed_cells``), as one height field.

    One box per merged run of cells came to 16,121 boxes on 9d2fe9b1, and
    MuJoCo pays for every geom on every step and on every ray. A wall on the
    ground is one height per cell, which is what a height field is: one geom
    whatever the count. MuJoCo interpolates between samples, so each cell is
    repeated over ``k`` x ``k`` samples and the side of a wall is a ramp one
    sample wide, centred on the cell's edge: it reaches at most half a sample,
    ``foot_m``, into the cell next to it. Outside the walls the field
    stays ``foot_m`` under the lowest ground, below anything the ground height
    field (whose lowest sample is that ground) can interpolate to.

    Samples sit at the centres of the ``k`` x ``k`` sub-cells, so the field
    lines up with the navmesh grid exactly.
    """
    from simkit.export.heightfield import HeightField

    base = _wall_base(navmesh)
    k = int(np.ceil(navmesh.cell_size / (2 * foot_m)))
    low = float(_filled_ground(navmesh).min()) - foot_m
    top = np.where(wall_cells, base + np.where(wall_cells, tops, 0.0), low)
    fine = np.kron(top, np.ones((k, k)))
    span = max(float(fine.max()) - low, 1e-3)
    spacing = navmesh.cell_size / k
    rows, cols = fine.shape
    return HeightField(
        elevation=((fine - low) / span).astype(np.float32),
        radius_x=(cols - 1) * spacing / 2,
        radius_y=(rows - 1) * spacing / 2,
        elevation_z=span,
        base_z=1.0,  # the ground field's own solid below its lowest sample
        centre=navmesh.origin + np.array([navmesh.grid.shape[1], navmesh.grid.shape[0]]) * navmesh.cell_size / 2,
        z_offset=low,
    )


def wall_boxes(navmesh, wall_cells: np.ndarray, tops: np.ndarray, bottoms: np.ndarray, *,
               step_m: float = 0.06):
    """Collision boxes tiling the wall cells exactly.

    Not the cluster-AABB merge used for splat obstacles: the axis-aligned box
    of a perimeter ring is the whole scene. Instead a greedy exact cover —
    horizontal runs grown downward while the full run stays present — so the
    boxes cover the wall cells and nothing else, all of them: a wall box left
    out is a hole in the fence. (A cap of 1500 boxes used to drop the rest,
    and 40 of the 50 sample scenes hit it.)

    ``tops`` and ``bottoms`` (per cell, from ``coverage_walls``) are measured
    from each cell's ground; a bottom of 0 is the ground. A box joins cells
    whose top, rounded up to ``step_m``, and bottom, rounded down, are the
    same, so no cell's box comes out smaller than measured, or more than
    ``step_m`` larger, on a slope too. A box on the ground stands from the
    lowest ground under it. ``step_m`` trades box count against over-height;
    6 cm is the step a stance already treats as flat
    (``filter_navmesh_by_step``).

    Wall cells have no observed ground by definition, so bases come from the
    nearest observed ground: the same fill the height field itself uses, which
    keeps each wall footed on the terrain it stands on.
    """
    filled, base = _filled_ground(navmesh), _wall_base(navmesh)
    tops, bottoms = np.where(wall_cells, tops, 0.0), np.where(wall_cells, bottoms, 0.0)
    level = np.ceil(np.round((base + tops) / step_m, 6)).astype(np.int64)
    raised = bottoms > 0
    floor_level = np.floor(np.round((base + bottoms) / step_m, 6)).astype(np.int64)
    # One label per (top, bottom) pair; boxes on the ground share one bottom,
    # whatever the slope under them.
    pairs = np.stack([level, np.where(raised, floor_level, level.min() - 1)], axis=-1).reshape(-1, 2)
    key = np.unique(pairs, axis=0, return_inverse=True)[1].reshape(level.shape)

    remaining = wall_cells.copy()
    rows_n, cols_n = remaining.shape
    boxes = []
    for row in range(rows_n):
        col = 0
        while col < cols_n:
            if not remaining[row, col]:
                col += 1
                continue
            here = key[row, col]
            end = col
            while end + 1 < cols_n and remaining[row, end + 1] and key[row, end + 1] == here:
                end += 1
            bottom = row
            while (bottom + 1 < rows_n and remaining[bottom + 1, col : end + 1].all()
                   and (key[bottom + 1, col : end + 1] == here).all()):
                bottom += 1
            remaining[row : bottom + 1, col : end + 1] = False
            top = float(level[row, col] * step_m)
            base = (float(floor_level[row, col] * step_m) if raised[row, col]
                    else float(filled[row : bottom + 1, col : end + 1].min()))
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
