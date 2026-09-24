"""S5 — walkable space, for indoor *and* outdoor scenes.

Produces a 2D grid of where a robot can stand, together with the ground height of
each walkable cell.

The naive definition — "near-horizontal surface at z close to 0" — only works on
a flat indoor floor. Outdoor captures in this corpus routinely span 2-4 m of
elevation within a single scene (streets, courtyards, beaches, waterfronts), so a
flat floor band would discard most of the genuinely walkable ground. Instead each
cell gets its own ground height, and walkability is decided against that:

  1. the cell must contain near-horizontal surface (its ground),
  2. there must be head clearance above *that* height, not above z = 0,
  3. the step up to a neighbouring cell must be climbable,

which degenerates to the flat-floor case indoors and stays correct on a slope.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Navmesh:
    grid: np.ndarray  # (H, W) bool, True where walkable
    ground_z: np.ndarray  # (H, W) float, ground height per cell (NaN where unknown)
    origin: np.ndarray  # (2,) world XY of cell (0, 0)
    cell_size: float

    @property
    def area_m2(self) -> float:
        return float(self.grid.sum()) * self.cell_size**2

    @property
    def elevation_span_m(self) -> float:
        """Height range of the walkable surface — 0 indoors, metres outdoors."""
        heights = self.ground_z[self.grid]
        heights = heights[np.isfinite(heights)]
        return float(heights.max() - heights.min()) if heights.size else 0.0

    def cell_centres(self) -> np.ndarray:
        """World XY of every walkable cell, (N, 2)."""
        rows, cols = np.nonzero(self.grid)
        return self.origin + np.stack([cols, rows], axis=1) * self.cell_size + self.cell_size / 2

    def sample_points(self, count: int, *, seed: int = 0) -> np.ndarray:
        """Walkable (x, y, ground_z) samples, (N, 3) — where a foot can be put."""
        rows, cols = np.nonzero(self.grid)
        if len(rows) == 0:
            return np.zeros((0, 3))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(rows), size=min(count, len(rows)), replace=False)
        xy = self.origin + np.stack([cols[chosen], rows[chosen]], axis=1) * self.cell_size + self.cell_size / 2
        return np.column_stack([xy, self.ground_z[rows[chosen], cols[chosen]]])

    def as_dict(self) -> dict:
        return {
            "cell_size_m": self.cell_size,
            "walkable_cells": int(self.grid.sum()),
            "walkable_area_m2": self.area_m2,
            "elevation_span_m": self.elevation_span_m,
            "grid_shape": list(self.grid.shape),
            "origin_xy": self.origin.tolist(),
        }


def cell_size_for(mesh, *, min_floor_hits: int = 2, margin: float = 3.0,
                  smallest: float = 0.15, largest: float = 0.60) -> float:
    """A cell size the mesh is actually dense enough to fill.

    A cell needs ``min_floor_hits`` vertices in it before it counts as ground,
    so the grid can only be as fine as the reconstruction is dense. These two
    corpora differ by more than two orders of magnitude — a LiDAR-fused indoor
    capture carries ~2,400 vertices per m², while a 360 mapping decimated to a
    fixed 500k triangles over open ground carries ~9. At 0.2 m the sparse one
    expects 0.4 vertices per cell, so almost every cell fails the threshold and
    the navmesh comes out empty: measured, 0.3 m² of walkable ground where 0.5 m
    cells find 1,950.

    Returning a size instead of a constant makes the grid follow the data. It is
    clamped: below ``smallest`` there is nothing to gain over the mesh's own
    resolution, and above ``largest`` a cell stops describing anything a foot
    could be placed on.
    """
    vertices = np.asarray(mesh.vertices)
    if len(vertices) < 100:
        return largest

    span = vertices[:, :2].max(axis=0) - vertices[:, :2].min(axis=0)
    area = float(max(span[0] * span[1], 1e-6))
    density = len(vertices) / area
    needed = np.sqrt(margin * min_floor_hits / max(density, 1e-9))
    return float(np.clip(needed, smallest, largest))


def build_navmesh(
    mesh,
    *,
    cell_size: float = 0.15,
    flat_normal: float = 0.85,  # applied to n_z with its sign, not to |n_z|
    clearance_m: float = 1.6,
    max_step_m: float = 0.25,
    min_floor_hits: int = 2,
) -> Navmesh:
    """Rasterise walkable ground from a surface mesh.

    ``clearance_m`` rejects ground a humanoid cannot occupy — under tables,
    inside shelving, beneath a low soffit. ``max_step_m`` rejects ground that is
    only reachable by a jump: a kerb is climbable, a 2 m wall is not.
    """
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    if len(vertices) == 0:
        return Navmesh(np.zeros((0, 0), dtype=bool), np.zeros((0, 0)), np.zeros(2), cell_size)

    lower = vertices[:, :2].min(axis=0)
    upper = vertices[:, :2].max(axis=0)
    width = max(1, int(np.ceil((upper[0] - lower[0]) / cell_size)))
    height = max(1, int(np.ceil((upper[1] - lower[1]) / cell_size)))

    columns = np.clip(((vertices[:, 0] - lower[0]) / cell_size).astype(int), 0, width - 1)
    rows = np.clip(((vertices[:, 1] - lower[1]) / cell_size).astype(int), 0, height - 1)
    flat_index = rows * width + columns
    cells = width * height

    # 1. Ground height per cell: the lowest *upward-facing* surface in it.
    #
    # The normal's sign matters, and |n_z| is a real bug: it accepts a ceiling, a
    # mezzanine underside or the bottom of a pallet as ground. On a factory scene
    # the ceiling carries 77k downward normals against 606 at floor level.
    horizontal = normals[:, 2] > flat_normal
    ground = np.full(cells, np.nan)
    hits = np.bincount(flat_index[horizontal], minlength=cells)
    if horizontal.any():
        order = np.lexsort((vertices[horizontal, 2], flat_index[horizontal]))
        sorted_index = flat_index[horizontal][order]
        sorted_z = vertices[horizontal, 2][order]
        first = np.ones(len(sorted_index), dtype=bool)
        first[1:] = sorted_index[1:] != sorted_index[:-1]
        ground[sorted_index[first]] = sorted_z[first]

    # 2. Head clearance measured from each cell's own ground, not from z = 0.
    cell_ground = ground[flat_index]
    above = vertices[:, 2] - cell_ground
    blocking = np.isfinite(above) & (above > 0.12) & (above < clearance_m)
    obstacles = np.bincount(flat_index[blocking], minlength=cells)

    walkable = (hits >= min_floor_hits) & (obstacles == 0) & np.isfinite(ground)
    grid = walkable.reshape(height, width)
    ground_grid = ground.reshape(height, width)
    navmesh = Navmesh(grid, ground_grid, lower, cell_size)

    # 3. Keep only what a robot can actually reach on foot.
    return largest_connected_region(navmesh, max_step_m=max_step_m)


def largest_connected_region(navmesh: Navmesh, *, max_step_m: float = 0.25) -> Navmesh:
    """Keep the biggest region a robot can actually traverse.

    Connectivity is **height-aware**: adjacent cells count as connected only if
    the step between their ground heights is climbable. Plan-view connectivity
    alone would merge a rooftop or balcony with the courtyard below it merely
    because they touch when seen from above — and then report the union as
    walkable area a policy could never use.
    """
    grid = navmesh.grid
    ground = navmesh.ground_z
    if not grid.any():
        return navmesh

    labels = np.zeros(grid.shape, dtype=np.int32)
    current = 0
    sizes: list[int] = []
    for start in zip(*np.nonzero(grid)):
        if labels[start]:
            continue
        current += 1
        size = 0
        stack = [start]
        labels[start] = current
        while stack:
            row, col = stack.pop()
            size += 1
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                r, c = row + dr, col + dc
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    continue
                if labels[r, c] or not grid[r, c]:
                    continue
                if abs(ground[r, c] - ground[row, col]) > max_step_m:
                    continue
                labels[r, c] = current
                stack.append((r, c))
        sizes.append(size)

    if not sizes:
        return navmesh
    keep = int(np.argmax(sizes)) + 1
    kept = labels == keep
    return Navmesh(kept, np.where(kept, ground, np.nan), navmesh.origin, navmesh.cell_size)
