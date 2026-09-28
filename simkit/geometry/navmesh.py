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


def cell_size_for(mesh, *, foot_m: float = 0.25, smallest: float = 0.15, largest: float = 0.60) -> float:
    """The cell size of the walkable grid: a foot, whatever the mesh's resolution.

    Walkability used to be decided from the mesh's *vertices*, so the grid could
    only be as fine as the reconstruction was dense, and this function derived a
    size from the vertex density. That broke on meshes whose floor is a few large
    coplanar panels (the MoGe floor fill merges them): a 30 m slab carries four
    vertices, so almost every cell on it held none and the floor read as void
    although a ray cast from every camera hit it. The grid now samples the
    surface of every triangle (``surface_samples``), which makes the density a
    parameter rather than a property of the mesh, and the cell size is simply
    the size of the thing that has to fit in it. Clamped to the same range as
    before so an explicit ``--cell-size`` keeps meaning what it meant.
    """
    return float(np.clip(foot_m, smallest, largest))


def surface_samples(vertices: np.ndarray, faces: np.ndarray, cell_size: float, *,
                    per_cell: float = 10.0, cap: int = 12_000_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Points spread uniformly over the surface, ``per_cell`` of them per cell of
    area, each carrying the normal of its triangle.

    Every triangle gets at least one sample, so a thin post still blocks the
    cell it stands in; a large panel gets as many as its area asks for, so a
    floor made of four triangles fills every cell it covers. Area-weighted and
    barycentric-uniform; ten per cell on average, so a cell of flat ground misses
    the two-sample ground threshold with probability 5e-4, not 9% as at four.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    live = twice_area > 1e-12
    normals = np.zeros_like(cross)
    normals[live] = cross[live] / twice_area[live, None]
    count = np.maximum(1, np.ceil(per_cell * (twice_area / 2.0) / cell_size**2)).astype(np.int64)
    count[~live] = 0
    total = int(count.sum())
    if total > cap:  # keep the run bounded on a pathological mesh; still >= 1 per triangle
        count = np.maximum(live.astype(np.int64), np.floor(count * (cap / total)).astype(np.int64))
    which = np.repeat(np.arange(len(faces)), count)
    rng = np.random.default_rng(seed)
    r1, r2 = rng.random(len(which)), rng.random(len(which))
    root = np.sqrt(r1)
    bary = np.column_stack([1.0 - root, root * (1.0 - r2), root * r2])
    points = np.einsum("nk,nkd->nd", bary, corners[which])
    return points, normals[which]


def build_navmesh(
    mesh,
    *,
    cell_size: float = 0.15,
    flat_normal: float = 0.85,  # applied to n_z with its sign, not to |n_z|
    clearance_m: float = 1.6,
    max_step_m: float = 0.25,
    min_floor_hits: int = 2,
    seed_points: np.ndarray | None = None,
) -> Navmesh:
    """Rasterise walkable ground from a surface mesh.

    The grid reads the *surface* of the mesh, not its vertices: every triangle is
    sampled in proportion to its area (``surface_samples``), so a floor made of
    a handful of large panels fills its cells exactly like a floor made of a
    million small ones. A mesh without triangles (a point set with normals) is
    read as it is.

    ``clearance_m`` rejects ground a humanoid cannot occupy — under tables,
    inside shelving, beneath a low soffit. ``max_step_m`` rejects ground that is
    only reachable by a jump: a kerb is climbable, a 2 m wall is not.
    """
    faces = np.asarray(getattr(mesh, "triangles", np.zeros((0, 3), dtype=np.int64)))
    if len(faces):
        vertices, normals = surface_samples(np.asarray(mesh.vertices), faces, cell_size)
    else:
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

    # 1. Ground height per cell: the lowest *upward-facing* surface sample in it.
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

    # 3. Keep only what a robot can actually reach on foot: from where the
    # operator walked when we know it (``seed_points``, world XY), else the
    # largest region.
    return largest_connected_region(navmesh, max_step_m=max_step_m, seed_points=seed_points)


def largest_connected_region(navmesh: Navmesh, *, max_step_m: float = 0.25,
                             seed_points: np.ndarray | None = None) -> Navmesh:
    """Keep the region(s) a robot can actually traverse.

    Connectivity is **height-aware**: adjacent cells count as connected only if
    the step between their ground heights is climbable. Plan-view connectivity
    alone would merge a rooftop or balcony with the courtyard below it merely
    because they touch when seen from above — and then report the union as
    walkable area a policy could never use.

    With ``seed_points`` (world XY, the ground under the cameras) every region
    that the walk touches is kept: that is where the operator went, so that is
    where a robot is expected to go. Without seeds the largest region is kept,
    which on a scene cut into pieces by kerbs and steps can be a courtyard the
    operator never entered while the walked corridor is dropped (measured: 895
    of 2199 m2 kept, 12% of the cameras on it).
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
    wanted: set[int] = set()
    if seed_points is not None and len(seed_points):
        seeds = np.asarray(seed_points, dtype=np.float64)[:, :2]
        cols = np.floor((seeds[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(int)
        rows = np.floor((seeds[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(int)
        inside = (rows >= 0) & (rows < grid.shape[0]) & (cols >= 0) & (cols < grid.shape[1])
        wanted = set(int(v) for v in labels[rows[inside], cols[inside]] if v)
    if not wanted:
        wanted = {int(np.argmax(sizes)) + 1}
    kept = np.isin(labels, sorted(wanted))
    return Navmesh(kept, np.where(kept, ground, np.nan), navmesh.origin, navmesh.cell_size)
