"""S4 — collision proxy.

Physics engines need convex geometry to compute contacts efficiently. The fused
surface is a dense concave mesh, so it is decimated and then split into convex
parts with CoACD (MIT). Explicitly not VHACD-via-NVIDIA tooling (licensing).

Import order note: importing ``coacd`` before ``pxr`` segfaults the interpreter.
Everything here imports coacd lazily, and USD export runs in a separate stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time
from pathlib import Path

import numpy as np


@dataclass
class CollisionStats:
    source_triangles: int
    decimated_triangles: int
    parts: int
    total_hull_triangles: int
    tiles: int = 0
    hulls_per_tile: int = 0
    dropped_into_ground: int = 0
    truncated: bool = False

    def as_dict(self) -> dict:
        return {
            "source_triangles": self.source_triangles,
            "decimated_triangles": self.decimated_triangles,
            "convex_parts": self.parts,
            "hull_triangles": self.total_hull_triangles,
            "tiles": self.tiles,
            "hulls_per_tile": self.hulls_per_tile,
            "dropped_into_ground": self.dropped_into_ground,
            "budget_reduced": self.truncated,
        }


def decimate(mesh, *, target_triangles: int = 120_000):
    """Reduce triangle count while keeping the surface where it is.

    The collision proxy only has to be accurate to roughly a foot's contact
    patch; carrying two million triangles into the solver buys nothing and costs
    every simulation step.
    """
    if len(mesh.triangles) <= target_triangles:
        return mesh
    reduced = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    reduced.remove_degenerate_triangles()
    reduced.remove_duplicated_vertices()
    reduced.remove_unreferenced_vertices()
    reduced.compute_vertex_normals()
    return reduced


def convex_decompose(
    mesh,
    *,
    threshold: float = 0.30,
    max_hulls: int = 64,
    resolution: int = 4_000,  # CoACD's own default is 2000; 1e6 runs for hours
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a mesh into convex parts. Returns a list of (vertices, faces)."""
    import coacd

    # One line of progress per tile times a hundred tiles buries everything else.
    if hasattr(coacd, "set_log_level"):
        coacd.set_log_level("error")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    if len(faces) == 0:
        return []

    result = coacd.run_coacd(
        coacd.Mesh(vertices, faces),
        threshold=threshold,
        max_convex_hull=max_hulls,
        resolution=resolution,
        seed=seed,
    )
    return [(np.asarray(v, dtype=np.float64), np.asarray(f, dtype=np.int32)) for v, f in result]


def tile_mesh(mesh, *, tile_size: float = 2.0) -> list:
    """Split a mesh into a grid of XY tiles.

    Decomposing a whole building into a handful of convex hulls is both slow and
    a poor fit: a room interior is deeply non-convex, and any global hull budget
    is spent badly. Tiling makes each sub-problem small and locally convex-ish,
    bounds the runtime, and is what keeps this tractable across 481 scenes.
    """
    import open3d as o3d

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return []

    centroids = vertices[triangles].mean(axis=1)
    lower = centroids[:, :2].min(axis=0)
    cell = np.floor((centroids[:, :2] - lower) / tile_size).astype(int)
    keys = cell[:, 0] * 100_000 + cell[:, 1]

    tiles = []
    for key in np.unique(keys):
        selected = triangles[keys == key]
        used, remapped = np.unique(selected, return_inverse=True)
        piece = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices[used]),
            o3d.utility.Vector3iVector(remapped.reshape(-1, 3)),
        )
        tiles.append(piece)
    return tiles


def voxel_box_proxy(mesh, output_dir, *, voxel_m: float = 0.25, max_boxes: int = 600):
    """A collision proxy of axis-aligned boxes, from a voxel occupancy grid.

    Convex decomposition answers a question this proxy does not need to ask.
    CoACD exists to represent a *concave* object faithfully enough to
    manipulate, and it pays for that with an iterative search whose cost is
    unpredictable: measured across the corpus, 20 of 23 scenes exceeded a
    twenty-five minute budget in it, with no relation to scene size — 1,685 m2
    finished in 5.5 minutes while 580 m2 ran past 22.

    Here the ground is already an exact height field, and the structure only has
    to stop a robot walking through a wall or a hedge. Occupancy answers that,
    and boxes merged out of it are exactly convex, so nothing downstream changes
    — each box is written as a small OBJ and travels the same path a hull would.

    The trade is fidelity for a pipeline that runs. It is measurable on the same
    terms as everything else: probe leak and rest penetration.
    """
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return [], 0

    # Sample each triangle at its corners and centroid: enough for surfaces
    # larger than a voxel, which after decimation they are.
    points = np.concatenate([vertices, vertices[triangles].mean(axis=1)])
    lower = points.min(axis=0)
    index = np.floor((points - lower) / voxel_m).astype(np.int64)
    shape = index.max(axis=0) + 1
    if np.prod(shape.astype(float)) > 4e8:  # a grid too large to hold
        return [], 0

    occupied = np.zeros(shape, dtype=bool)
    occupied[index[:, 0], index[:, 1], index[:, 2]] = True

    boxes = _greedy_boxes(occupied, max_boxes=max_boxes)
    paths = []
    for number, (start, size) in enumerate(boxes):
        centre = lower + (start + size / 2) * voxel_m
        extent = size * voxel_m
        box = o3d.geometry.TriangleMesh.create_box(*extent)
        box.translate(centre - extent / 2)
        path = output_dir / f"part_{number:03d}.obj"
        o3d.io.write_triangle_mesh(str(path), box)
        paths.append(path)
    return paths, int(occupied.sum())


def _greedy_boxes(occupied: np.ndarray, *, max_boxes: int):
    """Merge an occupancy grid into few axis-aligned boxes.

    Greedy meshing: take the first free cell, grow it along x while the run
    stays occupied, then along y, then along z, claim the block and repeat.
    Ordinary greedy meshing, which trades optimality for being O(cells).
    """
    remaining = occupied.copy()
    boxes = []
    while remaining.any() and len(boxes) < max_boxes:
        start = np.array(np.unravel_index(int(np.argmax(remaining)), remaining.shape))
        size = np.array([1, 1, 1])

        while start[0] + size[0] < remaining.shape[0] and remaining[
            start[0] + size[0], start[1], start[2]
        ]:
            size[0] += 1
        while start[1] + size[1] < remaining.shape[1] and remaining[
            start[0] : start[0] + size[0], start[1] + size[1], start[2]
        ].all():
            size[1] += 1
        while start[2] + size[2] < remaining.shape[2] and remaining[
            start[0] : start[0] + size[0], start[1] : start[1] + size[1], start[2] + size[2]
        ].all():
            size[2] += 1

        remaining[
            start[0] : start[0] + size[0],
            start[1] : start[1] + size[1],
            start[2] : start[2] + size[2],
        ] = False
        boxes.append((start.astype(float), size.astype(float)))
    return boxes


# Confining each worker to one thread was tried and is much worse. CoACD
# parallelises internally and does it well: one tile takes 3.8 s with the
# machine's 24 threads and about 138 s with one, so 22 single-threaded workers
# lose 36x to win 22x. The pool keeps its default threading and returns a
# marginal 1.3x — CoACD already saturates the machine, and there is no easy
# parallel win here. The lever that works is fewer tiles.


def _decompose_payload(payload):
    """Convex-decompose one tile, in whatever process picks it up.

    Takes plain arrays rather than an Open3D mesh: the pool has to pickle its
    arguments, and a TriangleMesh does not travel.
    """
    import open3d as o3d

    vertices, triangles, threshold, max_hulls = payload
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    return convex_decompose(mesh, threshold=threshold, max_hulls=max_hulls)


def build_collision_proxy(
    mesh,
    output_dir: str | Path,
    *,
    target_triangles: int = 120_000,
    threshold: float = 0.30,
    max_hulls: int = 4,
    tile_size: float | None = None,
    max_tiles: int = 60,
    max_parts: int = 400,
    workers: int = 0,  # 0 = all cores but two
    navmesh=None,
    base_margin: float = 0.10,
) -> tuple[list[Path], CollisionStats]:
    """Decimate, tile, convex-decompose each tile, write one OBJ per part.

    ``threshold`` is the concavity CoACD is allowed to leave unresolved, and it
    is the only parameter here that matters for cost. Measured on the heaviest
    tile of a real scene: 0.12 takes 27.75 s and returns 8 hulls, 0.30 takes
    3.45 s and returns 2. Resolution and the hull cap change neither.

    0.30 is the shipped value because of what this proxy is for. The ground is
    an exact height field and these hulls only have to stop a robot walking
    through a wall or a hedge; a pillar coming out slightly fuller than it is
    costs nothing there. It would cost something for manipulation, and that is
    the moment to lower it again — for one object, not for a whole scene.

    The defaults are tuned for *structure* rather than ground: the height field
    already carries the walkable surface, so what is left is walls and clutter,
    which are close to planar within a tile. Finer settings produced 1859 hulls
    on a single outdoor scene — slow to compute, heavy in the solver, and no more
    accurate where it matters.
    """
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_triangles = len(mesh.triangles)
    reduced = decimate(mesh, target_triangles=target_triangles)

    # Tile size follows the scene, because the cost is one convex decomposition
    # per tile and a fixed 3 m grid makes that cost scale with area. Measured:
    # scenes with 100 m2 of walkable ground finished in two minutes while ones
    # with a few thousand ran past twenty, having produced over a thousand tiles.
    # The grid exists to keep hulls local, not to be any particular size, so it
    # is sized to land near a tile budget instead.
    # 60 tiles, not 250. A tile costs a fixed ~3.8 s whatever it contains — 296
    # triangles or 99, the same — so tile count *is* the cost: 125 tiles took
    # 13.4 minutes on one scene, 43 take 4.0.
    #
    # The expected price was fidelity, since larger tiles cover more ground with
    # the same hull budget. It did not appear: leak stayed at 0%, rest
    # penetration at 0.71 cm p95, settle at 0.85 — identical to three decimal
    # places. Whatever those extra hulls were describing, no measurement of ours
    # could see it.
    if tile_size is None:
        span = np.asarray(reduced.vertices)[:, :2]
        footprint = float(np.ptp(span[:, 0]) * np.ptp(span[:, 1])) if len(span) else 0.0
        tile_size = float(np.clip(np.sqrt(max(footprint, 1.0) / max_tiles), 3.0, 25.0))

    tiles = [tile for tile in tile_mesh(reduced, tile_size=tile_size) if len(tile.triangles) >= 8]

    # Spend the budget across all tiles rather than truncating partway through.
    # Stopping early leaves whole regions with no collision geometry at all,
    # which is a hole in the wall; giving every tile fewer hulls degrades
    # gracefully instead.
    budget = max(1, min(max_hulls, max_parts // max(1, len(tiles))))

    # One CoACD call per tile, run across processes. Measured, the call costs
    # about 3.8 s whatever it is given — 296 triangles or 99, the same — so the
    # cost is a fixed overhead per call and the only lever is how many calls
    # happen at once. They are independent, and the machine has cores idle.
    payloads = [
        (np.asarray(tile.vertices, dtype=np.float64),
         np.asarray(tile.triangles, dtype=np.int32),
         threshold, budget)
        for tile in tiles
    ]
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    if len(payloads) > 4 and workers != 1:
        import multiprocessing as mp

        count = workers if workers and workers > 0 else max(1, (os.cpu_count() or 2) - 2)
        with mp.get_context("spawn").Pool(min(count, len(payloads))) as pool:
            for result in pool.imap_unordered(_decompose_payload, payloads, chunksize=1):
                parts.extend(result)
    else:
        for payload in payloads:
            parts.extend(_decompose_payload(payload))

    truncated = budget < max_hulls
    if truncated:
        print(
            f"note: {len(tiles)} tiles share a {max_parts}-part budget, "
            f"so each gets {budget} hull(s) instead of {max_hulls}",
            flush=True,
        )

    # A convex hull of geometry straddling the floor-wall junction bulges out
    # over the walkable surface, and a foot then rests on the bulge instead of
    # the ground. The height field already owns the ground, so any hull dipping
    # into it contributes nothing but contact noise.
    #
    # Measured on the mall scene: 55 of 397 hulls sat below the margin and took
    # the p95 foot rest error from 0.48 cm to 4.41 cm (max 16.19 cm). Dropping
    # them restores 0.48 cm while keeping 86% of the structure.
    dropped_into_ground = 0
    if navmesh is not None:
        kept = []
        for vertices, faces in parts:
            if _ground_clearance(vertices, navmesh) >= base_margin:
                kept.append((vertices, faces))
            else:
                dropped_into_ground += 1
        parts = kept

    paths: list[Path] = []
    hull_triangles = 0
    for index, (vertices, faces) in enumerate(parts):
        piece = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(faces),
        )
        piece.compute_vertex_normals()
        path = output_dir / f"part_{index:03d}.obj"
        o3d.io.write_triangle_mesh(str(path), piece)
        paths.append(path)
        hull_triangles += len(faces)

    stats = CollisionStats(
        truncated=truncated,
        dropped_into_ground=dropped_into_ground,
        hulls_per_tile=budget,
        tiles=len(tiles),
        source_triangles=source_triangles,
        decimated_triangles=len(reduced.triangles),
        parts=len(parts),
        total_hull_triangles=hull_triangles,
    )
    return paths, stats


def _ground_clearance(vertices: np.ndarray, navmesh) -> float:
    """Height of a hull's lowest point above the ground beneath it."""
    ground = navmesh.ground_z
    columns = np.clip(
        ((vertices[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(int),
        0,
        ground.shape[1] - 1,
    )
    rows = np.clip(
        ((vertices[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(int),
        0,
        ground.shape[0] - 1,
    )
    beneath = ground[rows, columns]
    if not np.isfinite(beneath).any():
        return float("inf")  # no ground claimed here, so nothing to bulge over
    return float(np.nanmin(vertices[:, 2] - beneath))
