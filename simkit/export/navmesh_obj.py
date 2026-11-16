"""The multi-level navmesh as a Wavefront OBJ, the walkable surface a NavMesh is baked from.

Unity has no navmesh file to import: its NavMesh is baked inside Unity from
geometry (NavMeshSurface in the editor, NavMeshBuilder at run time). This is
that geometry: the walkable surface of every storey, at its exact height, one
object per channel, no colours, no lift. Vertices are in the source frame of
the scene (metres, up = -Y), the same coordinates as ``mesh/model.glb``,
``gaussian_splatting.ply`` and ``training_cameras.json``, so whatever transform
a client applies to the scene mesh applies unchanged to this file.

Cells of a channel share their corners where the neighbouring cells are within
a step of each other, so a floor is one continuous surface rather than a field
of loose squares; a corner shared by cells at different heights takes their
mean. Faces wind so that their normal points up (-Y).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _channel_surface(grid: np.ndarray, ground: np.ndarray, origin: np.ndarray, cell: float,
                     max_step_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Welded quads of one channel in sim coordinates: (V, 3) vertices, (F, 3) triangles."""
    rows, cols = np.nonzero(grid)
    if len(rows) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    # Corner (r, c) of cell (r, c) is its lower-left; corners are keyed per cell
    # and merged with the neighbour's when the two grounds are within a step.
    parent: dict = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    offsets = ((0, 0), (0, 1), (1, 1), (1, 0))  # (dr, dc) of the four corners, CCW seen from +Z
    for r, c in zip(rows, cols):
        for dr, dc in offsets:
            parent[(r, c, dr, dc)] = (r, c, dr, dc)
    height, width = grid.shape
    for r, c in zip(rows, cols):
        z = ground[r, c]
        for nr, nc in ((r, c + 1), (r + 1, c), (r + 1, c + 1), (r + 1, c - 1)):
            if not (0 <= nr < height and 0 <= nc < width) or not grid[nr, nc]:
                continue
            if abs(ground[nr, nc] - z) > max_step_m:
                continue
            # corners the two cells share, by absolute grid position
            mine = {(r + dr, c + dc): (r, c, dr, dc) for dr, dc in offsets}
            theirs = {(nr + dr, nc + dc): (nr, nc, dr, dc) for dr, dc in offsets}
            for key in mine.keys() & theirs.keys():
                union(mine[key], theirs[key])

    groups: dict = {}
    for r, c in zip(rows, cols):
        for dr, dc in offsets:
            groups.setdefault(find((r, c, dr, dc)), []).append((r, c, dr, dc))
    index, vertices = {}, []
    for root, members in groups.items():
        r0, c0, dr0, dc0 = members[0]
        x = origin[0] + (c0 + dc0) * cell
        y = origin[1] + (r0 + dr0) * cell
        z = float(np.mean([ground[m[0], m[1]] for m in members]))
        index[root] = len(vertices)
        vertices.append((x, y, z))
    faces = []
    for r, c in zip(rows, cols):
        a, b, cc, d = (index[find((r, c, dr, dc))] for dr, dc in offsets)
        faces.append((a, b, cc))
        faces.append((a, cc, d))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def navmesh_levels_to_obj(bundle_dir: str | Path, out_path: str | Path, *, max_step_m: float = 0.25) -> dict:
    """Write ``out_path`` from a built ``<id8>.sre`` bundle; returns a summary."""
    bundle = Path(bundle_dir)
    grid = np.load(bundle / "collision" / "navmesh_levels_grid.npy")
    ground = np.load(bundle / "collision" / "navmesh_levels_ground_z.npy")
    nav = json.loads((bundle / "manifest.json").read_text())["stages"]["s5_navmesh_levels"]
    origin = np.asarray(nav["origin_xy"], dtype=np.float64)
    cell = float(nav["cell_size_m"])
    sim_from_source = np.asarray(json.loads((bundle / "frame" / "transform.json").read_text())["sim_from_source"])
    source_from_sim = np.linalg.inv(sim_from_source)
    up_source = source_from_sim[:3, :3] @ np.array([0.0, 0.0, 1.0])

    lines = [
        "# ovr-maps-360-simkit multi-level navmesh: walkable surface to bake a NavMesh from",
        "# frame: source scene frame, metres, up = -Y (same coordinates as mesh/model.glb and the splat)",
        f"# cell size {cell} m; one object per channel (channel k = k-th walkable surface of a cell, counted upwards)",
    ]
    offset, total_v, total_f, cells, normals = 1, 0, 0, 0, 0
    for k in range(grid.shape[0]):
        v_sim, f = _channel_surface(grid[k], ground[k], origin, cell, max_step_m)
        if len(f) == 0:
            continue
        v = v_sim @ source_from_sim[:3, :3].T + source_from_sim[:3, 3]
        lines.append(f"o navmesh_level_{k}")
        lines.extend(f"v {x:.4f} {y:.4f} {z:.4f}" for x, y, z in v)
        lines.append(f"vn {up_source[0]:.1f} {up_source[1]:.1f} {up_source[2]:.1f}")
        normals += 1
        vn = normals
        lines.extend(f"f {a + offset}//{vn} {b + offset}//{vn} {c + offset}//{vn}" for a, b, c in f)
        offset += len(v)
        total_v += len(v)
        total_f += len(f)
        cells += int(grid[k].sum())
    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n")
    return {"path": str(out_path), "bytes": out_path.stat().st_size, "vertices": total_v,
            "triangles": total_f, "cells": cells, "levels": int(grid.shape[0])}
