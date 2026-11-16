"""The multi-level navmesh as a glTF binary, in the frame of the source scene.

A viewer that already overlays ``mesh/model.glb`` on the splat can overlay this
file the same way: vertices are in the source frame (metres, up = -Y, the frame
of ``gaussian_splatting.ply`` and ``training_cameras.json``) and the root node
carries no transform. One quad per walkable cell and storey, lifted a little
above its ground so it does not z-fight with the floor of the mesh. Inner cells
are green, cells on the edge of the walkable region yellow.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

INNER_RGBA = (38, 255, 89, 255)
EDGE_RGBA = (255, 242, 51, 255)


def _edge_cells(grid: np.ndarray, ground: np.ndarray, max_step_m: float) -> np.ndarray:
    """(L, H, W) bool: walkable cells with a 4-neighbour holding no surface within a step."""
    levels, height, width = grid.shape
    edge = np.zeros_like(grid)
    for k in range(levels):
        rows, cols = np.nonzero(grid[k])
        for r, c in zip(rows, cols):
            z = ground[k, r, c]
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < height and 0 <= cc < width):
                    edge[k, r, c] = True
                    break
                near = ground[:, rr, cc][grid[:, rr, cc]]
                if not np.any(np.abs(near - z) <= max_step_m):
                    edge[k, r, c] = True
                    break
    return edge


def navmesh_levels_to_glb(bundle_dir: str | Path, out_path: str | Path, *,
                          lift_m: float = 0.02, opacity: float = 0.55, max_step_m: float = 0.25) -> dict:
    """Write ``out_path`` from a built ``<id8>.sre`` bundle; returns a summary."""
    bundle = Path(bundle_dir)
    collision = bundle / "collision"
    grid = np.load(collision / "navmesh_levels_grid.npy")
    ground = np.load(collision / "navmesh_levels_ground_z.npy")
    manifest = json.loads((bundle / "manifest.json").read_text())
    nav = manifest["stages"]["s5_navmesh_levels"]
    origin = np.asarray(nav["origin_xy"], dtype=np.float64)
    cell = float(nav["cell_size_m"])
    sim_from_source = np.asarray(json.loads((bundle / "frame" / "transform.json").read_text())["sim_from_source"])
    source_from_sim = np.linalg.inv(sim_from_source)

    k, rows, cols = np.nonzero(grid)
    z = ground[k, rows, cols] + lift_m
    x0 = origin[0] + cols * cell
    y0 = origin[1] + rows * cell
    # counter-clockwise seen from +Z (sim up), so the faces look up in either frame
    corners = np.stack([
        np.column_stack([x0, y0, z]),
        np.column_stack([x0 + cell, y0, z]),
        np.column_stack([x0 + cell, y0 + cell, z]),
        np.column_stack([x0, y0 + cell, z]),
    ], axis=1).reshape(-1, 3)
    positions = (corners @ source_from_sim[:3, :3].T + source_from_sim[:3, 3]).astype(np.float32)
    edge = _edge_cells(grid, ground, max_step_m)[k, rows, cols]
    colours = np.repeat(np.where(edge[:, None], EDGE_RGBA, INNER_RGBA), 4, axis=0).astype(np.uint8)
    base = np.arange(len(k), dtype=np.uint32) * 4
    indices = np.column_stack([base, base + 1, base + 2, base, base + 2, base + 3]).ravel().astype(np.uint32)

    def pad(blob: bytes, fill: bytes = b"\x00") -> bytes:
        return blob + fill * (-len(blob) % 4)

    pos_bytes, col_bytes, idx_bytes = positions.tobytes(), colours.tobytes(), indices.tobytes()
    views, offset = [], 0
    for blob, target in ((pos_bytes, 34962), (col_bytes, 34962), (idx_bytes, 34963)):
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": target})
        offset += len(pad(blob))
    binary = pad(pos_bytes) + pad(col_bytes) + pad(idx_bytes)

    gltf = {
        "asset": {"version": "2.0", "generator": "ovr-maps-360-simkit",
                  "extras": {"content": "multi-level navmesh, one quad per walkable cell and storey",
                             "frame": "source scene frame: metres, up = -Y (as gaussian_splatting.ply)",
                             "cell_size_m": cell, "lift_m": lift_m, "levels": int(grid.shape[0]),
                             "walkable_cells": int(len(k)), "edge_rgba": EDGE_RGBA, "inner_rgba": INNER_RGBA}},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "navmesh", "mesh": 0}],
        "meshes": [{"name": "navmesh", "primitives": [{
            "attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "material": 0, "mode": 4}]}],
        "materials": [{"name": "navmesh", "doubleSided": True, "alphaMode": "BLEND",
                       "pbrMetallicRoughness": {"baseColorFactor": [1.0, 1.0, 1.0, opacity],
                                                "metallicFactor": 0.0, "roughnessFactor": 1.0},
                       "extensions": {"KHR_materials_unlit": {}}}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": views,
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5121, "normalized": True, "count": len(colours), "type": "VEC4"},
            {"bufferView": 2, "componentType": 5125, "count": len(indices), "type": "SCALAR"},
        ],
    }
    json_bytes = pad(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    out_path = Path(out_path)
    with out_path.open("wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, total))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes)
        handle.write(struct.pack("<II", len(binary), 0x004E4942) + binary)
    return {"path": str(out_path), "bytes": total, "cells": int(len(k)), "edge_cells": int(edge.sum()),
            "levels": int(grid.shape[0])}
