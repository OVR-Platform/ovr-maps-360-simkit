"""Three levels of detail of the collision surface.

Source: ``simulation/<id8>.sre/collision/surface.ply``, the cropped and
simplified mesh the bundle was built from, already in the simulation frame.

1. Connected components with fewer than 50 triangles are pruned: they are
   reconstruction debris, not structure.
2. Each tier is the smallest quadric decimation whose symmetric deviation from
   the pruned source stays within budget at the 99th percentile:
   high 1 cm, mid 3 cm, low 10 cm. The triangle target is found by bisection.

Symmetric deviation is the mean of two one-sided distances: points sampled on
the decimated mesh to the source surface, and source vertices to the decimated
surface. The second term is what catches holes a decimation opens.

Output: ``<id8>_high.ply``, ``<id8>_mid.ply``, ``<id8>_low.ply`` and
``<id8>.json`` with triangle counts, reduction and deviation per tier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

TIERS = {"high": 1.0, "mid": 3.0, "low": 10.0}  # error budget, cm, at p99
MIN_COMPONENT_TRIANGLES = 50


def build_lods(surface_path: str | Path, output_dir: str | Path, scene_id: str, *, verbose: bool = True) -> dict:
    import open3d as o3d

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(str(surface_path))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    source_triangles = len(mesh.triangles)

    clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    clusters, cluster_sizes = np.asarray(clusters), np.asarray(cluster_sizes)
    small = cluster_sizes < MIN_COMPONENT_TRIANGLES
    remove = small[clusters]
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()

    reference = o3d.t.geometry.RaycastingScene()
    reference.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    source_vertices = np.asarray(mesh.vertices, dtype=np.float32)

    def deviation(decimated) -> tuple[float, float, float]:
        """Symmetric deviation in cm at p50, p95, p99."""
        points = np.asarray(decimated.sample_points_uniformly(number_of_points=60_000).points, dtype=np.float32)
        forward = reference.compute_distance(o3d.core.Tensor(points)).numpy() * 100
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(decimated))
        backward = scene.compute_distance(o3d.core.Tensor(source_vertices)).numpy() * 100

        def at(p):
            return 0.5 * (float(np.percentile(forward, p)) + float(np.percentile(backward, p)))

        return at(50), at(95), at(99)

    result = {
        "source_triangles": source_triangles,
        "pruned_faces": int(remove.sum()),
        "pruned_components": int(small.sum()),
        "tiers": {},
    }
    pruned_triangles = len(mesh.triangles)
    for name, budget in TIERS.items():
        low, high = 2000, pruned_triangles
        best = None
        for _ in range(9):
            target = int((low * high) ** 0.5)
            decimated = mesh.simplify_quadric_decimation(target_number_of_triangles=target)
            decimated.remove_unreferenced_vertices()
            p50, p95, p99 = deviation(decimated)
            if p99 <= budget:
                best = (decimated, p50, p95)
                high = target
            else:
                low = target
            if high / max(low, 1) < 1.08:
                break
        if best is None:
            best = (mesh, *deviation(mesh)[:2])
        decimated, p50, p95 = best
        decimated.compute_vertex_normals()
        path = output_dir / f"{scene_id}_{name}.ply"
        o3d.io.write_triangle_mesh(str(path), decimated, write_ascii=False, compressed=False,
                                   write_vertex_normals=True)
        result["tiers"][name] = {
            "error_budget_cm": budget,
            "triangles": len(decimated.triangles),
            "reduction": round(1 - len(decimated.triangles) / source_triangles, 3),
            "deviation_p50_cm": round(p50, 2),
            "deviation_p95_cm": round(p95, 2),
            "file_mb": round(os.path.getsize(path) / 1e6, 1),
        }
        if verbose:
            tier = result["tiers"][name]
            print(f"[{scene_id}] LOD {name}: {tier['triangles']:,} triangles, p95 {tier['deviation_p95_cm']} cm",
                  flush=True)
    (output_dir / f"{scene_id}.json").write_text(json.dumps(result, indent=1))
    return result
