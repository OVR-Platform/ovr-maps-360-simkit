import json

import numpy as np
import open3d as o3d

from simkit.lod import build_lods


def test_tiers_respect_their_budgets_and_debris_is_pruned(tmp_path):
    ground = o3d.geometry.TriangleMesh.create_box(20, 20, 0.2).subdivide_midpoint(4)
    bump = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=40).translate((5, 5, 1))
    debris = o3d.geometry.TriangleMesh.create_tetrahedron(0.05).translate((-5, -5, 1))
    mesh = ground + bump + debris
    o3d.io.write_triangle_mesh(str(tmp_path / "surface.ply"), mesh)

    result = build_lods(tmp_path / "surface.ply", tmp_path / "lod", "abcd1234", verbose=False)

    assert result["pruned_components"] >= 1
    counts = [result["tiers"][t]["triangles"] for t in ("high", "mid", "low")]
    assert counts == sorted(counts, reverse=True)
    for tier, budget in (("high", 1.0), ("mid", 3.0), ("low", 10.0)):
        assert result["tiers"][tier]["deviation_p95_cm"] <= budget
        assert (tmp_path / "lod" / f"abcd1234_{tier}.ply").stat().st_size > 0
    assert json.loads((tmp_path / "lod" / "abcd1234.json").read_text()) == result
