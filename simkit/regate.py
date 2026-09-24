"""Re-run the MuJoCo probes on a built bundle and compare with its manifest.

The probes are sampled from the shipped navmesh with the same seed the build
used, and MuJoCo is deterministic, so a bundle that has not been altered
reproduces its own gate measurements exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def regate(bundle: Path, *, probes: int = 40) -> dict:
    from simkit.geometry.navmesh import Navmesh
    from simkit.physics.probes import run_floor_probes, sample_floor_points

    manifest = json.loads((bundle / "manifest.json").read_text())
    info = manifest["stages"]["s5_navmesh"]
    navmesh = Navmesh(
        np.load(bundle / "collision" / "navmesh_grid.npy"),
        np.load(bundle / "collision" / "navmesh_ground_z.npy"),
        np.array(info["origin_xy"]),
        info["cell_size_m"],
    )
    report = run_floor_probes(bundle / "scene.xml", sample_floor_points(navmesh, count=probes)).as_dict()
    shipped = manifest["qa"]["measurements"]
    keys = ("probes", "leak_rate", "settle_rate", "p95_penetration_m")
    matches = all(
        np.isclose(report[k], shipped[k], atol=1e-6, equal_nan=True) for k in keys
    ) and np.isclose(navmesh.area_m2, shipped["walkable_area_m2"], rtol=1e-6)
    return {
        "bundle": bundle.name,
        "tier": manifest["tier"],
        "measured": {k: report[k] for k in keys},
        "manifest": {k: shipped[k] for k in keys},
        "matches_manifest": bool(matches),
    }
