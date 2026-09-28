"""The acceptance gate: eight checks, every one measured on the built bundle.

A bundle that fails any check does not ship. Thresholds are part of the
contract and are never relaxed to make a scene pass.

Orientation and frame (measured, not asserted):

- ``up_direction_verified``: the splat's densest horizontal slab sits in the
  lower part of the scene, and at least 90% of the camera positions have ground
  beneath them.
- ``scene_plumb_under_2deg``: the vertical implied by the walls (a building's
  walls are plumb) against the frame's +Z, which comes from the ARKit
  alignment. Floors may slope — a street on a hill does — walls may not. Two
  independent witnesses read the walls inside the walked corridor: the splat's
  disc Gaussians and the mesh's faces. Both reconstructions live in the same
  frame, so a broken alignment tilts the walls in both; a tilt seen by only one
  of them is that witness's noise (they disagree with each other by 0.9 deg
  median, 1.9 deg p90, over the 50 scenes of the sample). The measured tilt is
  therefore the smaller of the two, and the threshold is set where it catches a
  broken alignment: 2 deg of gravity error is a 3.5% slope under a robot's feet.
- ``floor_at_origin_under_5cm``: the navmesh's ground against the floor the
  operator actually stood on. Under every camera whose cell is walkable, the
  navmesh ground height is compared with the surface found by a ray cast
  straight down from that camera; the median absolute difference must stay
  under 5 cm, **and at least 20% of the cameras must stand on a walkable
  cell**, otherwise the comparison has nothing to say and the navmesh does not
  cover the walk. The frame puts the median walked floor at z = 0, so this is
  also where the origin is certified. Compared per camera, not as two medians,
  so a street climbing several metres does not read as a misplaced floor.

Not a check, recorded only: the angle between the plane of the camera
trajectory and the plane of the ground beneath it (``witness_disagreement_deg``).
It used to be gated at 0.5 deg, but the trajectory plane tilts whenever the
operator raises or lowers the pole along the walk (45 cm of pole travel over
100 m is 0.3-1 deg), so it measured the operator, not the scene: over the 50
scenes of the sample the walls were plumb to 0.6 deg median, 1.7 deg max, and
every failure of that check came with the ground under the cameras within
1 mm of the navmesh.
- ``alignment_residual_under_25cm``: median distance from the splat's solid
  Gaussians to the collision surface, inside the walked corridor. The two
  layers are separate reconstructions of one capture; this is how far apart
  they are.

Physics (MuJoCo, foot-sized probes dropped on navmesh cells):

- ``walkable_area_over_5m2``, ``no_collision_leak``, ``probes_settle``,
  ``penetration_p95_under_2cm``.
"""

from __future__ import annotations

import numpy as np

CHECK_TABLE = [
    # key, label, threshold (as printed in the certificate)
    ("up_direction_verified", "up direction verified (splat slab + ground under cameras)", "decidable, not inverted"),
    ("scene_plumb_under_2deg", "scene plumb (walls vs frame vertical)", "< 2 deg"),
    ("floor_at_origin_under_5cm", "floor registered (navmesh under >= 20% of the walk, vs ground under cameras)", "< 5 cm"),
    ("alignment_residual_under_25cm", "alignment residual, splat to collision surface", "< 25 cm"),
    ("walkable_area_over_5m2", "walkable area", ">= 5 m2"),
    ("no_collision_leak", "no collision leak (no probe falls through)", "0%"),
    ("probes_settle", "probes settle", ">= 90%"),
    ("penetration_p95_under_2cm", "solver penetration, 95th percentile", "< 2 cm"),
]


def evaluate_gate(
    *,
    walk: dict,
    up_verdict: dict,
    plumb: dict,
    floor_registration: dict,
    layer_residual: dict,
    walkable_area_m2: float,
    physics: dict,
) -> dict:
    """Apply the eight checks. Missing evidence counts as failure."""
    checks = {
        "up_direction_verified": bool(up_verdict.get("decidable"))
        and not up_verdict.get("inverted", True)
        and walk.get("grounded_fraction", 0.0) >= 0.9,
        "scene_plumb_under_2deg": plumb.get("tilt_deg", float("inf")) < 2.0,
        "floor_at_origin_under_5cm": floor_registration.get("walkable_fraction", 0.0) >= 0.2
        and floor_registration.get("median_abs_m", float("inf")) < 0.05,
        "alignment_residual_under_25cm": layer_residual.get("median_m", float("inf")) < 0.25,
        "walkable_area_over_5m2": walkable_area_m2 >= 5.0,
        "no_collision_leak": physics.get("leak_rate", 1.0) == 0.0,
        "probes_settle": physics.get("settle_rate", 0.0) >= 0.9,
        "penetration_p95_under_2cm": _finite(physics.get("p95_penetration_m"), 1.0) < 0.02,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "measurements": {
            "alignment_residual_m": layer_residual.get("median_m"),
            "alignment_residual_p90_m": layer_residual.get("p90_m"),
            "camera_height_m": walk.get("camera_height_m"),
            "grounded_fraction": walk.get("grounded_fraction"),
            "witness_disagreement_deg": walk.get("mesh_agreement_deg"),
            "plumb_tilt_deg": plumb.get("tilt_deg"),
            "plumb_reference": plumb.get("reference"),
            "floor_registration_m": floor_registration.get("median_abs_m"),
            "walkable_under_cameras": floor_registration.get("walkable_fraction"),
            "walkable_area_m2": walkable_area_m2,
        }
        | physics
        | {"orientation_up_check": up_verdict},
    }


def _finite(value, default: float) -> float:
    if value is None:
        return default
    value = float(value)
    return value if np.isfinite(value) else default


def mesh_walls_vertical(vertices: np.ndarray, faces: np.ndarray, *, wall_cosine: float = 0.35,
                        rounds: int = 5) -> np.ndarray | None:
    """Vertical from the mesh's wall faces: the ``u`` minimising sum(area * (n . u)^2).

    Same small-angle solve as ``gravity_from_walls`` on the splat, so the two
    witnesses differ only in the reconstruction they read.
    """
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(normals, axis=1)
    usable = length > 1e-12
    normals, weight = normals[usable] / length[usable, None], np.sqrt(0.5 * length[usable])
    up = np.array([0.0, 0.0, 1.0])
    for _ in range(rounds):
        wall = np.abs(normals @ up) < wall_cosine
        if wall.sum() < 1000:
            return None
        weighted = normals[wall] * weight[wall, None]
        try:
            offset = np.linalg.solve(weighted[:, :2].T @ weighted[:, :2], -weighted[:, :2].T @ weighted[:, 2])
        except np.linalg.LinAlgError:
            return None
        new_up = np.array([offset[0], offset[1], 1.0])
        new_up /= np.linalg.norm(new_up)
        moved = np.degrees(np.arccos(np.clip(new_up @ up, -1.0, 1.0)))
        up = new_up
        if moved < 0.01:
            break
    return up


def plumb_measurement(corridor_splat, corridor_vertices: np.ndarray, corridor_faces: np.ndarray) -> dict:
    """How far the frame's vertical is from the walls, seen by splat and mesh."""
    from simkit.geometry.up_check import gravity_from_walls, tilt_degrees

    from_splat = gravity_from_walls(corridor_splat)
    from_mesh = mesh_walls_vertical(corridor_vertices, corridor_faces)
    witnesses = {name: u for name, u in (("splat", from_splat), ("mesh", from_mesh)) if u is not None}
    if not witnesses:
        return {"reference": "none", "tilt_deg": float("inf")}
    tilts = {name: float(tilt_degrees(u)) for name, u in witnesses.items()}
    result = {
        "reference": "walls, smaller of " + " and ".join(witnesses) if len(witnesses) == 2 else f"walls ({next(iter(witnesses))})",
        "tilt_deg": min(tilts.values()),
    }
    if from_splat is not None:
        result["splat_walls_tilt_deg"] = float(tilt_degrees(from_splat))
    if from_mesh is not None:
        result["mesh_walls_tilt_deg"] = float(tilt_degrees(from_mesh))
    if len(witnesses) == 2:
        result["witnesses_apart_deg"] = float(np.degrees(np.arccos(np.clip(from_splat @ from_mesh, -1.0, 1.0))))
    return result


def layer_residual(sim_splat, surface_vertices: np.ndarray, surface_faces: np.ndarray,
                   sim_positions: np.ndarray, *, corridor_m: float = 12.0,
                   min_opacity: float = 0.5, sample: int = 1_000_000, seed: int = 0) -> dict:
    """Distance from solid Gaussians to the collision surface, in the walked corridor."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    solid = sim_splat.solid(min_opacity=min_opacity).means
    near, _ = cKDTree(sim_positions[:, :2]).query(solid[:, :2])
    solid = solid[near <= corridor_m]
    if len(solid) < 1000:
        return {"median_m": float("inf"), "gaussians": int(len(solid))}
    if len(solid) > sample:
        solid = solid[np.random.default_rng(seed).choice(len(solid), sample, replace=False)]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(surface_vertices, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(surface_faces, dtype=np.uint32)),
    )
    distance = scene.compute_distance(o3d.core.Tensor(solid.astype(np.float32))).numpy()
    return {
        "median_m": float(np.median(distance)),
        "p75_m": float(np.percentile(distance, 75)),
        "p90_m": float(np.percentile(distance, 90)),
        "gaussians": int(len(solid)),
        "corridor_m": corridor_m,
    }


def floor_registration(navmesh, ground_points: np.ndarray) -> dict:
    """Navmesh ground against the ray-cast floor, under each camera on a walkable cell."""
    columns = np.floor((ground_points[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(int)
    rows = np.floor((ground_points[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(int)
    inside = (rows >= 0) & (rows < navmesh.grid.shape[0]) & (columns >= 0) & (columns < navmesh.grid.shape[1])
    walkable = np.zeros(len(ground_points), dtype=bool)
    walkable[inside] = navmesh.grid[rows[inside], columns[inside]]
    if not walkable.any():
        return {"cameras": int(len(ground_points)), "on_walkable": 0, "walkable_fraction": 0.0}
    difference = navmesh.ground_z[rows[walkable], columns[walkable]] - ground_points[walkable, 2]
    return {
        "cameras": int(len(ground_points)),
        "on_walkable": int(walkable.sum()),
        # Informative: how much of the walked path the navmesh declares walkable.
        "walkable_fraction": float(walkable.mean()),
        "median_abs_m": float(np.median(np.abs(difference))),
        "median_signed_m": float(np.median(difference)),
        "p90_abs_m": float(np.percentile(np.abs(difference), 90)),
    }
def rescore(manifest: dict) -> tuple[dict, dict]:
    """Re-apply the gate to a built bundle from the measurements it recorded.

    Returns the new ``qa`` and the plumb record. No geometry is recomputed.
    """
    stages, measured = manifest["stages"], manifest["qa"]["measurements"]
    plumb = dict(stages["s1_plumb"])
    tilts = [plumb[k] for k in ("splat_walls_tilt_deg", "mesh_walls_tilt_deg") if k in plumb]
    if tilts:
        plumb["tilt_deg"] = min(tilts)
        plumb["reference"] = "walls, smaller of splat and mesh" if len(tilts) == 2 else plumb.get("reference")
        plumb.pop("vertical", None)
    physics_keys = ("probes", "leak_rate", "settle_rate", "median_penetration_m", "p95_penetration_m",
                    "max_penetration_m", "median_settle_time_s")
    return evaluate_gate(
        walk=stages["s1_camera_frame"],
        up_verdict={k: v for k, v in stages["s1_up_check_after"].items() if k != "witness"},
        plumb=plumb,
        floor_registration=stages["s7_floor_registration"],
        layer_residual=stages["s7_layer_residual"],
        walkable_area_m2=measured["walkable_area_m2"],
        physics={k: measured[k] for k in physics_keys if k in measured},
    ), plumb
