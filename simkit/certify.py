"""Certificates and datasheet of a built scene.

    simulation/certification/mujoco_physics_gate.json   the gate, machine-readable
    simulation/certification/mujoco_physics_gate.txt    the same, for a human
    simulation/certification/isaac_drop_test.json       cross-engine drop test
    datasheets/<uuid>.json                              tier, stages, gate

The gate is certified in MuJoCo. The Isaac drop test asks the other supported
engine one question — does a body released above a point MuJoCo calls solid
floor come to rest there — and is reported as agreement, not as certification.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

from simkit.gate import CHECK_TABLE

PLATE_HALF_THICKNESS_M = 0.06 * 0.3  # probes.py: foot plate half-size 0.06, thickness factor 0.3
DATASHEET_STAGES = (
    "s1_camera_frame", "s1_frame", "s1_walk_crop", "s1_up_check_after", "s1_plumb",
    "s3_surface", "s5_navmesh", "s7_layer_residual", "s7_floor_registration",
)


def _json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def write_gate_certificate(bundle: Path, certification: Path, uuid: str) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    qa, scene = manifest["qa"], manifest["scene_id"]
    m = qa["measurements"]
    record = {
        "scene": scene,
        "uuid": uuid,
        "engine": "MuJoCo",
        "tier": manifest["tier"],
        "passed": qa["passed"],
        "checks": qa["checks"],
        "measurements": {
            "probes": m.get("probes"),
            "settle_rate": m.get("settle_rate"),
            "leak_rate": m.get("leak_rate"),
            "penetration_p95_mm": _mm(m.get("p95_penetration_m")),
            "penetration_max_mm": _mm(m.get("max_penetration_m")),
            "median_settle_time_s": m.get("median_settle_time_s"),
            "witness_disagreement_deg": _round(m.get("witness_disagreement_deg"), 4),
            "plumb_tilt_deg": _round(m.get("plumb_tilt_deg"), 4),
            "plumb_reference": m.get("plumb_reference"),
            "floor_registration_m": _round(m.get("floor_registration_m"), 4),
            "walkable_under_cameras": _round(m.get("walkable_under_cameras"), 3),
            "alignment_residual_m": _round(m.get("alignment_residual_m"), 4),
            "camera_height_m": _round(m.get("camera_height_m"), 3),
            "walkable_area_m2": _round(m.get("walkable_area_m2"), 1),
        },
        "certified_geometry": (
            "mesh/model.glb, moved into the simulation frame and cropped to the walked corridor; "
            "the solver touches the ground height field and the convex parts derived from it"
        ),
        "reproduce": f"simkit regate simulation/{scene}.sre",
    }
    certification.mkdir(parents=True, exist_ok=True)
    (certification / "mujoco_physics_gate.json").write_text(json.dumps(record, indent=1, default=_json))
    (certification / "mujoco_physics_gate.txt").write_text(_gate_text(record))
    return record


def _mm(value):
    return None if value is None or not np.isfinite(value) else round(float(value) * 1000, 3)


def _round(value, digits):
    return None if value is None or not np.isfinite(value) else round(float(value), digits)


def _gate_text(record: dict) -> str:
    m = record["measurements"]
    lines = [
        f"MuJoCo physics gate - scene {record['scene']}",
        f"uuid {record['uuid']}",
        f"tier {record['tier']}    verdict: {'PASS' if record['passed'] else 'FAIL'}",
        "",
        f"{'#':<3}{'check':<60}{'threshold':<26}result",
    ]
    for number, (key, label, threshold) in enumerate(CHECK_TABLE, start=1):
        verdict = "PASS" if record["checks"].get(key) else "FAIL"
        lines.append(f"{number:<3}{label:<60}{threshold:<26}{verdict}")

    def show(value, unit="", scale=1.0, digits=2):
        return "n/a" if value is None else f"{value * scale:.{digits}f}{unit}"

    lines += [
        "",
        "measured",
        f"  probes dropped            {m['probes']}",
        f"  settled                   {show(m['settle_rate'], ' %', 100, 0)}",
        f"  fell through              {show(m['leak_rate'], ' %', 100, 0)}",
        f"  penetration p95           {show(m['penetration_p95_mm'], ' mm')}   (ceiling 20 mm)",
        f"  penetration max           {show(m['penetration_max_mm'], ' mm')}",
        f"  plumb tilt                {show(m['plumb_tilt_deg'], ' deg', digits=3)}   (witness: {m['plumb_reference'] or 'none'})",
        f"  witness disagreement      {show(m['witness_disagreement_deg'], ' deg', digits=3)}",
        f"  navmesh vs walked floor   {show(m['floor_registration_m'], ' cm', 100, 1)}",
        f"  walk on walkable cells    {show(m['walkable_under_cameras'], ' %', 100, 0)}",
        f"  splat to surface, median  {show(m['alignment_residual_m'], ' cm', 100, 1)}",
        f"  camera height (info)      {show(m['camera_height_m'], ' m')}",
        f"  walkable area             {show(m['walkable_area_m2'], ' m2', digits=1)}",
        "",
        "What is certified: the collision representation derived from mesh/model.glb.",
        "Isaac Sim is checked separately - see isaac_drop_test.json - not certified.",
        "",
    ]
    return "\n".join(lines)


def drop_point(bundle: Path) -> tuple[float, float, float]:
    """A probe MuJoCo settled cleanly, nearest the walkable area's centroid.

    Returns XY and the floor height MuJoCo reports there (rest height of the
    probe's centre less half the plate's thickness).
    """
    manifest = json.loads((bundle / "manifest.json").read_text())
    probes = [p for p in manifest["stages"].get("physics_probes", []) if p["settled"] and not p["fell_through"]]
    if not probes:
        raise ValueError("no settled MuJoCo probe to drop the Isaac probe onto")
    xy = np.array([[p["x"], p["y"]] for p in probes])
    chosen = probes[int(np.argmin(np.linalg.norm(xy - xy.mean(axis=0), axis=1)))]
    return chosen["x"], chosen["y"], chosen["final_z"] - PLATE_HALF_THICKNESS_M


def run_isaac_drop_test(bundle: Path, certification: Path, uuid: str, *, isaac_python: str,
                        drop_from: float = 1.0, steps: int = 300, timeout_s: int = 1800) -> dict:
    """Run the drop test in the Isaac Sim interpreter and write its certificate."""
    x, y, floor = drop_point(bundle)
    raw = certification / "_isaac_raw.json"
    certification.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    command = [
        isaac_python, "-m", "simkit.physics.isaac_drop",
        "--usd", str((bundle / "scene.usda").resolve()),
        "--at", f"{x}", f"{y}", f"{floor}",
        "--drop-from", str(drop_from), "--steps", str(steps), "--json", str(raw.resolve()),
    ]
    # One Isaac Sim at a time per machine: concurrent instances abort (SIGABRT)
    # while starting up, measured with six at once on two GPUs.
    import fcntl
    import tempfile

    with open(Path(tempfile.gettempdir()) / "simkit-isaac.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout_s)
    try:
        if completed.returncode != 0 or not raw.exists():
            raise RuntimeError(f"Isaac drop test failed (exit {completed.returncode}):\n{completed.stderr[-2000:]}")
        result = json.loads(raw.read_text())[0]
    finally:
        raw.unlink(missing_ok=True)

    manifest = json.loads((bundle / "manifest.json").read_text())
    record = {
        "scene": manifest["scene_id"],
        "uuid": uuid,
        "engine": result.get("engine", "Isaac Sim (PhysX), headless"),
        "check": (f"drop test - a 5 cm rigid sphere released {drop_from:.1f} m above a point the MuJoCo gate "
                  f"certifies as solid floor, integrated {steps} steps"),
        "this_is_not_a_certification": (
            "Isaac Sim is a supported export target, checked here for agreement with the certified MuJoCo floor. "
            "The gate itself runs in MuJoCo."
        ),
        "floor_from_mujoco_m": round(floor, 4),
        "resting_z_isaac_m": round(result["resting_z"], 5),
        "disagreement_cm": round(result["error_m"] * 100, 2),
        "fell_through": bool(result["fell_through"]),
        "navmesh_cell_size_m": manifest["stages"]["s5_navmesh"]["cell_size_m"],
        "verdict": "fell through the floor" if result["fell_through"] else "solid, no fall-through",
    }
    (certification / "isaac_drop_test.json").write_text(json.dumps(record, indent=1))
    return record


def write_datasheet(bundle: Path, datasheet: Path, uuid: str, *, isaac: dict | None) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    record = {
        "uuid": uuid,
        "tier": manifest["tier"],
        "seconds": manifest["stages"].get("build_seconds"),
        "stages": {k: manifest["stages"][k] for k in DATASHEET_STAGES if k in manifest["stages"]},
        "qa": manifest["qa"],
        "isaac_drop_test": None if isaac is None else {
            k: isaac[k] for k in ("disagreement_cm", "fell_through", "verdict")
        },
    }
    datasheet.parent.mkdir(parents=True, exist_ok=True)
    datasheet.write_text(json.dumps(record, indent=1, default=_json))
    return record
