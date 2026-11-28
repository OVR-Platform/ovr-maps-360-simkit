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
            "splat_to_surface_median_m": _round(m.get("splat_to_surface_median_m"), 4),
            "camera_height_m": _round(m.get("camera_height_m"), 3),
            "walkable_area_m2": _round(m.get("walkable_area_m2"), 1),
        },
        "certified_geometry": (
            "mesh/model.glb, moved into the simulation frame and cropped to the walked corridor; "
            "the solver touches the ground height field and the structure boxes derived from it"
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
        f"  splat offset, fitted p90  {show(m['alignment_residual_m'], ' cm', 100, 1)}",
        f"  splat to surface, median  {show(m['splat_to_surface_median_m'], ' cm', 100, 1)}   (info)",
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
    # Isaac gets only the scene around the drop point. The whole stage (Amazon
    # 04802a70: a 1M-triangle ground and 6548 boxes) took 14 GB here and was
    # OOM-killed on 62 GB machines; the sphere only ever touches a few metres.
    local = certification / "_isaac_local.usda"
    cropped = crop_usd_around(bundle / "scene.usda", local, (x, y), radius_m=ISAAC_RADIUS_M)
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    command = [
        isaac_python, "-m", "simkit.physics.isaac_drop",
        "--usd", str(local.resolve()),
        # Fixed-point: argparse reads "-1.2e-05" as an option, not a number
        # (Amazon 179f346d: "usage: isaac_drop.py ...").
        "--at", f"{x:.6f}", f"{y:.6f}", f"{floor:.6f}",
        "--drop-from", str(drop_from), "--steps", str(steps), "--json", str(raw.resolve()),
    ]
    # One Isaac Sim at a time per machine: concurrent instances abort (SIGABRT)
    # while starting up, measured with six at once on two GPUs.
    import fcntl
    import tempfile

    with open(Path(tempfile.gettempdir()) / "simkit-isaac.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        completed = run_capped(command, env=env, timeout_s=timeout_s, max_rss_gb=ISAAC_MAX_RSS_GB)
    try:
        if completed.returncode != 0 or not raw.exists():
            raise RuntimeError(f"Isaac drop test failed (exit {completed.returncode}):\n{completed.stderr[-2000:]}")
        result = json.loads(raw.read_text())[0]
    finally:
        raw.unlink(missing_ok=True)
        local.unlink(missing_ok=True)

    manifest = json.loads((bundle / "manifest.json").read_text())
    # Compare with the floor where the sphere came to rest: on a slope it rolls
    # (Amazon 2ffedfc0, an 8 deg street: 30.65 cm "below" the drop point's floor,
    # resting on the floor 1.9 m downhill).
    rest_xy = result.get("resting_xy", [x, y])
    floor_there = floor_at(bundle, manifest, rest_xy, default=floor)
    error_m = result["resting_z"] - (floor_there + 0.05)
    fell = bool(result["resting_z"] < floor_there - 0.5)
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
        "dropped_at_xy": [round(x, 3), round(y, 3)],
        "resting_xy_isaac": [round(v, 3) for v in rest_xy],
        "rolled_m": round(float(np.hypot(rest_xy[0] - x, rest_xy[1] - y)), 3),
        "floor_where_it_rests_m": round(floor_there, 4),
        "resting_z_isaac_m": round(result["resting_z"], 5),
        "disagreement_cm": round(error_m * 100, 2),
        "fell_through": fell,
        "stage_given_to_isaac": cropped,
        "isaac_peak_rss_gb": round(completed.peak_rss_gb, 2),
        "isaac_seconds": round(completed.seconds, 1),
        "navmesh_cell_size_m": manifest["stages"]["s5_navmesh"]["cell_size_m"],
        "verdict": "fell through the floor" if fell else "solid, no fall-through",
    }
    (certification / "isaac_drop_test.json").write_text(json.dumps(record, indent=1))
    return record


ISAAC_RADIUS_M = 15.0
# Isaac needs about 6 GB here (an empty stage takes as much). On the Amazon
# fleet it grew 0.25 GB/s with no plateau until the kernel killed it, taking
# the machine's memory with it; past this it is stopped and the build fails.
ISAAC_MAX_RSS_GB = float(os.environ.get("SIMKIT_ISAAC_MAX_RSS_GB", "16"))


def run_capped(command, *, env, timeout_s: float, max_rss_gb: float) -> subprocess.CompletedProcess:
    """``subprocess.run`` that kills the process tree when its resident memory passes ``max_rss_gb``."""
    import signal
    import tempfile
    import time

    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile("w+") as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err, text=True, start_new_session=True)
        started, peak, reason = time.monotonic(), 0.0, None
        while process.poll() is None:
            rss = _tree_rss_gb(process.pid)
            peak = max(peak, rss)
            if rss > max_rss_gb:
                reason = f"Isaac exceeded {max_rss_gb:g} GB resident ({rss:.1f} GB)"
            elif time.monotonic() - started > timeout_s:
                reason = f"Isaac took more than {timeout_s:g} s"
            if reason:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            time.sleep(1.0)
        out.seek(0)
        err.seek(0)
        stderr = err.read()
        if reason:
            raise RuntimeError(f"{reason}; stopped after {time.monotonic() - started:.0f} s:\n{stderr[-2000:]}")
        completed = subprocess.CompletedProcess(command, process.returncode, out.read(), stderr)
        completed.peak_rss_gb = peak
        completed.seconds = time.monotonic() - started
        return completed


def _tree_rss_gb(root: int) -> float:
    """Resident memory of a process and its descendants, from /proc."""
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            children.setdefault(int(fields[1]), []).append(int(entry.name))
            rss[int(entry.name)] = int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            continue
    total, stack = 0, [root]
    while stack:
        pid = stack.pop()
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total / 2**30


def crop_usd_around(source: Path, target: Path, xy, *, radius_m: float) -> dict:
    """Copy of the stage keeping the colliders within ``radius_m`` (XY) of ``xy``.

    Boxes farther away are dropped and the ground mesh keeps only the triangles
    with a vertex inside the square window; every kept prim is authored exactly
    as in the bundle, so the test is still a test of the exported USD.
    """
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.Open(str(source))
    x0, y0 = float(xy[0]), float(xy[1])
    boxes_kept = boxes_dropped = ground_before = ground_after = 0
    for prim in list(stage.Traverse()):
        if prim.IsA(UsdGeom.Cube):
            centre = translate = scale = None
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate = op.Get()
                elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale = op.Get()
            if translate is None:
                boxes_kept += 1
                continue
            centre = np.array([translate[0], translate[1]])
            half = np.abs(np.array([scale[0], scale[1]])) if scale is not None else np.ones(2)
            gap = np.maximum(np.abs(centre - [x0, y0]) - half, 0.0)
            if np.hypot(*gap) > radius_m:
                stage.RemovePrim(prim.GetPath())
                boxes_dropped += 1
            else:
                boxes_kept += 1
        elif prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if len(points) == 0 or not np.all(counts == 3):
                continue
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            inside = (np.abs(points[:, 0] - x0) <= radius_m) & (np.abs(points[:, 1] - y0) <= radius_m)
            keep = inside[faces].any(axis=1)
            ground_before += len(faces)
            faces = faces[keep]
            used, faces = np.unique(faces, return_inverse=True)
            faces = faces.reshape(-1, 3)
            points = points[used]
            ground_after += len(faces)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, dtype=np.int32)))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
            if len(points):
                mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, points.min(axis=0))),
                                                         Gf.Vec3f(*map(float, points.max(axis=0)))]))
    stage.Export(str(target))
    return {"window": f"{2 * radius_m:.0f} m square around the drop point",
            "boxes_kept": boxes_kept, "boxes_dropped": boxes_dropped,
            "mesh_triangles_kept": ground_after, "mesh_triangles_in_bundle": ground_before}


def floor_at(bundle: Path, manifest: dict, xy, *, default: float) -> float:
    """Navmesh ground under ``xy`` (the cell's height, single level)."""
    navmesh = manifest["stages"]["s5_navmesh"]
    ground = np.load(bundle / "collision" / "navmesh_ground_z.npy")
    col = int(np.floor((xy[0] - navmesh["origin_xy"][0]) / navmesh["cell_size_m"]))
    row = int(np.floor((xy[1] - navmesh["origin_xy"][1]) / navmesh["cell_size_m"]))
    if 0 <= row < ground.shape[0] and 0 <= col < ground.shape[1] and np.isfinite(ground[row, col]):
        return float(ground[row, col])
    return default


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
