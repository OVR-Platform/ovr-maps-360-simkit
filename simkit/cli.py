"""Command line.

    simkit build  <scene_dir> [--out <root>]     one scene: simulation, certification, lod, datasheet
    simkit batch  <dataset_root> [--jobs N]      every scene under a dataset root
    simkit lod    <surface.ply> <out_dir> <id8>  LODs only
    simkit regate <bundle.sre>                   re-run the MuJoCo probes on a built bundle
    simkit rescore <scene_out_dir>               re-apply the gate from recorded measurements
    simkit verify <scene_out_dir>                check the outputs of one scene

Outputs of scene ``<uuid>`` go to ``<root>/<uuid>/`` (default: the scene
directory itself):

    simulation/<id8>.sre/  simulation/certification/  lod/  datasheets/<uuid>.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DEFAULT_LICENCE = "see the dataset card"


def default_isaac_python() -> str | None:
    """The interpreter of the env made by install/install_env_isaac.sh."""
    configured = os.environ.get("SIMKIT_ISAAC_PYTHON")
    if configured:
        return configured
    try:
        base = subprocess.run(["conda", "info", "--base"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    candidate = Path(base) / "envs" / "simkit-isaac" / "bin" / "python"
    return str(candidate) if candidate.exists() else None


def scene_outputs(scene_dir: Path, out_root: Path | None) -> dict:
    uuid = scene_dir.name
    root = (out_root / uuid) if out_root else scene_dir
    return {
        "uuid": uuid,
        "id8": uuid[:8],
        "root": root,
        "bundle": root / "simulation" / f"{uuid[:8]}.sre",
        "certification": root / "simulation" / "certification",
        "lod": root / "lod",
        "datasheet": root / "datasheets" / f"{uuid}.json",
    }


def build_one(scene_dir: Path, args) -> dict:
    """All four products of one scene, resuming from whatever is already done."""
    from simkit.certify import run_isaac_drop_test, write_datasheet, write_gate_certificate
    from simkit.lod import build_lods
    from simkit.pipeline import build_scene

    out = scene_outputs(scene_dir, args.out)
    if out["datasheet"].exists() and not args.force:
        print(f"[{out['id8']}] done already ({out['datasheet']}), skipping", flush=True)
        return json.loads(out["datasheet"].read_text())

    isaac_python = args.isaac_python or default_isaac_python()
    if not isaac_python:
        raise RuntimeError("Isaac Sim interpreter not found: run install/install_env_isaac.sh "
                           "or pass --isaac-python / set SIMKIT_ISAAC_PYTHON")

    manifest_path = out["bundle"] / "manifest.json"
    if args.force or not manifest_path.exists():
        build_scene(scene_dir, out["bundle"], scene_id=out["id8"], licence=args.licence,
                    probe_count=args.probes, workers=args.workers)
    manifest = json.loads(manifest_path.read_text())

    write_gate_certificate(out["bundle"], out["certification"], out["uuid"])

    isaac_json = out["certification"] / "isaac_drop_test.json"
    if args.force or not isaac_json.exists():
        isaac = run_isaac_drop_test(out["bundle"], out["certification"], out["uuid"], isaac_python=isaac_python)
    else:
        isaac = json.loads(isaac_json.read_text())
    print(f"[{out['id8']}] Isaac drop test: {isaac['verdict']}, {isaac['disagreement_cm']:+.2f} cm", flush=True)

    lod_json = out["lod"] / f"{out['id8']}.json"
    if args.force or not lod_json.exists():
        build_lods(out["bundle"] / "collision" / "surface.ply", out["lod"], out["id8"])

    datasheet = write_datasheet(out["bundle"], out["datasheet"], out["uuid"], isaac=isaac)
    passed = manifest["qa"]["passed"] and not isaac["fell_through"]
    print(f"[{out['id8']}] {'PASS' if passed else 'FAIL'} tier {manifest['tier']}", flush=True)
    return datasheet


def cmd_build(args) -> int:
    build_one(Path(args.scene_dir), args)
    return 0


def cmd_batch(args) -> int:
    root = Path(args.dataset_root)
    if args.uuids:
        names = [line.strip() for line in Path(args.uuids).read_text().splitlines() if line.strip()]
    else:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and UUID.match(p.name))
    jobs = max(1, args.jobs)
    if args.workers == 0:
        args.workers = max(1, ((os.cpu_count() or 4) - 2) // jobs)
    log_dir = Path(args.log_dir) if args.log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    # One process per scene, so a scene that crashes the interpreter (CoACD and
    # pxr are C++ underneath) takes nothing else with it.
    base = [sys.executable, "-m", "simkit", "build", "--licence", args.licence,
            "--probes", str(args.probes), "--workers", str(args.workers)]
    if args.out:
        base += ["--out", str(args.out)]
    if args.isaac_python:
        base += ["--isaac-python", args.isaac_python]
    if args.force:
        base += ["--force"]
    pending, running, failed, finished = list(names), {}, [], 0
    while pending or running:
        while pending and len(running) < jobs:
            name = pending.pop(0)
            log = open(log_dir / f"{name}.log", "a") if log_dir else None
            process = subprocess.Popen(base + [str(root / name)], stdout=log, stderr=subprocess.STDOUT if log else None)
            running[name] = (process, log)
        for name, (process, log) in list(running.items()):
            if process.poll() is None:
                continue
            if log:
                log.close()
            finished += 1
            if process.returncode != 0:
                failed.append(name)
            print(f"{name}: {'ok' if process.returncode == 0 else 'ERROR'} ({finished}/{len(names)})", flush=True)
            del running[name]
        time.sleep(2)
    print(f"batch: {len(names) - len(failed)}/{len(names)} built, {len(failed)} errors"
          + (f": {' '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


def cmd_lod(args) -> int:
    from simkit.lod import build_lods

    build_lods(args.surface, args.out_dir, args.scene_id)
    return 0


def cmd_regate(args) -> int:
    from simkit.regate import regate

    result = regate(Path(args.bundle), probes=args.probes)
    print(json.dumps(result, indent=1))
    return 0 if result["matches_manifest"] else 1


def cmd_rescore(args) -> int:
    from simkit.certify import write_datasheet, write_gate_certificate
    from simkit.gate import rescore

    out = scene_outputs(Path(args.scene_out_dir), None)
    manifest_path = out["bundle"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    before = manifest["tier"]
    if "s7_floor_registration" not in manifest["stages"]:
        # Bundles built before the check existed: recompute it from the source
        # mesh (ray casts only) and the bundle's own navmesh.
        if not args.scene_dir:
            raise SystemExit("this bundle predates s7_floor_registration: pass --scene-dir <source scene>")
        manifest["stages"]["s7_floor_registration"] = _floor_registration_from_source(
            Path(args.scene_dir), out["bundle"], manifest)
    qa, plumb = rescore(manifest)
    manifest["qa"], manifest["stages"]["s1_plumb"] = qa, plumb
    manifest["tier"] = "T1a" if qa["passed"] else "FAILED"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    write_gate_certificate(out["bundle"], out["certification"], out["uuid"])
    isaac_json = out["certification"] / "isaac_drop_test.json"
    isaac = json.loads(isaac_json.read_text()) if isaac_json.exists() else None
    if out["datasheet"].exists():
        write_datasheet(out["bundle"], out["datasheet"], out["uuid"], isaac=isaac)
    failed = [k for k, v in qa["checks"].items() if not v]
    print(f"[{out['id8']}] {before} -> {manifest['tier']}" + (f" ({', '.join(failed)})" if failed else ""))
    return 0


def _floor_registration_from_source(scene_dir: Path, bundle: Path, manifest: dict) -> dict:
    import numpy as np

    from simkit.frame360 import scene_frame
    from simkit.gate import floor_registration
    from simkit.geometry.navmesh import Navmesh
    from simkit.pipeline import load_mesh, scene_inputs

    inputs = scene_inputs(scene_dir)
    mesh = load_mesh(inputs["mesh"])
    frame, _ = scene_frame(np.asarray(mesh.vertices), np.asarray(mesh.triangles), inputs["cameras"])
    shipped = np.array(json.loads((bundle / "frame" / "transform.json").read_text())["sim_from_source"])
    if not np.allclose(frame.transform, shipped, atol=1e-6):
        raise SystemExit(f"{scene_dir.name}: source does not reproduce the bundle's frame")
    info = manifest["stages"]["s5_navmesh"]
    navmesh = Navmesh(np.load(bundle / "collision" / "navmesh_grid.npy"),
                      np.load(bundle / "collision" / "navmesh_ground_z.npy"),
                      np.array(info["origin_xy"]), info["cell_size_m"])
    return floor_registration(navmesh, frame.ground_points)


def cmd_verify(args) -> int:
    from simkit.verify import verify_scene_outputs

    problems = verify_scene_outputs(Path(args.scene_out_dir))
    for problem in problems:
        print(f"FAIL {problem}")
    print("PASS" if not problems else f"FAIL: {len(problems)} problems")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simkit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def build_options(p):
        p.add_argument("--out", type=Path, help="output root (default: write into each scene directory)")
        p.add_argument("--isaac-python", help="Isaac Sim interpreter (default: env simkit-isaac)")
        p.add_argument("--licence", default=DEFAULT_LICENCE, help="licence recorded in every manifest")
        p.add_argument("--probes", type=int, default=40)
        p.add_argument("--workers", type=int, default=0, help="CoACD processes (0: all cores but two)")
        p.add_argument("--force", action="store_true", help="rebuild even if outputs exist")

    p = sub.add_parser("build", help="build one scene")
    p.add_argument("scene_dir")
    build_options(p)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("batch", help="build every scene of a dataset root")
    p.add_argument("dataset_root")
    p.add_argument("--uuids", help="file with one uuid per line (default: every uuid directory)")
    p.add_argument("--jobs", type=int, default=4, help="scenes built at once")
    p.add_argument("--log-dir", help="one log per scene here (default: stdout)")
    build_options(p)
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("lod", help="LODs of a collision surface")
    p.add_argument("surface")
    p.add_argument("out_dir")
    p.add_argument("scene_id")
    p.set_defaults(func=cmd_lod)

    p = sub.add_parser("regate", help="re-run the MuJoCo probes on a built bundle")
    p.add_argument("bundle")
    p.add_argument("--probes", type=int, default=40)
    p.set_defaults(func=cmd_regate)

    p = sub.add_parser("rescore", help="re-apply the gate from recorded measurements")
    p.add_argument("scene_out_dir")
    p.add_argument("--scene-dir", help="source scene, for bundles built before s7_floor_registration")
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("verify", help="check the outputs of one scene")
    p.add_argument("scene_out_dir")
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
