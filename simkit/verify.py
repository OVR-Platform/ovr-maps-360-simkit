"""Checks on the outputs of one scene, before they are published.

Structure (every expected file present), hygiene (no hidden files, no empty
files or directories, no absolute paths, MJCF and USD referencing only files
inside the bundle), verdict (gate passed, Isaac probe did not fall through) and
frame (the photoreal splat and the collision surface occupy the same space).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ABSOLUTE = re.compile(r"(?<![\w.:])/(?:tmp|home|mnt|root|media|Users|volume\d|srv|opt|data)/")
TEXT_SUFFIXES = {".json", ".txt", ".xml", ".usda", ".csv"}


def verify_scene_outputs(root: Path) -> list[str]:
    uuid = root.name
    id8 = uuid[:8]
    bundle = root / "simulation" / f"{id8}.sre"
    certification = root / "simulation" / "certification"
    problems: list[str] = []

    expected = [
        root / "lod" / f"{id8}_{tier}.ply" for tier in ("high", "mid", "low")
    ] + [
        root / "lod" / f"{id8}.json",
        bundle / "manifest.json", bundle / "scene.xml", bundle / "scene.usda",
        bundle / "frame" / "transform.json", bundle / "photoreal" / "splat.ply",
        bundle / "collision" / "surface.ply", bundle / "collision" / "ground_hfield.png",
        bundle / "collision" / "navmesh_grid.npy", bundle / "collision" / "navmesh_ground_z.npy",
        certification / "mujoco_physics_gate.json", certification / "mujoco_physics_gate.txt",
        certification / "isaac_drop_test.json",
        root / "datasheets" / f"{uuid}.json",
    ]
    for path in expected:
        if not path.is_file():
            problems.append(f"missing {path.relative_to(root)}")
    if not list((bundle / "collision").glob("part_*.obj")):
        problems.append("no convex parts in collision/")

    produced = [root / name for name in ("lod", "simulation", "datasheets") if (root / name).exists()]
    for top in produced:
        for path in [top, *top.rglob("*")]:
            relative = path.relative_to(root)
            if path.name.startswith(".") or path.name.startswith("_"):
                problems.append(f"hidden or temporary file {relative}")
            elif path.is_dir() and not any(path.iterdir()):
                problems.append(f"empty directory {relative}")
            elif path.is_file() and path.stat().st_size == 0:
                problems.append(f"empty file {relative}")
            elif path.is_file() and path.suffix in TEXT_SUFFIXES:
                text = path.read_text(errors="ignore")
                if ABSOLUTE.search(text):
                    problems.append(f"absolute path in {relative}")

    problems += _check_references(bundle)
    problems += _check_verdict(root, bundle, certification, uuid)
    problems += _check_frame(bundle)
    return problems


def _check_references(bundle: Path) -> list[str]:
    problems = []
    xml = bundle / "scene.xml"
    if xml.exists():
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml)
        for element in tree.iter():
            reference = element.get("file")
            if reference and not (bundle / reference).is_file():
                problems.append(f"scene.xml references {reference}, not in the bundle")
            if element.tag == "include":
                problems.append(f"scene.xml includes {element.get('file')}")
    usda = bundle / "scene.usda"
    if usda.exists():
        for reference in set(re.findall(r"@([^@]+)@", usda.read_text(errors="ignore"))):
            if not (bundle / reference).is_file():
                problems.append(f"scene.usda references {reference}, not in the bundle")
    return problems


def _check_verdict(root: Path, bundle: Path, certification: Path, uuid: str) -> list[str]:
    problems = []
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
        gate = json.loads((certification / "mujoco_physics_gate.json").read_text())
        isaac = json.loads((certification / "isaac_drop_test.json").read_text())
        sheet = json.loads((root / "datasheets" / f"{uuid}.json").read_text())
    except (OSError, ValueError) as error:
        return [f"unreadable certificate: {error}"]
    if not manifest["qa"]["passed"] or manifest["tier"] != "T1a":
        failed = [k for k, v in manifest["qa"]["checks"].items() if not v]
        problems.append(f"gate failed: {', '.join(failed)}")
    if gate["checks"] != manifest["qa"]["checks"] or sheet["qa"]["checks"] != manifest["qa"]["checks"]:
        problems.append("certificate or datasheet disagrees with the manifest")
    if gate["uuid"] != uuid or sheet["uuid"] != uuid:
        problems.append("uuid mismatch in certificate or datasheet")
    if isaac.get("fell_through", True):
        problems.append("Isaac drop test: probe fell through the floor")
    return problems


def _check_frame(bundle: Path) -> list[str]:
    """The splat and the surface must overlap: same frame, same place."""
    surface, splat = bundle / "collision" / "surface.ply", bundle / "photoreal" / "splat.ply"
    if not (surface.exists() and splat.exists()):
        return []
    import open3d as o3d
    from simkit.io.ply import read_vertices

    vertices = np.asarray(o3d.io.read_triangle_mesh(str(surface)).vertices)
    data = read_vertices(splat)
    means = np.stack([data["x"], data["y"], data["z"]], axis=1)
    solid = means[1 / (1 + np.exp(-data["opacity"])) > 0.5]
    low, high = np.percentile(solid, 1, axis=0), np.percentile(solid, 99, axis=0)
    inside = np.all((vertices >= low) & (vertices <= high), axis=1).mean()
    problems = []
    if inside < 0.9:
        problems.append(f"only {inside:.0%} of the collision surface lies inside the splat's extent")
    floor = float(np.median(vertices[:, 2]))
    if not (-1.0 < floor < 1.0) and abs(np.percentile(vertices[:, 2], 10)) > 1.0:
        problems.append(f"collision surface not near z = 0 (median z {floor:.2f} m)")
    return problems
