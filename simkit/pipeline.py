"""Build the simulation bundle of one OVR Maps 360 scene.

Input is a scene directory in the dataset layout; only three of its files are
read:

    mesh/model.glb            textured OpenMVS mesh, floor closed with MoGe
    gaussian_splatting.ply    3DGS trained on the perspective views
    training_cameras.json     camera centres of that training

All three share one metric, gravity-aligned frame (source up = -Y). Output:

    simulation/<id8>.sre/     manifest.json, frame/transform.json, scene.xml,
                              scene.usda, collision/, photoreal/splat.ply

The chain is: frame -> crop to the walked corridor -> navmesh -> height field
-> splat obstacles and coverage walls -> convex structure (CoACD) -> MJCF and
USD -> physics probes -> gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

MESH = Path("mesh") / "model.glb"
SPLAT = Path("gaussian_splatting.ply")
CAMERAS = Path("training_cameras.json")


def scene_inputs(scene_dir: Path) -> dict:
    paths = {"mesh": scene_dir / MESH, "splat": scene_dir / SPLAT, "cameras": scene_dir / CAMERAS}
    missing = [str(p.relative_to(scene_dir)) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{scene_dir.name}: missing {', '.join(missing)}")
    return paths


def load_mesh(path: Path):
    """The glb as an Open3D mesh, geometry only.

    Open3D's glTF reader is unreliable on large scenes; trimesh reads them and
    applies the node transforms, so it does the loading.
    """
    import open3d as o3d
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False, skip_materials=True)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(loaded.vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(loaded.faces, dtype=np.int32)),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    return mesh


def build_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
    *,
    scene_id: str | None = None,
    licence: str,
    target_triangles: int = 60_000,
    max_hulls: int = 8,
    probe_count: int = 40,
    simplify_to: int | None = 800_000,
    cell_size: float | None = None,
    walk_buffer_m: float = 12.0,
    workers: int = 0,
    verbose: bool = True,
) -> dict:
    """Build ``output_dir`` (the ``.sre`` bundle) from ``scene_dir``."""
    import open3d as o3d
    from scipy import ndimage
    from scipy.spatial import cKDTree

    from simkit.bundle import BundleLayout, export_usd_subprocess, write_manifest
    from simkit.export.heightfield import above_ground_mesh, build_heightfield, filter_navmesh_by_step
    from simkit.export.heightfield import to_mesh as heightfield_to_mesh
    from simkit.export.mjcf import write_mjcf
    from simkit.frame360 import scene_frame
    from simkit.gate import evaluate_gate, floor_registration, layer_residual, plumb_measurement
    from simkit.geometry.collision import build_collision_proxy
    from simkit.geometry.coverage_bounds import mesh_witness, uncovered_cells, wall_boxes, wall_heights
    from simkit.geometry.navmesh import Navmesh, build_navmesh, cell_size_for
    from simkit.geometry.splat_obstacles import obstacle_boxes, obstacle_cells
    from simkit.geometry.up_check import splat_up_verdict
    from simkit.io.splat_io import load_splat
    from simkit.photoreal import transform_splat, write_splat_ply
    from simkit.physics.probes import run_floor_probes, sample_floor_points

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    scene_dir = Path(scene_dir)
    inputs = scene_inputs(scene_dir)
    scene_id = scene_id or scene_dir.name[:8]
    layout = BundleLayout(Path(output_dir)).create()
    stages: dict = {}
    started = time.time()
    marks = {"last": started}

    # Every line carries the time since the previous one, so the expensive stage
    # is read off the log rather than guessed.
    def log(message: str) -> None:
        now = time.time()
        step, marks["last"] = now - marks["last"], now
        if verbose:
            print(f"[{scene_id}] +{step:6.1f}s  {message}", flush=True)

    mesh = load_mesh(inputs["mesh"])
    log(f"mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # S1 - frame: fixed axis change plus the measured floor height.
    frame, walk = scene_frame(np.asarray(mesh.vertices), np.asarray(mesh.triangles), inputs["cameras"])
    (layout.frame / "transform.json").write_text(json.dumps(frame.to_json(), indent=2))
    stages["s1_camera_frame"] = walk
    stages["s1_frame"] = {
        "frame_source": "ARKit-aligned capture",
        "up_axis": "Y",
        "up_sign": -1,
        "floor_z_source": frame.floor_up_m,
    }
    log(f"S1 frame: floor at {frame.floor_up_m:+.3f} m (source up), "
        f"{walk['grounded_fraction']:.0%} of {walk['positions']} positions grounded, "
        f"walk vs ground plane {walk['mesh_agreement_deg']:.3f} deg, "
        f"camera carried at {walk['camera_height_m']:.2f} m")

    mesh.transform(frame.transform)
    mesh.compute_vertex_normals()

    from simkit.frame360 import load_camera_positions

    positions = load_camera_positions(inputs["cameras"]) @ frame.transform[:3, :3].T + frame.transform[:3, 3]

    # The splat in the simulation frame: witness for orientation and obstacles,
    # and the bundle's photoreal layer.
    splat = transform_splat(load_splat(inputs["splat"]), frame.transform)
    log(f"splat: {len(splat):,} gaussians into the simulation frame")

    # Crop to the corridor that was actually walked. The reconstruction reaches
    # far past where the capture went; a robot cannot be evaluated where there
    # were no observations, and the convex decomposition pays for every
    # distant triangle.
    vertices_now = np.asarray(mesh.vertices)
    triangles_now = np.asarray(mesh.triangles)
    centroids = vertices_now[triangles_now].mean(axis=1)
    distance, _ = cKDTree(positions[:, :2]).query(centroids[:, :2])
    keep = distance <= walk_buffer_m
    if keep.any() and not keep.all():
        mesh.remove_triangles_by_mask(~keep)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()
    stages["s1_walk_crop"] = {
        "buffer_m": walk_buffer_m,
        "triangles_before": int(len(triangles_now)),
        "triangles_after": int(len(mesh.triangles)),
        "kept_fraction": float(keep.mean()),
    }
    log(f"S1 crop to the walked corridor (+-{walk_buffer_m:.0f} m): "
        f"{len(triangles_now):,} -> {len(mesh.triangles):,} triangles")

    # Orientation witnesses, read where the capture actually went. The whole
    # splat carries sky, canopy and distant background, and its densest slab
    # can be a roof; within 3 m of the walk the densest slab is the floor the
    # operator stood on, unless the scene is upside down.
    walked_xy, _ = cKDTree(positions[:, :2]).query(splat.means[:, :2])
    up_verdict = splat_up_verdict(splat.filter(walked_xy <= 3.0))
    plumb = plumb_measurement(splat.filter(walked_xy <= walk_buffer_m),
                              np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    stages["s1_up_check_after"] = up_verdict | {"witness": "splat within 3 m of the walk"}
    stages["s1_plumb"] = plumb
    log(f"S1 up-check: {'inverted' if up_verdict.get('inverted') else 'upright'}"
        f" (densest slab at {up_verdict.get('relative_position', float('nan')):.2f} of the walked height); "
        f"plumb {plumb['tilt_deg']:.3f} deg against the {plumb['reference']}")

    # The navmesh reads the full-resolution crop; the collision proxy reads a
    # simplified one. Walkability is decided per cell from samples spread over
    # the triangles' surface, so the grid is as dense as the cells ask, not as
    # dense as the mesh happens to be.
    if cell_size is None:
        cell_size = cell_size_for(mesh)
        log(f"S5 cell size {cell_size:.2f} m (a foot); surface sampled per cell")
    navmesh = build_navmesh(mesh, cell_size=cell_size, seed_points=frame.ground_points[:, :2])
    # The coverage walls are sized from the same full-resolution surface.
    mesh_points, mesh_normals = mesh_witness(mesh, cell_size)

    if simplify_to and len(mesh.triangles) > simplify_to:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=simplify_to)
        mesh.compute_vertex_normals()
        log(f"simplified to {len(mesh.triangles):,} triangles for collision")
    o3d.io.write_triangle_mesh(str(layout.collision / "surface.ply"), mesh)
    stages["s3_surface"] = {"source": MESH.as_posix(), "triangles": len(mesh.triangles)}

    np.save(layout.collision / "navmesh_grid.npy", navmesh.grid)
    np.save(layout.collision / "navmesh_ground_z.npy", navmesh.ground_z)
    log(f"S5 navmesh: {navmesh.area_m2:.1f} m2 walkable, elevation span {navmesh.elevation_span_m:.2f} m")

    heightfield = build_heightfield(navmesh)
    heightfield_png = heightfield.write_png(layout.collision / "ground_hfield.png")

    # Walkability re-decided against the ground under a whole stance: the
    # navmesh's step limit holds between neighbouring cells, a foot spans
    # several of them.
    navmesh, step_filter = filter_navmesh_by_step(navmesh, heightfield)

    # Obstacles the splat sees and the mesh lost (a car reconstructs as a smear
    # in the splat and near-nothing in the mesh): those cells leave the navmesh
    # and become box colliders.
    solid_means = splat.solid(min_opacity=0.5).means
    blocked = obstacle_cells(navmesh, solid_means)
    splat_boxes = []
    if blocked.any():
        splat_boxes = obstacle_boxes(navmesh, blocked)
        navmesh = Navmesh(navmesh.grid & ~blocked, navmesh.ground_z, navmesh.origin, navmesh.cell_size)
        log(f"S5 splat obstacles: {int(blocked.sum())} cells blocked, {len(splat_boxes)} boxes")
    stages["s5_splat_obstacles"] = {"blocked_cells": int(blocked.sum()), "boxes": len(splat_boxes)}

    # Containment: the height field fills ground past the mesh's coverage, and
    # nothing would stop a drifting robot from walking onto it. Each wall is as
    # tall as what the mesh or the splat saw there (a sofa, a pane the mesh lost).
    wall_grid = uncovered_cells(navmesh)
    heights = wall_heights(navmesh, wall_grid, mesh_points, mesh_normals, solid_means)
    coverage_walls = wall_boxes(navmesh, wall_grid, heights) if wall_grid.any() else []
    del mesh_points, mesh_normals
    stages["s5_coverage_walls"] = {"wall_cells": int(wall_grid.sum()), "boxes": len(coverage_walls)}
    log(f"S5 coverage walls: {int(wall_grid.sum())} cells fenced, {len(coverage_walls)} boxes")

    if navmesh.area_m2 < 5.0:
        raise ValueError(f"degenerate scene: {navmesh.area_m2:.1f} m2 walkable after the step filter")

    stages["s5_navmesh"] = navmesh.as_dict() | {"step_filter": step_filter}
    np.save(layout.collision / "navmesh_grid.npy", navmesh.grid)
    log(f"S5 step filter: {step_filter['area_before_m2']:.1f} -> {step_filter['area_after_m2']:.1f} m2 walkable")

    # S4 - structure only; the height field owns the ground.
    structure = above_ground_mesh(mesh, navmesh)
    parts, collision_stats = build_collision_proxy(
        structure, layout.collision, target_triangles=target_triangles, max_hulls=max_hulls, navmesh=navmesh,
        workers=workers,
    )
    stages["s4_collision"] = collision_stats.as_dict() | {"heightfield": heightfield.as_dict()}
    log(f"S4 collision: height field + {collision_stats.parts} convex parts")

    # Photoreal layer: the same splat, already carried into the frame.
    write_splat_ply(splat, layout.photoreal / "splat.ply")
    stages["photoreal"] = {
        "gaussians": len(splat),
        "source": SPLAT.as_posix(),
        "frame": "simulation (Z-up, metres, ground at the S1 floor)",
        "appearance": "degree-0 colour (f_dc); view-dependent SH stays in the source splat",
    }

    # S6 - exports, both from the same collision directory.
    mjcf_path = write_mjcf(
        parts, layout.root / "scene.xml", scene_name=f"sre_{scene_id}",
        heightfield=heightfield, heightfield_png=heightfield_png,
        extra_boxes=splat_boxes + coverage_walls,
    )
    usd_path = export_usd_subprocess(
        parts, layout.root / "scene.usda", f"sre_{scene_id}",
        ground_mesh=heightfield_to_mesh(heightfield, keep_mask=ndimage.binary_dilation(navmesh.grid, iterations=6)),
        extra_boxes=splat_boxes + coverage_walls,
    )
    stages["s6_export"] = {"mjcf": mjcf_path.name, "usd": usd_path.name}

    # Gate.
    points = sample_floor_points(navmesh, count=probe_count)
    report = run_floor_probes(mjcf_path, points)
    physics = report.as_dict() if len(points) else {"probes": 0}
    stages["physics_probes"] = [
        {"x": p.x, "y": p.y, "ground_z": p.ground_z, "final_z": p.final_z,
         "settled": p.settled, "fell_through": p.fell_through}
        for p in report.probes
    ]
    log(f"gate: {physics.get('probes')} probes, leak {physics.get('leak_rate', 1):.0%}, "
        f"rest p95 {physics.get('p95_penetration_m', float('nan')) * 100:.2f} cm")

    residual = layer_residual(splat, np.asarray(mesh.vertices), np.asarray(mesh.triangles), positions,
                              corridor_m=walk_buffer_m)
    stages["s7_layer_residual"] = residual
    registration = floor_registration(navmesh, frame.ground_points)
    stages["s7_floor_registration"] = registration
    log(f"layers: splat to surface median {residual['median_m'] * 100:.1f} cm; navmesh vs walked floor "
        f"{registration.get('median_abs_m', float('nan')) * 100:.1f} cm median over "
        f"{registration['on_walkable']} cameras ({registration['walkable_fraction']:.0%} of the walk walkable)")

    qa = evaluate_gate(
        walk=walk, up_verdict=up_verdict, plumb=plumb, floor_registration=registration,
        layer_residual=residual, walkable_area_m2=navmesh.area_m2, physics=physics,
    )
    manifest_path = write_manifest(
        layout, scene_id=scene_id, tier="T1a" if qa["passed"] else "FAILED",
        source={"mesh": MESH.as_posix(), "splat": SPLAT.as_posix(), "cameras": CAMERAS.as_posix(),
                "capture": "360 walk-through (Insta360 X5 dual fisheye, ARKit-aligned)"},
        stages=stages | {"build_seconds": round(time.time() - started, 1)},
        qa=qa, licence=licence,
    )
    failed = [k for k, v in qa["checks"].items() if not v]
    log(f"{'PASS' if qa['passed'] else 'FAIL ' + ', '.join(failed)} -> {manifest_path.name}")
    return json.loads(manifest_path.read_text())
