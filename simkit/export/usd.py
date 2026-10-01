"""USD export for Isaac Lab.

Isaac Lab consumes USD, so this is the artefact that makes a bundle drop-in for
an existing Isaac Lab stack. Collision parts are written as UsdGeom meshes with
UsdPhysics collision APIs and the convex-hull approximation, which is what a
physics engine wants for static environment geometry.

**Which engine matters.** Isaac Sim 6.0 runs Newton, not PhysX, and Newton reads
its own collision schemas. Authoring only the PhysX ones produced a bundle whose
meshes never became colliders — while a trivial `UsdGeom.Cube` in the same scene
collided correctly, because an analytic primitive needs no mesh-collision schema
at all. That asymmetry is what the defect looked like from outside, and it is
exactly what a wrong-solver schema produces.

The schema *libraries* are only present inside the Isaac runtime, so the names
are written straight into the prim's `apiSchemas` metadata instead of being
applied through a Python class. That is the same text NVIDIA's own converter
emits, and the runtime that has the schemas resolves it.

Import order note: ``pxr`` must be imported before ``coacd`` or the interpreter
segfaults, so USD export runs as its own stage, never in the same process as the
convex decomposition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_usd(
    collision_parts: list[Path],
    output_path: str | Path,
    *,
    ground_mesh: tuple[np.ndarray, np.ndarray] | None = None,
    scene_name: str = "sre_scene",
    extra_boxes: list | None = None,
    static_friction: float = 1.0,
    dynamic_friction: float = 0.9,
    restitution: float = 0.0,
) -> Path:
    """Write a USD stage with the static collision environment."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    # PhysxSchema only exists inside the Isaac runtime, not in standalone
    # usd-core, so the PhysX-specific APIs are applied when available and
    # skipped otherwise.
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)  # matches the S1 frame
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{scene_name}")
    stage.SetDefaultPrim(root.GetPrim())

    # A scene prim. NVIDIA's own MuJoCo converter authors one, and without it
    # the stage carries colliders with nothing told to simulate them.
    physics_scene = UsdPhysics.Scene.Define(stage, f"/{scene_name}/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    _add_api_schemas(physics_scene.GetPrim(), ["NewtonSceneAPI"])

    material_path = f"/{scene_name}/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(static_friction)
    physics_material.CreateDynamicFrictionAttr(dynamic_friction)
    physics_material.CreateRestitutionAttr(restitution)
    _add_api_schemas(material.GetPrim(), ["NewtonMaterialAPI"])

    collision_root = UsdGeom.Xform.Define(stage, f"/{scene_name}/Collision")

    # The ground, as a static triangle mesh collider. USD has no physics height
    # field, and without this the bundle arrives in a USD engine with no floor —
    # which a MuJoCo-side gate cannot detect, because MuJoCo reads the height
    # field natively.
    if ground_mesh is not None:
        vertices, faces = ground_mesh
        ground = UsdGeom.Mesh.Define(stage, f"/{scene_name}/Collision/ground")
        ground.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
        ground.CreateFaceVertexCountsAttr([3] * len(faces))
        ground.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        # UsdGeom.Mesh defaults to catmullClark subdivision, and PhysX does not
        # build a collider from a subdivision surface — the geometry loads, the
        # collider silently does not exist, and everything falls through.
        ground.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        ground.CreateDoubleSidedAttr(True)
        # Without an authored extent the physics parser skips the mesh: the prim
        # loads and renders, and no collider is ever created. Measured — a probe
        # over a trivial Cube collider rests correctly while the same probe over
        # this mesh falls straight through.
        _set_extent(ground, vertices)
        collision = UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        if PhysxSchema is not None:
            PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim())
            PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(ground.GetPrim())
        _add_api_schemas(ground.GetPrim(), ["NewtonCollisionAPI", "NewtonMeshCollisionAPI"])
        # "none" keeps the exact triangles: terrain must not be convexified.
        UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr(
            UsdPhysics.Tokens.none
        )
        UsdShade.MaterialBindingAPI(ground.GetPrim()).Bind(
            material, materialPurpose=UsdShade.Tokens.full
        )
        _add_api_schemas(ground.GetPrim(), ["MaterialBindingAPI"])
    for index, part in enumerate(collision_parts):
        vertices, faces = _read_obj(Path(part))
        if len(faces) == 0:
            continue
        prim_path = f"/{scene_name}/Collision/part_{index:03d}"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        # Gf.Vec3f needs Python floats; unpacking numpy scalars fails to match
        # any of its C++ overloads.
        mesh.CreatePointsAttr([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        _set_extent(mesh, vertices)

        collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        if PhysxSchema is not None:
            PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh.GetPrim())
        _add_api_schemas(mesh.GetPrim(), ["NewtonCollisionAPI", "NewtonMeshCollisionAPI"])
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_collision.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(
            material, materialPurpose=UsdShade.Tokens.full
        )
        _add_api_schemas(mesh.GetPrim(), ["MaterialBindingAPI"])

    # Splat-witnessed obstacle boxes — objects the splat sees at body height
    # where the mesh has nothing. Cube prims with exact-box collision.
    for number, box in enumerate(extra_boxes or []):
        prim_path = f"/{scene_name}/Collision/{box.get('kind', 'splat_obstacle')}_{number:03d}"
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(2.0)  # unit cube spans [-1, 1]; scale gives half-extents
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in box["centre"]]))
        xform.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in box["half_extents"]]))
        collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision.CreateCollisionEnabledAttr(True)
        _add_api_schemas(cube.GetPrim(), ["NewtonCollisionAPI"])
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(
            material, materialPurpose=UsdShade.Tokens.full
        )
        _add_api_schemas(cube.GetPrim(), ["MaterialBindingAPI"])

    _ = collision_root
    stage.GetRootLayer().Save()
    return output_path


def _add_api_schemas(prim, names: list[str]) -> None:
    """Append API schema names to a prim without needing the schema library.

    ``prim.ApplyAPI`` requires the schema to be registered in the running
    process, and Newton's are only registered inside Isaac. Writing the tokens
    directly produces identical USD, which is what matters: the file is the
    interface, not the process that wrote it.
    """
    from pxr import Sdf

    # Every bucket of the list op has to be carried over. Writing an explicit
    # list built from ``prependedItems`` alone silently dropped the schemas
    # applied before this call — PhysicsCollisionAPI among them — which is a
    # quieter version of the very bug this function exists to fix.
    existing = prim.GetMetadata("apiSchemas")
    current: list[str] = []
    if existing:
        for bucket in (existing.explicitItems, existing.prependedItems, existing.appendedItems):
            for name in bucket:
                if name not in current:
                    current.append(name)
    for name in names:
        if name not in current:
            current.append(name)
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(current))


def _set_extent(mesh, vertices: np.ndarray) -> None:
    """Author the bounding extent a physics parser needs to accept the mesh."""
    from pxr import Gf

    vertices = np.asarray(vertices)
    if len(vertices) == 0:
        # A degenerate scene (all walkable ground eaten by the step filter)
        # produced an empty ground mesh and this crashed the export with an
        # opaque numpy error — six scenes were recorded as ERROR when they
        # should have been rejected on their merits.
        raise ValueError(
            "ground mesh is empty: the scene has no walkable surface to export"
        )
    lower = vertices.min(axis=0)
    upper = np.asarray(vertices).max(axis=0)
    mesh.CreateExtentAttr(
        [Gf.Vec3f(*(float(v) for v in lower)), Gf.Vec3f(*(float(v) for v in upper))]
    )


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Minimal OBJ reader — the convex parts we write have no normals or UVs."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
            for k in range(1, len(indices) - 1):  # fan-triangulate
                faces.append([indices[0], indices[k], indices[k + 1]])
    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32).reshape(-1, 3)
