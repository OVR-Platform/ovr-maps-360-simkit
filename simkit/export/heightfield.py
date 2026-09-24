"""Ground as a height field.

Convex decomposition is the right tool for walls, furniture and clutter, but a
poor one for the ground: hulls approximate a large, gently varying surface badly,
and a foot then falls through the gaps between them. Measured on the mall scene,
a 48-hull global decomposition leaked 87% of foot probes straight through the
floor.

A height field represents exactly what the ground is — one elevation per cell —
and is exact for both a flat indoor floor and sloped outdoor terrain. It is also
what a walking robot actually contacts, so it is the representation that makes
the physics gate meaningful.

Vertical structure (walls, furniture) still goes through convex decomposition:
a height field cannot represent overhangs, and should not try.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class HeightField:
    """A MuJoCo/PhysX-ready elevation grid, in the S1 metric frame."""

    elevation: np.ndarray  # (H, W) float32 in [0, 1], MuJoCo's normalised form
    radius_x: float  # half-extent in metres
    radius_y: float
    elevation_z: float  # metres spanned by the [0, 1] range
    base_z: float  # metres of solid below the surface
    centre: np.ndarray  # (2,) world XY of the field centre
    z_offset: float  # world Z of elevation value 0

    def as_dict(self) -> dict:
        return {
            "grid_shape": list(self.elevation.shape),
            "radius_x_m": self.radius_x,
            "radius_y_m": self.radius_y,
            "elevation_span_m": self.elevation_z,
            "centre_xy": self.centre.tolist(),
            "z_offset_m": self.z_offset,
        }

    def write_png(self, path: str | Path) -> Path:
        """MuJoCo reads height fields from a 16-bit greyscale PNG.

        The image is flipped vertically: MuJoCo indexes height field rows from
        -Y upwards while a PNG stores its first row at the top. Measured on the
        mall scene, the flip takes the median foot rest error from 8.8 cm to
        **0.1 cm** (p90 18.6 cm to 0.4 cm) — the terrain was otherwise mirrored
        about the X axis and every probe landed on the wrong height.
        """
        import cv2

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.flipud(np.clip(self.elevation, 0.0, 1.0)) * 65535.0
        cv2.imwrite(str(path), image.astype(np.uint16))
        return path


def build_heightfield(navmesh, *, base_thickness: float = 1.0, fill_value: float | None = None) -> HeightField:
    """Turn per-cell navmesh ground heights into a height field.

    Cells with no observed ground are filled with the scene's lowest ground
    rather than left as holes: an unobserved cell is not a pit, and leaving it
    empty would create exactly the fall-through the gate is meant to catch.
    """
    ground = np.array(navmesh.ground_z, dtype=np.float64)
    known = np.isfinite(ground)
    if not known.any():
        raise ValueError("navmesh has no ground heights to build a height field from")

    if fill_value is not None:
        filled = np.where(known, ground, fill_value)
    else:
        # Nearest observed ground, not the global minimum. MuJoCo interpolates
        # between cells, so filling a hole with the scene's lowest point drags
        # the terrain down around it: on an outdoor scene with 0.79 m of
        # elevation this put the p95 foot rest error at 36 cm. Nearest-neighbour
        # fill keeps the surface locally continuous, which is also the honest
        # assumption — an unobserved cell most resembles the ground beside it.
        from scipy import ndimage

        indices = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
        filled = ground[tuple(indices)]

    z_min = float(filled.min())
    z_max = float(filled.max())
    span = max(z_max - z_min, 1e-3)  # MuJoCo rejects a zero elevation range
    elevation = ((filled - z_min) / span).astype(np.float32)

    height, width = elevation.shape
    cell = navmesh.cell_size
    centre = navmesh.origin + np.array([width, height]) * cell / 2.0

    return HeightField(
        elevation=elevation,
        radius_x=width * cell / 2.0,
        radius_y=height * cell / 2.0,
        elevation_z=span,
        base_z=base_thickness,
        centre=centre,
        z_offset=z_min,
    )


def above_ground_mesh(mesh, navmesh, *, margin: float = 0.10, reach_m: float | None = 3.0):
    """Keep only the surface that stands above the ground, within reach.

    What the height field already represents must not also be convex-decomposed:
    duplicate ground geometry produces contact fighting, and it wastes the hull
    budget on the one surface that does not need it.

    ``reach_m`` drops structure the robot could never touch. On the 360 corpus
    that is most of it — one scene offered 112 m² of walkable ground inside a
    corridor carrying 310,000 triangles of hedge, lawn and distant facade, and
    decomposing all of it exhausted a forty-minute budget without finishing. A
    collision proxy exists to be collided with; geometry beyond arm's length of
    anywhere the robot can stand is scenery, and the splat already draws it.
    """
    import open3d as o3d

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if len(triangles) == 0:
        return mesh

    if reach_m is not None and navmesh.grid.any():
        from scipy import ndimage

        cells = max(1, int(round(reach_m / navmesh.cell_size)))
        near = ndimage.binary_dilation(navmesh.grid, iterations=cells)
        vertex_columns = np.clip(
            ((vertices[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(int),
            0, near.shape[1] - 1)
        vertex_rows = np.clip(
            ((vertices[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(int),
            0, near.shape[0] - 1)
        reachable = near[vertex_rows, vertex_columns]
        keep_triangles = reachable[triangles].any(axis=1)
        if keep_triangles.any() and not keep_triangles.all():
            mesh = mesh.select_by_index(np.unique(triangles[keep_triangles]).tolist())
            mesh.compute_vertex_normals()
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)

    columns = np.clip(
        ((vertices[:, 0] - navmesh.origin[0]) / navmesh.cell_size).astype(int),
        0,
        navmesh.ground_z.shape[1] - 1,
    )
    rows = np.clip(
        ((vertices[:, 1] - navmesh.origin[1]) / navmesh.cell_size).astype(int),
        0,
        navmesh.ground_z.shape[0] - 1,
    )
    cell_ground = navmesh.ground_z[rows, columns]
    floor_value = np.nanmin(navmesh.ground_z) if np.isfinite(navmesh.ground_z).any() else 0.0
    cell_ground = np.where(np.isfinite(cell_ground), cell_ground, floor_value)

    keep_vertex = vertices[:, 2] > cell_ground + margin
    keep_triangle = keep_vertex[triangles].any(axis=1)

    selected = triangles[keep_triangle]
    if len(selected) == 0:
        return o3d.geometry.TriangleMesh()
    used, remapped = np.unique(selected, return_inverse=True)
    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices[used]),
        o3d.utility.Vector3iVector(remapped.reshape(-1, 3)),
    )
    result.compute_vertex_normals()
    return result


def filter_navmesh_by_step(navmesh, heightfield, *, max_step_m: float = 0.06, radius_m: float = 0.30):
    """Drop walkable cells whose *collision surface* presents a step under a foot.

    Walkability is decided from per-cell ground heights with a climbable-step
    constraint between *neighbouring* cells. That constraint says nothing about
    the range across a whole stance, and a stance spans several cells: ground
    can satisfy it everywhere and still put 6 cm of step under one foot.

    The steps are in the reconstruction, not in the fill. Replacing the
    nearest-neighbour fill with a smooth diffusion changes the median step from
    3.51 cm to 3.65 cm and the stand-test result from 11 of 30 to 13 — the known
    cells are preserved exactly either way, and they dominate.

    Measured on fabbrica with a passive G1: cells where the robot fell had a
    6.2 cm median step within 30 cm of the stance, against 2.8 cm for cells
    where it stood, and a 4.5 cm threshold separates the two populations 24
    times out of 30. Filtering at 6 cm keeps 77% of the walkable area and takes
    the stand-test pass rate from 37% to 70%; tightening further costs area
    without buying anything.

    The radius is the stance, not the foot: what topples the robot is a step
    anywhere its feet might be placed, not one directly underneath it.
    """
    from scipy import ndimage

    from simkit.geometry.navmesh import Navmesh

    # World-frame heights from MuJoCo's normalised form, in the navmesh's row order.
    surface = heightfield.elevation * heightfield.elevation_z + heightfield.z_offset
    cells = max(1, int(round(radius_m / navmesh.cell_size)))
    size = 2 * cells + 1
    span = ndimage.maximum_filter(surface, size=size) - ndimage.minimum_filter(surface, size=size)
    span = span[: navmesh.grid.shape[0], : navmesh.grid.shape[1]]

    grid = navmesh.grid & (span <= max_step_m)
    filtered = Navmesh(grid, navmesh.ground_z, navmesh.origin, navmesh.cell_size)
    return filtered, {
        "max_step_m": max_step_m,
        "radius_m": radius_m,
        "area_before_m2": navmesh.area_m2,
        "area_after_m2": filtered.area_m2,
    }


def to_mesh(heightfield, *, upsample: int = 4, keep_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate a height field into a triangle mesh, in world coordinates.

    USD has no physics height-field primitive, so engines that consume USD need
    the ground as geometry. PhysX accepts a static triangle mesh collider, which
    is exact for terrain and needs no convex approximation.

    Not having this is how a bundle can pass its gate in MuJoCo — which reads the
    height field natively — while arriving in Isaac with no floor at all.

    ``upsample`` matters as much as existing: at the navmesh's native 0.6 m
    facets every policy evaluated on the USD path fell within ~8 m while the
    same checkpoint completed 40 m routes on MuJoCo's bilinear height field —
    and the same policies walk Isaac's own procedural terrains, which are
    ~0.1 m meshes. Bilinear upsampling reproduces exactly the surface MuJoCo
    collides with, at a facet size a foot can span. ``keep_mask`` (bool, at
    native height-field resolution) crops the fine mesh to where a robot can
    actually go, so the vertex count stays sane.
    """
    from scipy import ndimage

    elevation = np.asarray(heightfield.elevation, dtype=np.float64)
    if upsample > 1:
        elevation = ndimage.zoom(elevation, upsample, order=1)  # bilinear
        if keep_mask is not None:
            keep_mask = ndimage.zoom(
                np.asarray(keep_mask, dtype=np.float64), upsample, order=0) > 0.5
    rows, columns = elevation.shape
    z = elevation * heightfield.elevation_z + heightfield.z_offset

    xs = np.linspace(
        heightfield.centre[0] - heightfield.radius_x,
        heightfield.centre[0] + heightfield.radius_x,
        columns,
    )
    ys = np.linspace(
        heightfield.centre[1] - heightfield.radius_y,
        heightfield.centre[1] + heightfield.radius_y,
        rows,
    )
    grid_x, grid_y = np.meshgrid(xs, ys)
    vertices = np.stack([grid_x.ravel(), grid_y.ravel(), z.ravel()], axis=1)

    if keep_mask is not None and keep_mask.shape == z.shape and not keep_mask.all():
        # Drop quads entirely outside the mask, then compact the vertex array.
        quad_keep = (keep_mask[:-1, :-1] | keep_mask[:-1, 1:]
                     | keep_mask[1:, :-1] | keep_mask[1:, 1:])
    else:
        quad_keep = None

    index = np.arange(rows * columns).reshape(rows, columns)
    top_left = index[:-1, :-1].ravel()
    top_right = index[:-1, 1:].ravel()
    bottom_left = index[1:, :-1].ravel()
    bottom_right = index[1:, 1:].ravel()
    # Wound so the face normals point up. The obvious order — top_left,
    # bottom_left, bottom_right — gives a cross product with a negative z, and a
    # ground whose collider faces downwards lets everything fall through it.
    # MuJoCo never sees this: it reads the height field natively, so the gate
    # stayed green for every bundle while the USD floor was inside out.
    faces = np.concatenate(
        [
            np.stack([top_left, bottom_right, bottom_left], axis=1),
            np.stack([top_left, top_right, bottom_right], axis=1),
        ]
    ).astype(np.int32)
    if quad_keep is not None:
        keep_flat = np.concatenate([quad_keep.ravel(), quad_keep.ravel()])
        faces = faces[keep_flat]
        used = np.unique(faces)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        vertices = vertices[used]
        faces = remap[faces].astype(np.int32)
    return vertices, faces
