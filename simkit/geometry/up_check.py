"""Orientation witnesses read from the splat.

Which way is up: a walk-through sweeps the floor from close range for the whole
capture, so near the walked path the floor is the densest horizontal slab of
Gaussians; a ceiling is seen from far away, at grazing angles, and reconstructs
thinly. If that slab sits in the upper part of the walked height, the scene is
upside down.

How far off plumb: walls are vertical by construction, and a Gaussian that
models a surface is a flat disc whose shortest axis is the surface normal, so
the vertical is the direction the wall discs' normals never point along.

These are checks on a frame that comes from the ARKit alignment; nothing here
changes the frame.
"""

from __future__ import annotations

import numpy as np


def densest_layer(heights: np.ndarray, *, bins: int = 80) -> tuple[float, float]:
    """Height of the densest horizontal slab, and where it sits in the scene.

    Returns ``(height, relative_position)`` with the position in [0, 1] measured
    between the 1st and 99th percentile, which trims the floaters every splat
    has above and below the real extent.
    """
    low, high = np.percentile(heights, [1, 99])
    trimmed = heights[(heights >= low) & (heights <= high)]
    if len(trimmed) < 100 or high - low < 1e-6:
        return float("nan"), float("nan")

    histogram, edges = np.histogram(trimmed, bins=bins, range=(low, high))
    peak = int(np.argmax(histogram))
    height = float((edges[peak] + edges[peak + 1]) / 2)
    return height, float((height - low) / (high - low))


def splat_up_verdict(splat, *, min_opacity: float = 0.5, inverted_above: float = 0.6) -> dict:
    """Whether a splat already in the simulation frame is the right way up.

    ``inverted_above`` is deliberately past the midpoint: a floor sits at the
    bottom of a scene by a wide margin, so a slab found near the middle is
    ambiguous — a mezzanine, say — and should not trigger a flip on its own.
    """
    solid = splat.solid(min_opacity=min_opacity)
    if len(solid) < 1000:
        return {"decidable": False, "reason": "too few solid Gaussians to judge"}

    height, position = densest_layer(solid.means[:, 2])
    if not np.isfinite(position):
        return {"decidable": False, "reason": "no height spread"}

    return {
        "decidable": True,
        "inverted": bool(position > inverted_above),
        "densest_layer_z": height,
        "relative_position": position,
        "solid_gaussians": int(len(solid)),
    }


def floor_plane_normal(splat, *, min_opacity: float = 0.5, band: float = 0.35) -> np.ndarray | None:
    """Unit normal of the real floor, fitted to the densest slab of Gaussians.

    Fallback plumb witness for scenes with too few wall Gaussians. A floor may
    legitimately slope, so this is the weaker of the two.

    Total least squares, with iterative rejection so pallets and machinery
    standing on the floor do not drag the plane with them.
    """
    solid = splat.solid(min_opacity=min_opacity)
    if len(solid) < 1000:
        return None

    height, _ = densest_layer(solid.means[:, 2])
    if not np.isfinite(height):
        return None

    points = solid.means[np.abs(solid.means[:, 2] - height) < band]
    if len(points) < 500:
        return None

    keep = np.ones(len(points), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(6):
        selected = points[keep]
        centre = selected.mean(axis=0)
        normal = np.linalg.svd(selected - centre, full_matrices=False)[2][-1]
        distance = np.abs((points - centre) @ normal)
        keep = distance < max(0.05, 3 * np.median(distance[keep]))

    normal = normal / np.linalg.norm(normal)
    return normal if normal[2] >= 0 else -normal






# On the older OpenMVS meshes (no MoGe floor, ~500k faces) mesh walls were a poor
# plumb witness: 2.80 deg on a scene whose true tilt was 0.005 deg. On the
# current meshes they agree with the splat walls to about a degree, so the gate
# uses both (simkit.gate.plumb_measurement) and a threshold that sits above the
# witnesses' own disagreement.


def gaussian_normals(splat, *, flatness: float = 0.4) -> np.ndarray:
    """Surface normal of every disc-shaped Gaussian, in world axes.

    A Gaussian that models a surface is flattened along it: its shortest axis is
    the surface normal. Round Gaussians model volume rather than surface and
    carry no orientation worth reading, so they are dropped.
    """
    scales, quats = splat.scales, splat.quats
    order = np.argsort(scales, axis=1)
    shortest, middle = order[:, 0], order[:, 1]
    rows = np.arange(len(scales))
    flat = scales[rows, shortest] < flatness * scales[rows, middle]

    w, x, y, z = quats[flat].T
    rotation = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)

    axis = np.zeros((flat.sum(), 3))
    axis[rows[flat] - rows[flat], :] = 0.0  # placeholder, filled below
    local = np.eye(3)[shortest[flat]]
    return np.einsum("nij,nj->ni", rotation, local)


def gravity_from_walls(
    splat, *, min_opacity: float = 0.5, wall_cosine: float = 0.35, rounds: int = 5
) -> np.ndarray | None:
    """Vertical direction, from the fact that a building's walls are plumb.

    A plane fitted to the floor is only as good as the floor is flat, and this
    factory's undulates by 15 cm — enough that two slabs of it disagree by more
    than a degree. Walls are the stronger reference: vertical by construction,
    and there are hundreds of thousands of them here.

    Gravity is the direction wall normals never point along, i.e. the ``u``
    minimising the sum of ``(n · u)²``. Taking that as the smallest eigenvector
    of the normals' covariance fails in exactly this building: a long hall's
    wall area is dominated by its two side walls, so the normals cluster on one
    horizontal axis and the vertical and the *other* horizontal axis are equally
    unrepresented — the eigenvector can come back horizontal. Writing
    ``u ∝ (a, b, 1)`` instead confines the answer to the neighbourhood of the
    current vertical, which is where the true one is, and reduces the problem to
    a 2×2 solve.
    """
    solid = splat.solid(min_opacity=min_opacity)
    if len(solid) < 5000:
        return None

    normals = gaussian_normals(solid)
    if len(normals) < 2000:
        return None

    up = np.array([0.0, 0.0, 1.0])
    for _ in range(rounds):
        vertical_component = normals @ up
        wall = np.abs(vertical_component) < wall_cosine
        if wall.sum() < 1000:
            return None

        # Work in a frame whose z is the current estimate, so the small-angle
        # parametrisation stays valid as the estimate moves.
        basis = _orthonormal_basis(up)
        local = normals[wall] @ basis  # columns: two horizontals, then up
        matrix = local[:, :2].T @ local[:, :2]
        rhs = -local[:, :2].T @ local[:, 2]
        try:
            offset = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            return None

        step = basis @ np.array([offset[0], offset[1], 1.0])
        new_up = step / np.linalg.norm(step)
        moved = np.degrees(np.arccos(np.clip(new_up @ up, -1.0, 1.0)))
        up = new_up
        if moved < 0.01:
            break

    return up if up[2] >= 0 else -up


def _orthonormal_basis(up: np.ndarray) -> np.ndarray:
    """3x3 whose third column is ``up``; the first two span the horizontal."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    first = np.cross(seed, up)
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    return np.stack([first, second, up], axis=1)




def tilt_degrees(normal: np.ndarray) -> float:
    """Angle between a floor normal and the frame's vertical."""
    return float(np.degrees(np.arccos(np.clip(normal[2] / np.linalg.norm(normal), -1.0, 1.0))))


