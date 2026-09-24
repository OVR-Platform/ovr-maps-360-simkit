"""Move the Gaussian splat into the simulation frame.

The bundle's rendering layer has to live in the same frame as its collision
layer, or the renderer draws a floor where the physics has none. Since S1 is a
similarity — rotation, uniform scale, translation — a splat can be carried into
it exactly: means transform as points, scales by the scale factor, and rotations
compose with the rotation. No retraining, no resampling.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from an orthonormal 3x3, via the numerically stable branch."""
    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product of (w, x, y, z) quaternions, broadcasting over rows."""
    left = np.atleast_2d(left)
    right = np.atleast_2d(right)
    w1, x1, y1, z1 = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    w2, x2, y2, z2 = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def transform_splat(splat, transform: np.ndarray):
    """Carry a splat through a similarity transform (4x4 with uniform scale)."""
    scale = float(np.cbrt(abs(np.linalg.det(transform[:3, :3]))))
    rotation = transform[:3, :3] / scale
    translation = transform[:3, 3]

    from simkit.io.splat_io import Splat

    return Splat(
        means=splat.means @ rotation.T * scale + translation,
        opacities=splat.opacities.copy(),
        scales=splat.scales * scale,  # uniform scale acts on the Gaussian extent
        quats=quaternion_multiply(
            np.tile(_quaternion_from_matrix(rotation), (len(splat.means), 1)), splat.quats
        ),
        sh_dc=splat.sh_dc.copy(),
    )


def write_splat_ply(splat, path: str | Path) -> Path:
    """Write an INRIA-convention 3DGS PLY (values stored pre-activation)."""
    from simkit.io.ply import write_vertices

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = len(splat.means)
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    fields += [(f"f_dc_{i}", "f4") for i in range(3)]
    fields += [("opacity", "f4")]
    fields += [(f"scale_{i}", "f4") for i in range(3)]
    fields += [(f"rot_{i}", "f4") for i in range(4)]

    data = np.zeros(count, dtype=fields)
    data["x"], data["y"], data["z"] = splat.means.T
    for i in range(3):
        data[f"f_dc_{i}"] = splat.sh_dc[:, i]
    clipped = np.clip(splat.opacities, 1e-6, 1 - 1e-6)
    data["opacity"] = np.log(clipped / (1 - clipped))  # inverse sigmoid
    for i in range(3):
        data[f"scale_{i}"] = np.log(np.maximum(splat.scales[:, i], 1e-9))
    for i in range(4):
        data[f"rot_{i}"] = splat.quats[:, i]

    write_vertices(path, data)
    return path


def add_photoreal_layer(splat_path: str | Path, transform: np.ndarray, output_path: str | Path) -> dict:
    """Load, transform into the simulation frame, and write the bundle's splat."""
    from simkit.io.splat_io import load_splat

    splat = load_splat(splat_path)
    moved = transform_splat(splat, transform)
    write_splat_ply(moved, output_path)
    return {
        "gaussians": len(moved),
        "source": str(splat_path),
        "frame": "simulation (Z-up, metres, ground at the S1 floor)",
    }
