"""The simulation frame of an OVR Maps 360 scene.

The reconstructions are already metric and gravity-aligned: the fisheye rig is
registered rigidly to the ARKit poses of the smartphone that records alongside
the Insta360, so the source frame's -Y is gravity to within the accuracy of
ARKit. Nothing about rotation is estimated here. The only rotation is the fixed
axis change from the source convention (Y down) to the simulation convention
(Z up, as MuJoCo and USD expect).

What the alignment does not fix is the height of the floor: the ARKit origin is
wherever the phone session started, so the walked floor sits anywhere between
a few decimetres and a few metres from zero. It is measured directly, by casting
a ray straight down from every camera position and taking the first surface it
meets. That is the ground the operator was standing on at that instant,
whatever height the pole was held at, and the median over the walk is the level
the scene is placed on.

The same rays give two more measurements used by the gate: the fraction of
camera positions that have ground beneath them (which way is up), and the plane
through the hit points, compared with the plane of the walked trajectory (two
independent witnesses to the floor's attitude).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Source convention: x right, y down, z forward (OpenCV / COLMAP world after the
# ARKit alignment). Simulation convention: z up. sim = (x, z, -y).
SIM_FROM_SOURCE_ROTATION = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])


@dataclass
class SceneFrame:
    """Rigid transform from the source frame into the simulation frame."""

    transform: np.ndarray  # 4x4, sim_from_source
    floor_up_m: float  # floor height along source up (-Y), before the shift
    evidence: dict = field(default_factory=dict)
    ground_points: np.ndarray | None = None  # (N, 3) sim frame: surface under each grounded camera

    def to_json(self) -> dict:
        return {
            "convention": {"up_axis": "Z", "units": "metres", "floor_at": 0.0},
            "sim_from_source": self.transform.tolist(),
            "source_up_axis": "Y",
            "source_up_sign": -1,
            "source_floor_value": self.floor_up_m,
            "evidence": self.evidence,
        }


def load_camera_positions(path: str | Path) -> np.ndarray:
    """Distinct camera centres from ``training_cameras.json``.

    The splat is trained on five perspective views per fisheye frame, all
    sharing one centre, so duplicates are collapsed: otherwise every stop would
    be weighted by the number of views cut from it.
    """
    cameras = json.loads(Path(path).read_text())
    positions = np.array([c["position"] for c in cameras], dtype=np.float64)
    return np.unique(positions.round(6), axis=0)


def _plane_normal(points: np.ndarray, *, rounds: int = 6, tolerance: float = 0.05) -> tuple[np.ndarray, int]:
    """Robust plane normal (sign pointing to sim +Z) and the inlier count."""
    keep = np.ones(len(points), dtype=bool)
    normal = np.array([0.0, 0.0, 1.0])
    for _ in range(rounds):
        subset = points[keep]
        centre = subset.mean(axis=0)
        normal = np.linalg.svd(subset - centre, full_matrices=False)[2][-1]
        residual = np.abs((points - centre) @ normal)
        keep = residual < max(tolerance, 3 * np.median(residual[keep]))
    normal = normal / np.linalg.norm(normal)
    return (normal if normal[2] >= 0 else -normal), int(keep.sum())


def _degrees_from_vertical(normal: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0))))


def scene_frame(vertices: np.ndarray, faces: np.ndarray, cameras_path: str | Path) -> tuple[SceneFrame, dict]:
    """Frame for one scene, plus the walk measurements the gate reads.

    ``vertices`` and ``faces`` are the source-frame mesh. Returns the frame and a
    dict of measurements recorded under ``s1_camera_frame`` in the manifest.
    """
    import open3d as o3d

    positions = load_camera_positions(cameras_path)
    sim_positions = positions @ SIM_FROM_SOURCE_ROTATION.T

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.asarray(vertices @ SIM_FROM_SOURCE_ROTATION.T, dtype=np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    )
    rays = np.concatenate(
        [sim_positions, np.tile([0.0, 0.0, -1.0], (len(sim_positions), 1))], axis=1
    ).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    distance = hit["t_hit"].numpy()
    grounded = np.isfinite(distance)
    if grounded.sum() < 20:
        raise ValueError(
            f"only {int(grounded.sum())} of {len(positions)} camera positions have "
            "ground beneath them: the mesh is not in the cameras' frame, or is upside down"
        )

    ground = sim_positions[grounded].copy()
    ground[:, 2] -= distance[grounded]
    floor_up = float(np.median(ground[:, 2]))

    transform = np.eye(4)
    transform[:3, :3] = SIM_FROM_SOURCE_ROTATION
    transform[2, 3] = -floor_up

    carry = distance[grounded]
    walk_normal, walk_inliers = _plane_normal(sim_positions)
    floor_normal, _ = _plane_normal(ground)
    agreement = float(np.degrees(np.arccos(np.clip(abs(walk_normal @ floor_normal), -1.0, 1.0))))

    measurements = {
        "positions": int(len(positions)),
        "grounded_positions": int(grounded.sum()),
        "grounded_fraction": float(grounded.mean()),
        # Informative only: how high the pole was held. It varies by operator
        # and within a capture, so it is recorded and never gated.
        "camera_height_m": float(np.median(carry)),
        "camera_height_p05_p95_m": [float(np.percentile(carry, 5)), float(np.percentile(carry, 95))],
        "floor_z": floor_up,
        "floor_spread_p05_p95_m": [
            float(np.percentile(ground[:, 2], 5) - floor_up),
            float(np.percentile(ground[:, 2], 95) - floor_up),
        ],
        "walk_plane_inliers": walk_inliers,
        "walk_slope_deg": _degrees_from_vertical(walk_normal),
        "floor_slope_deg": _degrees_from_vertical(floor_normal),
        "mesh_agreement_deg": agreement,
    }
    ground_sim = ground.copy()
    ground_sim[:, 2] -= floor_up
    frame = SceneFrame(
        transform=transform,
        floor_up_m=floor_up,
        ground_points=ground_sim,
        evidence={
            "rotation": "fixed axis change: source -Y (ARKit-aligned gravity) to simulation +Z",
            "floor": "median of the first surface below each camera position",
            "grounded_fraction": measurements["grounded_fraction"],
            "floor_slope_deg": measurements["floor_slope_deg"],
        },
    )
    return frame, measurements
