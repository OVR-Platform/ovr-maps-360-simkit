"""Reading 3D Gaussian Splatting PLY files.

The splat is the source of truth for geometry: collision and appearance must be
extracted from the same representation, or the robot's foot ends up penetrating
a floor that the renderer draws in a different place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Splat:
    """A Gaussian splat, in whatever frame the PLY was written in."""

    means: np.ndarray  # (N, 3)
    opacities: np.ndarray  # (N,) post-sigmoid, in [0, 1]
    scales: np.ndarray  # (N, 3) metres (post-exp)
    quats: np.ndarray  # (N, 4) (w, x, y, z), normalised
    sh_dc: np.ndarray  # (N, 3) degree-0 spherical harmonics
    sh_rest: np.ndarray | None = None  # (N, K, 3) higher-order SH, if present

    def __len__(self) -> int:
        return len(self.means)

    def filter(self, mask: np.ndarray) -> "Splat":
        return Splat(
            means=self.means[mask],
            opacities=self.opacities[mask],
            scales=self.scales[mask],
            quats=self.quats[mask],
            sh_dc=self.sh_dc[mask],
            sh_rest=self.sh_rest[mask] if self.sh_rest is not None else None,
        )

    @property
    def sh_degree(self) -> int:
        """Spherical-harmonic degree available, 0 if only the DC term."""
        if self.sh_rest is None or self.sh_rest.shape[1] == 0:
            return 0
        return int(round(np.sqrt(self.sh_rest.shape[1] + 1) - 1))

    def solid(self, min_opacity: float = 0.3, max_scale_m: float = 1.0) -> "Splat":
        """Keep Gaussians that plausibly represent real surface.

        Transparent or huge Gaussians are the splat's way of painting distant
        background and sky; treating them as geometry produces phantom collision
        surfaces.
        """
        keep = (self.opacities >= min_opacity) & (self.scales.max(axis=1) <= max_scale_m)
        return self.filter(keep)


def load_splat(ply_path: str | Path) -> Splat:
    """Load an INRIA-convention 3DGS PLY (raw, pre-activation, as written)."""
    from simkit.io.ply import read_vertices

    vertex = read_vertices(ply_path)
    names = set(vertex.dtype.names)

    def column(name: str) -> np.ndarray:
        return np.asarray(vertex[name], dtype=np.float64)

    means = np.stack([column("x"), column("y"), column("z")], axis=1)

    # Stored pre-activation: opacity through a sigmoid, scales through exp.
    opacities = 1.0 / (1.0 + np.exp(-column("opacity"))) if "opacity" in names else np.ones(len(means))
    scale_names = sorted(n for n in names if n.startswith("scale_"))
    scales = (
        np.exp(np.stack([column(n) for n in scale_names], axis=1))
        if scale_names
        else np.full((len(means), 3), 0.01)
    )
    rot_names = sorted(n for n in names if n.startswith("rot_"))
    quats = (
        np.stack([column(n) for n in rot_names], axis=1)
        if rot_names
        else np.tile([1.0, 0.0, 0.0, 0.0], (len(means), 1))
    )
    quats = quats / np.clip(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12, None)
    sh_dc = np.stack([column(f"f_dc_{i}") for i in range(3)], axis=1) if "f_dc_0" in names else np.zeros((len(means), 3))

    # Higher-order spherical harmonics carry the view-dependent appearance.
    # Dropping them renders a scene flat and washed out — visibly so on a splat
    # trained at degree 3, which is what these reconstructions ship.
    rest_names = sorted(
        (n for n in names if n.startswith("f_rest_")), key=lambda n: int(n.split("_")[-1])
    )
    sh_rest = None
    if rest_names:
        # Stored channel-major: all red coefficients, then green, then blue.
        flat = np.stack([column(n) for n in rest_names], axis=1)
        per_channel = flat.shape[1] // 3
        sh_rest = flat.reshape(len(means), 3, per_channel).transpose(0, 2, 1)

    return Splat(
        means=means, opacities=opacities, scales=scales, quats=quats,
        sh_dc=sh_dc, sh_rest=sh_rest,
    )
