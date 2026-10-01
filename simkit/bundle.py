"""The `.sre/` bundle (simulation-ready environment): layout, manifest, USD stage.

One bundle, many exports. USD and MJCF are both generated from the same
collision directory so they can never describe different worlds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BUNDLE_VERSION = "0.1.0"


@dataclass
class BundleLayout:
    """Directory layout of a `.sre/` bundle."""

    root: Path

    @property
    def collision(self) -> Path:
        return self.root / "collision"

    @property
    def frame(self) -> Path:
        return self.root / "frame"

    @property
    def photoreal(self) -> Path:
        return self.root / "photoreal"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def create(self) -> "BundleLayout":
        for directory in (self.collision, self.frame, self.photoreal):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def write_manifest(
    layout: BundleLayout,
    *,
    scene_id: str,
    tier: str,
    source: dict,
    stages: dict,
    qa: dict,
    licence: str,
) -> Path:
    """Write the manifest: provenance, what ran, and what the QA gate measured."""
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "scene_id": scene_id,
        "tier": tier,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "licence": licence,
        "source": source,
        "stages": stages,
        "qa": qa,
    }
    layout.manifest.write_text(json.dumps(manifest, indent=2, default=_json_safe))
    return layout.manifest


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def export_usd_subprocess(
    collision_parts: list[Path],
    output_path: Path,
    scene_name: str,
    static_meshes: dict | None = None,
    extra_boxes: list | None = None,
) -> Path:
    """Run the USD export in a fresh interpreter.

    ``coacd`` and ``pxr`` cannot coexist in one process (segfault, see
    README.md), and the build pipeline needs both. Isolating the USD stage is
    simpler and more robust than trying to control import order across modules.

    The part list travels through a file, not the command line: a scene with a
    few hundred convex parts overruns the argument limit outright.
    """
    output_path = Path(output_path)
    listing = output_path.with_suffix(".parts.json")
    listing.write_text(json.dumps({
        "parts": [str(p) for p in collision_parts],
        "boxes": extra_boxes or [],
    }))

    mesh_paths = {name: output_path.with_suffix(f".{name}.npz") for name in (static_meshes or {})}
    for name, (vertices, faces) in (static_meshes or {}).items():
        np.savez(mesh_paths[name], vertices=vertices, faces=faces)

    script = (
        "import sys, json, numpy as np; sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from simkit.export.usd import write_usd\n"
        "payload = json.loads(Path(%r).read_text())\n"
        "parts = [Path(p) for p in payload['parts']]\n"
        "meshes = {n: tuple(np.load(p)[k] for k in ('vertices', 'faces')) for n, p in %r.items()}\n"
        "write_usd(parts, %r, static_meshes=meshes, scene_name=%r, extra_boxes=payload['boxes'])\n"
    ) % (
        str(Path(__file__).resolve().parents[1]),
        str(listing),
        {name: str(path) for name, path in mesh_paths.items()},
        str(output_path),
        scene_name,
    )
    try:
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        if result.returncode != 0:
            # Re-raise with the child's stderr attached: capture_output alone
            # swallows it, and a USD export failure then reads as a bare
            # non-zero exit — four scenes went ERROR before anyone saw why.
            raise RuntimeError(f"USD export subprocess failed:\n{result.stderr[-2000:]}")
    finally:
        listing.unlink(missing_ok=True)
        for path in mesh_paths.values():
            path.unlink(missing_ok=True)
    return output_path
