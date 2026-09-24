import json

import numpy as np
import pytest


def grid_floor(size=20.0, step=0.5, height_up=0.0, slope_deg=0.0):
    """A floor in the SOURCE frame (up = -Y), optionally sloped along x."""
    xs = np.arange(-size / 2, size / 2 + 1e-9, step)
    gx, gz = np.meshgrid(xs, xs)
    up = height_up + np.tan(np.radians(slope_deg)) * gx
    vertices = np.stack([gx.ravel(), -up.ravel(), gz.ravel()], axis=1)
    n = len(xs)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, i * n + j + 1, (i + 1) * n + j, (i + 1) * n + j + 1
            faces += [[a, c, b], [b, c, d]]
    return vertices, np.array(faces)


@pytest.fixture
def cameras_file(tmp_path):
    def write(positions_up, height=2.1):
        cams = [{"id": i, "img_name": f"{i:05d}_perspective_{f}", "width": 1600, "height": 1600,
                 "position": [float(x), float(-(u + height)), float(z)], "rotation": np.eye(3).tolist(),
                 "fx": 800.0, "fy": 800.0}
                for i, (x, z, u) in enumerate(positions_up) for f in (0, 1, 2, 4, 5)]
        path = tmp_path / "training_cameras.json"
        path.write_text(json.dumps(cams))
        return path
    return write
