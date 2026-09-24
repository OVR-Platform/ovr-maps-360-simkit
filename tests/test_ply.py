import numpy as np
import pytest

from simkit.io.ply import read_vertices, write_vertices


def test_round_trip(tmp_path):
    data = np.zeros(1000, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4"), ("red", "u1")])
    rng = np.random.default_rng(0)
    for name in ("x", "y", "z", "opacity"):
        data[name] = rng.normal(size=1000)
    data["red"] = rng.integers(0, 255, 1000)
    back = read_vertices(write_vertices(tmp_path / "a.ply", data))
    assert back.dtype.names == data.dtype.names
    for name in data.dtype.names:
        np.testing.assert_array_equal(back[name], data[name])


def test_reads_what_open3d_writes(tmp_path):
    o3d = pytest.importorskip("open3d")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.arange(30, dtype=float).reshape(10, 3)))
    o3d.io.write_point_cloud(str(tmp_path / "b.ply"), cloud, write_ascii=False)
    back = read_vertices(tmp_path / "b.ply")
    np.testing.assert_allclose(np.stack([back["x"], back["y"], back["z"]], 1), np.asarray(cloud.points))


def test_ascii_is_refused(tmp_path):
    (tmp_path / "c.ply").write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
    with pytest.raises(ValueError, match="binary"):
        read_vertices(tmp_path / "c.ply")
