"""The navmesh glb lands in the source frame, on the ground it was read from."""

import json

import numpy as np
import pytest

from simkit.export.navmesh_glb import navmesh_levels_to_glb


def test_glb_is_in_the_source_frame(tmp_path):
    bundle = tmp_path / "abcd1234.sre"
    (bundle / "collision").mkdir(parents=True)
    (bundle / "frame").mkdir()
    grid = np.zeros((2, 3, 4), dtype=bool)
    grid[0] = True
    grid[1, 1, 1] = True
    ground = np.where(grid, 0.0, np.nan)
    ground[1, 1, 1] = 3.0
    np.save(bundle / "collision" / "navmesh_levels_grid.npy", grid)
    np.save(bundle / "collision" / "navmesh_levels_ground_z.npy", ground)
    (bundle / "manifest.json").write_text(json.dumps(
        {"stages": {"s5_navmesh_levels": {"origin_xy": [10.0, 20.0], "cell_size_m": 0.5}}}))
    floor = 1.7  # sim = (x, z, -y + floor)
    sim_from_source = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, floor], [0, 0, 0, 1]]
    (bundle / "frame" / "transform.json").write_text(json.dumps({"sim_from_source": sim_from_source}))

    summary = navmesh_levels_to_glb(bundle, tmp_path / "navmesh.glb", lift_m=0.0)
    assert summary["cells"] == 13 and summary["levels"] == 2

    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.load(tmp_path / "navmesh.glb", force="mesh", process=False)
    v = np.asarray(mesh.vertices)
    # source x = sim x, source z = sim y, source y = floor - sim z
    assert v[:, 0].min() == pytest.approx(10.0) and v[:, 0].max() == pytest.approx(12.0)
    assert v[:, 2].min() == pytest.approx(20.0) and v[:, 2].max() == pytest.approx(21.5)
    assert sorted(np.unique(np.round(v[:, 1], 4))) == pytest.approx([floor - 3.0, floor])
    assert np.allclose(mesh.face_normals[:, 1], -1.0)  # facing up, source up is -Y


def test_obj_is_welded_in_the_source_frame(tmp_path):
    from simkit.export.navmesh_obj import navmesh_levels_to_obj

    bundle = tmp_path / "abcd1234.sre"
    (bundle / "collision").mkdir(parents=True)
    (bundle / "frame").mkdir()
    grid = np.zeros((2, 3, 4), dtype=bool)
    grid[0] = True
    grid[1, 1, 1] = True
    ground = np.where(grid, 0.0, np.nan)
    ground[1, 1, 1] = 3.0
    np.save(bundle / "collision" / "navmesh_levels_grid.npy", grid)
    np.save(bundle / "collision" / "navmesh_levels_ground_z.npy", ground)
    (bundle / "manifest.json").write_text(json.dumps(
        {"stages": {"s5_navmesh_levels": {"origin_xy": [10.0, 20.0], "cell_size_m": 0.5}}}))
    floor = 1.7
    (bundle / "frame" / "transform.json").write_text(json.dumps(
        {"sim_from_source": [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, floor], [0, 0, 0, 1]]}))

    summary = navmesh_levels_to_obj(bundle, tmp_path / "navmesh.obj")
    assert summary["cells"] == 13
    assert summary["vertices"] == 4 * 5 + 4  # a 3 x 4 floor shares its 20 corners; the upper cell has its own 4
    assert summary["triangles"] == 26

    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.load(tmp_path / "navmesh.obj", force="mesh", process=False)
    v = np.asarray(mesh.vertices)
    assert sorted(np.unique(np.round(v[:, 1], 4))) == pytest.approx([floor - 3.0, floor])
    assert np.allclose(mesh.face_normals[:, 1], -1.0)
    assert mesh.area == pytest.approx(13 * 0.25)
