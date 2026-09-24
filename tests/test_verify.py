import re

from simkit.verify import ABSOLUTE, verify_scene_outputs


def test_absolute_paths_are_caught():
    for text in ('"mesh": "/tmp/work/mesh.obj"', '"x": "/home/someone/a.ply"', "file=/mnt/data/x", "/volume1/share/a"):
        assert ABSOLUTE.search(text), text
    for text in ('"source": "mesh/model.glb"', "https://example.org/home/x", "collision/part_000.obj", "m/s"):
        assert not ABSOLUTE.search(text), text


def test_an_empty_scene_reports_everything_missing(tmp_path):
    scene = tmp_path / "0c05c38a-8b2f-4c56-b48e-05c136e287f6"
    scene.mkdir()
    problems = verify_scene_outputs(scene)
    assert any(re.match(r"missing lod/0c05c38a_high\.ply", p) for p in problems)
    assert any("datasheets" in p for p in problems)
