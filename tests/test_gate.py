"""The nine checks, and that missing evidence fails closed."""

import numpy as np

from simkit.gate import CHECK_TABLE, evaluate_gate, mesh_walls_vertical

GOOD = dict(
    walk={"grounded_fraction": 1.0, "mesh_agreement_deg": 0.07, "camera_height_m": 2.1},
    up_verdict={"decidable": True, "inverted": False},
    plumb={"tilt_deg": 0.4, "reference": "walls, smaller of splat and mesh"},
    floor_registration={"median_abs_m": 0.01, "walkable_fraction": 0.6},
    layer_residual={"median_m": 0.04, "p90_m": 0.2},
    walkable_area_m2=900.0,
    physics={"probes": 40, "leak_rate": 0.0, "settle_rate": 1.0, "p95_penetration_m": 0.004},
)


def test_a_good_bundle_passes():
    qa = evaluate_gate(**GOOD)
    assert qa["passed"] and len(qa["checks"]) == 9 == len(CHECK_TABLE)
    assert set(qa["checks"]) == {key for key, _, _ in CHECK_TABLE}


def test_camera_height_is_not_a_check():
    qa = evaluate_gate(**GOOD | {"walk": GOOD["walk"] | {"camera_height_m": 0.3}})
    assert qa["passed"] and qa["measurements"]["camera_height_m"] == 0.3


def test_each_threshold_fails_on_its_own():
    cases = {
        "up_direction_verified": {"up_verdict": {"decidable": True, "inverted": True}},
        "scene_plumb_under_2deg": {"plumb": {"tilt_deg": 2.1}},
        "witnesses_agree_under_0p5deg": {"walk": GOOD["walk"] | {"mesh_agreement_deg": 0.6}},
        "floor_at_origin_under_5cm": {"floor_registration": {"median_abs_m": 0.06}},
        "alignment_residual_under_25cm": {"layer_residual": {"median_m": 0.3}},
        "walkable_area_over_5m2": {"walkable_area_m2": 4.0},
        "no_collision_leak": {"physics": GOOD["physics"] | {"leak_rate": 0.025}},
        "probes_settle": {"physics": GOOD["physics"] | {"settle_rate": 0.85}},
        "penetration_p95_under_2cm": {"physics": GOOD["physics"] | {"p95_penetration_m": 0.03}},
    }
    for key, change in cases.items():
        qa = evaluate_gate(**GOOD | change)
        assert not qa["passed"] and [k for k, v in qa["checks"].items() if not v] == [key], key


def test_missing_evidence_fails_closed():
    qa = evaluate_gate(walk={}, up_verdict={}, plumb={}, floor_registration={},
                       layer_residual={}, walkable_area_m2=0.0, physics={})
    assert not any(qa["checks"].values())


def test_mesh_walls_recover_a_known_tilt():
    rng = np.random.default_rng(0)
    tilt = np.radians(1.5)
    rotation = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    faces, vertices = [], []
    for k in range(3000):  # small vertical wall triangles facing random horizontal directions
        a = rng.uniform(0, 2 * np.pi)
        normal = np.array([np.cos(a), np.sin(a), 0.0])
        along = np.array([-np.sin(a), np.cos(a), 0.0])
        base = rng.uniform(-10, 10, 3)
        tri = np.array([base, base + along * 0.3, base + np.array([0, 0, 0.3])])
        vertices += list(tri @ rotation.T)
        faces.append([3 * k, 3 * k + 1, 3 * k + 2])
    up = mesh_walls_vertical(np.array(vertices), np.array(faces))
    assert abs(np.degrees(np.arccos(up[2])) - 1.5) < 0.05


def test_rescore_reproduces_the_gate_from_the_manifest():
    from simkit.gate import rescore

    manifest = {
        "stages": {
            "s1_camera_frame": GOOD["walk"],
            "s1_up_check_after": GOOD["up_verdict"] | {"witness": "splat within 3 m of the walk"},
            "s1_plumb": {"tilt_deg": 2.5, "splat_walls_tilt_deg": 0.4, "mesh_walls_tilt_deg": 4.6},
            "s7_floor_registration": GOOD["floor_registration"],
            "s7_layer_residual": GOOD["layer_residual"],
        },
        "qa": {"measurements": {"walkable_area_m2": GOOD["walkable_area_m2"]} | GOOD["physics"]},
    }
    qa, plumb = rescore(manifest)
    assert plumb["tilt_deg"] == 0.4  # a tilt seen by one witness only is that witness's noise
    assert qa["passed"]
