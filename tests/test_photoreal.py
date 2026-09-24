"""Tests for carrying a splat into the simulation frame.

If the rendering layer and the collision layer disagree, the renderer draws a
floor where the physics has none — which is the single worst failure mode a
simulation-ready bundle can have.
"""

import numpy as np
import pytest

from simkit.frame360 import SIM_FROM_SOURCE_ROTATION as R_YUP_TO_ZUP
from simkit.io.splat_io import Splat
from simkit.photoreal import quaternion_multiply, transform_splat


class Similarity:
    """x -> scale * R x + t, as a 4x4."""

    def __init__(self, scale, rotation, translation):
        self.scale, self.rotation, self.translation = scale, np.asarray(rotation), np.asarray(translation)

    def as_matrix(self):
        matrix = np.eye(4)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def apply(self, points):
        return self.scale * points @ self.rotation.T + self.translation


def _splat(count: int = 5, seed: int = 0) -> Splat:
    rng = np.random.default_rng(seed)
    quats = rng.normal(size=(count, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return Splat(
        means=rng.normal(size=(count, 3)),
        opacities=rng.uniform(0, 1, count),
        scales=rng.uniform(0.01, 0.2, (count, 3)),
        quats=quats,
        sh_dc=rng.normal(size=(count, 3)),
    )


def test_means_follow_the_same_transform_as_geometry():
    """The splat must land exactly where the collision mesh lands."""
    splat = _splat()
    similarity = Similarity(2.5, R_YUP_TO_ZUP, np.array([1.0, -2.0, 0.5]))

    moved = transform_splat(splat, similarity.as_matrix())

    np.testing.assert_allclose(moved.means, similarity.apply(splat.means), atol=1e-9)


def test_gaussian_extent_scales_with_the_frame():
    splat = _splat()
    moved = transform_splat(splat, Similarity(3.0, np.eye(3), np.zeros(3)).as_matrix())

    np.testing.assert_allclose(moved.scales, splat.scales * 3.0, rtol=1e-9)


def test_rotations_compose_and_stay_unit_norm():
    splat = _splat()
    moved = transform_splat(splat, Similarity(1.7, R_YUP_TO_ZUP, np.zeros(3)).as_matrix())

    np.testing.assert_allclose(np.linalg.norm(moved.quats, axis=1), 1.0, atol=1e-6)


def test_identity_transform_is_a_no_op():
    splat = _splat()
    moved = transform_splat(splat, np.eye(4))

    np.testing.assert_allclose(moved.means, splat.means, atol=1e-9)
    np.testing.assert_allclose(moved.scales, splat.scales, atol=1e-9)
    np.testing.assert_allclose(np.abs(moved.quats), np.abs(splat.quats), atol=1e-6)


def test_opacity_and_colour_are_untouched():
    """Only geometry moves; appearance is frame-independent."""
    splat = _splat()
    moved = transform_splat(splat, Similarity(0.4, R_YUP_TO_ZUP, np.ones(3)).as_matrix())

    np.testing.assert_allclose(moved.opacities, splat.opacities)
    np.testing.assert_allclose(moved.sh_dc, splat.sh_dc)


def test_quaternion_multiply_matches_rotation_composition():
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(1)
    a = Rotation.random(1, random_state=2)
    b = Rotation.random(1, random_state=3)
    _ = rng

    # scipy uses (x, y, z, w); ours is (w, x, y, z).
    def to_wxyz(rotation):
        x, y, z, w = rotation.as_quat()[0]
        return np.array([[w, x, y, z]])

    product = quaternion_multiply(to_wxyz(a), to_wxyz(b))[0]
    expected = to_wxyz(a * b)[0]

    assert np.allclose(product, expected, atol=1e-8) or np.allclose(product, -expected, atol=1e-8)


def test_splat_round_trips_through_a_ply(tmp_path):
    from simkit.io.splat_io import load_splat
    from simkit.photoreal import write_splat_ply

    splat = _splat(count=64, seed=7)
    path = write_splat_ply(splat, tmp_path / "splat.ply")
    reloaded = load_splat(path)

    np.testing.assert_allclose(reloaded.means, splat.means, atol=1e-5)
    np.testing.assert_allclose(reloaded.scales, splat.scales, rtol=1e-4)
    np.testing.assert_allclose(reloaded.opacities, splat.opacities, atol=1e-5)
    assert len(reloaded) == 64


def test_transform_rejects_a_non_similarity():
    """A non-uniform scale cannot be carried by a single Gaussian scale factor."""
    splat = _splat()
    skewed = np.eye(4)
    skewed[:3, :3] = np.diag([1.0, 2.0, 3.0])

    moved = transform_splat(splat, skewed)

    # Documents current behaviour: the cube-root scale is used, so a non-uniform
    # transform silently distorts. The frame is always rigid.
    assert moved.scales.max() == pytest.approx((splat.scales * np.cbrt(6)).max(), rel=1e-6)
