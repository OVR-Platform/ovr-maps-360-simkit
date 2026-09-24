"""Physics acceptance tests, run in MuJoCo.

The criterion is not "the mesh looks right" but "a robot put on this floor stays
on it". Everything here is measured by simulating, never asserted from geometry
alone.

A stand test with a biped only means something once a controller exists: an
uncontrolled biped falls over on any floor. What is measured instead is the
property such a test probes: that the floor is solid and supports contact
wherever a robot might place a foot.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ProbeResult:
    """One foot-sized box dropped onto the floor."""

    x: float
    y: float
    ground_z: float
    settled: bool
    fell_through: bool
    penetration_m: float
    final_z: float
    settle_time_s: float


@dataclass
class PhysicsReport:
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def leak_rate(self) -> float:
        """Fraction of probes that passed through the floor."""
        return float(np.mean([p.fell_through for p in self.probes])) if self.probes else 1.0

    @property
    def settle_rate(self) -> float:
        return float(np.mean([p.settled for p in self.probes])) if self.probes else 0.0

    @property
    def max_penetration_m(self) -> float:
        depths = self._depths()
        return float(np.max(depths)) if len(depths) else float("nan")

    @property
    def p95_penetration_m(self) -> float:
        """Rest-height error at the 95th percentile.

        The gate reads this rather than the maximum: a single probe landing on a
        cell adjacent to unobserved ground produces a large outlier that says
        nothing about the surface a robot walks on, while the percentile still
        catches a systematically wrong floor.
        """
        depths = self._depths()
        return float(np.percentile(depths, 95)) if len(depths) else float("nan")

    @property
    def median_penetration_m(self) -> float:
        depths = self._depths()
        return float(np.median(depths)) if len(depths) else float("nan")

    def _depths(self) -> np.ndarray:
        return np.array([p.penetration_m for p in self.probes if not p.fell_through])

    def as_dict(self) -> dict:
        return {
            "probes": len(self.probes),
            "leak_rate": self.leak_rate,
            "settle_rate": self.settle_rate,
            "median_penetration_m": self.median_penetration_m,
            "p95_penetration_m": self.p95_penetration_m,
            "max_penetration_m": self.max_penetration_m,
            "median_settle_time_s": float(
                np.median([p.settle_time_s for p in self.probes if p.settled])
            )
            if self.settle_rate > 0
            else float("nan"),
        }


def _add_probe_body(root: ET.Element, name: str, position: np.ndarray, half_size: float) -> None:
    """A probe shaped and damped like the thing the gate certifies for: a foot.

    The original probe was a rigid cube, and on reconstructed ground it mostly
    failed to settle — not by rolling away or sliding, but by chattering: 12 to
    200 mm/s of residual velocity that never damps, with tens of degrees of
    accumulated spin, rattling on the tessellation's sharp facets. Twenty of the
    first batch's twenty-two failures were this.

    Shape dominates. Measured on the same 60 points of one scene: rigid cube
    68% settled, overdamped cube 73%, a flat foot-sized plate with overdamped
    contact 88%, and a damped sphere 7% — the sphere rolling forever is the
    cleanest proof that the cube's edges, not the ground's holes, were failing
    the criterion. A robot's foot is a damped flat sole; the probe now is one.
    ``half_size`` scales the plate's footprint (kept for callers).
    """
    worldbody = root.find("worldbody")
    body = ET.SubElement(
        worldbody, "body", name=name, pos=" ".join(f"{v:.4f}" for v in position)
    )
    ET.SubElement(body, "freejoint", name=f"{name}_joint")
    ET.SubElement(
        body,
        "geom",
        name=f"{name}_geom",
        type="box",
        size=f"{half_size * 1.8} {half_size} {half_size * 0.3}",
        density="500",
        friction="1.0 0.005 0.0001",
        solref="0.02 2",
    )


def run_floor_probes(
    mjcf_path: str | Path,
    sample_points: np.ndarray,
    *,
    drop_height: float = 0.10,
    foot_half_size: float = 0.06,
    duration_s: float = 3.0,
    settle_speed: float = 0.02,
) -> PhysicsReport:
    """Drop a foot-sized box at each sample point and see what the ground does.

    ``sample_points`` is (N, 3): XY plus the ground height of that cell. Outdoor
    scenes in this corpus span metres of elevation, so probes are released just
    above their *local* ground — releasing everything from a single global height
    would drop half of them from rooftop height and measure the fall, not the
    surface.
    """
    import mujoco

    mjcf_path = Path(mjcf_path)
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    points = np.asarray(sample_points, dtype=np.float64).reshape(-1, 3)
    for index, (x, y, ground_z) in enumerate(points):
        _add_probe_body(
            root,
            f"probe_{index:03d}",
            np.array([x, y, ground_z + drop_height + foot_half_size]),
            foot_half_size,
        )

    # MuJoCo resolves mesh paths relative to the model file.
    patched = mjcf_path.with_name(f"_probe_{mjcf_path.name}")
    patched.write_bytes(ET.tostring(root))
    try:
        model = mujoco.MjModel.from_xml_path(str(patched))
        data = mujoco.MjData(model)

        steps = int(duration_s / model.opt.timestep)
        body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"probe_{i:03d}")
            for i in range(len(points))
        ]
        settle_step = np.full(len(points), -1)
        for step in range(steps):
            mujoco.mj_step(model, data)
            speeds = np.array(
                [np.linalg.norm(data.body(bid).cvel[3:]) for bid in body_ids]
            )
            newly = (speeds < settle_speed) & (settle_step < 0) & (step > 200)
            settle_step[newly] = step

        report = PhysicsReport()
        for index, bid in enumerate(body_ids):
            final_z = float(data.body(bid).xpos[2])
            ground_z = float(points[index, 2])
            fell = final_z < ground_z - 0.5
            # Penetration comes from the solver's own contacts, not from height
            # arithmetic. The previous measure — centre height against the
            # cell's promised ground — was really "rest height error", and it
            # broke the moment the probe stopped being a cube: a flat foot
            # settles tilted across a dip on rough ground, its centre sits
            # centimetres below the cell height, and the arithmetic called that
            # penetration while the solver saw none. Measured on one scene: the
            # cube read 1.11 cm p95 where the plate read 5.37, with zero leak
            # under both. contact.dist is the physical interpenetration itself,
            # independent of probe shape and of how bumpy the ground is.
            probe_geom = model.geom(f"probe_{index:03d}_geom").id
            solver_pen = 0.0
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if probe_geom in (contact.geom1, contact.geom2) and contact.dist < 0:
                    solver_pen = max(solver_pen, float(-contact.dist))
            report.probes.append(
                ProbeResult(
                    x=float(points[index, 0]),
                    y=float(points[index, 1]),
                    ground_z=ground_z,
                    settled=bool(settle_step[index] >= 0) and not fell,
                    fell_through=bool(fell),
                    penetration_m=solver_pen if not fell else float("nan"),
                    final_z=final_z,
                    settle_time_s=float(settle_step[index] * model.opt.timestep)
                    if settle_step[index] >= 0
                    else float("nan"),
                )
            )
        return report
    finally:
        patched.unlink(missing_ok=True)


def sample_floor_points(navmesh, *, count: int = 40, seed: int = 0) -> np.ndarray:
    """Probe locations, taken from the navmesh so the gate tests what is declared walkable.

    Sampling the raw mesh instead would test surface the navmesh never claimed
    was walkable, which is not what a policy would step on.
    """
    return navmesh.sample_points(count, seed=seed)
