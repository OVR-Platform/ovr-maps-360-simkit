"""MJCF export for MuJoCo / MJX.

MuJoCo contacts require convex geometry, so the static environment ships as the
convex parts produced by S4. The same bundle also exports to USD for Isaac; the
two must never diverge, so both read the same collision directory.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


def write_mjcf(
    collision_parts: list[Path],
    output_path: str | Path,
    *,
    scene_name: str = "sre_scene",
    extra_boxes: list | None = None,
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001),
    include_ground_plane: bool = False,
    heightfield=None,
    heightfield_png: Path | None = None,
    wall_field=None,
    wall_field_png: Path | None = None,
) -> Path:
    """Write an MJCF describing the static environment.

    Ground comes from ``heightfield`` when given, and vertical structure from the
    convex parts. Convex hulls approximate a large gently-varying floor badly and
    a foot falls between them — measured at 87% leak on a 48-hull decomposition
    of the mall scene — so the two surfaces use the representation that fits.

    ``include_ground_plane`` adds an infinite plane at z = 0. Off by default: the
    point of these bundles is that the real floor comes from the capture, and a
    helper plane would mask exactly the collision leaks the gate looks for.

    ``wall_field`` is the coverage walls that stand on the ground
    (``coverage_bounds.wall_field``), a second height field.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("mujoco", model=scene_name)
    ET.SubElement(root, "compiler", angle="radian", meshdir=".")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")

    asset = ET.SubElement(root, "asset")
    fields = [(name, field, png) for name, field, png in
              (("ground", heightfield, heightfield_png), ("walls", wall_field, wall_field_png))
              if field is not None and png is not None]
    for name, field, png in fields:
        ET.SubElement(
            asset,
            "hfield",
            name=name,
            file=str(Path(png).resolve().relative_to(output_path.parent.resolve())),
            size=f"{field.radius_x:.4f} {field.radius_y:.4f} {field.elevation_z:.4f} {field.base_z:.4f}",
        )
    for index, part in enumerate(collision_parts):
        relative = Path(part).resolve().relative_to(output_path.parent.resolve())
        ET.SubElement(asset, "mesh", name=f"part_{index:03d}", file=str(relative))

    worldbody = ET.SubElement(root, "worldbody")

    # Splat-witnessed obstacle boxes: objects the splat sees at body height
    # where the mesh has nothing — a badly reconstructed car, typically. Their
    # job is to stop a robot, not to model the object. This block must sit
    # after worldbody exists: its first placement was before, which crashed the
    # export with an UnboundLocalError and silently left the previous night's
    # scene.xml in place — the manifest existed, the wait condition passed, and
    # an evaluation ran against a bundle that had never heard of the car.
    for number, box in enumerate(extra_boxes or []):
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{box.get('kind', 'splat_obstacle')}_{number:03d}",
            type="box",
            pos=" ".join(f"{v:.4f}" for v in box["centre"]),
            size=" ".join(f"{v:.4f}" for v in box["half_extents"]),
            friction="1.0 0.005 0.0001",
            contype="1",
            conaffinity="1",
            rgba="0.3 0.4 0.8 0.3" if box.get("kind") == "coverage_wall" else "0.8 0.3 0.2 0.4",
        )
    ET.SubElement(
        worldbody,
        "light",
        name="overhead",
        pos="0 0 6",
        dir="0 0 -1",
        directional="true",
    )
    for name, field, _ in fields:
        # MuJoCo places elevation 0 at the geom origin, so lift it to world Z.
        ET.SubElement(
            worldbody,
            "geom",
            name=f"{name}_hfield",
            type="hfield",
            hfield=name,
            pos=f"{field.centre[0]:.4f} {field.centre[1]:.4f} {field.z_offset:.4f}",
            friction=" ".join(str(f) for f in friction),
            contype="1",
            conaffinity="1",
        )
    if include_ground_plane:
        ET.SubElement(
            worldbody, "geom", name="ground", type="plane", size="50 50 0.1", pos="0 0 0"
        )
    for index, _ in enumerate(collision_parts):
        ET.SubElement(
            worldbody,
            "geom",
            name=f"static_{index:03d}",
            type="mesh",
            mesh=f"part_{index:03d}",
            friction=" ".join(str(f) for f in friction),
            contype="1",
            conaffinity="1",
        )

    pretty = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    output_path.write_text(pretty)
    return output_path
