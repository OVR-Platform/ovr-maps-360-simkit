"""Drop a probe onto a bundle's USD inside Isaac Sim, and report where it lands.

The bundle's gate runs in MuJoCo. That is one engine's opinion, and a pipeline
that only ever asks one engine cannot see the class of defect where the export
is wrong rather than the geometry — which is exactly what happened here twice:
once with a USD that shipped no ground at all, and once with meshes authored
against the wrong solver's collision schema.

So this is deliberately the smallest possible question asked of the other
engine: put a rigid body above a point the MuJoCo gate says is solid, let go,
and see whether it stops where MuJoCo says the floor is.

Run with the Isaac Sim interpreter (install/install_env_isaac.sh), which is kept
separate because Isaac pins its own numpy and USD against everything else the
build needs:

    python -m simkit.physics.isaac_drop --usd scene.usda --at X Y FLOOR_Z --json out.json
"""

from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", required=True, action="append",
                        help="repeatable; each stage is opened in the same session")
    parser.add_argument("--at", nargs=3, type=float, action="append",
                        help="world XY and the floor height MuJoCo reports there, per stage")
    parser.add_argument("--drop-from", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--json", help="write the result here")
    args = parser.parse_args()

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    from isaacsim.simulation_app import SimulationApp

    app = SimulationApp({"headless": True})

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    import omni.usd

    from isaacsim.core.api import SimulationContext

    places = args.at or [[0.0, 0.0, 0.0]] * len(args.usd)
    results = []
    context = omni.usd.get_context()

    for usd_path, place in zip(args.usd, places):
        context.open_stage(usd_path)
        stage = context.get_stage()
        x, y, floor = place
        probe_path = "/DropProbe"
        probe = UsdGeom.Sphere.Define(stage, probe_path)
        probe.CreateRadiusAttr(0.05)
        probe.AddTranslateOp().Set(Gf.Vec3d(x, y, floor + args.drop_from))
        UsdPhysics.CollisionAPI.Apply(probe.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(probe.GetPrim())
        UsdPhysics.MassAPI.Apply(probe.GetPrim()).CreateMassAttr(1.0)

        simulation = SimulationContext(stage_units_in_meters=1.0)
        simulation.initialize_physics()
        simulation.play()
        for _ in range(args.steps):
            simulation.step(render=False)

        prim = stage.GetPrimAtPath(probe_path)
        resting = float(
            UsdGeom.Xformable(prim)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .ExtractTranslation()[2]
        )
        simulation.stop()
        simulation.clear_instance()

        results.append({
            "engine": "Isaac Sim (PhysX), headless",
            "floor_from_mujoco": floor,
            "resting_z": resting,
            "error_m": resting - (floor + 0.05),
            "fell_through": resting < floor - 0.5,
        })
        print(json.dumps(results[-1]), flush=True)

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)

    app.close()


if __name__ == "__main__":
    main()
