# ovr-maps-360-simkit

Turns an **OVR Maps 360** scene into a simulation-ready environment for MuJoCo
and Isaac Sim, certifies it with a physics gate, and derives three levels of
detail of its collision surface. This is the code that produced the
`simulation/`, `lod/` and `datasheets/` folders of the OVR Maps 360 dataset;
run it on a scene of the dataset and you get the same files.

## What goes in

A scene directory of the dataset, `<uuid>/`. Only three files are read:

| file | what it is |
|---|---|
| `mesh/model.glb` | textured surface mesh: OpenMVS dense reconstruction and meshing, with the floor closed by MoGe-2 plane panels where the photogrammetry left holes |
| `gaussian_splatting.ply` | 3D Gaussian splat (5M Gaussians, SH degree 3, INRIA layout) |
| `training_cameras.json` | the cameras the splat was trained on: `{id, img_name, width, height, position, rotation, fx, fy}` |

How those were made, because it decides what the frame is:

- **Capture.** An Insta360 X5 walked through the place, recording its two
  fisheye lenses (3840×3840 each), with a smartphone running ARKit alongside.
- **Structure from motion** runs on the fisheye frames as a two-camera rig
  (OPENCV_FISHEYE). The reconstruction is then registered **rigidly to the
  ARKit poses** of the phone, which makes it metric and gravity-aligned.
- **Splat training** uses five perspective views cut from every fisheye frame
  (1600×1600, 90° field of view: three horizontal, one up, one down), so
  `training_cameras.json` holds five views per camera centre.
- All three files share that frame: metres, **up = −Y**, origin wherever the
  ARKit session started.

## What comes out

```
<uuid>/
  simulation/
    <id8>.sre/                       the bundle
      manifest.json                  tier, sources, every stage's measurements, gate
      frame/transform.json           sim_from_source (4x4) and how it was obtained
      scene.xml                      MJCF (MuJoCo / MJX)
      scene.usda                     USD (Isaac Sim / Isaac Lab), same collision geometry
      collision/
        surface.ply                  the collision surface: mesh cropped to the walk, simplified
        ground_hfield.png            ground height field (16-bit)
        navmesh_grid.npy             walkable cells (bool, rows = y, cols = x)
        navmesh_ground_z.npy         ground height per cell (float, NaN off the navmesh)
        part_NNN.obj                 convex parts (CoACD) for structure above the ground
      photoreal/splat.ply            the splat, moved into the simulation frame
    certification/
      mujoco_physics_gate.json       the gate, machine-readable
      mujoco_physics_gate.txt        the same, for a human
      isaac_drop_test.json           cross-engine drop test
  lod/
    <id8>_high.ply  <id8>_mid.ply  <id8>_low.ply  <id8>.json
  datasheets/
    <uuid>.json                      tier, stage measurements, gate, Isaac verdict
```

`<id8>` is the first eight characters of the uuid. Nothing in these files
carries an absolute path; the MJCF and USD reference only files inside the
bundle.

## Install

Two conda environments, because Isaac Sim pins its own numpy and USD:

```bash
bash install/install_env_simkit.sh     # env "simkit": build, MuJoCo gate, LOD (CPU)
bash install/install_env_isaac.sh      # env "simkit-isaac": Isaac Sim 5.1 headless (NVIDIA GPU)
```

`simkit` finds the Isaac interpreter by env name; elsewhere, pass
`--isaac-python` or set `SIMKIT_ISAAC_PYTHON`. The Isaac installer accepts the
NVIDIA Isaac Sim EULA on your behalf, so read it first (link in the script).

## Run

```bash
conda activate simkit

simkit build  /data/ovrmaps360/<uuid>                      # outputs written into the scene
simkit build  /data/ovrmaps360/<uuid> --out /data/out      # ... or into /data/out/<uuid>/
simkit batch  /data/ovrmaps360 --out /data/out --jobs 6 --log-dir /data/out/logs

simkit verify  /data/out/<uuid>                  # PASS / FAIL with the reasons
simkit regate  /data/out/<uuid>/simulation/<id8>.sre   # re-run the MuJoCo probes, compare with the manifest
simkit rescore /data/out/<uuid>                  # re-apply the gate from the recorded measurements
simkit lod     surface.ply out_dir <id8>         # LODs only
```

A build resumes: an existing bundle, Isaac result or LOD set is kept unless
`--force`. `batch` runs one process per scene, so a crash in native code takes
out that scene only.

Measured on a 32-core machine with an RTX 4090: about 4 minutes per scene with
the whole machine, peak 6–7 GB of RAM per scene; `--jobs 6` builds 50 scenes in
about two and a half hours. Only one Isaac Sim instance runs at a time per
machine (a file lock serialises the drop tests): concurrent instances abort at
start-up.

## The frame

The source frame is already gravity-aligned by the ARKit registration, so the
rotation into the simulation frame is **fixed**, not estimated: `sim = (x, z, −y)`,
Z up, as MuJoCo and USD expect. The ARKit origin, however, is wherever the
phone session started, so the floor can sit metres from zero. It is measured:
a ray is cast straight down from every camera centre, the first surface it
meets is the ground the operator stood on at that moment (whatever height the
pole was held at), and the median of those heights becomes z = 0.

Floors are allowed to slope. A street climbing a hill keeps its slope, because
levelling the walk or the floor would tilt gravity for the robot. What must be
vertical is the walls, and that is what the gate checks.

## The gate

A bundle ships as tier **T1a** only if all eight checks pass. Thresholds are
part of the contract and are not relaxed to make a scene pass.

| # | check | measured as | threshold |
|---|---|---|---|
| 1 | `up_direction_verified` | the densest horizontal slab of the splat within 3 m of the walk sits in the lower part of the walked height, and ≥ 90% of camera centres have ground beneath them | decidable, not inverted |
| 2 | `scene_plumb_under_2deg` | vertical implied by the walls, from the splat's disc Gaussians and from the mesh faces, against the frame's +Z; the tilt is the **smaller** of the two witnesses, since a broken alignment tilts the walls in both reconstructions | < 2° |
| 3 | `floor_at_origin_under_5cm` | under every camera on a walkable cell, navmesh ground height against the ray-cast floor: median absolute difference, and the navmesh must cover at least 20% of the cameras | < 5 cm, >= 20% covered |
| 4 | `alignment_residual_under_25cm` | median distance from the splat's solid Gaussians (opacity > 0.5) to the collision surface, within 12 m of the walk | < 25 cm |
| 5 | `walkable_area_over_5m2` | navmesh area after the stance step filter and splat obstacles | ≥ 5 m² |
| 6 | `no_collision_leak` | foot-sized damped plates dropped on 40 navmesh cells in MuJoCo: none falls through | 0% |
| 7 | `probes_settle` | plates at rest within 3 s | ≥ 90% |
| 8 | `penetration_p95_under_2cm` | solver contact penetration, 95th percentile | < 2 cm |

Why the threshold of check 2 is where it is: over the 50 scenes of the sample,
the two wall witnesses disagree with each other by 0.9° median (1.9° p90), so
2° is where a real alignment error stands out from witness noise, and 2° of
gravity error is a 3.5% slope under a robot's feet.

The angle between the plane of the camera trajectory and the plane of the
ground beneath it is recorded (`witness_disagreement_deg`) but no longer
gated: it tilts with the pole, not with the scene (45 cm of pole travel over a
100 m walk is 0.3–1°), and every scene it rejected had its navmesh ground
within 1 mm of the ray-cast floor.

The walkable grid samples the *surface* of every triangle (ten samples per
cell of area, at least one per triangle), so a floor of four large panels is
read exactly like a floor of a million small ones; the cell is a foot,
0.25 m.

The camera's height above the floor is recorded (`camera_height_m`) but not
gated: operators hold the pole at different heights, and sometimes lower it
to the ground. The share of the walk that lies on walkable cells
(`walkable_under_cameras`) is recorded too.

**Isaac Sim** is checked separately and reported, not certified: a 5 cm rigid
sphere is released 1 m above a point where a MuJoCo probe settled, and its
resting height is compared with the floor MuJoCo reported there
(`isaac_drop_test.json`). A sphere that falls through fails the scene.

## The bundle, in detail

- **Collision** comes from the mesh, **appearance** from the splat. They are two
  reconstructions of one capture in one frame, so no registration is needed;
  check 5 measures how far apart they are (about 4–6 cm median on the sample).
- The mesh is **cropped** to 12 m around the walked path: a robot cannot be
  evaluated where the capture had no observations, and the convex
  decomposition pays for every distant triangle.
- The **navmesh** is 2.5D: per cell, the lowest upward-facing surface, with
  1.6 m of head clearance, a 0.25 m step limit between cells, a 6 cm limit
  under a whole stance, and the largest connected region kept. Cells where the
  splat sees a solid obstacle the mesh lost (typically a car) are removed and
  get box colliders. Uncovered ground at the navmesh edge is fenced with wall
  boxes, so a drifting robot cannot walk onto invented height field. Each
  wall cell is as tall as what the mesh saw in it (a sofa at its seat), raised
  by the splat where the mesh lost glass, and 1.6 m where the mesh saw nothing
  above the 0.25 m climbable step or has wall above 1.6 m.
  Being 2.5D, it cannot represent a walk that passes over another level: the
  lower level wins, and such scenes fail check 4.
- The **ground** is a height field (MJCF `hfield`, USD mesh); the structure
  above it is convex parts from **CoACD** (concavity threshold 0.30, up to 8
  hulls per tile, about 60 tiles).
- `photoreal/splat.ply` carries the degree-0 colour only (17 floats per
  Gaussian): rotating higher-order spherical harmonics into a new frame is not
  implemented. The view-dependent source splat is `gaussian_splatting.ply` at
  the scene root.

## LODs

From `collision/surface.ply`: connected components with fewer than 50
triangles are pruned, then each tier is the smallest quadric decimation whose
symmetric deviation from the pruned surface stays within budget at the 99th
percentile: **high 1 cm, mid 3 cm, low 10 cm** (triangle target found by
bisection). `<id8>.json` records triangles, reduction, p50/p95 deviation and
file size per tier.

## Tests

```bash
python -m pytest tests
```

## Traps

- Importing `coacd` before `pxr` segfaults the interpreter (both bundle TBB).
  The USD export runs in its own process for that reason.
- CoACD's `resolution` default is 2000. Passing 1e6 runs for hours.
- MuJoCo indexes height field rows from −Y, a PNG stores its first row at the
  top; without the vertical flip the terrain is mirrored about X.
- Open3D reads large glTF files unreliably; `trimesh` loads the glb, without
  materials (0.4 s instead of 45 s with textures).

## Licence

Code: Apache License 2.0, see `LICENSE`. Every dependency of the build is
under a permissive licence: numpy, scipy (BSD), open3d, trimesh, CoACD (MIT),
Pillow (MIT-CMU), OpenCV, MuJoCo (Apache-2.0), usd-core (TOST-1.0, a modified
Apache-2.0). Isaac Sim is used only for the drop test, in its own environment,
under NVIDIA's terms. The dataset has
its own licence, stated in its dataset card; `--licence` sets the string
written into each manifest.
