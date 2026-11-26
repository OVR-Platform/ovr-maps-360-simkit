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
        navmesh_levels_grid.npy      walkable cells on every storey (bool, L x rows x cols)
        navmesh_levels_ground_z.npy  ground height per storey and cell (float, NaN where none)
        structure_boxes.json         structure above the ground: box columns on the navmesh cells
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
levelling the walk or the floor would tilt gravity for the robot. Walls are not
required to be plumb either: how far they lean from the frame's vertical is
measured and recorded (`s1_plumb`), not gated.

## The gate

A bundle ships as tier **T1a** only if all seven checks pass. Thresholds are
part of the contract and are not relaxed to make a scene pass.

| # | check | measured as | threshold |
|---|---|---|---|
| 1 | `up_direction_verified` | the bottom of the horizontal training views points down: image-down axis along gravity, median over ≥ 100 views; and ≥ 50% of camera centres have ground beneath them (the mesh is in the cameras' frame) | decidable, not inverted |
| 2 | `floor_at_origin_under_5cm` | under every camera on a walkable cell, navmesh ground height (the storey nearest the camera's floor, from the multi-level navmesh) against the ray-cast floor: median absolute difference, and the navmesh must cover at least 20% of the cameras | < 5 cm, >= 20% covered |
| 3 | `alignment_residual_under_25cm` | offset between splat and collision surface: the similarity transform that best moves the solid Gaussians (opacity > 0.5, within 12 m of the walk) onto the surface (trimmed ICP), and how far it moves them, 90th percentile | < 25 cm |
| 4 | `walkable_area_over_5m2` | navmesh area after the stance step filter and splat obstacles | ≥ 5 m² |
| 5 | `no_collision_leak` | foot-sized damped plates dropped on 40 navmesh cells in MuJoCo: none falls through | 0% |
| 6 | `probes_settle` | plates at rest within 3 s | ≥ 90% |
| 7 | `penetration_p95_under_2cm` | solver contact penetration, 95th percentile | < 2 cm |

The Amazon corpus (1,101 scenes) changed four rules, none by relaxing a
threshold on what a robot meets:

- *Walls are recorded, not gated.* `scene_plumb_under_2deg` failed 190 scenes
  at 2-12° of wall tilt (median 3.7° splat, 4.5° mesh). The frame's vertical is
  ARKit's gravity and nothing built here depends on walls being plumb.
- *The step under a stance is measured against the local slope.* The filter
  took the height range over a stance, so a street at 8° read as a 9 cm step:
  on a hillside scene it dropped 86% of the walkable area and left 12% of the
  walk on the navmesh. The slope is now the minmod of the one-cell differences
  (a kerb on a hill still counts in full; on level ground nothing changes).
  Same rule in the multi-level navmesh.
- *The floor is compared with the storey the walk was on.* The single-level
  navmesh keeps the lowest surface of a cell, so a walk over a ramp or a
  bridge was compared with the ground beneath it (3.04 m, against 1 mm on the
  multi-level navmesh).
- *Structure is columns of boxes, not convex hulls.* The convex hull of a
  tile's walls is the room between them: on one hall 24 of 25 CoACD tiles had
  a hull holding the space a standing robot occupies, and MuJoCo pushed probes
  out through an exact, flat height field (leak 10%). Every cell that is not
  walkable and holds surface between 10 cm and 2 m above its ground now gets a
  box column up to that surface; a walkable cell never does, by construction,
  and the build records how many walkable points end up inside any box
  (`walkable_points_inside_boxes`). Splat obstacle boxes cover their cells
  exactly instead of their cluster's bounding box. A scene now carries
  2,000-11,000 static boxes; the gate's probe run loads only those within 1 m
  of a probe (180 s to 2.5 s on 10,000 boxes, same result).

Checks 1 and 3 hold indoors as well as outdoors. Check 1 used to read up
from the splat: near the walk the floor should be its densest horizontal slab.
Under a station vault 8 m above the walk, or a subway ceiling 1.25 m above the
camera, the ceiling is denser, and an upright scene read as upside down. The
cameras do not depend on what the scene holds: the horizontal views are cut
with the rig upright on the pole, so the bottom of each image looks at the
ground, and a frame turned over by the registration turns them over with it.
On the 50 scenes of the sample and two covered stations, image-down is within
15° of gravity in every horizontal view (median cosine 0.96–0.999). The slab
is still recorded (`s1_splat_slab`). Check 3 used to be the median distance
from the splat to the surface, but the splat holds much the mesh does not
(chandeliers, foliage, glass, what is seen through an opening): on a station
hall it read 32 cm with the two reconstructions within 2 cm and 0.03° of each
other, and a 30 cm shift of the splat moved it by 3 cm. The fitted offset
measures the shift itself (31 cm for 30 cm, about 1.6 m at the far end for 1°
of yaw); the plain distance is still recorded (`splat_to_surface_median_m`).

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
resting height is compared with the navmesh floor where it came to rest
(`isaac_drop_test.json`, with `rolled_m`). A sphere that falls through fails
the scene. On a sloping street the sphere rolls: Amazon 2ffedfc0 read 30.65 cm
below the floor of the drop point, resting 3.0 m downhill 2.07 cm above the
floor there.

## The bundle, in detail

- **Collision** comes from the mesh, **appearance** from the splat. They are two
  reconstructions of one capture in one frame, so no registration is needed;
  check 3 measures how far apart they are.
- The mesh is **cropped** to 12 m around the walked path: a robot cannot be
  evaluated where the capture had no observations.
- The **navmesh** is 2.5D: per cell, the lowest upward-facing surface, with
  1.6 m of head clearance, a 0.25 m step limit between cells, a 6 cm limit
  under a whole stance (measured against the local slope, so a ramp passes and
  a kerb does not), and the regions the walk reaches kept. Cells where the
  splat sees a solid obstacle the mesh lost (typically a car) are removed and
  get box colliders. Uncovered ground at the navmesh edge is fenced with wall
  boxes, so a drifting robot cannot walk onto invented height field.
  Being 2.5D, it cannot represent a walk that passes over another level: the
  lower level wins. It is what the height field and the physics probes use.
- The **multi-level navmesh** (`navmesh_levels_*.npy`) keeps every storey.
  Per cell, the upward-facing surface is split wherever two heights are more
  than 12 cm apart, and every surface is judged by the rules above: head
  clearance, steps to the neighbouring cells, the stance limit, splat
  obstacles. A surface is kept only if the walk reaches it (the ground under a
  camera within 30 cm of its height) and its region still holds at least 1 m²
  of stance, so a desk a camera was carried over does not become a floor.
  The gate's floor check (2) reads this navmesh.
  Channel *k* is the *k*-th walkable surface of the cell counted upwards.
  Stairs stay out, as in the single level: every tread is a step over 6 cm.
- The **ground** is a height field (MJCF `hfield`, USD mesh); the structure
  above it, within 3 m of a walkable cell and up to 2 m above the ground, is
  axis-aligned boxes over the navmesh's cells, merged where they share a
  height (tops rounded up to 25 cm). Overhangs above 2 m (ceilings, canopies,
  lintels) are not collision: a walking robot does not reach them. CoACD
  (`simkit.geometry.collision`) is no longer called by the build.
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
