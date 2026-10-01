"""Coverage walls: as tall as what the capture saw there, full height where it saw nothing.

The scene: walkable floor on x < 2 m, and past it a strip the navmesh never
kept (ground unknown), which therefore gets a wall two cells deep (x 2.0-2.5 m).
What stands on that strip changes per test; the assertions are on the
collision boxes that come out, measured from the ground under each cell.
"""

import numpy as np
import pytest

from simkit.geometry.coverage_bounds import (GO2_FOOT_RADIUS_M, boxed_cells, coverage_walls, mesh_witness, uncovered_cells,
                                             wall_boxes, wall_field)
from simkit.geometry.splat_obstacles import obstacle_boxes
from simkit.geometry.navmesh import Navmesh

CELL = 0.25
FULL = 1.6
STEP = 0.06  # wall_boxes rounds wall tops up by at most this
WALL_X = (2.125, 2.375)  # centres of the two wall columns
NONE = np.zeros((0, 3))


def _ground(y):
    return 0.0 * y


def _navmesh(ground=_ground, cols=16):
    rows = 16  # 4 m along y, cols * 0.25 m along x
    grid = np.zeros((rows, cols), dtype=bool)
    grid[:, :8] = True  # walkable for x < 2 m
    y = (np.arange(rows)[:, None] + 0.5) * CELL * np.ones((1, cols))
    return Navmesh(grid, np.where(grid, ground(y), np.nan), np.zeros(2), CELL)


def _block(x0, x1, y0, y1, top, step=0.02, ground=_ground):
    """Points filling an object from the ground to ``top`` above it."""
    xs, ys, zs = np.arange(x0, x1, step), np.arange(y0, y1, step), np.arange(0.0, top + 1e-9, step)
    gx, gy, gz = np.meshgrid(xs, ys, zs)
    # The ground of a cell is read at the cell centre, as the navmesh stores it.
    cell_y = (np.floor(gy / CELL) + 0.5) * CELL
    return np.column_stack([gx.ravel(), gy.ravel(), (gz + ground(cell_y)).ravel()])


def _side(points):
    """Normals of a vertical face, for mesh points."""
    return np.tile([1.0, 0.0, 0.0], (len(points), 1))


def _walls(mesh=NONE, splat=NONE, normals=None, ground=_ground, navmesh=None):
    navmesh = _navmesh(ground) if navmesh is None else navmesh
    fence = uncovered_cells(navmesh)
    assert fence[:, 8:10].all() and not fence[:, 10:].any()
    normals = _side(mesh) if normals is None else normals
    return wall_boxes(navmesh, *coverage_walls(navmesh, mesh, normals, splat))


def _floor(x0, x1, y0, y1, step=0.05):
    """Up-facing floor samples at the ground, with their normals."""
    points = _block(x0, x1, y0, y1, top=0.0, step=step)
    return points, np.tile([0.0, 0.0, 1.0], (len(points), 1))


def _bottom_at(boxes, x, y):
    found = _boxes_over(boxes, x, y)
    assert len(found) == 1, f"expected one wall box over ({x}, {y}), found {len(found)}"
    return found[0]["centre"][2] - found[0]["half_extents"][2]


def _boxes_over(boxes, x, y):
    return [b for b in boxes
            if abs(x - b["centre"][0]) < b["half_extents"][0] and abs(y - b["centre"][1]) < b["half_extents"][1]]


def _height_at(boxes, x, y, ground=_ground):
    """Wall top over (x, y), above the ground of that cell."""
    tops = [b["centre"][2] + b["half_extents"][2] for b in boxes
            if abs(x - b["centre"][0]) < b["half_extents"][0] and abs(y - b["centre"][1]) < b["half_extents"][1]]
    assert len(tops) == 1, f"expected one wall box over ({x}, {y}), found {len(tops)}"
    return tops[0] - ground((np.floor(y / CELL) + 0.5) * CELL)


def _assert_full(boxes, x, y, ground=_ground):
    assert FULL <= _height_at(boxes, x, y, ground) <= FULL + STEP


def test_unobserved_strip_keeps_a_full_wall():
    boxes = _walls()
    for x in WALL_X:
        for y in np.arange(0, 4, CELL) + CELL / 2:
            _assert_full(boxes, x, y)


def test_nothing_seen_past_the_fence_gets_no_box():
    boxes = _walls(_block(2.0, 4.0, 0.0, 4.0, top=0.0))  # floor only
    assert not _boxes_over(boxes, 2.875, 1.0) and not _boxes_over(boxes, 3.625, 1.0)


def test_the_whole_sofa_is_in_collision_seat_and_backrest_past_the_fence():
    """A sofa 1 m deep (seat 0.42 m) with a 0.86 m backrest at its far end:
    only its front half lies in the fence ring."""
    seat = _block(2.0, 3.0, 0.0, 2.0, top=0.42)
    back = _block(3.0, 3.25, 0.0, 2.0, top=0.86)
    boxes = _walls(np.vstack([seat, back]))
    for x in (2.125, 2.625, 2.875):  # seat, inside the ring and past it
        assert 0.42 <= _height_at(boxes, x, 1.0) <= 0.42 + STEP
    assert 0.86 <= _height_at(boxes, 3.125, 1.0) <= 0.86 + STEP  # backrest
    assert not _boxes_over(boxes, 3.625, 1.0)  # floor behind the sofa


def test_low_furniture_gets_its_own_height_and_the_wall_beside_it_stays_full():
    sofa = _block(2.0, 2.5, 0.0, 2.0, top=0.42)  # seat 0.42 m, along y < 2 m
    wall = _block(2.0, 2.5, 2.0, 4.0, top=2.6)  # a real wall to the ceiling, y > 2 m
    boxes = _walls(np.vstack([sofa, wall]))
    for x in WALL_X:
        for y in (0.125, 0.875, 1.875):
            assert 0.42 <= _height_at(boxes, x, y) <= 0.42 + STEP
        for y in (2.125, 3.125, 3.875):
            _assert_full(boxes, x, y)


def test_furniture_on_a_slope_is_measured_from_its_own_ground():
    """Ground rising 10% along y: 40 cm over the strip."""
    def ramp(y):
        return 0.5 + 0.1 * y
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42, ground=ramp)
    boxes = _walls(sofa, ground=ramp)
    for x in WALL_X:
        for y in np.arange(0, 4, CELL) + CELL / 2:
            assert 0.42 <= _height_at(boxes, x, y, ramp) <= 0.42 + STEP


def test_a_pane_the_mesh_lost_is_closed_by_the_splat():
    """Mesh: only the bottom of the pane, 40 cm. Splat: the frosted pane up to 1.5 m."""
    rail = _block(2.0, 2.1, 0.0, 4.0, top=0.40)
    pane = _block(2.0, 2.1, 0.0, 4.0, top=1.50, step=0.04)
    boxes = _walls(rail, pane)
    for y in (0.125, 1.0, 3.875):
        assert _height_at(boxes, 2.125, y) >= 1.5


def test_a_wall_with_its_middle_lost_keeps_full_height():
    """Mesh: a 30 cm skirting and the wall again from 1.8 m up, nothing between."""
    skirting = _block(2.0, 2.1, 0.0, 4.0, top=0.30)
    upper = _block(2.0, 2.1, 0.0, 4.0, top=2.6)
    upper = upper[upper[:, 2] > 1.8]
    _assert_full(_walls(np.vstack([skirting, upper])), 2.125, 1.0)


def test_a_ceiling_over_low_furniture_does_not_raise_it():
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42)
    ceiling = _block(2.0, 2.5, 0.0, 4.0, top=0.0) + [0.0, 0.0, 2.7]
    mesh = np.vstack([sofa, ceiling])
    normals = np.vstack([_side(sofa), np.tile([0.0, 0.0, -1.0], (len(ceiling), 1))])
    assert 0.42 <= _height_at(_walls(mesh, normals=normals), 2.125, 1.0) <= 0.42 + STEP


def test_a_few_splat_floaters_do_not_raise_a_seen_cell():
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42)
    floaters = np.array([[2.1, 1.0, 1.3], [2.2, 1.05, 1.35], [2.15, 1.1, 1.5]])
    assert 0.42 <= _height_at(_walls(sofa, floaters), 2.125, 1.0) <= 0.42 + STEP


def test_mesh_debris_does_not_size_a_wall_down_but_furniture_does():
    """A sofa made of a few hundred triangles, and one stray triangle at 30 cm."""
    import open3d as o3d

    sofa = o3d.geometry.TriangleMesh.create_box(0.5, 1.0, 0.42).translate((2.0, 0.5, 0.0))
    sofa = sofa.subdivide_midpoint(number_of_iterations=2)
    stray = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector([[2.05, 3.05, 0.3], [2.2, 3.05, 0.3], [2.05, 3.2, 0.3]]),
        o3d.utility.Vector3iVector([[0, 1, 2]]))
    points, normals = mesh_witness(sofa + stray, CELL)
    boxes = _walls(points, normals=normals)
    assert 0.42 <= _height_at(boxes, 2.125, 1.0) <= 0.42 + STEP
    _assert_full(boxes, 2.125, 3.125)


def test_tilted_facets_in_a_ceiling_do_not_raise_low_furniture():
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42)
    facets = np.array([[2.1, 1.0, 2.7], [2.2, 1.05, 2.72], [2.15, 1.1, 2.69]])
    mesh = np.vstack([sofa, facets])
    assert 0.42 <= _height_at(_walls(mesh), 2.125, 1.0) <= 0.42 + STEP


def test_floor_or_a_climbable_kerb_keeps_the_wall_full():
    """Past the wall lies filled, unobserved terrain: a wall the robot can step
    over (the navmesh climbs 0.25 m) would not contain it."""
    floor = _block(2.0, 4.0, 0.0, 2.0, top=0.0)
    kerb = _block(2.0, 4.0, 2.0, 4.0, top=0.20)
    boxes = _walls(np.vstack([floor, kerb]))
    _assert_full(boxes, 2.125, 1.0)
    _assert_full(boxes, 2.125, 3.0)


def test_every_wall_cell_gets_a_box_of_its_own_height_however_many_boxes_that_takes():
    """Alternating heights make one box per cell; none may be dropped."""
    size = 420
    grid = np.zeros((size, size), dtype=bool)
    grid[2:-2, 2:-2] = True
    navmesh = Navmesh(grid, np.where(grid, 0.0, np.nan), np.zeros(2), CELL)
    cells = uncovered_cells(navmesh)
    rows, cols = np.indices(cells.shape)
    heights = np.where((rows + cols) % 2 == 0, 0.42, FULL) * cells
    boxes = wall_boxes(navmesh, cells, heights, np.zeros_like(heights))
    assert len(boxes) > 1500
    covered = np.zeros(cells.shape, dtype=int)
    for box in boxes:
        c0 = int(round((box["centre"][0] - box["half_extents"][0]) / CELL))
        c1 = int(round((box["centre"][0] + box["half_extents"][0]) / CELL))
        r0 = int(round((box["centre"][1] - box["half_extents"][1]) / CELL))
        r1 = int(round((box["centre"][1] + box["half_extents"][1]) / CELL))
        covered[r0:r1, c0:c1] += 1
        top = box["centre"][2] + box["half_extents"][2]
        assert heights[r0:r1, c0:c1].max() <= top <= heights[r0:r1, c0:c1].min() + STEP
    assert (covered[cells] == 1).all() and not covered[~cells].any()


def test_the_splat_alone_never_lowers_a_wall():
    """A hole in the mesh where the splat holds something low stays a full wall."""
    blob = _block(2.0, 2.5, 0.0, 4.0, top=0.6, step=0.04)
    _assert_full(_walls(splat=blob), 2.125, 1.0)


def test_every_splat_obstacle_cluster_gets_a_box():
    """More than the 150 clusters the old cap kept."""
    size = 100
    grid = np.ones((size, size), dtype=bool)
    navmesh = Navmesh(grid, np.zeros((size, size)), np.zeros(2), CELL)
    blocked = np.zeros((size, size), dtype=bool)
    blocked[::4, ::4] = True  # 625 single-cell clusters
    assert len(obstacle_boxes(navmesh, blocked)) == int(blocked.sum())


def test_walkable_ground_never_gets_a_box():
    """A wall face hanging above walkable floor (a soffit from 1.8 m up)."""
    soffit = _block(0.9, 1.1, 0.0, 4.0, top=2.6)
    soffit = soffit[soffit[:, 2] > 1.8]
    assert not _boxes_over(_walls(soffit), 1.0, 1.0)



def test_structure_beyond_reach_gets_no_box():
    """Walkable for x < 2 m; a cabinet 2.5 m past it, another 4 m past it."""
    navmesh = _navmesh(cols=40)
    near = _block(4.5, 4.75, 1.0, 1.25, top=1.0)
    far = _block(6.0, 6.25, 1.0, 1.25, top=1.0)
    boxes = _walls(np.vstack([near, far]), navmesh=navmesh)
    assert 1.0 <= _height_at(boxes, 4.625, 1.125) <= 1.0 + STEP
    assert not _boxes_over(boxes, 6.125, 1.125)


def test_a_high_ceiling_or_vault_alone_makes_no_wall():
    """Past the fence, nothing in the band and a curved vault from 2.3 m up."""
    vault = _block(2.5, 4.0, 0.0, 4.0, top=3.0)
    vault = vault[vault[:, 2] > 2.3]
    normals = np.tile([0.6, 0.0, -0.8], (len(vault), 1))  # tilted: |n_z| < 0.85
    boxes = _walls(vault, normals=normals)
    assert not _boxes_over(boxes, 2.875, 1.0) and not _boxes_over(boxes, 3.625, 1.0)


def test_a_vault_over_low_furniture_does_not_raise_it():
    sofa = _block(2.0, 2.5, 0.0, 4.0, top=0.42)
    vault = _block(2.0, 2.5, 0.0, 4.0, top=2.6)
    vault = vault[vault[:, 2] > 2.1]
    mesh = np.vstack([sofa, vault])
    assert 0.42 <= _height_at(_walls(mesh), 2.125, 1.0) <= 0.42 + STEP


def test_the_go2_fits_under_a_table_over_observed_floor():
    """Table top 0.74-0.78 m over the whole strip, floor seen under it."""
    top = _block(2.0, 4.0, 0.0, 4.0, top=0.78)
    top = top[top[:, 2] >= 0.74 - 1e-9]
    floor, floor_n = _floor(2.0, 4.0, 0.0, 4.0)
    boxes = _walls(np.vstack([top, floor]), normals=np.vstack([_side(top), floor_n]))
    for x in (2.125, 2.875):
        assert 0.74 - STEP <= _bottom_at(boxes, x, 1.0) <= 0.74
        assert 0.78 <= _height_at(boxes, x, 1.0) <= 0.78 + STEP


def test_an_overhang_over_unobserved_ground_stands_on_the_ground():
    """A sign from 1.4 m up at the edge of the capture, no floor seen under it."""
    sign = _block(2.0, 2.5, 0.0, 4.0, top=1.55)
    sign = sign[sign[:, 2] >= 1.4]
    boxes = _walls(sign)
    assert _bottom_at(boxes, 2.125, 1.0) == pytest.approx(0.0)


def test_a_gap_lower_than_the_go2_is_closed():
    """A bench seat from 0.3 m up over observed floor: the Go2 (0.40 m) does not fit."""
    seat = _block(2.0, 2.5, 0.0, 4.0, top=0.45)
    seat = seat[seat[:, 2] >= 0.3]
    floor, floor_n = _floor(2.0, 2.5, 0.0, 4.0)
    boxes = _walls(np.vstack([seat, floor]), normals=np.vstack([_side(seat), floor_n]))
    assert _bottom_at(boxes, 2.125, 1.0) == pytest.approx(0.0)


def test_a_small_pocket_inside_the_walkable_area_stays_open():
    """A 0.5 m x 0.5 m hole in the walkable floor with a post in it."""
    navmesh = _navmesh()
    navmesh.grid[6:8, 3:5] = False
    navmesh.ground_z[6:8, 3:5] = np.nan
    post = _block(0.8, 1.2, 1.55, 1.95, top=1.0)
    boxes = _walls(post, navmesh=navmesh)
    assert not _boxes_over(boxes, 1.0, 1.75)



def test_the_wall_field_holds_a_probe_on_the_seat_and_leaves_the_floor_beside_it_alone(tmp_path):
    """MuJoCo, ground and wall height fields only: a foot-sized plate dropped on
    the sofa rests on the seat, one dropped on the floor beside it rests on the
    floor, and one on unobserved ground past the fence stays up on the wall."""
    from simkit.export.heightfield import build_heightfield
    from simkit.export.mjcf import write_mjcf
    from simkit.physics.probes import run_floor_probes

    def ramp(y):
        return 0.5 + 0.1 * y
    navmesh = _navmesh(ramp)
    sofa = _block(2.0, 2.5, 0.0, 2.0, top=0.42, ground=ramp)
    cells, tops, bottoms = coverage_walls(navmesh, sofa, _side(sofa), NONE)
    ground, walls = build_heightfield(navmesh), wall_field(navmesh, cells, tops)
    xml = write_mjcf([], tmp_path / "scene.xml", heightfield=ground,
                     heightfield_png=ground.write_png(tmp_path / "ground.png"),
                     wall_field=walls, wall_field_png=walls.write_png(tmp_path / "walls.png"))
    g = ramp(1.125), ramp(3.125)  # ground under the probes' cells
    probes = run_floor_probes(xml, np.array([[2.125, 1.1, g[0] + 0.42], [1.875, 1.1, g[0]], [2.125, 3.1, g[1] + FULL]])).probes
    assert not any(p.fell_through for p in probes)
    rests = [p.final_z - p.ground_z for p in probes]
    assert rests == pytest.approx([rests[1]] * 3, abs=0.07)  # each on its own support, within one rounding step


def test_the_wall_field_reaches_no_further_than_a_foot_into_walkable_ground():
    navmesh = _navmesh()
    cells = uncovered_cells(navmesh)
    walls = wall_field(navmesh, cells, np.where(cells, FULL, np.nan))
    k = walls.elevation.shape[1] // navmesh.grid.shape[1]
    z = walls.elevation * walls.elevation_z + walls.z_offset
    spacing = 2 * walls.radius_x / (walls.elevation.shape[1] - 1)
    x = walls.centre[0] - walls.radius_x + np.arange(walls.elevation.shape[1]) * spacing
    row = z[0]
    last_low = x[(row < 0.0) & (x < 2.0)].max()  # last sample below the floor before the wall
    assert 2.0 - last_low <= GO2_FOOT_RADIUS_M + 1e-9
    assert row[(x > 2.0) & (x < 2.5)].min() >= FULL
    assert k >= navmesh.cell_size / (2 * GO2_FOOT_RADIUS_M)
    # samples at the centres of the sub-cells, as MuJoCo spaces them
    assert spacing == pytest.approx(navmesh.cell_size / k)
    assert x[0] == pytest.approx(navmesh.cell_size / (2 * k))



def test_walls_within_the_go2_reach_are_boxes_and_raised_ones_too():
    """Walkable 4 x 4 cells in the middle; walls everywhere else."""
    grid = np.zeros((10, 10), dtype=bool)
    grid[3:7, 3:7] = True
    navmesh = Navmesh(grid, np.where(grid, 0.0, np.nan), np.zeros(2), CELL)
    walls = ~grid
    bottoms = np.zeros(grid.shape)
    bottoms[0, 0] = 0.8  # a sign far out, the Go2 fits under it
    boxed = boxed_cells(navmesh, walls, bottoms)
    assert boxed[2, 2] and boxed[2, 4] and boxed[7, 7]  # next to walkable, diagonal included
    assert not boxed[1, 4] and not boxed[8, 8]          # out of the Go2's reach: the field
    assert boxed[0, 0]                                  # raised: only a box can hold it
    assert not boxed[grid].any()


def test_a_wall_beside_a_drop_is_measured_from_the_floor_above():
    """Lower floor for x < 2 m at -0.8 m, upper floor from x = 2.25 m at 0, one
    wall cell between them holding steps 0.3-0.6 m above the lower floor. Seen
    from the upper floor the steps are a hole: the wall must stand there.
    (The nearest-ground fill breaks the tie towards the lower floor.)"""
    rows, cols = 16, 16
    grid = np.zeros((rows, cols), dtype=bool)
    grid[:, :8] = True
    grid[:, 9:] = True
    ground = np.where(grid, np.where(np.arange(cols) < 8, -0.8, 0.0)[None, :] * np.ones((rows, 1)), np.nan)
    navmesh = Navmesh(grid, ground, np.zeros(2), CELL)
    steps = _block(2.0, 2.25, 0.0, 4.0, top=0.6) + [0.0, 0.0, -0.8]
    steps = steps[steps[:, 2] >= -0.5]
    cells, tops, bottoms = coverage_walls(navmesh, steps, _side(steps), NONE)
    boxes = wall_boxes(navmesh, cells, tops, bottoms)
    for y in (0.125, 2.0, 3.875):
        assert _height_at(boxes, 2.125, y) >= FULL  # measured from the upper floor, at 0
