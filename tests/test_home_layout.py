"""Layout tests for the `home` world (no simulator): the rules of the plan's sections 6.2 and 6.3.

Run: python3 -m pytest sim/tests/test_home_layout.py
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import house_layout as hl  # noqa: E402  (robot constants)
from worlds import home  # noqa: E402
from worlds import house  # noqa: E402  (to compare the dock with, whatever VACUUM_WORLD says)
from worlds import reach  # noqa: E402
from worlds.shapes import Box, Cylinder, footprint, overlaps_3d  # noqa: E402

EPS = 1e-9
LIDAR_Z = hl.LIDAR_XYZ[2]
CLEAR_UNDER = 0.20  # rule 4: an object stands on the floor or leaves this much space underneath
PATH_HALF_WIDTH = 0.30  # rule 2: a path at least 60 cm wide reaches every room

# The plan's section 6.2, inside measurements.
PLAN_ROOMS = {
    "living_room": ((0.0, 0.0, 6.4, 4.0), "living_room"),
    "kitchen": ((6.5, 0.0, 10.6, 3.2), "kitchen"),
    "corridor": ((4.0, 4.1, 7.0, 5.2), "corridor"),
    "bedroom_1": ((7.1, 3.3, 10.6, 7.6), "bedroom"),
    "bedroom_2": ((0.0, 4.1, 3.9, 7.6), "bedroom"),
    "bathroom": ((4.0, 5.3, 7.0, 7.6), "bathroom"),
}
PLAN_OPENINGS = {
    # name: (rect of the gap in the wall, width, has a door)
    "living_kitchen": ((6.4, 1.0, 6.5, 2.4), 1.4, False),
    "living_corridor": ((4.6, 4.0, 5.5, 4.1), 0.9, False),
    "corridor_bedroom_2": ((3.9, 4.25, 4.0, 5.05), 0.8, True),
    "corridor_bathroom": ((5.1, 5.2, 5.9, 5.3), 0.8, True),
    "corridor_bedroom_1": ((7.0, 4.25, 7.1, 5.05), 0.8, True),
    "front_door": ((5.3, -0.1, 6.1, 0.0), 0.8, True),
}
# What the plan's section 6.3 puts in each room (by kind).
PLAN_KINDS = {
    "living_room": {"sofa", "armchair", "coffee_table", "media_table", "tv", "bookshelf", "lamp", "plant", "rug",
                    "dining_table", "chair", "shoe_rack", "dock"},
    "kitchen": {"counter", "fridge", "stove", "table", "stool", "bin"},
    "corridor": {"console_table"},
    "bedroom_1": {"bed", "bedside_table", "wardrobe", "dresser", "chair", "rug"},
    "bedroom_2": {"bed", "desk", "chair", "bookshelf", "wardrobe", "toy_box"},
    "bathroom": {"toilet", "basin", "shower", "washing_machine", "basket"},
}


def inside(inner, outer):
    return (inner[0] >= outer[0] - EPS and inner[1] >= outer[1] - EPS
            and inner[2] <= outer[2] + EPS and inner[3] <= outer[3] + EPS)


def rects_overlap(a, b):
    return a[0] < b[2] - EPS and b[0] < a[2] - EPS and a[1] < b[3] - EPS and b[1] < a[3] - EPS


def colliding(thing):
    return [s for s in thing.shapes if s.collide]


# --- rooms, walls, openings --------------------------------------------------------------------

def test_rooms_are_the_plans():
    rooms = {r.name: r for r in home.rooms()}
    assert set(rooms) == set(PLAN_ROOMS)
    for name, (rect, kind) in PLAN_ROOMS.items():
        assert rooms[name].rect == pytest.approx(rect)
        assert rooms[name].kind == kind
        assert home.ROOMS[name] == rooms[name].rect
        assert home.FLOOR_TYPES[name] == rooms[name].floor_type
    assert sum(r.area for r in rooms.values()) == pytest.approx(77.6, abs=0.05)


def test_rooms_do_not_overlap_each_other_or_the_walls():
    rooms = home.rooms()
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            assert not rects_overlap(a.rect, b.rect), (a.name, b.name)
        for w in home.walls():
            assert not rects_overlap(a.rect, footprint(w)), (a.name, w.name)


def test_walls_are_ten_centimetres_thick_and_along_the_axes():
    for w in home.walls():
        if w.name in home.NOT_A_WALL:
            continue
        assert isinstance(w, Box), w.name
        assert min(w.x1 - w.x0, w.y1 - w.y0) == pytest.approx(home.WALL_T), w.name


def test_house_is_closed_except_through_its_openings():
    """Walking round each room just outside its edge meets a wall everywhere except in an opening."""
    walls = [footprint(w) for w in home.walls() if w.collide and w.z0 <= 0.05 and w.z1 >= 0.5]  # sills count
    openings = [o.rect for o in home.openings()]
    step = 0.01
    for room in home.rooms():
        x0, y0, x1, y1 = room.rect
        probes = []
        n = int(round((x1 - x0) / step))
        m = int(round((y1 - y0) / step))
        probes += [(x0 + (i + 0.5) * step, y0 - 0.05) for i in range(n)]
        probes += [(x0 + (i + 0.5) * step, y1 + 0.05) for i in range(n)]
        probes += [(x0 - 0.05, y0 + (j + 0.5) * step) for j in range(m)]
        probes += [(x1 + 0.05, y0 + (j + 0.5) * step) for j in range(m)]
        for px, py in probes:
            in_wall = any(r[0] <= px <= r[2] and r[1] <= py <= r[3] for r in walls)
            in_opening = any(r[0] <= px <= r[2] and r[1] <= py <= r[3] for r in openings)
            assert in_wall != in_opening, (room.name, px, py, in_wall, in_opening)


def test_openings_are_the_plans_and_open():
    openings = {o.name: o for o in home.openings()}
    assert set(openings) == set(PLAN_OPENINGS)
    low = [s for s in home.all_shapes() if s.collide and s.z0 < home.DOOR_H - EPS and s.z1 > 0.0]
    for name, (rect, width, has_door) in PLAN_OPENINGS.items():
        o = openings[name]
        assert o.rect == pytest.approx(rect), name
        assert max(rect[2] - rect[0], rect[3] - rect[1]) == pytest.approx(width), name
        assert bool(o.door) == has_door, name
        blockers = [s.name for s in low if rects_overlap(footprint(s), o.rect)]
        assert not blockers, (name, blockers)


def test_every_door_stands_open_beside_its_opening():
    things = {t.name: t for t in home.things()}
    for o in home.openings():
        if not o.door:
            continue
        door = things[o.door]
        assert door.kind == "door"
        assert door.room in o.rooms
        (leaf,) = door.shapes
        assert leaf.z0 == 0.0 and leaf.z1 == pytest.approx(home.DOOR_H)
        thin = min(leaf.x1 - leaf.x0, leaf.y1 - leaf.y0)
        long = max(leaf.x1 - leaf.x0, leaf.y1 - leaf.y0)
        assert thin == pytest.approx(0.04)
        width = max(o.rect[2] - o.rect[0], o.rect[3] - o.rect[1])
        assert width - 0.05 <= long <= width
        # at a right angle to the doorway, touching the opening's edge, and inside the gap's span
        wall_along_x = (o.rect[2] - o.rect[0]) > (o.rect[3] - o.rect[1])
        if wall_along_x:  # the gap runs along x, so the open leaf runs along y
            assert leaf.y1 - leaf.y0 == pytest.approx(long)
            assert leaf.y0 == pytest.approx(o.rect[3]) or leaf.y1 == pytest.approx(o.rect[1])
            assert o.rect[0] - EPS <= leaf.x0 and leaf.x1 <= o.rect[2] + EPS
        else:
            assert leaf.x1 - leaf.x0 == pytest.approx(long)
            assert leaf.x0 == pytest.approx(o.rect[2]) or leaf.x1 == pytest.approx(o.rect[0])
            assert o.rect[1] - EPS <= leaf.y0 and leaf.y1 <= o.rect[3] + EPS


def test_stairs_outside_the_front_door():
    x0, y0, x1, y1 = home.STAIRS
    door = next(o for o in home.openings() if o.name == "front_door").rect
    assert (x0, x1) == pytest.approx((door[0], door[2]))
    assert y1 == pytest.approx(door[1])  # the drop starts at the wall's outer face
    assert home.STAIRS_DROP == pytest.approx(0.15)
    # the landing is lower than the floor, outside the floor slab, and walled in on three sides
    slab = next(s for s in home.floors() if s.name == "floor_collision")
    assert y1 <= slab.y0 + EPS
    landing = next(s for s in home.walls() if s.name == "stairs_floor")
    assert landing.z1 == pytest.approx(-home.STAIRS_DROP)
    for px, py in [(x0 - 0.05, (y0 + y1) / 2), (x1 + 0.05, (y0 + y1) / 2), ((x0 + x1) / 2, y0 - 0.05)]:
        assert any(w.collide and w.z1 >= 1.0 and footprint(w)[0] <= px <= footprint(w)[2]
                   and footprint(w)[1] <= py <= footprint(w)[3] for w in home.walls()), (px, py)


def test_floor_is_one_slab_and_finishes_cover_every_room():
    colliders = [s for s in home.floors() if s.collide]
    assert [s.name for s in colliders] == ["floor_collision"]
    slab = colliders[0]
    assert slab.z1 == 0.0 and not slab.visible
    assert (slab.x0, slab.y0, slab.x1, slab.y1) == pytest.approx(home.FOOTPRINT)
    finishes = [s for s in home.floors() if s.visible]
    for r in home.rooms():
        assert any(footprint(f) == pytest.approx(r.rect) and f.floor_type == r.floor_type for f in finishes), r.name
    for o in home.openings():
        assert any(footprint(f) == pytest.approx(o.rect) for f in finishes), o.name


# --- objects ----------------------------------------------------------------------------------

def test_names_are_unique_and_known():
    shapes = home.all_shapes()
    names = [s.name for s in shapes]
    assert len(names) == len(set(names)), sorted(n for n in names if names.count(n) > 1)
    things = home.things()
    tnames = [t.name for t in things]
    assert len(tnames) == len(set(tnames))
    for t in things:
        assert t.room in home.ROOMS, t.name
        assert t.kind in home.KINDS, (t.name, t.kind)
        assert t.shapes, t.name


def test_furniture_and_dock_are_the_things():
    from_things = [s for t in home.things() for s in t.shapes]
    assert sorted(s.name for s in from_things) == sorted(s.name for s in home.furniture() + home.dock())
    assert {s.name for s in home.dock()} == {s.name for s in next(t for t in home.things() if t.kind == "dock").shapes}


def test_each_room_has_what_the_plan_lists():
    for room, kinds in PLAN_KINDS.items():
        have = {t.kind for t in home.things() if t.room == room}
        assert kinds <= have, (room, sorted(kinds - have))


def test_rule_1_inside_its_room():
    for t in home.things():
        room = home.ROOMS[t.room]
        for s in t.shapes:
            assert inside(footprint(s), room), (t.name, s.name)


def test_rule_1_no_two_things_overlap():
    things = home.things()
    for i, a in enumerate(things):
        for b in things[i + 1:]:
            for sa in colliding(a):
                for sb in colliding(b):
                    assert not overlaps_3d(sa, sb), (sa.name, sb.name)


def test_rule_3_nothing_the_robot_can_hit_is_hidden_from_the_lidar():
    for t in home.things():
        for s in colliding(t):
            if s.z0 < CLEAR_UNDER:
                assert s.z0 <= LIDAR_Z - 0.02 and s.z1 >= LIDAR_Z + 0.02, (t.name, s.name, s.z0, s.z1)


def test_rule_4_on_the_floor_or_enough_space_underneath():
    for t in home.things():
        for s in colliding(t):
            assert s.z0 == 0.0 or s.z0 >= CLEAR_UNDER - EPS, (t.name, s.name, s.z0)
        if t.kind != "tv" and colliding(t):  # the television stands on the media table; rugs are flat
            assert any(s.z0 == 0.0 for s in colliding(t)), t.name


def test_rule_5_dock_has_clear_floor_in_front_and_beside():
    zone = (0.0, home.DOCK_Y - 0.5, 1.6, home.DOCK_Y + 0.5)  # 1.5 m in front of the tower, 50 cm either side
    dock_names = {s.name for s in home.dock()}
    for t in home.things():
        for s in colliding(t):
            if s.name in dock_names or s.z0 >= CLEAR_UNDER:
                continue
            assert not rects_overlap(footprint(s), zone), (t.name, s.name)


def test_dock_and_start_pose_match_the_house():
    """docking.yaml, the tag settings and the start pose are shared by both worlds."""
    sx, sy, syaw = home.START_POSE_HOUSE
    hx, hy, hyaw = house.START_POSE_HOUSE
    assert syaw == hyaw == 0.0
    assert sy == pytest.approx(home.DOCK_Y)
    assert hy == pytest.approx(house.DOCK_Y)

    def relative(module, rect):
        x, y = module.START_POSE_HOUSE[:2]
        return tuple(round(v, 9) for v in (rect[0] - x, rect[1] - y, rect[2] - x, rect[3] - y))

    assert relative(home, home.DOCK_TOWER) == relative(house, house.DOCK_TOWER)
    assert relative(home, home.DOCK_PLATE) == relative(house, house.DOCK_PLATE)
    assert home.DOCK_TOWER[0] == 0.0  # against the west wall
    assert home.DOCK_TOWER_H == house.DOCK_TOWER_H
    tag = home.DOCK_TAG_HOUSE
    htag = house.DOCK_TAG_HOUSE
    assert (tag[0] - sx, tag[1] - sy, tag[2]) == pytest.approx((htag[0] - hx, htag[1] - hy, htag[2]))
    assert (home.DOCK_TAG_FAMILY, home.DOCK_TAG_ID, home.DOCK_TAG_SIZE, home.DOCK_TAG_TEXTURE) == (
        house.DOCK_TAG_FAMILY, house.DOCK_TAG_ID, house.DOCK_TAG_SIZE, house.DOCK_TAG_TEXTURE)


# --- the robot's view and reach ---------------------------------------------------------------

def test_frames_and_ground_truth_ranges():
    assert home.to_world(*home.START_POSE_HOUSE[:2]) == (0.0, 0.0)
    assert home.to_house(0.0, 0.0) == pytest.approx(home.START_POSE_HOUSE[:2])
    # straight back from the start: the dock tower's face; straight left: the living room's north wall
    assert home.expected_range(0.0, 0.0, math.pi, LIDAR_Z, hl.LIDAR_MAX_RANGE) == pytest.approx(0.39)
    assert home.expected_range(0.0, 0.0, math.pi / 2, LIDAR_Z, hl.LIDAR_MAX_RANGE) == pytest.approx(
        home.ROOMS["living_room"][3] - home.START_POSE_HOUSE[1])
    # shapes the lidar plane crosses: no rugs, no table tops, every leg
    boxes, cyls = home.shapes_at_height(LIDAR_Z)
    n_low = sum(1 for s in home.walls() + home.furniture() + home.dock() if s.collide and s.z0 <= LIDAR_Z <= s.z1)
    assert len(boxes) + len(cyls) == n_low
    assert home.room_world("kitchen") == pytest.approx(
        home.to_world(6.5, 0.0) + home.to_world(10.6, 3.2))


@pytest.fixture(scope="module")
def reachable():
    return reach.reachable(home, PATH_HALF_WIDTH)


def test_rule_2_a_wide_path_reaches_every_room(reachable):
    for room in home.rooms():  # at least 1 m2 of each room is reachable on a 60 cm wide path
        assert reachable.in_rect(room.rect) * reach.RES ** 2 >= 1.0, room.name


def test_bedroom_1_bed_can_be_cleaned_underneath(reachable):
    bed = next(t for t in home.things() if t.name == "bed_double")
    frame = next(s for s in bed.shapes if s.name == "bed_double_frame")
    assert frame.z0 >= CLEAR_UNDER - EPS
    assert reachable.at(((frame.x0 + frame.x1) / 2, (frame.y0 + frame.y1) / 2))


def test_bedroom_2_gap_beside_the_bed_is_too_narrow_for_the_robot():
    bed = next(t for t in home.things() if t.name == "bed_single")
    room = home.ROOMS["bedroom_2"]
    x0 = min(footprint(s)[0] for s in colliding(bed))
    assert x0 - room[0] == pytest.approx(0.30)
    assert 0.30 < 2 * hl.ROBOT_RADIUS
    tight = reach.reachable(home, hl.ROBOT_RADIUS)
    gap = (room[0], min(footprint(s)[1] for s in colliding(bed)), x0, room[3])
    assert tight.in_rect(gap) == 0


def test_bedroom_1_dead_end_gap_beside_the_wardrobe():
    wardrobe = next(t for t in home.things() if t.name == "wardrobe_bedroom_1")
    (w,) = colliding(wardrobe)
    room = home.ROOMS["bedroom_1"]
    assert w.x0 - room[0] == pytest.approx(0.50)  # wide enough to drive into, closed at the end
    assert w.y1 == pytest.approx(room[3])


# --- the picture ------------------------------------------------------------------------------

def test_plan_picture_is_drawn(tmp_path):
    import draw_plan

    out = tmp_path / "plan.png"
    labels = draw_plan.draw(home, str(out))
    assert out.stat().st_size > 50_000
    assert {t.name for t in home.things()} <= set(labels)  # every object has its name on the picture
    assert {r.name for r in home.rooms()} <= set(labels)


def test_floor_collider_is_thick_enough():
    """PhysX finding (2026-11-20): on the home's 10.8 x 7.8 m floor slab only 10 cm thick, the robot
    rested 7 mm low and tipped back 3 degrees, and could not turn. The same slab 1 m thick (top
    still at z = 0) gives exactly the house's rest pose. The visible floors stay 10 cm."""
    slab = next(s for s in home.floors() if s.name == "floor_collision")
    assert slab.z1 == 0.0
    assert slab.z1 - slab.z0 >= 0.5
    assert max(slab.x1 - slab.x0, slab.y1 - slab.y0) / (slab.z1 - slab.z0) <= 25.0


# --- looks above the robot: windows, ceiling, lamps ---------------------------------------------

def test_windows_are_above_the_robot_in_outer_walls():
    glass = home.windows()
    assert len(glass) == sum(len(v) for v in home.WINDOWS.values()) >= 6
    for g in glass:
        assert not g.collide and g.z0 >= 0.8 and g.z1 <= home.DOOR_H + 0.1, g.name
        # outside every room: in the wall
        assert not any(rects_overlap(footprint(g), r.rect) for r in home.rooms()), g.name
    for wall, spans in home.WINDOWS.items():
        assert wall in ("wall_west", "wall_south", "wall_east", "wall_north")
        for w0, w1, sill, top in spans:
            sill_box = next(w for w in home.walls() if w.name.startswith(wall + "_sill_") and
                            (w.x0 if (w.x1 - w.x0) > (w.y1 - w.y0) else w.y0) == pytest.approx(w0))
            assert sill_box.z0 == 0.0 and sill_box.z1 == pytest.approx(sill)


def test_ceiling_and_lamps():
    ceil = [s for s in home.ceiling() if s.name.endswith("_ceiling")]
    for r in home.rooms():
        c = next(s for s in ceil if s.name == f"{r.name}_ceiling")
        assert footprint(c) == pytest.approx(r.rect) and c.z1 == pytest.approx(home.WALL_H) and not c.collide
    lamps = home.lamps()
    assert sorted(room for _, room, _, _ in lamps) == sorted(home.ROOMS)
    for name, room, x, y in lamps:
        x0, y0, x1, y1 = home.ROOMS[room]
        assert x0 < x < x1 and y0 < y < y1, name
    assert all(not s.collide for s in home.ceiling())
