"""Two worlds, one switch (no simulator): house_layout.py hands out the chosen world's layout.

Run: python3 -m pytest sim/tests/test_worlds.py
"""

import dataclasses
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.normpath(os.path.join(HERE, "..", "scripts"))
sys.path.insert(0, SCRIPTS)
os.environ.pop("VACUUM_WORLD", None)

import house_layout as hl  # noqa: E402

SNAPSHOT = os.path.join(HERE, "data", "house_layout_snapshot.json")


def _json(value):
    """The value as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(value))


def _shape(s):
    return {"class": type(s).__name__, **dataclasses.asdict(s)}


@pytest.fixture(scope="module")
def snap():
    with open(SNAPSHOT) as f:
        return json.load(f)


def test_default_world_is_the_house():
    assert hl.WORLD == "house"


def test_house_constants_are_what_they_were(snap):
    for name, value in snap["constants"].items():
        assert _json(getattr(hl, name)) == value, name


@pytest.mark.parametrize("part", ["floors", "walls", "furniture", "dock", "all_shapes"])
def test_house_shapes_are_what_they_were(snap, part):
    assert [_shape(s) for s in getattr(hl, part)()] == snap[part]


def test_house_ground_truth_is_what_it_was(snap):
    assert _json(hl.obstacles_world()) == snap["obstacles_world"]
    for z, value in snap["shapes_at_height"].items():
        assert _json(hl.shapes_at_height(float(z))) == value, z
    for name, value in snap["room_world"].items():
        assert _json(hl.room_world(name)) == value
    assert _json([hl.to_world(x, y) for x, y in [(0, 0), (1.5, -2.25), (8.6, 4.5)]]) == snap["to_world"]
    assert _json([hl.to_house(x, y) for x, y in [(0, 0), (1.5, -2.25), (8.6, 4.5)]]) == snap["to_house"]
    for ox, oy, a, r in snap["expected_range"][:-1]:
        assert hl.expected_range(ox, oy, a, hl.LIDAR_XYZ[2]) == r, (ox, oy, a)
    ox, oy, a, r = snap["expected_range"][-1]
    assert hl.expected_range(ox, oy, a, 0.3, 5.0) == r


def test_house_shapes_are_the_shared_classes():
    from worlds.shapes import Box, Cylinder

    assert hl.Box is Box and hl.Cylinder is Cylinder
    assert all(isinstance(s, (Box, Cylinder)) for s in hl.all_shapes())


def test_every_world_gives_its_outline():
    from worlds import home, house

    for world in (house, home):
        x0, y0, x1, y1 = world.FOOTPRINT
        slab = next(s for s in world.floors() if s.name == "floor_collision")
        assert (slab.x0, slab.y0, slab.x1, slab.y1) == pytest.approx((x0, y0, x1, y1))


def _in_world(world, code):
    env = dict(os.environ)
    if world is None:
        env.pop("VACUUM_WORLD", None)
    else:
        env["VACUUM_WORLD"] = world
    out = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {SCRIPTS!r}); {code}"],
                         env=env, capture_output=True, text=True)
    return out


def test_home_world_through_house_layout():
    code = (
        "import house_layout as hl; from worlds import home; "
        "assert hl.WORLD == 'home'; "
        "assert [s.name for s in hl.all_shapes()] == [s.name for s in home.all_shapes()]; "
        "assert hl.START_POSE_HOUSE == home.START_POSE_HOUSE and hl.FOOTPRINT == home.FOOTPRINT; "
        "assert hl.expected_range(0.0, 0.0, 3.14159265, hl.LIDAR_XYZ[2]) == home.expected_range(0.0, 0.0, 3.14159265, hl.LIDAR_XYZ[2], hl.LIDAR_MAX_RANGE); "
        "assert hl.ROBOT_RADIUS == 0.171; "
        # the house-only names the 32 house phases use are not there: they cannot run on the home by mistake
        "assert not any(hasattr(hl, n) for n in ('DOORWAY', 'STAIRWELL', 'BEDROOM', 'LIVING_ROOM')); "
        "print('OK')"
    )
    out = _in_world("home", code)
    assert out.stdout.strip() == "OK", out.stderr


def test_unknown_world_is_refused_by_name():
    out = _in_world("castle", "import house_layout")
    assert out.returncode != 0
    assert "castle" in out.stderr and "house" in out.stderr and "home" in out.stderr


def test_world_names():
    out = _in_world(None, "import house_layout as hl; print(sorted(hl.WORLDS))")
    assert out.stdout.strip() == "['home', 'house']", out.stderr


def test_floor_areas_and_drops():
    from worlds import home, house

    # the house: exactly what the house's phases measured against before (rooms + doorway, the stairwell)
    assert house.floor_areas() == [house.LIVING_ROOM, house.BEDROOM, house.DOORWAY]
    assert house.drops() == [house.STAIRWELL]
    assert home.floor_areas() == [r.rect for r in home.rooms()] + [o.rect for o in home.openings()]
    assert home.drops() == [home.STAIRS]
