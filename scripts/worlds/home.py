"""The `home` world: a two-bedroom home of six rooms (77.6 m2), as plain data.

Plan: docs/2026-11-19-bigger-home-plan.md, sections 6.2 and 6.3. Picture: `python3 sim/scripts/draw_plan.py`.
No Isaac Sim imports (see shapes.py). Same frames as the house:
  * house frame: origin at the inside south-west corner of the living room, x east, y north.
  * world frame: origin at the robot's start pose in front of the dock, x into the room.

What collides is the ground truth: boxes and cylinders. Every object is a Thing with a name, a kind
and a room. Rules every object obeys are tested in sim/tests/test_home_layout.py.
"""

from .shapes import Box, Cylinder, Opening, Room, Thing, chair, ray_range, seat_with_back, table

WALL_T = 0.10  # wall thickness
WALL_H = 2.40  # wall height
DOOR_H = 2.00  # height of every opening (a lintel closes the wall above it)
FLOOR_T = 0.10  # thickness of the visible floors (top surface at z = 0)
# Thickness of the one invisible floor collider. PhysX finding: on a 10.8 x 7.8 m box only 10 cm
# thick, the robot rested 7 mm low, tipped back 3 degrees and could not turn; 1 m thick, it rests
# exactly as in the house (whose 8.8 x 4.7 m slab is 10 cm thick and fine). Top still at z = 0.
SLAB_T = 1.0
EPS = 1e-9

_ROOMS = (
    Room("living_room", "living_room", (0.0, 0.0, 6.4, 4.0), "hardwood"),
    Room("kitchen", "kitchen", (6.5, 0.0, 10.6, 3.2), "tile"),
    Room("corridor", "corridor", (4.0, 4.1, 7.0, 5.2), "hardwood"),
    Room("bedroom_1", "bedroom", (7.1, 3.3, 10.6, 7.6), "carpet"),
    Room("bedroom_2", "bedroom", (0.0, 4.1, 3.9, 7.6), "carpet"),
    Room("bathroom", "bathroom", (4.0, 5.3, 7.0, 7.6), "tile"),
)
ROOMS = {r.name: r.rect for r in _ROOMS}
FLOOR_TYPES = {r.name: r.floor_type for r in _ROOMS}
FLOOR_MATERIALS = {"hardwood": "floor_wood", "tile": "floor_tile", "carpet": "floor_carpet"}

# Outer faces of the outer walls (the floor slab), not counting the stairs outside the front door.
FOOTPRINT = (-WALL_T, -WALL_T, 10.6 + WALL_T, 7.6 + WALL_T)

_OPENINGS = (
    Opening("living_kitchen", (6.4, 1.0, 6.5, 2.4), ("living_room", "kitchen")),  # an archway
    Opening("living_corridor", (4.6, 4.0, 5.5, 4.1), ("living_room", "corridor")),
    Opening("corridor_bedroom_2", (3.9, 4.25, 4.0, 5.05), ("corridor", "bedroom_2"), "door_bedroom_2"),
    Opening("corridor_bathroom", (5.1, 5.2, 5.9, 5.3), ("corridor", "bathroom"), "door_bathroom"),
    Opening("corridor_bedroom_1", (7.0, 4.25, 7.1, 5.05), ("corridor", "bedroom_1"), "door_bedroom_1"),
    Opening("front_door", (5.3, -0.1, 6.1, 0.0), ("living_room", "outside"), "door_front"),
)

# Outside the front door: a landing STAIRS_DROP lower, walled in on three sides. The lidar sees it
# as open floor; only the cliff sensors and a no-go zone keep the robot out. It lies outside the
# floor slab, so the home's floor stays one seamless collider.
STAIRS = (5.3, -0.9, 6.1, -WALL_T)
STAIRS_DROP = 0.15

# Shapes in walls() that are not 10 cm walls.
NOT_A_WALL = {"service_shaft", "stairs_floor"}

# Robot start pose (x, y, yaw) and the dock: the same relative to each other as in the house, so
# docking.yaml, the tag settings and the start pose serve both worlds.
START_POSE_HOUSE = (0.45, 2.0, 0.0)
DOCK_Y = 2.0
DOCK_TOWER = (0.0, DOCK_Y - 0.11, 0.06, DOCK_Y + 0.11)
DOCK_TOWER_H = 0.16
DOCK_PLATE = (0.06, DOCK_Y - 0.14, 0.30, DOCK_Y + 0.14)
DOCK_TAG_FAMILY = "36h11"
DOCK_TAG_ID = 0
DOCK_TAG_SIZE = 0.10
DOCK_TAG_TEXTURE = "textures/apriltag_36h11_0.png"
DOCK_TAG_HOUSE = (DOCK_TOWER[2], DOCK_Y, 0.075)

# The kinds of things in this world (names for programs).
KINDS = {
    "sofa", "armchair", "coffee_table", "media_table", "tv", "bookshelf", "lamp", "plant", "rug",
    "dining_table", "chair", "shoe_rack", "dock", "door", "counter", "fridge", "stove", "table",
    "stool", "bin", "console_table", "bed", "bedside_table", "wardrobe", "dresser", "desk", "toy_box",
    "toilet", "basin", "shower", "washing_machine", "basket",
}

# How each room's floor looks: a picture (made by sim/scripts/make_textures.py) and the size of
# floor it covers in metres, so boards and tiles have their true size. Doorways take the look of
# the first room they join. Looks only: the robot rolls on the one invisible collider.
FLOOR_LOOKS = {
    "living_room": ("textures/floor_wood.png", 1.8),
    "corridor": ("textures/floor_wood.png", 1.8),
    "kitchen": ("textures/floor_tile.png", 1.2),
    "bathroom": ("textures/floor_tile_bath.png", 0.6),
    "bedroom_1": ("textures/floor_carpet.png", 1.0),
    "bedroom_2": ("textures/floor_carpet_beige.png", 1.0),
}


def floor_look(floor_name):
    """(texture, metres) for a visible floor shape ('<room>_floor' or '<opening>_floor'), or None."""
    for r in _ROOMS:
        if floor_name == f"{r.name}_floor":
            return FLOOR_LOOKS.get(r.name)
    for o in _OPENINGS:
        if floor_name == f"{o.name}_floor":
            return FLOOR_LOOKS.get(o.rooms[0])
    return None


# name -> (r, g, b), roughness
MATERIALS = {
    "floor_wood": ((0.55, 0.38, 0.22), 0.55),
    "floor_carpet": ((0.42, 0.45, 0.52), 0.95),
    "floor_tile": ((0.78, 0.78, 0.76), 0.35),
    "wall": ((0.92, 0.90, 0.86), 0.9),
    "sofa": ((0.25, 0.33, 0.45), 0.9),
    "wood_dark": ((0.30, 0.20, 0.13), 0.5),
    "wood_light": ((0.72, 0.58, 0.40), 0.5),
    "metal": ((0.55, 0.56, 0.58), 0.3),
    "steel": ((0.68, 0.69, 0.70), 0.25),
    "fabric_light": ((0.85, 0.83, 0.78), 0.9),
    "bedding": ((0.55, 0.65, 0.78), 0.9),
    "rug": ((0.62, 0.25, 0.22), 0.95),
    "rug_blue": ((0.25, 0.35, 0.55), 0.95),
    "plant_pot": ((0.75, 0.42, 0.28), 0.7),
    "plant_leaf": ((0.18, 0.42, 0.20), 0.8),
    "lamp_shade": ((0.95, 0.90, 0.78), 0.9),
    "screen": ((0.04, 0.04, 0.05), 0.2),
    "window_glass": ((0.80, 0.90, 0.95), 0.05),
    "ceiling": ((0.95, 0.95, 0.93), 0.9),
    "cabinet": ((0.82, 0.80, 0.74), 0.5),
    "white_gloss": ((0.93, 0.93, 0.92), 0.25),
    "glass": ((0.75, 0.85, 0.90), 0.05),
    "door": ((0.95, 0.94, 0.90), 0.6),
    "toy": ((0.85, 0.30, 0.25), 0.6),
    "wicker": ((0.70, 0.58, 0.38), 0.8),
    "dock": ((0.10, 0.10, 0.12), 0.4),
}


def rooms():
    """The rooms, with their kind and floor."""
    return list(_ROOMS)


def openings():
    """The gaps in the walls: archways, doorways and the front door."""
    return list(_OPENINGS)


def floor_areas():
    """Where the floor is, as rectangles (x0, y0, x1, y1) in the house frame: the rooms and the openings."""
    return [r.rect for r in _ROOMS] + [o.rect for o in _OPENINGS]


def drops():
    """Floors lower than the home's, which only the cliff sensors find: the stairs outside the front door."""
    return [STAIRS]


def to_world(x, y):
    """Convert a house-frame point to the world frame."""
    return (x - START_POSE_HOUSE[0], y - START_POSE_HOUSE[1])


def to_house(x, y):
    """Convert a world-frame point to the house frame."""
    return (x + START_POSE_HOUSE[0], y + START_POSE_HOUSE[1])


def room_world(name):
    """Room rectangle (x0, y0, x1, y1) in the world frame."""
    x0, y0, x1, y1 = ROOMS[name]
    return to_world(x0, y0) + to_world(x1, y1)


def floors():
    """One invisible collision slab under the whole home, and a visible finish per room and doorway.

    The robot must roll on one continuous collider: with one slab per room the caster ball catches
    on the seams.
    """
    x0, y0, x1, y1 = FOOTPRINT
    out = [Box("floor_collision", "Floor", x0, y0, x1, y1, -SLAB_T, 0.0, "floor_wood", True, "", False)]
    for r in _ROOMS:
        rx0, ry0, rx1, ry1 = r.rect
        out.append(Box(f"{r.name}_floor", "Floor", rx0, ry0, rx1, ry1, -FLOOR_T, 0.0,
                       FLOOR_MATERIALS[r.floor_type], False, r.floor_type))
    for o in _OPENINGS:
        ft = FLOOR_TYPES[o.rooms[0]]
        ox0, oy0, ox1, oy1 = o.rect
        out.append(Box(f"{o.name}_floor", "Floor", ox0, oy0, ox1, oy1, -FLOOR_T, 0.0, FLOOR_MATERIALS[ft], False, ft))
    return out


# Windows in the outer walls: wall name -> [(from, to along the wall, sill height, top)]. All sit
# well above the robot and the lidar plane, so for the robot a wall with a window is still a wall.
WINDOWS = {
    "wall_west": [(0.5, 1.3, 0.9, 2.1), (4.6, 5.3, 0.9, 2.1)],  # living room, bedroom 2
    "wall_south": [(1.2, 2.5, 0.9, 2.1), (8.2, 9.6, 1.05, 2.1)],  # living room (behind the sofa), kitchen
    "wall_east": [(2.1, 2.7, 0.9, 2.1)],  # kitchen
    "wall_north": [(4.15, 4.95, 1.4, 2.0), (9.6, 10.4, 0.9, 2.1)],  # bathroom, bedroom 1
}


def _wall(name, axis, a0, a1, c0, gaps=()):
    """A wall WALL_T thick running along `axis` ('x' or 'y') from a0 to a1, from c0 across.

    Each gap (g0, g1) is left open up to DOOR_H and closed above by a lintel. Each window the
    WINDOWS table gives this wall leaves an opening between its sill and its top (the glass is in
    windows()).
    """

    def rect(p0, p1):
        return (p0, c0, p1, c0 + WALL_T) if axis == "x" else (c0, p0, c0 + WALL_T, p1)

    gaps = sorted(gaps)
    windows = sorted(WINDOWS.get(name, []))
    cuts = sorted([v for g in gaps for v in g] + [v for w in windows for v in w[:2]])
    cuts = [a0] + cuts + [a1]
    pieces = [(p0, p1) for p0, p1 in zip(cuts[0::2], cuts[1::2]) if p1 - p0 > EPS]
    out = []
    for i, (p0, p1) in enumerate(pieces):
        piece = name if len(pieces) == 1 else f"{name}_{i}"
        out.append(Box(piece, "Walls", *rect(p0, p1), 0.0, WALL_H, "wall"))
    for i, (g0, g1) in enumerate(gaps):
        out.append(Box(f"{name}_lintel_{i}", "Walls", *rect(g0, g1), DOOR_H, WALL_H, "wall"))
    for i, (w0, w1, sill, top) in enumerate(windows):
        out.append(Box(f"{name}_sill_{i}", "Walls", *rect(w0, w1), 0.0, sill, "wall"))
        out.append(Box(f"{name}_header_{i}", "Walls", *rect(w0, w1), top, WALL_H, "wall"))
    return out


def windows():
    """The glass in each window: drawn only (it is above the robot and the lidar plane)."""
    out = []
    for wall in walls():
        if "_sill_" not in wall.name:
            continue
        header = next(w for w in walls() if w.name == wall.name.replace("_sill_", "_header_"))
        along_x = (wall.x1 - wall.x0) > (wall.y1 - wall.y0)
        mid = (wall.y0 + wall.y1) / 2 if along_x else (wall.x0 + wall.x1) / 2
        x0, y0, x1, y1 = (wall.x0, mid - 0.01, wall.x1, mid + 0.01) if along_x else (mid - 0.01, wall.y0, mid + 0.01, wall.y1)
        out.append(Box(wall.name.replace("_sill_", "_glass_"), "Windows", x0, y0, x1, y1, wall.z1, header.z0, "window_glass", False))
    return out


def ceiling():
    """A ceiling over every room and doorway at the top of the walls, and a lamp shade in each room.
    Drawn only, and only from below (build_world.py makes the ceiling one-sided, so the overview
    camera above the home sees in)."""
    out = [Box(f"{r.name}_ceiling", "Ceiling", *r.rect, WALL_H - 0.005, WALL_H, "ceiling", False) for r in _ROOMS]
    out += [Box(f"{o.name}_ceiling", "Ceiling", *o.rect, WALL_H - 0.005, WALL_H, "ceiling", False) for o in _OPENINGS
            if o.rooms[1] != "outside"]
    for name, _room, x, y in lamps():
        out.append(Cylinder(f"{name}_shade", "Ceiling", x, y, 0.18, WALL_H - 0.12, WALL_H - 0.005, "lamp_shade", False))
    return out


def lamps():
    """One ceiling lamp per room, over the middle of the room: (name, room, x, y). Its light hangs
    LAMP_DROP below the ceiling."""
    return [(f"{r.name}_lamp", r.name, (r.rect[0] + r.rect[2]) / 2, (r.rect[1] + r.rect[3]) / 2) for r in _ROOMS]


LAMP_DROP = 0.15


def walls():
    """Outer and inner walls with their openings, the closed service shaft, and the stairs landing."""
    x_end, y_end = 10.6, 7.6
    sx0, sy0, sx1, sy1 = STAIRS
    low = -STAIRS_DROP
    out = []
    out += _wall("wall_south", "x", -WALL_T, x_end + WALL_T, -WALL_T, [(5.3, 6.1)])
    out += _wall("wall_north", "x", -WALL_T, x_end + WALL_T, y_end)
    out += _wall("wall_west", "y", 0.0, y_end, -WALL_T)
    out += _wall("wall_east", "y", 0.0, y_end, x_end)
    out += _wall("wall_living_north", "x", 0.0, 7.0, 4.0, [(4.6, 5.5)])
    out += _wall("wall_living_east", "y", 0.0, 4.0, 6.4, [(1.0, 2.4)])
    out += _wall("wall_kitchen_north", "x", 6.5, x_end, 3.2)
    out += _wall("wall_bedroom_1_west", "y", 3.3, y_end, 7.0, [(4.25, 5.05)])
    out += _wall("wall_bedroom_2_east", "y", 4.1, y_end, 3.9, [(4.25, 5.05)])
    out += _wall("wall_bathroom_south", "x", 4.0, 7.0, 5.2, [(5.1, 5.9)])
    # the corner between the living room, the kitchen and bedroom 1 is closed, like a service shaft
    out.append(Box("service_shaft", "Walls", 6.5, 3.3, 7.0, 4.0, 0.0, WALL_H, "wall"))
    # the stairs landing outside the front door, lower than the floor, walled in on three sides
    out += [
        Box("stairs_floor", "Walls", sx0 - WALL_T, sy0 - WALL_T, sx1 + WALL_T, sy1, low - FLOOR_T, low, "floor_wood"),
        Box("stairs_wall_west", "Walls", sx0 - WALL_T, sy0, sx0, sy1, low, WALL_H, "wall"),
        Box("stairs_wall_east", "Walls", sx1, sy0, sx1 + WALL_T, sy1, low, WALL_H, "wall"),
        Box("stairs_wall_south", "Walls", sx0 - WALL_T, sy0 - WALL_T, sx1 + WALL_T, sy0, low, WALL_H, "wall"),
    ]
    return out


def _door(name, room, x0, y0, x1, y1):
    """A door leaf standing open at a right angle to its doorway. It does not move."""
    return Thing(name, "door", room, (Box(name, "Doors", x0, y0, x1, y1, 0.0, DOOR_H, "door"),))


def _thing(name, kind, room, *shapes):
    return Thing(name, kind, room, tuple(shapes))


def _living_room():
    f, r = "Furniture", "living_room"
    return [
        _thing("sofa", "sofa", r, *seat_with_back("sofa", f, 0.85, 0.05, 2.85, 0.95, "s", "sofa")),
        _thing("armchair", "armchair", r, *seat_with_back("armchair", f, 3.35, 0.10, 4.15, 0.90, "s", "fabric_light")),
        _thing("coffee_table", "coffee_table", r,
               *table("coffee_table", f, 1.8, 1.5, 2.8, 2.1, 0.44, 0.04, 0.04, 0.03, "wood_light", "metal")),
        _thing("media_table", "media_table", r, Box("media_table", f, 1.05, 3.55, 2.65, 4.0, 0.0, 0.50, "wood_dark")),
        _thing("tv", "tv", r,
               Box("tv_stand", f, 1.75, 3.72, 1.95, 3.86, 0.50, 0.56, "screen"),
               Box("tv_screen", f, 1.25, 3.76, 2.45, 3.82, 0.56, 1.26, "screen")),
        _thing("bookshelf", "bookshelf", r, Box("bookshelf", f, 0.0, 3.0, 0.35, 4.0, 0.0, 1.90, "wood_light")),
        _thing("floor_lamp", "lamp", r,
               Cylinder("floor_lamp_base", f, 0.30, 0.30, 0.12, 0.0, 0.01, "metal", False),
               Cylinder("floor_lamp_post", f, 0.30, 0.30, 0.02, 0.0, 1.45, "metal"),
               Cylinder("floor_lamp_shade", f, 0.30, 0.30, 0.20, 1.30, 1.60, "lamp_shade", False)),
        _thing("plant", "plant", r,
               Cylinder("plant_pot", f, 3.30, 3.68, 0.20, 0.0, 0.40, "plant_pot"),
               Cylinder("plant_leaves", f, 3.30, 3.68, 0.30, 0.40, 1.20, "plant_leaf", False)),
        _thing("rug_living", "rug", r, Box("rug_living", f, 1.6, 1.15, 3.4, 2.45, 0.001, 0.005, "rug", False)),
        _thing("dining_table", "dining_table", r,
               *table("dining_table", f, 4.2, 1.9, 5.6, 2.7, 0.75, 0.04, 0.05, 0.05, "wood_dark", "wood_dark")),
        _thing("dining_chair_1", "chair", r,
               *chair("dining_chair_1", f, 4.34, 1.43, 0.42, "s", 0.46, 0.90, 0.035, "wood_light", "wood_light")),
        _thing("dining_chair_2", "chair", r,
               *chair("dining_chair_2", f, 5.04, 1.43, 0.42, "s", 0.46, 0.90, 0.035, "wood_light", "wood_light")),
        _thing("dining_chair_3", "chair", r,
               *chair("dining_chair_3", f, 4.34, 2.75, 0.42, "n", 0.46, 0.90, 0.035, "wood_light", "wood_light")),
        _thing("dining_chair_4", "chair", r,
               *chair("dining_chair_4", f, 5.04, 2.75, 0.42, "n", 0.46, 0.90, 0.035, "wood_light", "wood_light")),
        _thing("shoe_rack", "shoe_rack", r, Box("shoe_rack", f, 6.1, 0.1, 6.4, 0.9, 0.0, 0.50, "wood_light")),
        _door("door_front", r, 5.30, 0.0, 5.34, 0.78),
    ]


def _kitchen():
    f, r = "Furniture", "kitchen"
    return [
        _thing("fridge", "fridge", r, Box("fridge", f, 6.55, 0.05, 7.25, 0.75, 0.0, 1.85, "steel")),
        _thing("counter", "counter", r,  # cabinets in an L, solid down to the floor
               Box("counter_south_west", f, 7.25, 0.0, 8.25, 0.6, 0.0, 0.90, "cabinet"),
               Box("counter_south_east", f, 8.85, 0.0, 10.6, 0.6, 0.0, 0.90, "cabinet"),
               Box("counter_east", f, 10.0, 0.6, 10.6, 1.8, 0.0, 0.90, "cabinet")),
        _thing("stove", "stove", r, Box("stove", f, 8.25, 0.0, 8.85, 0.6, 0.0, 0.92, "steel")),
        _thing("kitchen_table", "table", r,
               *table("kitchen_table", f, 7.6, 2.55, 8.6, 3.15, 0.75, 0.03, 0.04, 0.03, "wood_light", "metal")),
        _thing("stool_1", "stool", r, *table("stool_1", f, 7.70, 2.10, 8.04, 2.44, 0.65, 0.03, 0.03, 0.01, "wood_dark", "metal")),
        _thing("stool_2", "stool", r, *table("stool_2", f, 8.16, 2.10, 8.50, 2.44, 0.65, 0.03, 0.03, 0.01, "wood_dark", "metal")),
        _thing("bin", "bin", r, Cylinder("bin", f, 10.30, 2.85, 0.15, 0.0, 0.60, "steel")),
    ]


def _corridor():
    f, r = "Furniture", "corridor"
    return [
        _thing("console_table", "console_table", r,
               *table("console_table", f, 5.8, 4.1, 6.8, 4.4, 0.80, 0.03, 0.03, 0.02, "wood_dark", "wood_dark")),
    ]


def _bedroom_1():
    f, r = "Furniture", "bedroom_1"
    legs = [Box(f"bed_double_leg_{i}", f, x, y, x + 0.06, y + 0.06, 0.0, 0.20, "wood_light")
            for i, (x, y) in enumerate([(8.52, 4.62), (10.42, 4.62), (8.52, 6.12), (10.42, 6.12)])]
    return [
        # a double bed on legs, 20 cm of space underneath: the robot drives and cleans under it
        _thing("bed_double", "bed", r, *legs,
               Box("bed_double_frame", f, 8.5, 4.6, 10.5, 6.2, 0.20, 0.45, "wood_light"),
               Box("bed_double_mattress", f, 8.55, 4.65, 10.45, 6.15, 0.45, 0.65, "bedding"),
               Box("bed_double_headboard", f, 10.5, 4.6, 10.6, 6.2, 0.0, 1.10, "wood_light")),
        _thing("bedside_table_1", "bedside_table", r, Box("bedside_table_1", f, 10.15, 4.10, 10.6, 4.55, 0.0, 0.55, "wood_dark")),
        _thing("bedside_table_2", "bedside_table", r, Box("bedside_table_2", f, 10.15, 6.25, 10.6, 6.70, 0.0, 0.55, "wood_dark")),
        # 50 cm between the wardrobe and the west wall: a dead end the robot can drive into
        _thing("wardrobe_bedroom_1", "wardrobe", r, Box("wardrobe_bedroom_1", f, 7.6, 7.0, 9.4, 7.6, 0.0, 2.10, "wood_light")),
        _thing("dresser", "dresser", r, Box("dresser", f, 8.0, 3.3, 9.2, 3.8, 0.0, 0.85, "wood_dark")),
        _thing("chair_bedroom_1", "chair", r,
               *chair("chair_bedroom_1", f, 7.45, 6.20, 0.42, "w", 0.46, 0.90, 0.035, "fabric_light", "wood_dark")),
        _thing("rug_bedroom_1", "rug", r, Box("rug_bedroom_1", f, 7.9, 4.3, 9.6, 6.5, 0.001, 0.005, "rug_blue", False)),
        _door("door_bedroom_1", r, 7.10, 4.25, 7.88, 4.29),
    ]


def _bedroom_2():
    f, r = "Furniture", "bedroom_2"
    return [
        # a single bed on a solid base, 30 cm from the wall: the lidar sees the gap, the robot cannot enter it
        _thing("bed_single", "bed", r,
               Box("bed_single_base", f, 0.3, 5.5, 1.2, 7.5, 0.0, 0.30, "wood_light"),
               Box("bed_single_mattress", f, 0.32, 5.52, 1.18, 7.5, 0.30, 0.50, "bedding"),
               Box("bed_single_headboard", f, 0.3, 7.5, 1.2, 7.6, 0.0, 0.90, "wood_light")),
        _thing("wardrobe_bedroom_2", "wardrobe", r, Box("wardrobe_bedroom_2", f, 1.25, 7.0, 2.65, 7.6, 0.0, 2.10, "wood_light")),
        _thing("desk", "desk", r, *table("desk", f, 3.3, 6.2, 3.9, 7.6, 0.75, 0.03, 0.04, 0.03, "wood_dark", "metal")),
        _thing("desk_chair", "chair", r,
               *chair("desk_chair", f, 2.80, 6.55, 0.42, "w", 0.46, 0.90, 0.035, "fabric_light", "metal")),
        _thing("bookshelf_bedroom_2", "bookshelf", r, Box("bookshelf_bedroom_2", f, 0.1, 4.1, 1.0, 4.45, 0.0, 1.60, "wood_light")),
        _thing("toy_box", "toy_box", r, Box("toy_box", f, 1.6, 4.1, 2.2, 4.5, 0.0, 0.45, "toy")),
        _thing("rug_bedroom_2", "rug", r, Box("rug_bedroom_2", f, 1.5, 5.0, 3.0, 6.3, 0.001, 0.005, "rug", False)),
        _door("door_bedroom_2", r, 3.12, 4.25, 3.90, 4.29),
    ]


def _bathroom():
    f, r = "Furniture", "bathroom"
    return [
        _thing("toilet", "toilet", r,
               Cylinder("toilet_pedestal", f, 4.6, 7.2, 0.12, 0.0, 0.25, "white_gloss"),
               Cylinder("toilet_bowl", f, 4.6, 7.2, 0.20, 0.25, 0.42, "white_gloss"),
               Box("toilet_tank", f, 4.4, 7.42, 4.8, 7.6, 0.25, 0.85, "white_gloss")),
        _thing("basin", "basin", r,
               Cylinder("basin_pedestal", f, 5.5, 7.42, 0.08, 0.0, 0.80, "white_gloss"),
               Box("basin_bowl", f, 5.25, 7.15, 5.75, 7.6, 0.80, 0.95, "white_gloss")),
        _thing("shower", "shower", r,  # closed off by glass: the lidar (a physics ray cast) sees glass as a wall
               Box("shower_glass_west", f, 6.10, 6.70, 6.12, 7.6, 0.0, 2.0, "glass"),
               Box("shower_glass_south", f, 6.12, 6.70, 7.0, 6.72, 0.0, 2.0, "glass"),
               Box("shower_tray", f, 6.12, 6.72, 7.0, 7.6, 0.0, 0.03, "white_gloss", False)),
        _thing("washing_machine", "washing_machine", r,
               Box("washing_machine", f, 6.4, 5.3, 7.0, 5.9, 0.0, 0.85, "white_gloss")),
        _thing("laundry_basket", "basket", r, Cylinder("laundry_basket", f, 4.3, 5.62, 0.20, 0.0, 0.55, "wicker")),
        _door("door_bathroom", r, 5.10, 5.30, 5.14, 6.08),
    ]


def _dock():
    tx0, ty0, tx1, ty1 = DOCK_TOWER
    px0, py0, px1, py1 = DOCK_PLATE
    return Thing("dock", "dock", "living_room", (
        Box("dock_tower", "Dock", tx0, ty0, tx1, ty1, 0.0, DOCK_TOWER_H, "dock"),
        Box("dock_plate", "Dock", px0, py0, px1, py1, 0.0, 0.003, "dock", False),
    ))


def things():
    """Every object in the home, the dock and the open doors included."""
    return _living_room() + _kitchen() + _corridor() + _bedroom_1() + _bedroom_2() + _bathroom() + [_dock()]


def furniture():
    """The shapes of every object except the dock, in the house frame."""
    return [s for t in things() if t.kind != "dock" for s in t.shapes]


def dock():
    """The charging dock: a solid tower and a floor plate that is only drawn (build_world.py adds the tag)."""
    return list(_dock().shapes)


def all_shapes():
    """Every shape in the home, in the house frame."""
    return floors() + walls() + windows() + ceiling() + furniture() + dock()


def obstacles_world():
    """Colliding boxes that reach the floor, as world-frame (name, x0, y0, x1, y1)."""
    out = []
    for s in walls() + furniture() + dock():
        if isinstance(s, Box) and s.collide and s.z0 < 0.2:
            out.append((s.name,) + to_world(s.x0, s.y0) + to_world(s.x1, s.y1))
    return out


def shapes_at_height(z):
    """Colliding shapes that cross height z, in the world frame: (boxes, cylinders).

    Boxes as (x0, y0, x1, y1), cylinders as (x, y, radius): what a lidar plane at height z sees.
    """
    boxes, cyls = [], []
    for s in walls() + furniture() + dock():
        if not s.collide or not (s.z0 <= z <= s.z1):
            continue
        if isinstance(s, Box):
            boxes.append(to_world(s.x0, s.y0) + to_world(s.x1, s.y1))
        else:
            cyls.append(to_world(s.x, s.y) + (s.radius,))
    return boxes, cyls


def expected_range(ox, oy, angle, z, max_range):
    """Distance from (ox, oy) along `angle` (world frame, radians) to the nearest shape at height z."""
    boxes, cyls = shapes_at_height(z)
    return ray_range(boxes, cyls, ox, oy, angle, max_range)
