"""The `house` world: the first, two-room house (38.2 m2), as plain data. No Isaac Sim imports.

The 32 acceptance phases 1 to 32 run on it. Its shapes must not change: sim/tests/test_worlds.py
compares them with a snapshot taken before the worlds were split out of house_layout.py.

Two frames are used:
  * house frame: origin at the inside south-west corner of the living room, x east, y north.
  * world frame: origin at the robot start pose in front of the dock, x pointing into the room.
    world = house - START_POSE_HOUSE[:2]. The robot starts at the world origin with yaw 0,
    so the odom frame and (later) the SLAM map frame coincide with the world frame.

All lengths are in meters.
"""

from .shapes import Box, Cylinder, ray_range, table

WALL_T = 0.10  # wall thickness
WALL_H = 2.40  # wall height
DOOR_H = 2.00  # door opening height (lintel above)
FLOOR_T = 0.10  # floor slab thickness (top surface at z = 0)

# Rooms as (x0, y0, x1, y1) interior rectangles in the house frame.
LIVING_ROOM = (0.0, 0.0, 5.0, 4.5)
BEDROOM = (5.1, 0.0, 8.6, 4.5)
# Opening in the dividing wall between the rooms (0.8 m wide).
DOORWAY = (5.0, 2.6, 5.1, 3.4)

# Stairwell alcove off the bedroom's north wall: an open doorway onto a floor that is
# STAIRWELL_DROP lower. The lidar sees it as open space; only the cliff sensors can save the robot.
# It lies outside the main floor slab, so the house floor stays one seamless collider.
STAIRWELL = (5.15, 4.6, 5.95, 5.4)
STAIRWELL_DROP = 0.15

ROOMS = {"living_room": LIVING_ROOM, "bedroom": BEDROOM}
FLOOR_TYPES = {"living_room": "hardwood", "bedroom": "carpet"}

# Outer faces of the outer walls (the floor slab), not counting the stairwell alcove.
FOOTPRINT = (-WALL_T, -WALL_T, BEDROOM[2] + WALL_T, LIVING_ROOM[3] + WALL_T)

# Robot start pose (x, y, yaw) in the house frame: just in front of the dock on the west wall.
START_POSE_HOUSE = (0.45, 2.25, 0.0)
# Charging dock on the west wall, on the robot's start line: a tower against the wall with an
# AprilTag on its front face, and a thin plate on the floor in front of it (for looks only).
DOCK_Y = 2.25  # centre line of the dock
DOCK_TOWER = (0.0, DOCK_Y - 0.11, 0.06, DOCK_Y + 0.11)  # x0, y0, x1, y1
DOCK_TOWER_H = 0.16  # taller than the lidar plane, so the lidar sees it too
DOCK_PLATE = (0.06, DOCK_Y - 0.14, 0.30, DOCK_Y + 0.14)
DOCK_TAG_FAMILY = "36h11"
DOCK_TAG_ID = 0
DOCK_TAG_SIZE = 0.10  # edge of the tag's black square: the size the detector is told
DOCK_TAG_TEXTURE = "textures/apriltag_36h11_0.png"  # relative to sim/assets; made by make_apriltag.py
# Centre of the tag (x, y, z): on the tower's front face, facing +x into the room, at camera height.
DOCK_TAG_HOUSE = (DOCK_TOWER[2], DOCK_Y, 0.075)

# name -> (r, g, b), roughness
MATERIALS = {
    "floor_wood": ((0.55, 0.38, 0.22), 0.55),
    "floor_carpet": ((0.42, 0.45, 0.52), 0.95),
    "wall": ((0.92, 0.90, 0.86), 0.9),
    "sofa": ((0.25, 0.33, 0.45), 0.9),
    "wood_dark": ((0.30, 0.20, 0.13), 0.5),
    "wood_light": ((0.72, 0.58, 0.40), 0.5),
    "metal": ((0.55, 0.56, 0.58), 0.3),
    "fabric_light": ((0.85, 0.83, 0.78), 0.9),
    "rug": ((0.62, 0.25, 0.22), 0.95),
    "plant_pot": ((0.75, 0.42, 0.28), 0.7),
    "plant_leaf": ((0.18, 0.42, 0.20), 0.8),
    "screen": ((0.04, 0.04, 0.05), 0.2),
    "dock": ((0.10, 0.10, 0.12), 0.4),
}


def to_world(x, y):
    """Convert a house-frame point to the world frame."""
    return (x - START_POSE_HOUSE[0], y - START_POSE_HOUSE[1])


def to_house(x, y):
    """Convert a world-frame point to the house frame."""
    return (x + START_POSE_HOUSE[0], y + START_POSE_HOUSE[1])


def room_world(name):
    """Room rectangle (x0, y0, x1, y1) in the world frame."""
    x0, y0, x1, y1 = ROOMS[name]
    wx0, wy0 = to_world(x0, y0)
    wx1, wy1 = to_world(x1, y1)
    return (wx0, wy0, wx1, wy1)


def floor_areas():
    """Where the floor is, as rectangles (x0, y0, x1, y1) in the house frame: the rooms and the doorway."""
    return list(ROOMS.values()) + [DOORWAY]


def drops():
    """Floors lower than the house's, which only the cliff sensors find: the stairwell."""
    return [STAIRWELL]


def floors():
    """Two visible floor finishes (one per room) over a single invisible collision slab.

    The robot must roll on one continuous collider: with one slab per room, the caster ball
    catches on the seam in the doorway and the robot pitches by several degrees.
    """
    seam = (DOORWAY[0] + DOORWAY[2]) / 2.0
    xmin, xmax = -WALL_T, BEDROOM[2] + WALL_T
    ymin, ymax = -WALL_T, LIVING_ROOM[3] + WALL_T
    return [
        Box("floor_collision", "Floor", xmin, ymin, xmax, ymax, -FLOOR_T, 0.0, "floor_wood", True, "", False),
        Box("living_room_floor", "Floor", xmin, ymin, seam, ymax, -FLOOR_T, 0.0, "floor_wood", False, "hardwood"),
        Box("bedroom_floor", "Floor", seam, ymin, xmax, ymax, -FLOOR_T, 0.0, "floor_carpet", False, "carpet"),
    ]


def walls():
    """Outer walls, the dividing wall with its doorway, lintels, and the stairwell alcove."""
    x_end = BEDROOM[2]
    y_end = LIVING_ROOM[3]
    dx0, dy0, dx1, dy1 = DOORWAY
    sx0, sy0, sx1, sy1 = STAIRWELL
    h = WALL_H
    low = -STAIRWELL_DROP
    return [
        Box("wall_south", "Walls", -WALL_T, -WALL_T, x_end + WALL_T, 0.0, 0.0, h, "wall"),
        Box("wall_north_west", "Walls", -WALL_T, y_end, sx0, y_end + WALL_T, 0.0, h, "wall"),
        Box("wall_north_east", "Walls", sx1, y_end, x_end + WALL_T, y_end + WALL_T, 0.0, h, "wall"),
        Box("stairwell_lintel", "Walls", sx0, y_end, sx1, y_end + WALL_T, DOOR_H, h, "wall"),
        Box("wall_west", "Walls", -WALL_T, 0.0, 0.0, y_end, 0.0, h, "wall"),
        Box("wall_east", "Walls", x_end, 0.0, x_end + WALL_T, y_end, 0.0, h, "wall"),
        Box("wall_divider_south", "Walls", dx0, 0.0, dx1, dy0, 0.0, h, "wall"),
        Box("wall_divider_north", "Walls", dx0, dy1, dx1, y_end, 0.0, h, "wall"),
        Box("door_lintel", "Walls", dx0, dy0, dx1, dy1, DOOR_H, h, "wall"),
        # stairwell: lowered landing enclosed on three sides
        Box("stairwell_floor", "Walls", sx0 - WALL_T, sy0, sx1 + WALL_T, sy1 + WALL_T, low - FLOOR_T, low, "floor_wood"),
        Box("stairwell_wall_west", "Walls", sx0 - WALL_T, sy0, sx0, sy1, low, h, "wall"),
        Box("stairwell_wall_east", "Walls", sx1, sy0, sx1 + WALL_T, sy1, low, h, "wall"),
        Box("stairwell_wall_north", "Walls", sx0 - WALL_T, sy1, sx1 + WALL_T, sy1 + WALL_T, low, h, "wall"),
    ]


def furniture():
    """Furniture built from boxes and cylinders, in the house frame."""
    f = "Furniture"
    items = [
        # --- living room ---
        Box("sofa_base", f, 1.2, 0.05, 3.2, 0.95, 0.0, 0.42, "sofa"),
        Box("sofa_back", f, 1.2, 0.05, 3.2, 0.30, 0.42, 0.85, "sofa"),
        Box("sofa_arm_left", f, 1.2, 0.30, 1.4, 0.95, 0.42, 0.62, "sofa"),
        Box("sofa_arm_right", f, 3.0, 0.30, 3.2, 0.95, 0.42, 0.62, "sofa"),
        Box("tv_stand", f, 1.4, 4.05, 3.0, 4.45, 0.0, 0.50, "wood_dark"),
        Box("tv_screen", f, 1.6, 4.20, 2.8, 4.26, 0.55, 1.25, "screen"),
        Box("armchair_base", f, 3.9, 0.20, 4.7, 1.00, 0.0, 0.42, "fabric_light"),
        Box("armchair_back", f, 3.9, 0.20, 4.7, 0.40, 0.42, 0.85, "fabric_light"),
        Box("rug", f, 1.3, 1.1, 3.1, 2.4, 0.001, 0.005, "rug", False),
        Cylinder("plant_pot", f, 4.6, 4.1, 0.20, 0.0, 0.40, "plant_pot"),
        Cylinder("plant_leaves", f, 4.6, 4.1, 0.30, 0.40, 1.20, "plant_leaf", False),
        # --- bedroom ---
        Box("bed_frame", f, 6.55, 0.60, 8.55, 2.20, 0.0, 0.30, "wood_light"),
        Box("bed_mattress", f, 6.60, 0.65, 8.45, 2.15, 0.30, 0.50, "fabric_light"),
        Box("bed_headboard", f, 8.45, 0.60, 8.55, 2.20, 0.30, 1.00, "wood_light"),
        Box("nightstand_south", f, 8.10, 0.08, 8.55, 0.53, 0.0, 0.50, "wood_dark"),
        Box("nightstand_north", f, 8.10, 2.27, 8.55, 2.72, 0.0, 0.50, "wood_dark"),
        Box("wardrobe", f, 6.0, 3.85, 7.8, 4.45, 0.0, 2.00, "wood_light"),
    ]
    items += table("coffee_table", f, 1.7, 1.4, 2.7, 2.0, 0.44, 0.04, 0.04, 0.03, "wood_light", "metal")
    items += table("desk", f, 5.15, 0.05, 6.05, 0.65, 0.75, 0.03, 0.04, 0.03, "wood_dark", "metal")
    return items


def dock():
    """The charging dock: a solid tower and a floor plate that is only drawn (the tag is added
    separately by build_world.py, since it needs a picture and not just a colour)."""
    tx0, ty0, tx1, ty1 = DOCK_TOWER
    px0, py0, px1, py1 = DOCK_PLATE
    return [
        Box("dock_tower", "Dock", tx0, ty0, tx1, ty1, 0.0, DOCK_TOWER_H, "dock"),
        Box("dock_plate", "Dock", px0, py0, px1, py1, 0.0, 0.003, "dock", False),
    ]


def all_shapes():
    """Every shape in the house, in the house frame."""
    return floors() + walls() + furniture() + dock()


def obstacles_world():
    """Colliding boxes that reach the floor, as world-frame (x0, y0, x1, y1). Useful for tests."""
    out = []
    for s in walls() + furniture() + dock():
        if isinstance(s, Box) and s.collide and s.z0 < 0.2:
            wx0, wy0 = to_world(s.x0, s.y0)
            wx1, wy1 = to_world(s.x1, s.y1)
            out.append((s.name, wx0, wy0, wx1, wy1))
    return out


def shapes_at_height(z):
    """Colliding shapes that cross height z, in the world frame.

    Returns (boxes, cylinders): boxes as (x0, y0, x1, y1), cylinders as (x, y, radius).
    This is what a horizontal lidar plane at height z should see.
    """
    boxes, cyls = [], []
    for s in walls() + furniture() + dock():
        if not s.collide or not (s.z0 <= z <= s.z1):
            continue
        if isinstance(s, Box):
            wx0, wy0 = to_world(s.x0, s.y0)
            wx1, wy1 = to_world(s.x1, s.y1)
            boxes.append((wx0, wy0, wx1, wy1))
        else:
            wx, wy = to_world(s.x, s.y)
            cyls.append((wx, wy, s.radius))
    return boxes, cyls


def expected_range(ox, oy, angle, z, max_range):
    """Distance from (ox, oy) along `angle` (world frame, radians) to the nearest shape at height z."""
    boxes, cyls = shapes_at_height(z)
    return ray_range(boxes, cyls, ox, oy, angle, max_range)
