"""The robot, and the world it is in, as plain data.

No Isaac Sim imports: this file is read both by the USD builder (Isaac's Python 3.12)
and by ROS-side tools and tests (system Python 3.10).

The robot's constants (size, wheels, sensors, topics) are defined here. The world's layout
(rooms, walls, furniture, dock, start pose, ground truth) comes from one module per world in
worlds/, and is handed out here under the same names for whichever world is chosen:

    VACUUM_WORLD=house   the two-room house (the default); the 32 acceptance phases run on it
    VACUUM_WORLD=home    the six-room home (docs/2026-11-19-bigger-home-plan.md)

build_world.py and run_sim.py set it from their --world argument; the robot software is told the
world by the simulator (topic /sim/world). Every world gives at least: WALL_T, WALL_H, DOOR_H,
FLOOR_T, ROOMS, FLOOR_TYPES, FOOTPRINT, START_POSE_HOUSE, the DOCK_* values, MATERIALS, Box,
Cylinder, to_world(), to_house(), room_world(), floors(), walls(), furniture(), dock(),
all_shapes(), obstacles_world(), shapes_at_height() and expected_range(). Names only one world has
(the house's DOORWAY, STAIRWELL, LIVING_ROOM, BEDROOM) are there only in that world.

Two frames are used:
  * house frame: origin at the inside south-west corner of the world's living room, x east, y north.
  * world frame: origin at the robot start pose in front of the dock, x pointing into the room.
    world = house - START_POSE_HOUSE[:2]. The robot starts at the world origin with yaw 0,
    so the odom frame and (later) the SLAM map frame coincide with the world frame.

All lengths are in meters.
"""

import importlib
import os

# Create 3 geometry, read from the Isaac Sim asset (matches iRobot's published numbers).
ROBOT_RADIUS = 0.171
WHEEL_RADIUS = 0.03575
WHEEL_SEPARATION = 0.233
MAX_LINEAR_SPEED = 0.306
MAX_ANGULAR_SPEED = 1.9
# --- sensors (positions in the base_link frame: x forward, y left, z up from the floor) ---
# Lidar backend: "raycast" casts physics rays against collision geometry (deterministic, cheap).
# "rtx" uses Isaac Sim's RTX lidar, which on this setup returns scans with drifting gaps of
# missing beams next to depth edges (up to 15 % of a scan), so it is not the default.
LIDAR_BACKEND = "raycast"
LIDAR_BEAMS = 720  # raycast backend only: 0.5 degree resolution
LIDAR_XYZ = (0.0, 0.0, 0.115)  # scan plane just above the 0.0925 m tall body
LIDAR_MIN_RANGE = 0.05
LIDAR_MAX_RANGE = 12.0
LIDAR_RATE_HZ = 10.0
CAMERA_XYZ = (0.165, 0.0, 0.070)  # behind the front of the bumper, looking forward
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_HFOV_DEG = 87.0
CAMERA_RATE_HZ = 15.0
# Cliff rays point straight down from iRobot's four sensor positions.
CLIFF_SENSORS = {
    "cliff_front_left": (0.160, 0.045),
    "cliff_front_right": (0.160, -0.045),
    "cliff_side_left": (0.060, 0.145),
    "cliff_side_right": (0.060, -0.145),
}
CLIFF_RAY_Z = 0.012  # ray origin height above the floor
CLIFF_MAX_RANGE = 0.25
CLIFF_THRESHOLD = 0.05  # a reading beyond this (or no hit) means "no floor here"
# Where the front caster touches the floor, ahead of the wheel axle. It is the first support to
# leave the floor at a drop, 3.5 cm behind the front cliff sensors: the room there is to stop in.
CASTER_X = 0.125
# Bumper zones as (name, bit, min angle, max angle) in degrees, 0 = straight ahead, + = left.
BUMPER_ZONES = [
    ("right", 16, -100.0, -54.0),
    ("front_right", 8, -54.0, -18.0),
    ("front_center", 4, -18.0, 18.0),
    ("front_left", 2, 18.0, 54.0),
    ("left", 1, 54.0, 100.0),
]

# The wheels obey one topic only. Everything upstream (keyboard, Nav2) goes through the safety
# gate, which is the only publisher on it. Tests of the simulator alone publish here directly.
WHEEL_CMD_TOPIC = "cmd_vel_safe"
# Like a real motor controller, the simulated base stops by itself if commands stop arriving.
BASE_CMD_TIMEOUT = 0.5  # s

# Acceleration limits applied in the simulator's drive controller (our choice, not an iRobot spec).
MAX_LINEAR_ACCEL = 0.9  # m/s^2
# Hard braking: a cliff stop from full speed must fit in the 3.5 cm between the front cliff sensors
# and the caster. About 1.5 cm of that goes before braking even starts (the sensor reading, the
# safety gate and the wheel command each wait for a simulation frame: 45 to 60 ms in all), and
# braking itself takes 0.306^2 / (2 * 4.5) = 1.0 cm. At 3.0 m/s^2 only 0.5 cm was left over.
# The wheels grip well enough for this: braking at 4.5 needs a friction coefficient of 0.66, they have 1.0.
MAX_LINEAR_DECEL = 4.5  # m/s^2
MAX_ANGULAR_ACCEL = 6.0  # rad/s^2


# --- the world ----------------------------------------------------------------------------------
WORLDS = ("house", "home")
WORLD = os.environ.get("VACUUM_WORLD", "") or "house"
if WORLD not in WORLDS:
    raise ImportError(f"unknown world '{WORLD}' in VACUUM_WORLD; the worlds are: {', '.join(WORLDS)}")
_world = importlib.import_module(f"worlds.{WORLD}")
_names = {k: v for k, v in vars(_world).items() if not k.startswith("_") and k not in ("expected_range",)}
_clash = sorted(k for k in _names if k in globals() and k not in ("importlib", "os"))
if _clash:  # a world describes a place, never the robot: it must not change the robot's constants
    raise ImportError(f"world '{WORLD}' redefines names of house_layout.py: {', '.join(_clash)}")
globals().update(_names)


def expected_range(ox, oy, angle, z, max_range=LIDAR_MAX_RANGE):
    """Distance from (ox, oy) along `angle` (world frame, radians) to the nearest shape at height z."""
    return _world.expected_range(ox, oy, angle, z, max_range)
