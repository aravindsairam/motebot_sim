"""Open the vacuum world and run the simulation with the ROS 2 bridge active.

Usage (from a shell WITHOUT ROS sourced):
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/run_sim.py              # with the Isaac Sim window
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/run_sim.py --headless   # no window
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/run_sim.py --no-noise   # perfect sensors
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/run_sim.py --world home # the home instead of the house

Topics while running: /sim/world (latched: which world this is), /clock, /odom, /tf, /joint_states, /scan, /imu, /camera/image_raw,
/camera/depth/image_raw, /camera/camera_info, /bumper, /cliff/* (out) and /cmd_vel_safe (in).
The wheels stop by themselves if /cmd_vel_safe goes quiet for half a second.
Stop with Ctrl+C or by closing the window.
"""

import argparse
import os
import signal
import sys

from isaacsim import SimulationApp

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--world", default=None, help="which world: house (default) or home")
parser.add_argument("--usd", default=None, help="world USD to open (default: the world's file in sim/assets)")
parser.add_argument("--headless", action="store_true", help="run without a window")
parser.add_argument("--rate", type=float, default=60.0, help="physics and frame rate in Hz")
parser.add_argument("--no-noise", action="store_true", help="perfect sensors and ground-truth odometry")
parser.add_argument("--seed", type=int, default=0, help="random seed for the sensor noise")
args, _ = parser.parse_known_args()
if args.world is None:  # a world file named <world>_world.usd says which world it is
    name = os.path.basename(args.usd or "")
    args.world = name[: -len("_world.usd")] if name.endswith("_world.usd") and name != "vacuum_world.usd" else "house"
args.usd = args.usd or os.path.join(ASSETS, "vacuum_world.usd" if args.world == "house" else f"{args.world}_world.usd")
os.environ["VACUUM_WORLD"] = args.world  # read by house_layout.py (imported below, and by ros_publishers.py)
sys.path.insert(0, HERE)
import house_layout as hl  # noqa: E402  (plain data; refuses an unknown world by name)

if not os.path.isfile(args.usd):
    sys.exit(f"World file not found: {args.usd}\nRun build_world.py --world {args.world} first.")

# Worlds with a ceiling: it is one-sided, facing down (build_world.py). With back faces culled from
# the start (it has no effect when set later), cameras above the home (the window's view) look
# into the rooms while the robot's camera still sees a ceiling.
CULL = ["--/rtx/hydra/faceCulling/enabled=true"] if hasattr(hl, "ceiling") else []
# Without a window nobody sees the window's own view of the world, and drawing it took about a
# tenth of every frame. The robot's camera is drawn separately and is not affected. (The setting
# only acts when headless.)
simulation_app = SimulationApp(
    {"headless": args.headless, "width": 1280, "height": 720, "extra_args": CULL, "disable_viewport_updates": True}
)

import carb  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402

# The bridge must be loaded before the stage so the ROS 2 graph nodes in the USD resolve.
app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

ok, _stage = stage_utils.open_stage(args.usd)
if not ok:
    simulation_app.close()
    sys.exit(f"Could not open {args.usd}")
simulation_app.update()
while stage_utils.is_stage_loading():
    simulation_app.update()

# Keep physics time, timeline time and the app loop on the same clock so sim time tracks wall time.
# The loop at the end of this file holds the frame rate itself (FramePacer): Isaac Sim's own
# limiter times only its update, not the sensor messages published after it.
settings = carb.settings.get_settings()
settings.set_bool("/app/viewport/grid/enabled", False)  # the grid sits exactly on the floor and shows through
settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
settings.set_int("/persistent/simulation/minFrameRate", int(args.rate))
SimulationManager.setup_simulation(dt=1.0 / args.rate, device="cpu")
simulation_app.update()

if not args.headless:
    # Start the window on the angled overview camera so the whole house is visible.
    try:
        from omni.kit.viewport.utility import get_active_viewport

        get_active_viewport().camera_path = "/World/Cameras/Overview"
    except Exception as exc:  # the default camera still works; just say why the nicer view is missing
        print(f"Could not switch the viewport to the overview camera: {exc}", flush=True)

_stop = False


def _request_stop(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)

app_utils.play()
for _ in range(10):
    simulation_app.update()

# The raycast lidar, bumper and cliff sensors have no ROS 2 graph nodes, so they are read and
# published from this process. Fixed sensor frames go out once on /tf_static.
from ros_publishers import InProcessPublishers  # noqa: E402

publishers = InProcessPublishers(
    robot_path="/World/create3",
    base_link_path="/World/create3/base_link",
    bumper_path="/World/create3/base_link/bumper_contact",
    cliff_path="/World/create3/base_link/cliff_rays",
    imu_path="/World/create3/base_link/imu_link/imu_sensor",
    lidar_path="/World/create3/base_link/laser" if hl.LIDAR_BACKEND == "raycast" else None,
    frame_rate=args.rate,
    noise=not args.no_noise,
    seed=args.seed,
    world=hl.WORLD,
)
mode = "perfect sensors, ground-truth odometry" if args.no_noise else f"sensor noise on (seed {args.seed}), wheel odometry"
print(f"SIM_MODE {mode}; world {hl.WORLD}", flush=True)
print(
    "SIM_READY topics: /sim/world /clock /odom /ground_truth/odom /tf /joint_states /scan /imu /camera/* /bumper /cliff/* <- /cmd_vel_safe",
    flush=True,
)

from frame_pacer import FramePacer  # noqa: E402

pacer = FramePacer(args.rate)
while simulation_app.is_running() and not _stop:
    simulation_app.update()
    if app_utils.is_playing():
        publishers.step()
    pacer.wait()

publishers.shutdown()
app_utils.stop()
simulation_app.close()
