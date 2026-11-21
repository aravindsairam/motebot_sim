"""Render still images of the vacuum world from its preview cameras.

Usage (from a shell WITHOUT ROS sourced):
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/capture_preview.py                # the house
    cd ~/isaacsim && ./python.sh <repo>/sim/scripts/capture_preview.py --world home   # the home

Writes overview.png and topdown.png (the house) or <world>_overview.png and <world>_topdown.png
to <repo>/sim/assets/previews/.
The timeline is not played, so nothing is published to ROS while this runs.
"""

import argparse
import os
import sys

from isaacsim import SimulationApp

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.normpath(os.path.join(HERE, "..", "assets"))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "assets", "previews"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--world", default="house", help="house (default) or home")
parser.add_argument("--usd", default=None, help="world file (default: the world's file in sim/assets)")
parser.add_argument("--out", default=DEFAULT_OUT)
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--robot-camera", action="store_true", help="also render what the robot's camera sees at its start pose")
args, _ = parser.parse_known_args()
args.usd = args.usd or os.path.join(ASSETS, "vacuum_world.usd" if args.world == "house" else f"{args.world}_world.usd")
PREFIX = "" if args.world == "house" else f"{args.world}_"

# Worlds with a one-sided ceiling (see run_sim.py): back faces culled from the start, so the
# cameras above the home look into the rooms while the robot's camera sees the ceiling.
CULL = [] if args.world == "house" else ["--/rtx/hydra/faceCulling/enabled=true"]
simulation_app = SimulationApp({"headless": True, "width": args.width, "height": args.height, "extra_args": CULL})

import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import isaacsim.core.experimental.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from PIL import Image  # noqa: E402

import carb  # noqa: E402

carb.settings.get_settings().set_bool("/app/viewport/grid/enabled", False)

CAMERAS = {"overview": "/World/Cameras/Overview", "topdown": "/World/Cameras/TopDown"}
if args.robot_camera:
    CAMERAS["robot_camera"] = "/World/create3/base_link/camera_link/front_camera"

app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
ok, _stage = stage_utils.open_stage(args.usd)
if not ok:
    simulation_app.close()
    sys.exit(f"Could not open {args.usd}")
while stage_utils.is_stage_loading():
    simulation_app.update()

os.makedirs(args.out, exist_ok=True)
annotators = {}
for name, cam in CAMERAS.items():
    size = (640, 480) if name == "robot_camera" else (args.width, args.height)  # the robot camera's own size
    product = rep.create.render_product(cam, size)
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([product])
    annotators[name] = annot

for _ in range(120):  # let lighting and materials settle
    simulation_app.update()
rep.orchestrator.step()

failed = False
for name, annot in annotators.items():
    data = np.asarray(annot.get_data())
    if data.size == 0:
        print(f"CAPTURE_FAILED {name}: empty frame")
        failed = True
        continue
    rgb = data[:, :, :3]
    path = os.path.join(args.out, f"{PREFIX}{name}.png")
    Image.fromarray(rgb).save(path)
    print(f"CAPTURE_OK {name} mean_brightness={rgb.mean():.1f} path={path}")
    if rgb.mean() < 5.0:
        print(f"CAPTURE_WARNING {name}: image is almost black")

simulation_app.close()
sys.exit(1 if failed else 0)
