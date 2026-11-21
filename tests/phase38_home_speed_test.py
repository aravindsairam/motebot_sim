#!/usr/bin/env python3
"""Phase 38 acceptance test: the home is fast enough to use (plan section 6.7).

Needs the simulator on the home, headless, and the robot stack, the home's map in use:
    tools/start_sim.sh --headless --world home ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase38_home_speed_test.py
or: tools/run_tests.sh 38

While a scripted phone watches the camera video (as the app does), the robot drives from room to
room on Nav2 for five minutes of simulated time.
  1. speed        the simulator keeps up at least 0.6 of real time over those five minutes
  2. video        the phone receives pictures the whole time (reported: pictures a second, and
                  what the camera published a second of real and of simulated time)
"""

import math
import os
import sys
import threading
import time

os.environ["VACUUM_WORLD"] = "home"  # before house_layout is imported (by cleaning_client)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from phase17_camera_test import Viewer  # noqa: E402
from phase36_home_going_places_test import ROOM_GOALS  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

hl = cc.hl
DRIVE_SIM_SECONDS = 300.0
TARGET = 0.6


def main():
    run = cc.Run("phase38")
    n, modes = run.n, run.modes
    camera = {"count": 0}
    lock = threading.Lock()

    def on_image(_msg):
        with lock:
            camera["count"] += 1

    n.create_subscription(Image, "/camera/image_raw", on_image, 5)
    viewer = None
    try:
        if not run.ready():
            return run.results
        ok, msg = modes.request("navigation", 180.0)
        if not ok:
            run.check("speed", False, f"navigation mode: {msg}")
            return run.results
        viewer = Viewer().start()
        end = time.monotonic() + 15.0
        while viewer.connected_at is None and viewer.error is None and time.monotonic() < end:
            n.spin(0.1)
        if viewer.connected_at is None:
            run.check("video", False, f"the scripted phone did not connect: {viewer.error}")
            return run.results
        n.wait_sim(2.0)

        sim0, wall0 = n.sim_time, time.monotonic()
        with lock:
            cam0 = camera["count"]
        pics0 = len(viewer.pictures)
        trips, i = [], 0
        while n.sim_time - sim0 < DRIVE_SIM_SECONDS:
            room, xy, yaw = ROOM_GOALS[(i + 1) % len(ROOM_GOALS)]  # start with the kitchen: the living room is where we are
            wx, wy = hl.to_world(*xy)
            left = DRIVE_SIM_SECONDS - (n.sim_time - sim0)
            status, secs, dist = n.navigate(wx, wy, yaw, sim_timeout=max(5.0, left))
            trips.append(f"{room} {status}")
            i += 1
        sim1, wall1 = n.sim_time, time.monotonic()
        with lock:
            cam1 = camera["count"]
        pics1 = len(viewer.pictures)
        rtf = (sim1 - sim0) / (wall1 - wall0)
        done = sum(1 for t in trips[:-1] if t.endswith("succeeded"))  # the last trip is cut short by the clock
        drove = len(trips) > 1 and done >= 0.8 * (len(trips) - 1)
        run.check("speed", rtf >= TARGET and drove,
                  f"{sim1 - sim0:.0f} s of simulated driving took {wall1 - wall0:.0f} s: {rtf:.2f} of real time "
                  f"(target {TARGET}); trips: {', '.join(trips)} ({done} of {len(trips) - 1} finished trips arrived)")
        gaps = [b[0] - a[0] for a, b in zip(viewer.pictures[pics0:], viewer.pictures[pics0 + 1:pics1])]
        run.check("video", pics1 - pics0 > 0 and (max(gaps) if gaps else 99.0) < 3.0,
                  f"the phone got {(pics1 - pics0) / (wall1 - wall0):.1f} pictures a second (longest gap "
                  f"{max(gaps) if gaps else float('nan'):.1f} s); the camera published {(cam1 - cam0) / (wall1 - wall0):.1f} "
                  f"a second of real time, {(cam1 - cam0) / (sim1 - sim0):.1f} a second of simulated time")
    finally:
        if viewer is not None:
            viewer.stop()
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
