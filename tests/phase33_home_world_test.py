#!/usr/bin/env python3
"""Phase 33 acceptance test: the home world itself, without the robot software.

Run the simulator alone on the home, with perfect sensors (as phase 2 does for the house):
    tools/start_sim.sh --headless --world home --no-noise
    source /opt/ros/humble/setup.bash
    python3 sim/tests/phase33_home_world_test.py
or: tools/run_tests.sh 33

Checks:
  1. world       the simulator says it runs the home; the robot rests on the floor at its start
                 pose in front of the dock, level as in the house (height within 8 mm, tilt under
                 0.5 deg), and all four cliff sensors see the floor
  2. dock tag    the camera sees the dock's tag (OpenCV, tag36h11 id 0) from 0.2 m and 1.0 m;
                 2.5 m, 10 degrees off to the side, is reported
  3. tour        a drive through all six rooms and under the double bed: in every room the lidar
                 agrees with the layout (phase 2's limits: 97 % of beams return, median error
                 under 3 cm, 90 % under 15 cm); no bump, level all the way (tilt under 2 degrees)
  4. stairs      creeping toward the front door, the front cliff sensors trip at the edge of the
                 stairs before the robot drops, and clear when it backs off
The real-time factor during the tour is reported.
"""

import math
import os
import statistics
import sys
import time

os.environ["VACUUM_WORLD"] = "home"  # before house_layout is imported (here and in phase 2's helpers)

import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from std_msgs.msg import String  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from phase1_drive_test import tilt_of, wrap  # noqa: E402
from phase2_sensor_test import SensorDriver, scan_errors  # noqa: E402

# Stops of the tour, in the house frame, with the room each one is in. The legs between them run
# in straight lines clear of every object (the driver steers straight at the next stop).
TOUR = [
    ((1.20, 2.45), None),
    ((3.80, 2.45), "living_room"),
    ((3.80, 3.50), None),
    ((5.90, 3.50), None),
    ((5.90, 1.70), None),
    ((7.20, 1.70), None),
    ((8.50, 1.40), "kitchen"),
    ((9.40, 1.40), None),
    ((7.20, 1.70), None),
    ((5.90, 1.70), None),
    ((5.90, 3.50), None),
    ((5.05, 3.50), None),
    ((5.05, 4.65), "corridor"),
    ((5.52, 4.65), None),
    ((5.52, 6.40), "bathroom"),
    ((5.52, 4.70), None),
    ((7.60, 4.70), None),
    ((7.80, 5.20), "bedroom_1"),
    ((9.40, 5.40), "under the double bed"),
    ((7.80, 5.20), None),
    ((7.60, 4.70), None),
    ((3.40, 4.67), None),
    ((2.30, 5.60), "bedroom_2"),
    ((3.40, 4.67), None),
    ((5.05, 4.67), None),
    ((5.05, 3.50), None),
    ((3.80, 3.50), None),
    ((3.80, 2.45), None),
    ((1.20, 2.45), None),
]
# then straight south through the front door
FRONT_DOOR_APPROACH = [(3.80, 2.45), (3.80, 3.50), (5.90, 3.50), (5.90, 1.70), (5.72, 1.10)]


class HomeDriver(SensorDriver):
    def __init__(self):
        super().__init__()
        self.world = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "sim/world", lambda m: setattr(self, "world", m.data), latched)

    def go_house(self, x, y):
        return self.go_to(*hl.to_world(x, y))

    def face_point_house(self, x, y):
        wx, wy = hl.to_world(x, y)
        rx, ry, _ = self.pose()
        self.face(math.atan2(wy - ry, wx - rx))


def find_tag(image_msg):
    """Ids of tag36h11 tags OpenCV finds in a camera picture."""
    import cv2

    arr = np.frombuffer(bytes(image_msg.data), dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    _corners, ids, _ = detector.detectMarkers(gray)
    return [] if ids is None else [int(i) for i in ids.flatten()]


def main():
    rclpy.init()
    HomeDriver.CMD_TOPIC = hl.WHEEL_CMD_TOPIC  # simulator only, no safety gate: talk to the wheels
    n = HomeDriver()
    results = []

    def check(name, ok, detail):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if not n.wait_ready():
            check("world", False, "no /odom or /clock within 60 s; is run_sim.py --world home playing?")
            return results
        missing = n.wait_sensors()
        wall0 = time.time()
        while n.world is None and time.time() - wall0 < 10.0:
            n.spin(0.1)
        if missing or n.world != "home":
            check("world", False, f"simulator world={n.world!r} (must be 'home'); no messages on: {missing or 'none'}")
            return results

        # 1. at rest at the start pose
        n.run_for(0.0, 0.0, 1.0)
        x, y, yaw = n.pose()
        z = n.odom.pose.pose.position.z
        tilt = tilt_of(n.odom.pose.pose.orientation)
        cl = {k[6:]: v.range for k, v in n.cliff.items()}
        floor_ok = all(math.isfinite(v) and v < hl.CLIFF_THRESHOLD for v in cl.values())
        hx, hy = hl.to_house(x, y)
        # The house's robot rests at z -4.4 mm, tilt 0.01 deg (measured 2026-11-20); a robot tipped
        # back on a bad floor collider rested at -11 mm, 3 deg.
        check("world", math.hypot(x, y) < 0.03 and abs(yaw) < 0.03 and abs(z) < 0.008 and tilt < 0.5 and floor_ok,
              f"simulator runs '{n.world}'; robot at house ({hx:.3f}, {hy:.3f}), heading {math.degrees(yaw):.1f} deg "
              f"(start pose {hl.START_POSE_HOUSE}); height {z * 1000:.1f} mm, tilt {tilt:.2f} deg; cliff "
              + ", ".join(f"{k}={v * 1000:.0f} mm" for k, v in cl.items()))

        # 2. the dock's tag in the camera picture
        tag_x, tag_y = hl.DOCK_TAG_HOUSE[:2]
        seen = {}
        near_x, far_x = tag_x + 1.0 + hl.CAMERA_XYZ[0], tag_x + 2.5 + hl.CAMERA_XYZ[0]
        for label, path in (("0.2 m", []),
                            ("1.0 m", [(near_x, tag_y)]),
                            ("2.5 m, 10 deg off", [(near_x, tag_y + 0.45), (far_x, tag_y + 0.45)])):  # past the coffee table
            for px, py in path:
                n.go_house(px, py)
            n.face_point_house(tag_x, tag_y)
            n.run_for(0.0, 0.0, 1.0)
            frames = []
            t0 = n.sim_time
            while n.sim_time - t0 < 2.0:
                last = n.rgb
                n.spin()
                if n.rgb is not last:
                    frames.append(hl.DOCK_TAG_ID in find_tag(n.rgb))
            seen[label] = (sum(frames), len(frames))
        near_ok = all(seen[k][1] > 0 and seen[k][0] >= 0.9 * seen[k][1] for k in ("0.2 m", "1.0 m"))
        check("dock tag", near_ok, "; ".join(f"{k}: tag {hl.DOCK_TAG_ID} found in {a} of {b} pictures"
                                             for k, (a, b) in seen.items()))

        # 3. the tour
        n.go_house(*TOUR[0][0])
        n.max_tilt, n.max_abs_z, n.bumper_seen = 0.0, 0.0, 0
        stops, reached = [], True
        sim0, wall0 = n.sim_time, time.time()
        for (px, py), room in TOUR:
            if not n.go_house(px, py):
                reached = False
                stops.append(f"did not reach ({px}, {py})")
                break
            if room is None:
                continue
            n.run_for(0.0, 0.0, 0.6)
            rx, ry, ryaw = n.pose()
            errs, valid = scan_errors(n.scan, rx, ry, ryaw, +1)
            med = statistics.median(errs) if errs else float("inf")
            p90 = sorted(errs)[int(0.9 * len(errs))] if errs else float("inf")
            ok = valid > 0.97 and med < 0.03 and p90 < 0.15
            stops.append((room, ok, f"{room}: {valid * 100:.0f} % return, median {med * 100:.1f} cm, 90 % under {p90 * 100:.1f} cm"))
        rtf = (n.sim_time - sim0) / max(1e-6, time.time() - wall0)
        rooms_ok = [s for s in stops if isinstance(s, tuple)]
        lidar_ok = len(rooms_ok) == 7 and all(ok for _, ok, _ in rooms_ok)
        check("tour", reached and lidar_ok and n.bumper_seen == 0 and n.max_tilt < 2.0 and n.max_abs_z < 0.01,
              "; ".join(s if isinstance(s, str) else s[2] for s in stops)
              + f"; bumper pressed: {'no' if n.bumper_seen == 0 else 'yes'}; max tilt {n.max_tilt:.2f} deg, "
              f"max height {n.max_abs_z * 1000:.1f} mm; {n.sim_time - sim0:.0f} s of simulated driving at "
              f"{rtf:.2f} of real time")

        # 4. the stairs outside the front door
        reached = all(n.go_house(px, py) for px, py in FRONT_DOOR_APPROACH)
        n.face(-math.pi / 2.0)
        z_before = n.odom.pose.pose.position.z
        n.max_tilt = 0.0
        tripped = n.drive_until(0.08, n.front_cliff_tripped, 20.0)
        _, hy = hl.to_house(*n.pose()[:2])
        z_drop = z_before - n.odom.pose.pose.position.z
        edge = hl.STAIRS[3]
        expected_y = edge + hl.CLIFF_SENSORS["cliff_front_left"][0]
        n.run_for(-0.12, 0.0, 3.0)
        n.stop()
        cleared = not n.front_cliff_tripped()
        check("stairs", reached and tripped and abs(hy - expected_y) < 0.06 and z_drop < 0.01 and n.max_tilt < 2.0 and cleared,
              f"reached the door={reached}; front cliff sensors tripped={tripped} with the robot centre at house "
              f"y={hy:.3f} (edge at {edge}, expected about {expected_y:.2f}); height change {z_drop * 1000:.1f} mm, "
              f"max tilt {n.max_tilt:.2f} deg; cleared after backing off={cleared}")
    finally:
        try:
            for _ in range(20):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        n.destroy_node()
        rclpy.shutdown()

    print(f"SUMMARY {sum(results)}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(res) else 1)
