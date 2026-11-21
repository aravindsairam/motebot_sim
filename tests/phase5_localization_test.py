#!/usr/bin/env python3
"""Phase 5 acceptance test: localize on the saved map with AMCL and compare with the true pose.

Needs two things running first, with the robot at its start pose:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase5_localization_test.py

The test asks the mode manager for the mode it needs (manual: driven by this test, localized by AMCL).

Checks:
  0. map served as saved  map_server's /map has exactly the saved file's free, occupied and unknown
                          cells (until 2026-11-14 the saved YAML made unknown cells load as free)
  1. starts localized  AMCL knows where the robot is at start-up without being told
  2. tracking          AMCL's pose stays close to the truth over a tour of both rooms
  3. recovery          given a start pose that is wrong by 58 cm and 25 degrees, AMCL finds the robot again
"""

import json
import math
import os
import statistics
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.msg import ParticleCloud
from rclpy.qos import qos_profile_sensor_data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase1_drive_test import wrap  # noqa: E402
from phase4_mapping_test import TOUR_HOUSE, MappingDriver  # noqa: E402


MAPS_DIR = os.path.join(HERE, "..", "..", "ros2_ws", "maps")


def saved_cells(name):
    """The saved map <name>.pgm as OccupancyGrid values, by the colours map_files writes:
    254 free (0), 0 occupied (100), 205 unknown (-1). Rows bottom first."""
    raw = open(os.path.join(MAPS_DIR, name + ".pgm"), "rb").read()
    head = raw.split(b"\n", 3)
    width, height = map(int, head[1].split())
    pixels = np.frombuffer(head[3][: width * height], dtype=np.uint8).reshape(height, width)
    cells = np.full(pixels.shape, -1, dtype=np.int8)
    cells[pixels == 254] = 0
    cells[pixels == 0] = 100
    return np.flipud(cells), int(np.count_nonzero((pixels != 254) & (pixels != 0) & (pixels != 205)))


class LocalizationDriver(MappingDriver):
    def __init__(self):
        super().__init__()
        self.grid = None
        self.particles = 0
        self.samples = []  # (sim time, position error m, heading error rad)
        self.sampling = False
        self._last_sample = -1.0
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 5)
        self.create_subscription(ParticleCloud, "particle_cloud", self._on_cloud, qos_profile_sensor_data)

    def _on_cloud(self, msg):
        self.particles = len(msg.particles)

    def _on_map(self, msg):
        super()._on_map(msg)
        self.grid = msg

    def error(self):
        """(position error m, heading error rad) of AMCL's pose against the truth."""
        tx, ty, tyaw = self.pose()
        ax, ay, ayaw = self.slam_pose()  # same maths: (map -> odom) composed with (odom -> base_link)
        return math.hypot(ax - tx, ay - ty), wrap(ayaw - tyaw)

    def spin(self, seconds=0.02):
        super().spin(seconds)
        if self.sampling and self.map_to_odom is not None and self.sim_time is not None:
            if self.sim_time - self._last_sample >= 0.25:
                self._last_sample = self.sim_time
                e, a = self.error()
                self.samples.append((self.sim_time, e, a))

    def set_pose_estimate(self, x, y, yaw, sigma_xy=0.5, sigma_yaw=0.45):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp.sec = int(self.sim_time)
        msg.header.stamp.nanosec = int((self.sim_time - int(self.sim_time)) * 1e9)
        msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        cov = [0.0] * 36
        cov[0] = cov[7] = sigma_xy**2
        cov[35] = sigma_yaw**2
        msg.pose.covariance = cov
        for _ in range(3):
            self.initialpose_pub.publish(msg)
            self.spin(0.05)


def main():
    rclpy.init()
    n = LocalizationDriver()
    results, numbers = [], {}

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if not n.wait_ready():
            check("starts localized", False, "no /odom, /ground_truth/odom or /clock within 60 s; is run_sim.py playing?")
            return results
        modes = ModeClient(n)
        if modes.wait_ready() is None:
            check("starts localized", False, "no /robot_state from the mode manager; is the robot stack running?")
            return results
        ok, why = modes.request("manual")
        if not ok:
            check("starts localized", False, f"could not enter manual mode: {why}")
            return results
        wall0 = time.time()
        while (n.map_msgs == 0 or n.map_to_odom is None) and time.time() - wall0 < 40.0:
            n.spin(0.1)
        if n.map_msgs == 0 or n.map_to_odom is None:
            check("starts localized", False,
                  f"in manual mode but map received={n.map_msgs > 0}, "
                  f"map->odom transform received={n.map_to_odom is not None}; is there a saved map?")
            return results

        # 0. the map map_server serves is the saved file, unknown cells included
        name = modes.state.current_map if modes.state is not None else ""
        expected, odd_pixels = saved_cells(name) if name else (None, 0)
        served = (np.asarray(n.grid.data, dtype=np.int8).reshape(n.grid.info.height, n.grid.info.width)
                  if n.grid is not None else None)
        if expected is None or served is None or served.shape != expected.shape:
            check("map served as saved", False, f"map '{name}': file {None if expected is None else expected.shape}, "
                  f"/map {None if served is None else served.shape}")
        else:
            counts = {label: (int(np.count_nonzero(expected == v)), int(np.count_nonzero(served == v)))
                      for label, v in (("free", 0), ("occupied", 100), ("unknown", -1))}
            differ = int(np.count_nonzero(served != expected))
            check("map served as saved", differ == 0 and odd_pixels == 0 and counts["unknown"][0] > 0,
                  f"map '{name}': {differ} cells differ; file / served: " +
                  ", ".join(f"{k} {a} / {b}" for k, (a, b) in counts.items()) +
                  (f"; {odd_pixels} pixels in the file are none of the three colours" if odd_pixels else ""))

        # 1. localized at start, with no pose given by us
        n.run_for(0.0, 0.0, 2.0)
        e0, a0 = n.error()
        check("starts localized", e0 < 0.08 and abs(math.degrees(a0)) < 2.0,
              f"before moving, AMCL is {e0 * 100:.1f} cm and {math.degrees(a0):+.1f} deg from the truth")

        # 2. tracking over the tour
        n.sampling = True
        reached = True
        for hx, hy in TOUR_HOUSE:
            reached = n.go_to(*hl.to_world(hx, hy)) and reached
        n.face(0.0)
        n.run_for(0.0, 0.0, 1.0)
        n.sampling = False
        pos = sorted(s[1] for s in n.samples)
        ang = sorted(abs(math.degrees(s[2])) for s in n.samples)
        med, p95, worst = statistics.median(pos), pos[int(0.95 * len(pos))], pos[-1]
        a_med, a_p95 = statistics.median(ang), ang[int(0.95 * len(ang))]
        tx, ty, tyaw = n.pose()
        ox, oy, oyaw = n.odom_pose()
        odom_err, odom_yaw = math.hypot(ox - tx, oy - ty), math.degrees(wrap(oyaw - tyaw))
        # AMCL only publishes its cloud after the robot has moved, so the count is checked here.
        check("tracking", reached and len(pos) > 100 and med < 0.06 and p95 < 0.12 and worst < 0.20 and a_p95 < 3.0
              and n.particles >= 500,
              f"tour completed={reached}, {len(pos)} samples; position error median {med * 100:.1f} cm, "
              f"95th percentile {p95 * 100:.1f} cm, worst {worst * 100:.1f} cm; heading error median {a_med:.1f} deg, "
              f"95th percentile {a_p95:.1f} deg; {n.particles} guesses in the cloud; "
              f"wheel odometry alone ended {odom_err * 100:.1f} cm and {odom_yaw:+.1f} deg off")
        numbers.update({"median_cm": med * 100, "p95_cm": p95 * 100, "worst_cm": worst * 100,
                        "odom_err_cm": odom_err * 100, "particles": n.particles})

        # 3. recovery from a wrong pose estimate (what clicking the wrong spot in RViz would do)
        tx, ty, tyaw = n.pose()
        n.set_pose_estimate(tx + 0.5, ty + 0.3, tyaw + math.radians(25.0))
        n.run_for(0.0, 0.0, 1.5)
        e_wrong, a_wrong = n.error()
        n.samples, n.sampling = [], True
        t0 = n.sim_time
        n.run_for(0.0, 0.8, 2.0 * math.pi / 0.8)  # one full turn on the spot
        n.go_to(1.5, 0.6)
        n.go_to(0.6, 0.0)
        n.run_for(0.0, 0.0, 2.0)
        n.sampling = False
        converged = None
        for i, (t, e, _a) in enumerate(n.samples):  # first moment the error drops under 10 cm and stays there
            if all(s[1] < 0.10 for s in n.samples[i:]):
                converged = t - t0
                break
        e_end, a_end = n.error()
        check("recovery", e_wrong > 0.3 and converged is not None and converged < 40.0 and e_end < 0.08
              and abs(math.degrees(a_end)) < 3.0,
              f"after the wrong estimate AMCL was {e_wrong * 100:.0f} cm and {math.degrees(a_wrong):+.0f} deg off; "
              + (f"back within 10 cm after {converged:.0f} s of driving" if converged is not None else "never came back within 10 cm")
              + f"; final error {e_end * 100:.1f} cm and {math.degrees(a_end):+.1f} deg")
        numbers.update({"recovery_s": converged, "recovery_end_cm": e_end * 100})
    finally:
        try:
            for _ in range(20):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE5_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "numbers": numbers}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
