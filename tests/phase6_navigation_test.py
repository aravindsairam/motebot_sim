#!/usr/bin/env python3
"""Phase 6 acceptance test: send Nav2 goals around the house and check where the robot really ends up.

Needs two things running first, with the robot at its start pose:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase6_navigation_test.py

The test asks the mode manager for the mode it needs (navigation).

Checks:
  1. nav2 ready      the NavigateToPose action is available and AMCL is localizing
  2. reaches goals   five goals in both rooms, through the doorway both ways, each reached within tolerance
  3. no-go zone      a goal inside the stairwell is refused and the robot never enters the zone
  4. no collisions   the bumper is never pressed and the cliff sensors never trip while navigating
"""

import json
import math
import os
import sys
import time

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from PIL import Image, ImageDraw
from rclpy.action import ActionClient
from sensor_msgs.msg import Range
from std_msgs.msg import UInt8

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase1_drive_test import wrap  # noqa: E402
from phase4_mapping_test import MappingDriver, load_map  # noqa: E402

MAP_STEM = os.path.join(REPO, "ros2_ws", "maps", "house")

# (name, house x, house y, heading in degrees, should Nav2 accept it)
GOALS = [
    ("living room, far corner", 4.0, 3.7, 90.0, True),
    ("bedroom, through the doorway", 6.0, 1.6, -90.0, True),
    ("bedroom, beside the stairwell", 5.6, 3.6, 90.0, True),
    ("inside the stairwell", 5.55, 5.0, 90.0, False),
    ("living room, between sofa and armchair", 3.55, 1.25, 180.0, True),
    ("home, in front of the dock", 0.45, 2.25, 0.0, True),
]
# The stairwell no-go zone from maps/house_zones.yaml, shrunk by 5 cm for localization error.
ZONE = (4.60, 2.05, 5.60, 3.25)
STATUS_NAMES = {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_ABORTED: "aborted",
                GoalStatus.STATUS_CANCELED: "canceled"}


class NavDriver(MappingDriver):
    def __init__(self):
        super().__init__()
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.bumper_seen = 0
        self.cliff_trips = 0
        self.zone_intrusions = 0
        self.monitoring = False
        self.create_subscription(UInt8, "bumper", self._on_bumper, 50)
        for side in ("front_left", "front_right"):
            self.create_subscription(Range, f"cliff/{side}", self._on_cliff, 20)

    def _on_bumper(self, m):
        if self.monitoring:
            self.bumper_seen |= m.data

    def _on_cliff(self, m):
        if self.monitoring and (math.isinf(m.range) or m.range > hl.CLIFF_THRESHOLD):
            self.cliff_trips += 1

    def spin(self, seconds=0.02):
        super().spin(seconds)
        if self.monitoring and self.odom is not None:
            x, y, _ = self.pose()
            if ZONE[0] < x < ZONE[2] and ZONE[1] < y < ZONE[3]:
                self.zone_intrusions += 1

    def navigate(self, x, y, yaw, sim_timeout=150.0):
        """Send one goal in the map frame and wait. Returns (status name, sim seconds, metres driven)."""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        send = self.nav.send_goal_async(goal)
        while not send.done():
            self.spin()
        handle = send.result()
        if not handle.accepted:
            return "rejected", 0.0, 0.0
        result = handle.get_result_async()
        t0, dist, prev = self.sim_time, 0.0, self.pose()[:2]
        while not result.done():
            self.spin()
            cur = self.pose()[:2]
            dist += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            prev = cur
            if self.sim_time - t0 > sim_timeout:
                cancel = handle.cancel_goal_async()
                while not cancel.done():
                    self.spin()
                return "timed out", self.sim_time - t0, dist
        return STATUS_NAMES.get(result.result().status, str(result.result().status)), self.sim_time - t0, dist


def draw_track(track, goals, out_path):
    """Picture of the saved map with the true path driven and the goals marked."""
    img, res, origin = load_map(MAP_STEM)
    h, w = img.shape
    k = 6
    canvas = Image.fromarray(img).resize((w * k, h * k), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(canvas)

    def px(x, y):
        return ((x - origin[0]) / res * k, (h - (y - origin[1]) / res) * k)

    zx0, zy0 = px(ZONE[0] - 0.05, ZONE[3] + 0.05)
    zx1, zy1 = px(ZONE[2] + 0.05, ZONE[1] - 0.05)
    d.rectangle([zx0, zy0, zx1, zy1], outline=(200, 40, 40), width=3)
    if len(track) > 1:
        d.line([px(x, y) for x, y in track], fill=(30, 110, 220), width=3)
    for (name, gx, gy, ok) in goals:
        cx, cy = px(gx, gy)
        colour = (30, 160, 60) if ok else (200, 40, 40)
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=colour, width=3)
    canvas.save(out_path)


def main():
    rclpy.init()
    n = NavDriver()
    results, legs = [], []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if not n.wait_ready():
            check("nav2 ready", False, "no /odom, /ground_truth/odom or /clock within 60 s; is run_sim.py playing?")
            return results
        modes = ModeClient(n)
        if modes.wait_ready() is None:
            check("nav2 ready", False, "no /robot_state from the mode manager; is the robot stack running?")
            return results
        ok, why = modes.request("navigation")
        if not ok:
            check("nav2 ready", False, f"could not enter navigation mode: {why}")
            return results
        server = n.nav.wait_for_server(timeout_sec=30.0)
        wall0 = time.time()
        while n.map_to_odom is None and time.time() - wall0 < 30.0:
            n.spin(0.1)
        check("nav2 ready", server and n.map_to_odom is not None,
              f"navigate_to_pose action available={server}, AMCL publishing map->odom={n.map_to_odom is not None}")
        if not (server and n.map_to_odom is not None):
            return results
        n.run_for(0.0, 0.0, 3.0)  # let the costmaps fill in

        n.monitoring = True
        n.track = []
        for name, hx, hy, yaw_deg, reachable in GOALS:
            gx, gy = hl.to_world(hx, hy)
            yaw = math.radians(yaw_deg)
            status, secs, dist = n.navigate(gx, gy, yaw)
            n.stop(0.5)
            tx, ty, tyaw = n.pose()
            legs.append({"name": name, "goal": [gx, gy], "reachable": reachable, "status": status, "seconds": secs,
                         "metres": dist, "pos_err": math.hypot(tx - gx, ty - gy), "yaw_err": abs(wrap(tyaw - yaw)),
                         "end": [tx, ty]})
            print(f"      {name}: {status} in {secs:.0f} s, {dist:.1f} m driven, "
                  f"ended {legs[-1]['pos_err'] * 100:.0f} cm from the goal", flush=True)
        n.monitoring = False

        good = [g for g in legs if g["reachable"]]
        ok_legs = [g for g in good if g["status"] == "succeeded" and g["pos_err"] < 0.15 and g["yaw_err"] < 0.25]
        worst_pos = max(g["pos_err"] for g in good)
        worst_yaw = max(g["yaw_err"] for g in good)
        total_m, total_s = sum(g["metres"] for g in good), sum(g["seconds"] for g in good)
        failed = [f"{g['name']} ({g['status']}, {g['pos_err'] * 100:.0f} cm off)" for g in good if g not in ok_legs]
        check("reaches goals", len(ok_legs) == len(good),
              f"{len(ok_legs)} of {len(good)} goals reached; worst final error {worst_pos * 100:.1f} cm and "
              f"{math.degrees(worst_yaw):.1f} deg; {total_m:.1f} m driven in {total_s:.0f} s of sim time "
              f"({total_m / max(total_s, 1e-6):.2f} m/s average)" + (f"; failed: {'; '.join(failed)}" if failed else ""))

        bad = [g for g in legs if not g["reachable"]][0]
        check("no-go zone", bad["status"] != "succeeded" and n.zone_intrusions == 0,
              f"goal inside the stairwell was {bad['status']} after {bad['seconds']:.0f} s with the robot moving "
              f"{bad['metres']:.1f} m; robot was inside the no-go zone on {n.zone_intrusions} samples of the whole run")

        from phase2_sensor_test import zones  # noqa: E402  (bumper zone names)
        check("no collisions", n.bumper_seen == 0 and n.cliff_trips == 0,
              f"bumper zones pressed during navigation: {zones(n.bumper_seen)}; cliff sensor trips: {n.cliff_trips}")

        out_png = os.path.join(REPO, "ros2_ws", "maps", "navigation_test.png")
        draw_track(n.track, [(g["name"], g["goal"][0], g["goal"][1], g["reachable"]) for g in legs], out_png)
    finally:
        try:
            for _ in range(20):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE6_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "legs": legs}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
