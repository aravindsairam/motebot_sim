#!/usr/bin/env python3
"""Phase 9 acceptance test: does the robot's camera find the dock, and how accurately?

Needs two things running first, with the robot at its start pose and a saved map present:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase9_dock_detection_test.py

The robot is driven (in manual mode) to a set of viewpoints. At each one it stands still, and the
dock tag's position reported by the detector is compared with where the tag truly is, worked out
from the simulator's true robot pose and the house layout.

Checks:
  1. no false dock    facing away from the dock, nothing is detected
  2. straight ahead   on the dock's centre line from 0.2 m to 1.0 m: seen in nearly every frame, placed within centimetres
  3. off to the side  from about 20 degrees to either side, and with the tag off-centre in the image
  4. docked view      10 cm from the dock, where the robot will stop, the whole tag is still seen
  5. far away         1.5 m to 2.5 m: reported, to show how detection degrades (not pass/fail)

Limits: detected in at least 90% of frames, position within 2 cm plus 2% of the distance, and the
tag's facing direction within 5 degrees (1 cm and 5 degrees when docked).

The pass/fail range ends at 1.0 m on purpose. A flat tag seen straight on from further away is
still found and placed well, but which way it faces becomes uncertain (measured: 0.2 degrees off
at 0.5 m, 1.2 at 1.0 m, 5.9 at 1.5 m, 7.3 at 2.0 m). Docking should therefore start its final
approach from 1 m or closer.
"""

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase1_drive_test import wrap  # noqa: E402
from phase7_mode_manager_test import ModeTestDriver  # noqa: E402

TAG_FRAME = "dock_tag"
TAG_X, TAG_Y, TAG_Z = hl.DOCK_TAG_HOUSE[0] + 0.001, hl.DOCK_TAG_HOUSE[1], hl.DOCK_TAG_HOUSE[2]
DOCKED_X = hl.DOCK_TOWER[2] + 0.10 + hl.ROBOT_RADIUS  # robot centre when its front is 10 cm from the tower

# Camera optical frame (x right, y down, z forward) in base_link (x forward, y left, z up).
BASE_TO_OPTICAL = np.eye(4)
BASE_TO_OPTICAL[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
BASE_TO_OPTICAL[:3, 3] = hl.CAMERA_XYZ
# The tag's own frame as apriltag_ros reports it (measured, not assumed): z points out of the tag
# towards whoever is looking at it, y points down, and x therefore to the viewer's left.
# The tag faces +x (east) into the room, so z is +x, y is -z and x is -y of the world.
WORLD_TO_TAG = np.eye(4)
WORLD_TO_TAG[:3, :3] = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
WORLD_TO_TAG[:3, 3] = (*hl.to_world(TAG_X, TAG_Y), TAG_Z)


def matrix(x, y, z, qx, qy, qz, qw):
    m = np.eye(4)
    m[:3, :3] = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    m[:3, 3] = (x, y, z)
    return m


def angle_between(a, b):
    """Angle in degrees of the rotation that takes orientation a to orientation b."""
    c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class DockTestDriver(ModeTestDriver):
    def __init__(self):
        super().__init__()
        self.detections = []  # 4x4 camera -> tag matrices, while recording
        self.frames = 0  # camera frames, while recording
        self.recording = False
        self.create_subscription(TFMessage, "tf", self._on_tag_tf, 100)
        self.create_subscription(CameraInfo, "camera/camera_info", self._on_frame, 20)

    def _on_tag_tf(self, msg):
        if not self.recording:
            return
        for t in msg.transforms:
            if t.child_frame_id == TAG_FRAME:
                tr, q = t.transform.translation, t.transform.rotation
                self.detections.append(matrix(tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))

    def _on_frame(self, _msg):
        if self.recording:
            self.frames += 1

    def true_camera_to_tag(self):
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        world_to_camera = matrix(p.x, p.y, p.z, q.x, q.y, q.z, q.w) @ BASE_TO_OPTICAL
        return np.linalg.inv(world_to_camera) @ WORLD_TO_TAG

    def face_tag(self, extra_deg=0.0):
        x, y, _ = self.pose()
        tx, ty = hl.to_world(TAG_X, TAG_Y)
        self.face(math.atan2(ty - y, tx - x) + math.radians(extra_deg))

    def creep_to(self, house_x, speed=0.05):
        """Drive slowly straight at the dock until the robot centre reaches house_x."""
        wall0 = time.time()
        while hl.to_house(*self.pose()[:2])[0] > house_x and time.time() - wall0 < 60.0:
            self.cmd(speed, 0.0)
            self.spin()
        self.stop(0.5)

    def measure(self, sim_seconds=2.0):
        """Stand still and compare what the detector reports with the truth."""
        self.run_for(0.0, 0.0, 1.0)  # at rest, and any image taken while moving is gone
        truth = self.true_camera_to_tag()
        self.detections, self.frames, self.recording = [], 0, True
        self.wait_sim(sim_seconds)
        self.recording = False
        seen = self.detections
        out = {
            "distance": float(np.linalg.norm(truth[:3, 3])),
            "view_angle": math.degrees(math.atan2(abs(truth[0, 2]), abs(truth[2, 2]))),  # how obliquely the tag is seen
            "bearing": math.degrees(math.atan2(truth[0, 3], truth[2, 3])),  # where it is in the image, + = right
            "frames": self.frames,
            "rate": min(1.0, len(seen) / max(1, self.frames)),
        }
        if seen:
            errors = np.array([d[:3, 3] - truth[:3, 3] for d in seen])
            out["position_error"] = float(np.linalg.norm(errors.mean(axis=0)))
            out["sideways_error"] = float(abs(errors[:, 0].mean()))
            out["depth_error"] = float(abs(errors[:, 2].mean()))
            out["jitter"] = float(np.linalg.norm(errors.std(axis=0)))
            out["angle_error"] = float(np.median([angle_between(d, truth) for d in seen]))
        return out


def good(m, position_limit, angle_limit=5.0):
    return (m["rate"] >= 0.9 and "position_error" in m and m["position_error"] <= position_limit
            and m["angle_error"] <= angle_limit)


def text(name, m):
    head = f"{name}: {m['distance']:.2f} m away, seen at {m['view_angle']:.0f} deg"
    if "position_error" not in m:
        return head + f", not detected in {m['frames']} frames"
    return (head + f", detected in {m['rate'] * 100:.0f}% of {m['frames']} frames, position off by "
            f"{m['position_error'] * 100:.1f} cm (sideways {m['sideways_error'] * 100:.1f}, depth {m['depth_error'] * 100:.1f}), "
            f"facing off by {m['angle_error']:.1f} deg")


def main():
    rclpy.init()
    n = DockTestDriver()
    modes = ModeClient(n)
    results, table = [], {}

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def view(name, house_x, house_y, extra_deg=0.0):
        n.go_to(*hl.to_world(house_x, house_y))
        n.face_tag(extra_deg)
        m = n.measure()
        table[name] = m
        return m

    try:
        if not n.wait_ready():
            check("no false dock", False, "no /odom, /ground_truth/odom or /clock within 60 s; is the simulator running?")
            return results
        if modes.wait_ready() is None:
            check("no false dock", False, "no /robot_state from the mode manager; is the robot stack running?")
            return results
        ok, why = modes.request("manual")
        if not ok:
            check("no false dock", False, f"could not enter manual mode: {why}")
            return results

        # 1. facing away from the dock (the start pose): nothing may be detected
        m = n.measure(3.0)
        check("no false dock", m["frames"] > 20 and m["rate"] == 0.0,
              f"facing away from the dock: {m['frames']} camera frames, {int(round(m['rate'] * m['frames']))} detections")

        # 4 (done first, the robot is right there). the view from where the robot will sit when docked
        n.face_tag()
        start = n.measure()
        table["start pose, turned around"] = start
        n.creep_to(DOCKED_X)
        n.face_tag()
        docked = n.measure()
        table["docked"] = docked
        gap = hl.to_house(*n.pose()[:2])[0] - hl.ROBOT_RADIUS - hl.DOCK_TOWER[2]

        # 2. straight ahead, from close to 1.5 m
        line = [start] + [view(f"centre line {d} m", TAG_X + d + hl.CAMERA_XYZ[0], hl.DOCK_Y) for d in (0.5, 1.0)]
        check("straight ahead", all(good(m, 0.02 + 0.02 * m["distance"]) for m in line),
              "; ".join(text(f"{m['distance']:.1f} m", m) for m in line))

        # 5. far away: where does it give out?
        far = [view(f"centre line {d} m", TAG_X + d + hl.CAMERA_XYZ[0], hl.DOCK_Y) for d in (1.5, 2.0, 2.5)]

        # 3. from the side, and with the tag away from the middle of the image
        side = [
            view("1.0 m, from one side", 1.15, hl.DOCK_Y + 0.42),
            view("1.0 m, from the other side", 1.15, hl.DOCK_Y - 0.42),
            view("0.6 m, from one side", 0.75, hl.DOCK_Y + 0.30),
            view("0.6 m, from the other side", 0.75, hl.DOCK_Y - 0.30),
            view("1.0 m, tag 20 deg off-centre", TAG_X + 1.0 + hl.CAMERA_XYZ[0], hl.DOCK_Y, 20.0),
            view("1.0 m, tag 20 deg off-centre the other way", TAG_X + 1.0 + hl.CAMERA_XYZ[0], hl.DOCK_Y, -20.0),
        ]
        check("off to the side", all(good(m, 0.02 + 0.02 * m["distance"]) for m in side),
              "; ".join(text(name, m) for name, m in list(table.items())[-len(side):]))

        check("docked view", good(docked, 0.01) and 0.07 < gap < 0.13,
              f"robot's front {gap * 100:.1f} cm from the dock tower; " + text("tag", docked))

        check("far away", True, "; ".join(text(f"{m['distance']:.1f} m", m) for m in far) + " (reported only)")
        modes.request("idle")
    finally:
        try:
            for _ in range(10):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE9_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "table": table}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
