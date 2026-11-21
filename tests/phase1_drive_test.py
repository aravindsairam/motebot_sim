#!/usr/bin/env python3
"""Phase 1 acceptance test: drive the simulated Create 3 over ROS 2 and check what it does.

Tests the simulator alone: run it WITHOUT the robot stack, while run_sim.py --no-noise is playing:
    source /opt/ros/humble/setup.bash
    python3 sim/tests/phase1_drive_test.py

Poses are read from /ground_truth/odom (the robot starts at the world origin facing +x).
Intended for `run_sim.py --no-noise`; with noise on it still works, since it only uses the true pose.
Checks:
  1. topics     /clock, /odom, /tf (odom -> base_link) and /joint_states are publishing
  2. straight   0.2 m/s for 5 s of sim time covers about 1.0 m without drifting sideways
  3. rotate     1.0 rad/s for pi seconds turns about 180 degrees in place
  4. speed cap  a 1.0 m/s command is limited to the Create 3 maximum of 0.306 m/s
  5. doorway    drives from the living room through the 0.8 m doorway into the bedroom
  6. collision  driving at the wardrobe stops at its face instead of passing through
  7. timeout    the wheels stop by themselves half a second after commands stop arriving
"""

import json
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import house_layout as hl  # noqa: E402


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def tilt_of(q):
    """Angle between the robot's up axis and world up, in degrees."""
    up_z = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    return math.degrees(math.acos(max(-1.0, min(1.0, up_z))))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Driver(Node):
    # Where this driver sends velocity commands. With the robot stack running, /cmd_vel is the
    # manual-driving input of the safety gate. Tests of the simulator alone set this to the wheel
    # topic (house_layout.WHEEL_CMD_TOPIC) before creating a Driver.
    CMD_TOPIC = "cmd_vel"

    def __init__(self):
        super().__init__("phase1_drive_test")
        self.pub = self.create_publisher(Twist, self.CMD_TOPIC, 10)
        # The true pose drives the test; /odom is the robot's own estimate and may drift.
        self.create_subscription(Odometry, "ground_truth/odom", self._on_odom, 50)
        self.create_subscription(Odometry, "odom", self._on_robot_odom, 50)
        self.robot_odom = None
        self.create_subscription(Clock, "clock", self._on_clock, 50)
        self.create_subscription(TFMessage, "tf", self._on_tf, 50)
        self.create_subscription(JointState, "joint_states", self._on_joints, 50)
        self.odom = None
        self.sim_time = None
        self.tf_pairs = set()
        self.joint_names = []
        self.counts = {"odom": 0, "clock": 0, "tf": 0, "joint_states": 0}
        self.max_tilt = 0.0
        self.max_abs_z = 0.0
        self.track = []

    def _on_robot_odom(self, msg):
        self.robot_odom = msg
        self.counts["odom"] += 1

    def _on_odom(self, msg):
        self.odom = msg
        self.counts["truth"] = self.counts.get("truth", 0) + 1
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.max_tilt = max(self.max_tilt, tilt_of(q))
        self.max_abs_z = max(self.max_abs_z, abs(p.z))
        if self.counts["truth"] % 6 == 0:
            self.track.append((round(p.x, 3), round(p.y, 3)))

    def _on_clock(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        self.counts["clock"] += 1

    def _on_tf(self, msg):
        self.counts["tf"] += 1
        for t in msg.transforms:
            self.tf_pairs.add((t.header.frame_id, t.child_frame_id))

    def _on_joints(self, msg):
        self.counts["joint_states"] += 1
        self.joint_names = list(msg.name)

    # --- helpers -------------------------------------------------------------
    def spin(self, seconds=0.02):
        rclpy.spin_once(self, timeout_sec=seconds)

    def wait_clock(self, limit=30.0):
        """The simulator's time, once the first clock message is in. A node that has just started can
        get it seconds after its first sensor data, so nothing may count on sim_time before this."""
        end = time.time() + limit
        while self.sim_time is None:
            if time.time() > end:
                raise RuntimeError(f"no clock message from the simulator after {limit:.0f} s")
            self.spin()
        return self.sim_time

    def pose(self):
        p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
        return p.x, p.y, yaw_of(q)

    def cmd(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub.publish(m)

    def stop(self, settle=1.0):
        self.run_for(0.0, 0.0, settle)

    def run_for(self, v, w, sim_seconds, wall_timeout=120.0):
        """Hold a command for a duration measured in simulation time."""
        t0, wall0 = self.wait_clock(), time.time()
        peak_v = 0.0
        while self.sim_time - t0 < sim_seconds:
            self.cmd(v, w)
            self.spin()
            peak_v = max(peak_v, abs(self.odom.twist.twist.linear.x))
            if time.time() - wall0 > wall_timeout:
                raise TimeoutError("simulation time is not advancing")
        return peak_v

    def wait_ready(self, wall_timeout=60.0):
        wall0 = time.time()
        while time.time() - wall0 < wall_timeout:
            self.spin(0.1)
            if self.odom is not None and self.robot_odom is not None and self.sim_time is not None:
                return True
        return False

    def go_to(self, gx, gy, tol=0.06, wall_timeout=180.0):
        """Turn toward a point, then drive to it with simple proportional steering."""
        wall0 = time.time()
        while True:
            x, y, yaw = self.pose()
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            err = wrap(math.atan2(dy, dx) - yaw)
            if abs(err) > 0.35:
                self.cmd(0.0, max(-1.2, min(1.2, 2.0 * err)))
            else:
                self.cmd(min(0.28, 0.15 + dist), max(-1.0, min(1.0, 2.5 * err)))
            self.spin()
            if time.time() - wall0 > wall_timeout:
                self.stop()
                return False
        self.stop(0.5)
        return True

    def face(self, target_yaw, tol=0.02, wall_timeout=60.0):
        wall0 = time.time()
        while True:
            err = wrap(target_yaw - self.pose()[2])
            if abs(err) < tol:
                break
            self.cmd(0.0, max(-1.0, min(1.0, 2.5 * err)))
            self.spin()
            if time.time() - wall0 > wall_timeout:
                break
        self.stop(0.5)


def main():
    rclpy.init()
    Driver.CMD_TOPIC = hl.WHEEL_CMD_TOPIC  # simulator only, no safety gate: talk to the wheels
    n = Driver()
    results = []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if not n.wait_ready():
            check("topics", False, "no /odom or /clock within 60 s; is run_sim.py playing?")
            return results

        # 1. topics and rates, measured over 3 s of sim time
        c0, t0, wall0 = dict(n.counts), n.sim_time, time.time()
        n.run_for(0.0, 0.0, 3.0)
        dt_sim, dt_wall = n.sim_time - t0, time.time() - wall0
        rates = {k: (n.counts[k] - c0[k]) / dt_sim for k in n.counts}
        rtf = dt_sim / dt_wall
        tf_ok = ("odom", "base_link") in n.tf_pairs
        wheels_ok = {"left_wheel_joint", "right_wheel_joint"} <= set(n.joint_names)
        check(
            "topics",
            all(r > 5.0 for r in rates.values()) and tf_ok and wheels_ok,
            "rates Hz " + ", ".join(f"{k}={v:.0f}" for k, v in rates.items())
            + f"; tf odom->base_link={tf_ok}; wheel joints in joint_states={wheels_ok}; real-time factor={rtf:.2f}",
        )
        x0, y0, yaw0 = n.pose()
        check("start pose", abs(x0) < 0.02 and abs(y0) < 0.02 and abs(yaw0) < 0.02,
              f"x={x0:.3f} y={y0:.3f} yaw={math.degrees(yaw0):.1f} deg (expected the origin)")

        # 2. straight line
        n.run_for(0.2, 0.0, 5.0)
        n.stop()
        x1, y1, yaw1 = n.pose()
        fwd = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
        lat = -(x1 - x0) * math.sin(yaw0) + (y1 - y0) * math.cos(yaw0)
        check("straight", abs(fwd - 1.0) < 0.05 and abs(lat) < 0.02 and abs(wrap(yaw1 - yaw0)) < math.radians(2),
              f"forward={fwd:.3f} m (target 1.000), sideways={lat * 100:.1f} cm, "
              f"heading change={math.degrees(wrap(yaw1 - yaw0)):.2f} deg")

        # 3. rotate in place: accumulate yaw so a half turn is not lost at the +/-180 wrap
        total, prev = 0.0, n.pose()[2]
        t0 = n.sim_time
        xr, yr, _ = n.pose()
        while n.sim_time - t0 < math.pi:
            n.cmd(0.0, 1.0)
            n.spin()
            cur = n.pose()[2]
            total += wrap(cur - prev)
            prev = cur
        t_stop = n.sim_time
        while n.sim_time - t_stop < 1.0:
            n.cmd(0.0, 0.0)
            n.spin()
            cur = n.pose()[2]
            total += wrap(cur - prev)
            prev = cur
        xs, ys, _ = n.pose()
        shift = math.hypot(xs - xr, ys - yr)
        check("rotate", abs(math.degrees(total) - 180.0) < 8.0 and shift < 0.03,
              f"turned {math.degrees(total):.1f} deg (target 180), centre moved {shift * 100:.1f} cm")

        # 4. speed cap
        n.face(0.0)
        n.max_tilt = 0.0
        peak = n.run_for(1.0, 0.0, 2.5)
        n.stop()
        check("speed cap", 0.25 < peak < hl.MAX_LINEAR_SPEED + 0.03 and n.max_tilt < 2.0,
              f"commanded 1.0 m/s, peak measured {peak:.3f} m/s (limit {hl.MAX_LINEAR_SPEED}); "
              f"max tilt speeding up and braking hard={n.max_tilt:.2f} deg")

        # 5. through the doorway into the bedroom
        door_y = (hl.DOORWAY[1] + hl.DOORWAY[3]) / 2.0
        door_x = (hl.DOORWAY[0] + hl.DOORWAY[2]) / 2.0
        waypoints = [hl.to_world(door_x - 0.9, door_y), hl.to_world(door_x + 0.9, door_y), hl.to_world(6.45, 2.75)]
        tilt_before = n.max_tilt
        n.max_tilt = 0.0
        reached = all(n.go_to(wx, wy) for wx, wy in waypoints)
        hx, hy = hl.to_house(*n.pose()[:2])
        b = hl.BEDROOM
        in_bedroom = b[0] < hx < b[2] and b[1] < hy < b[3]
        check("doorway", reached and in_bedroom and n.max_tilt < 3.0 and n.max_abs_z < 0.02,
              f"reached waypoints={reached}, final house position=({hx:.2f}, {hy:.2f}) in bedroom={in_bedroom}, "
              f"max tilt on this leg={n.max_tilt:.2f} deg (earlier checks {tilt_before:.2f}), max |z|={n.max_abs_z * 1000:.1f} mm")

        # 6. collision with the wardrobe (its south face is at house y = 3.85)
        n.face(math.pi / 2.0)
        n.max_tilt = 0.0
        n.run_for(0.25, 0.0, 8.0)
        n.stop()
        hx, hy = hl.to_house(*n.pose()[:2])
        face_y = 3.85
        expected = face_y - hl.ROBOT_RADIUS
        check("collision", expected - 0.06 < hy < expected + 0.03 and n.max_tilt < 5.0,
              f"stopped with centre at house y={hy:.3f} (wardrobe face {face_y}, expected about {expected:.3f}); "
              f"max tilt while pushing={n.max_tilt:.2f} deg")

        # 7. motor-controller timeout and braking: the wheels stop by themselves when commands stop
        n.run_for(-0.25, 0.0, 1.5)  # back away from the wardrobe, and keep moving
        n.max_tilt = 0.0
        moving = abs(n.odom.twist.twist.linear.x)
        t0, wall0, stopped_after = n.sim_time, time.time(), None
        while n.sim_time - t0 < 3.0 and time.time() - wall0 < 30.0:
            n.spin()  # no commands published from here on
            if abs(n.odom.twist.twist.linear.x) < 0.01:
                stopped_after = n.sim_time - t0
                break
        check("command timeout", moving > 0.2 and stopped_after is not None
              and hl.BASE_CMD_TIMEOUT - 0.15 < stopped_after < hl.BASE_CMD_TIMEOUT + 0.5 and n.max_tilt < 2.0,
              f"moving at {moving:.2f} m/s when commands stopped; at rest after "
              f"{'%.2f s' % stopped_after if stopped_after is not None else 'never'} "
              f"(timeout {hl.BASE_CMD_TIMEOUT} s plus braking); max tilt while braking={n.max_tilt:.2f} deg")
    finally:
        try:
            for _ in range(20):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE1_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "track_world_xy": n.track}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
