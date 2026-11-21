#!/usr/bin/env python3
"""Phase 2 acceptance test: check every simulated sensor against what the house should produce.

Tests the simulator alone: run it WITHOUT the robot stack, while run_sim.py --no-noise is playing,
with the robot at its start pose:
    source /opt/ros/humble/setup.bash
    python3 sim/tests/phase2_sensor_test.py

Checks:
  1. rates       every sensor topic publishes at its expected rate
  2. frames      /tf_static has laser, imu_link, camera_link, camera_optical_frame, cliff_* under base_link
  3. lidar       /scan ranges match ray casts against the known house layout
  4. imu         gravity on z at rest; gyro z matches the turn rate while spinning
  5. camera      640x480 colour and depth; depth at the image centre matches the layout
  6. cliff idle  all four cliff sensors see the floor
  7. bumper      head-on and angled hits on the west wall press the right zones, release clears
  8. cliff edge  creeping to the stairwell trips the front cliff sensors before the robot falls
"""

import json
import math
import os
import statistics
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan, Range
from std_msgs.msg import UInt8
from tf2_msgs.msg import TFMessage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from phase1_drive_test import Driver, wrap  # noqa: E402

ZONE_BITS = {name: bit for name, bit, _lo, _hi in hl.BUMPER_ZONES}


def zones(mask):
    return [name for name, bit in ZONE_BITS.items() if mask & bit] or ["none"]


class SensorDriver(Driver):
    def __init__(self):
        super().__init__()
        self.scan = None
        self.imu = None
        self.rgb = None
        self.depth = None
        self.info = None
        self.bumper = None
        self.bumper_seen = 0
        self.cliff = {}
        self.tf_xyz = {}
        self.counts.update({k: 0 for k in ("scan", "imu", "rgb", "depth", "info", "bumper", "cliff")})
        # Fixed sensor frames are sent once on /tf_static, so the subscription must be latched too.
        latched = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(TFMessage, "tf_static", self._on_tf, latched)
        self.create_subscription(LaserScan, "scan", self._on_scan, 10)
        self.create_subscription(Imu, "imu", self._on_imu, 50)
        self.create_subscription(Image, "camera/image_raw", self._on_rgb, 2)
        self.create_subscription(Image, "camera/depth/image_raw", self._on_depth, 2)
        self.create_subscription(CameraInfo, "camera/camera_info", self._on_info, 5)
        self.create_subscription(UInt8, "bumper", self._on_bumper, 50)
        for name in hl.CLIFF_SENSORS:
            topic = "cliff/" + name[len("cliff_"):]
            self.create_subscription(Range, topic, lambda m, n=name: self._on_cliff(n, m), 20)

    def _on_scan(self, m):
        self.scan = m
        self.counts["scan"] += 1

    def _on_imu(self, m):
        self.imu = m
        self.counts["imu"] += 1

    def _on_rgb(self, m):
        self.rgb = m
        self.counts["rgb"] += 1

    def _on_depth(self, m):
        self.depth = m
        self.counts["depth"] += 1

    def _on_info(self, m):
        self.info = m
        self.counts["info"] += 1

    def _on_bumper(self, m):
        self.bumper = m.data
        self.bumper_seen |= m.data
        self.counts["bumper"] += 1

    def _on_cliff(self, name, m):
        self.cliff[name] = m
        if name == "cliff_front_left":
            self.counts["cliff"] += 1

    def _on_tf(self, msg):
        super()._on_tf(msg)
        for t in msg.transforms:
            tr = t.transform.translation
            self.tf_xyz[(t.header.frame_id, t.child_frame_id)] = (tr.x, tr.y, tr.z)

    def wait_sensors(self, wall_timeout=90.0):
        wall0 = time.time()
        need = ("scan", "imu", "rgb", "depth", "info")
        while time.time() - wall0 < wall_timeout:
            self.spin(0.1)
            if all(getattr(self, k) is not None for k in need) and len(self.cliff) == 4 and self.bumper is not None:
                return []
        missing = [k for k in need if getattr(self, k) is None]
        if len(self.cliff) < 4:
            missing.append("cliff")
        if self.bumper is None:
            missing.append("bumper")
        return missing

    def drive_until(self, v, predicate, max_sim_seconds):
        """Drive straight until predicate() is true. Returns True if it fired."""
        t0 = self.sim_time
        while self.sim_time - t0 < max_sim_seconds:
            if predicate():
                self.stop(0.6)
                return True
            self.cmd(v, 0.0)
            self.spin()
        self.stop(0.6)
        return False

    def front_cliff_tripped(self):
        return any(self.is_cliff(self.cliff[n].range) for n in ("cliff_front_left", "cliff_front_right"))

    @staticmethod
    def is_cliff(r):
        """A cliff is no reading at all, or a floor further away than normal."""
        return math.isinf(r) or r > hl.CLIFF_THRESHOLD


def scan_errors(scan, x, y, yaw, sign):
    """Absolute range errors for one angle convention (sign=+1 is the ROS standard, counter-clockwise)."""
    errs, valid = [], 0
    n = len(scan.ranges)
    step = max(1, n // 720)
    for i in range(0, n, step):
        r = scan.ranges[i]
        if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
            continue
        valid += 1
        a = scan.angle_min + i * scan.angle_increment
        exp = hl.expected_range(x, y, yaw + sign * a, hl.LIDAR_XYZ[2])
        errs.append(abs(r - exp))
    return errs, valid / max(1, len(range(0, n, step)))


def main():
    rclpy.init()
    SensorDriver.CMD_TOPIC = hl.WHEEL_CMD_TOPIC  # simulator only, no safety gate: talk to the wheels
    n = SensorDriver()
    results = []
    extra = {}

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if not n.wait_ready():
            check("rates", False, "no /odom or /clock within 60 s; is run_sim.py playing?")
            return results
        missing = n.wait_sensors()
        if missing:
            check("rates", False, f"no messages on: {', '.join(missing)}")
            return results

        # 1. rates over 5 s of sim time
        c0, t0, wall0 = dict(n.counts), n.sim_time, time.time()
        n.run_for(0.0, 0.0, 5.0)
        dt_sim, rtf = n.sim_time - t0, (n.sim_time - t0) / (time.time() - wall0)
        hz = {k: (n.counts[k] - c0[k]) / dt_sim for k in ("scan", "imu", "rgb", "depth", "info", "cliff", "bumper", "odom")}
        want = {"scan": (8, 12), "imu": (40, 70), "rgb": (10, 20), "depth": (10, 20), "info": (10, 20),
                "cliff": (50, 70), "bumper": (15, 70), "odom": (50, 70)}
        bad = [k for k, (lo, hi) in want.items() if not lo <= hz[k] <= hi]
        check("rates", not bad and rtf > 0.5,
              "Hz " + ", ".join(f"{k}={v:.1f}" for k, v in hz.items()) + f"; real-time factor={rtf:.2f}"
              + (f"; out of range: {bad}" if bad else ""))
        extra["rates_hz"] = hz
        extra["rtf"] = rtf

        # 2. frames
        needed = ["laser", "imu_link", "camera_link", "camera_optical_frame"] + list(hl.CLIFF_SENSORS)
        have = {child: parent for parent, child in n.tf_pairs}
        missing_frames = [f for f in needed if f not in have]
        laser = n.tf_xyz.get((have.get("laser", ""), "laser"))
        cam = n.tf_xyz.get((have.get("camera_link", ""), "camera_link"))
        geom_ok = (laser is not None and abs(laser[2] - hl.LIDAR_XYZ[2]) < 0.005
                   and cam is not None and abs(cam[0] - hl.CAMERA_XYZ[0]) < 0.005)
        check("frames", not missing_frames and geom_ok,
              f"missing={missing_frames or 'none'}; laser parent={have.get('laser')} xyz={laser}; "
              f"camera_link xyz={cam}; optical frame parent={have.get('camera_optical_frame')}")

        # 3. lidar against the layout, at a heading where the room is not mirror-symmetric
        n.face(0.6)
        n.run_for(0.0, 0.0, 1.0)
        x, y, yaw = n.pose()
        scan = n.scan
        e_std, valid = scan_errors(scan, x, y, yaw, +1)
        e_mir, _ = scan_errors(scan, x, y, yaw, -1)
        med_std = statistics.median(e_std) if e_std else float("inf")
        med_mir = statistics.mean(e_mir) if e_mir else float("inf")
        mean_std = statistics.mean(e_std) if e_std else float("inf")
        p90 = sorted(e_std)[int(0.9 * len(e_std))] if e_std else float("inf")
        check("lidar", valid > 0.97 and med_std < 0.03 and p90 < 0.15 and mean_std < 0.3 * med_mir
              and scan.header.frame_id == "laser",
              f"{len(scan.ranges)} beams, range {scan.range_min:.2f}-{scan.range_max:.1f} m, frame={scan.header.frame_id}, "
              f"valid={valid * 100:.0f}%, median error={med_std * 100:.1f} cm, 90th pct={p90 * 100:.1f} cm "
              f"mean={mean_std * 100:.1f} cm (a mirrored angle convention would give {med_mir * 100:.0f} cm)")
        extra["scan"] = {"angle_min": scan.angle_min, "angle_increment": scan.angle_increment,
                         "ranges": [r if math.isfinite(r) else -1.0 for r in scan.ranges], "pose": [x, y, yaw]}

        # 4. imu at rest, then while spinning
        a = n.imu.linear_acceleration
        g = n.imu.angular_velocity
        rest_ok = abs(a.z - 9.81) < 0.4 and abs(a.x) < 0.4 and abs(a.y) < 0.4 and abs(g.z) < 0.05
        rest_detail = f"at rest accel=({a.x:.2f}, {a.y:.2f}, {a.z:.2f}) m/s^2, gyro z={g.z:.3f} rad/s"
        n.run_for(0.0, 0.8, 2.0)
        gz = [n.imu.angular_velocity.z]
        wz = [n.odom.twist.twist.angular.z]
        t0 = n.sim_time
        while n.sim_time - t0 < 1.0:
            n.cmd(0.0, 0.8)
            n.spin()
            gz.append(n.imu.angular_velocity.z)
            wz.append(n.odom.twist.twist.angular.z)
        n.stop()
        gz_m, wz_m = statistics.mean(gz), statistics.mean(wz)
        q = n.imu.orientation
        imu_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        yaw_err = abs(wrap(imu_yaw - n.pose()[2]))
        check("imu", rest_ok and abs(gz_m - wz_m) < 0.05 and abs(gz_m - 0.8) < 0.08 and yaw_err < 0.05
              and n.imu.header.frame_id == "imu_link",
              f"{rest_detail}; spinning at 0.8 rad/s gyro z={gz_m:.3f} (odometry {wz_m:.3f}); "
              f"orientation yaw differs from odometry by {math.degrees(yaw_err):.2f} deg; frame={n.imu.header.frame_id}")

        # 5. camera, facing straight into the room
        n.face(0.0)
        n.run_for(0.0, 0.0, 1.0)
        rgb, depth, info = n.rgb, n.depth, n.info
        rgb_arr = np.frombuffer(bytes(rgb.data), dtype=np.uint8).reshape(rgb.height, rgb.width, -1)
        depth_arr = np.frombuffer(bytes(depth.data), dtype=np.float32).reshape(depth.height, depth.width)
        cy_, cx_ = depth.height // 2, depth.width // 2
        centre = float(np.nanmedian(depth_arr[cy_ - 10:cy_ + 10, cx_ - 10:cx_ + 10]))
        x, y, yaw = n.pose()
        cam_x, cam_y = x + hl.CAMERA_XYZ[0] * math.cos(yaw), y + hl.CAMERA_XYZ[0] * math.sin(yaw)
        exp_depth = hl.expected_range(cam_x, cam_y, yaw, hl.CAMERA_XYZ[2])
        fx_exp = hl.CAMERA_WIDTH / (2.0 * math.tan(math.radians(hl.CAMERA_HFOV_DEG) / 2.0))
        size_ok = (rgb.width, rgb.height, depth.width, depth.height) == (hl.CAMERA_WIDTH, hl.CAMERA_HEIGHT) * 2
        check("camera", size_ok and rgb_arr.std() > 8.0 and abs(centre - exp_depth) < 0.15
              and abs(info.k[0] - fx_exp) < 5.0 and rgb.header.frame_id == "camera_optical_frame",
              f"rgb {rgb.width}x{rgb.height} {rgb.encoding} (brightness mean {rgb_arr.mean():.0f}, spread {rgb_arr.std():.0f}); "
              f"depth {depth.width}x{depth.height} {depth.encoding}, centre={centre:.2f} m (layout says {exp_depth:.2f}); "
              f"fx={info.k[0]:.1f} (expected {fx_exp:.1f}); frame={rgb.header.frame_id}")
        out_dir = os.environ.get("PHASE2_OUT_DIR")
        if out_dir:
            np.save(os.path.join(out_dir, "camera_rgb.npy"), rgb_arr[:, :, :3])
            np.save(os.path.join(out_dir, "camera_depth.npy"), depth_arr)

        # 6. cliff sensors over normal floor
        cl = {k: v.range for k, v in n.cliff.items()}
        check("cliff idle", all(math.isfinite(v) and v < hl.CLIFF_THRESHOLD for v in cl.values()),
              "floor distance " + ", ".join(f"{k[6:]}={v * 1000:.0f} mm" for k, v in cl.items()))

        # 7. bumper: head-on into the west wall, release, then an angled hit
        idle_mask = n.bumper
        n.face(math.pi)
        n.bumper_seen = 0
        hit = n.drive_until(0.15, lambda: n.bumper != 0, 6.0)
        head_on = n.bumper_seen
        n.run_for(-0.15, 0.0, 2.0)
        n.stop()
        released = n.bumper
        n.face(math.radians(-145.0))  # heading 35 deg left of straight-at-the-wall puts the wall on the robot's right-front
        n.bumper_seen = 0
        hit2 = n.drive_until(0.15, lambda: n.bumper != 0, 6.0)
        angled = n.bumper_seen
        n.run_for(-0.15, 0.0, 2.0)
        n.stop()
        check("bumper", idle_mask == 0 and hit and head_on & ZONE_BITS["front_center"] and released == 0
              and hit2 and angled & (ZONE_BITS["front_right"] | ZONE_BITS["right"]) and not angled & ZONE_BITS["left"],
              f"idle={zones(idle_mask)}; head-on hit pressed {zones(head_on)}; after backing off={zones(released)}; "
              f"angled hit pressed {zones(angled)}")

        # 8. cliff: creep to the stairwell edge
        door_y = (hl.DOORWAY[1] + hl.DOORWAY[3]) / 2.0
        door_x = (hl.DOORWAY[0] + hl.DOORWAY[2]) / 2.0
        stair_x = (hl.STAIRWELL[0] + hl.STAIRWELL[2]) / 2.0
        route = [hl.to_world(door_x - 0.9, door_y), hl.to_world(door_x + 0.9, door_y), hl.to_world(stair_x, 3.9)]
        reached = all(n.go_to(wx, wy) for wx, wy in route)
        n.face(math.pi / 2.0)
        z_before = n.odom.pose.pose.position.z
        n.max_tilt = 0.0
        tripped = n.drive_until(0.08, n.front_cliff_tripped, 15.0)
        hx, hy = hl.to_house(*n.pose()[:2])
        edge_y = hl.STAIRWELL[1]
        z_drop = z_before - n.odom.pose.pose.position.z
        at_edge = {k[6:]: (f"CLIFF ({v.range * 1000:.0f} mm)" if n.is_cliff(v.range) else f"{v.range * 1000:.0f} mm")
                   for k, v in n.cliff.items()}
        n.run_for(-0.12, 0.0, 3.0)
        n.stop()
        cleared = not n.front_cliff_tripped()
        expected_y = edge_y - hl.CLIFF_SENSORS["cliff_front_left"][0]
        check("cliff edge", reached and tripped and abs(hy - expected_y) < 0.06 and z_drop < 0.01 and n.max_tilt < 2.0 and cleared,
              f"reached approach point={reached}; front sensors tripped={tripped} with robot centre at house y={hy:.3f} "
              f"(edge at {edge_y}, expected about {expected_y:.2f}); readings at the edge {at_edge}; "
              f"height change={z_drop * 1000:.1f} mm, max tilt={n.max_tilt:.2f} deg; cleared after backing off={cleared}")
    finally:
        try:
            for _ in range(20):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE2_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "extra": extra}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
