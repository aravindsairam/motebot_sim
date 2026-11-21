#!/usr/bin/env python3
"""Phase 3 acceptance test: measure the noise each sensor actually produces and compare it with
the settings in sim/scripts/sensor_noise.py, then check that wheel odometry drifts by a sane amount.

Tests the simulator alone: run it WITHOUT the robot stack, while run_sim.py is playing WITH noise
(the default), robot at its start pose:
    source /opt/ros/humble/setup.bash
    python3 sim/tests/phase3_noise_test.py

Checks:
  1. lidar noise   range error spread matches the model at every distance; dropouts at the set rate
  2. imu noise     gyro and accelerometer jitter match; offsets are within the set bias; heading drifts
  3. cliff noise   millimetre jitter around the true floor distance
  4. odometry      /odom starts equal to the true pose and /tf agrees with /odom
  5. drift         after a 4.8 m square with four turns, /odom is close to the truth but not identical
"""

import json
import math
import os
import statistics
import sys

import numpy as np
import rclpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
import sensor_noise as sn  # noqa: E402
from phase1_drive_test import wrap, yaw_of  # noqa: E402
from phase2_sensor_test import SensorDriver  # noqa: E402


class NoiseDriver(SensorDriver):
    def __init__(self):
        super().__init__()
        self.scans, self.imus, self.cliffs = [], [], []
        self.record = False

    def _on_scan(self, m):
        super()._on_scan(m)
        if self.record:
            self.scans.append((m, self.pose()))

    def _on_imu(self, m):
        super()._on_imu(m)
        if self.record:
            self.imus.append((m, self.pose()[2]))

    def _on_cliff(self, name, m):
        super()._on_cliff(name, m)
        if self.record:
            self.cliffs.append(m.range)

    def odom_pose(self):
        p, q = self.robot_odom.pose.pose.position, self.robot_odom.pose.pose.orientation
        return p.x, p.y, yaw_of(q)


def main():
    rclpy.init()
    NoiseDriver.CMD_TOPIC = hl.WHEEL_CMD_TOPIC  # simulator only, no safety gate: talk to the wheels
    n = NoiseDriver()
    results, table = [], {}

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if not n.wait_ready():
            check("lidar noise", False, "no /odom, /ground_truth/odom or /clock within 60 s; is run_sim.py playing?")
            return results
        missing = n.wait_sensors()
        if missing:
            check("lidar noise", False, f"no messages on: {', '.join(missing)}")
            return results

        # odometry at the start, before anything moves
        tx, ty, tyaw = n.pose()
        ox, oy, oyaw = n.odom_pose()
        tf0 = n.tf_xyz.get(("odom", "base_link"))
        start_ok = math.hypot(ox - tx, oy - ty) < 0.01 and abs(wrap(oyaw - tyaw)) < 0.01
        tf_ok = tf0 is not None and math.hypot(tf0[0] - ox, tf0[1] - oy) < 0.01

        # record 10 s standing still
        n.record = True
        n.run_for(0.0, 0.0, 10.0)
        n.record = False

        # 1. lidar
        cfg = sn.LIDAR
        bins = [(0.0, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 12.0)]
        errs = {b: [] for b in bins}
        sigmas = {b: [] for b in bins}
        total = dropped = 0
        for scan, (x, y, yaw) in n.scans:
            count = len(scan.ranges)
            exp = [hl.expected_range(x, y, yaw + scan.angle_min + i * scan.angle_increment, hl.LIDAR_XYZ[2])
                   for i in range(count)]
            for i, r in enumerate(scan.ranges):
                # skip beams at depth edges, where a tiny angle error swaps one surface for another
                if abs(exp[(i + 1) % count] - exp[i - 1]) > 0.05:
                    continue
                total += 1
                if not math.isfinite(r):
                    dropped += 1
                    continue
                for b in bins:
                    if b[0] <= exp[i] < b[1]:
                        errs[b].append(r - exp[i])
                        sigmas[b].append(cfg.range_std(exp[i]))
        ok = bool(n.scans)
        parts = []
        for b in bins:
            if len(errs[b]) < 500:
                continue
            meas, model, bias = float(np.std(errs[b])), float(np.mean(sigmas[b])), float(np.mean(errs[b]))
            ok = ok and 0.8 < meas / model < 1.25 and abs(bias) < 0.005
            parts.append(f"{b[0]:g}-{b[1]:g} m: spread {meas * 100:.2f} cm (set {model * 100:.2f}), offset {bias * 1000:+.1f} mm")
            table[f"lidar {b[0]:g}-{b[1]:g} m spread (cm)"] = (meas * 100, model * 100)
        drop = dropped / max(1, total)
        ok = ok and 0.5 * cfg.dropout_prob < drop < 2.0 * cfg.dropout_prob
        table["lidar dropouts (%)"] = (drop * 100, cfg.dropout_prob * 100)
        check("lidar noise", ok, f"{len(n.scans)} scans; " + "; ".join(parts)
              + f"; dropouts {drop * 100:.2f}% (set {cfg.dropout_prob * 100:.2f}%)")

        # 2. imu
        icfg = sn.IMU
        gyro = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for m, _ in n.imus])
        acc = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for m, _ in n.imus])
        g_std, a_std = gyro.std(axis=0), acc.std(axis=0)
        g_bias, a_bias = gyro.mean(axis=0), acc.mean(axis=0) - np.array([0.0, 0.0, 9.81])
        yaw_err = [wrap(yaw_of(m.orientation) - true_yaw) for m, true_yaw in n.imus]
        t_span = (n.imus[-1][0].header.stamp.sec + n.imus[-1][0].header.stamp.nanosec * 1e-9) - (
            n.imus[0][0].header.stamp.sec + n.imus[0][0].header.stamp.nanosec * 1e-9)
        drift_rate = (statistics.mean(yaw_err[-30:]) - statistics.mean(yaw_err[:30])) / t_span
        cov = n.imus[-1][0].angular_velocity_covariance[0]
        ok = (all(0.8 < s / icfg.gyro_noise_std < 1.25 for s in g_std)
              and all(0.8 < s / icfg.accel_noise_std < 1.25 for s in a_std)
              and all(abs(b) < 4 * icfg.gyro_bias_std for b in g_bias)
              and all(abs(b) < 4 * icfg.accel_bias_std + 0.02 for b in a_bias)
              and abs(drift_rate - g_bias[2]) < 0.0005
              and abs(cov - icfg.gyro_noise_std ** 2) < 1e-9)
        table["gyro jitter (rad/s)"] = (float(g_std.mean()), icfg.gyro_noise_std)
        table["accelerometer jitter (m/s^2)"] = (float(a_std.mean()), icfg.accel_noise_std)
        check("imu noise", ok,
              f"{len(n.imus)} samples; gyro jitter {g_std.round(4).tolist()} (set {icfg.gyro_noise_std}), "
              f"gyro offset {g_bias.round(4).tolist()} (set up to about {2 * icfg.gyro_bias_std}); "
              f"accel jitter {a_std.round(3).tolist()} (set {icfg.accel_noise_std}), offset {a_bias.round(3).tolist()}; "
              f"heading drift {math.degrees(drift_rate) * 60:+.2f} deg/min, matching the gyro z offset "
              f"({math.degrees(g_bias[2]) * 60:+.2f} deg/min)")

        # 3. cliff
        c = np.array([v for v in n.cliffs if math.isfinite(v)])
        c_std, c_mean = float(c.std()), float(c.mean())
        table["cliff jitter (mm)"] = (c_std * 1000, sn.CLIFF.range_std * 1000)
        check("cliff noise", len(c) > 300 and 0.75 < c_std / sn.CLIFF.range_std < 1.3 and abs(c_mean - 0.015) < 0.002
              and len(c) == len(n.cliffs),
              f"{len(c)} readings; jitter {c_std * 1000:.2f} mm (set {sn.CLIFF.range_std * 1000:.2f}), mean {c_mean * 1000:.1f} mm")

        # 4. odometry plumbing
        check("odometry", start_ok and tf_ok and n.robot_odom.header.frame_id == "odom" and n.odom.header.frame_id == "world",
              f"at start /odom=({ox:.3f}, {oy:.3f}, {math.degrees(oyaw):.1f} deg) vs truth=({tx:.3f}, {ty:.3f}, "
              f"{math.degrees(tyaw):.1f} deg); /tf odom->base_link={tf0}; frames {n.robot_odom.header.frame_id} / {n.odom.header.frame_id}")

        # 5. drift over a square: 1.2 m sides, four left turns
        side = 1.2
        speed_err = []
        route = [(side, 0.0), (side, side), (0.0, side), (0.0, 0.0)]
        reached = True
        for wx, wy in route:
            reached = n.go_to(wx, wy) and reached
            speed_err.append(abs(n.robot_odom.twist.twist.linear.x - n.odom.twist.twist.linear.x))
        n.face(0.0)
        n.run_for(0.0, 0.0, 1.0)
        tx, ty, tyaw = n.pose()
        ox, oy, oyaw = n.odom_pose()
        pos_err = math.hypot(ox - tx, oy - ty)
        yaw_err_deg = math.degrees(wrap(oyaw - tyaw))
        # sample the reported speed against the truth on one more straight run
        n.run_for(0.2, 0.0, 2.0)
        v_odom, v_true = n.robot_odom.twist.twist.linear.x, n.odom.twist.twist.linear.x
        n.stop()
        table["odometry position error after 4.8 m (cm)"] = (pos_err * 100, None)
        table["odometry heading error after 360 deg (deg)"] = (yaw_err_deg, None)
        check("drift", reached and 0.001 < pos_err < 0.25 and abs(yaw_err_deg) < 12.0 and abs(v_odom - v_true) < 0.02,
              f"route completed={reached}; after 4.8 m and 360 deg of turning /odom is off by {pos_err * 100:.1f} cm and "
              f"{yaw_err_deg:+.1f} deg; speed reads {v_odom:.3f} m/s against a true {v_true:.3f}")
    finally:
        try:
            for _ in range(20):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE3_RESULT_JSON")
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
