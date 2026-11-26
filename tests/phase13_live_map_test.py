#!/usr/bin/env python3
"""Phase 13 acceptance test: the live map for the app (step 3 of the Android app), against the robot.

Needs two things running first, with the robot at its start pose and a saved map:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false     (it starts the gateway)
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase13_live_map_test.py

Scripted phones (phase 11's) receive what the app receives. The map frame and the simulator's
world frame coincide (the robot starts at the origin of both), so positions compare directly.
  1. map at once        a phone that connects gets the map within 0.5 s, equal to /map cell for cell
  2. position           while the robot drives, every position sent is within 10 cm of the truth
                        at the same moment
  3. rates              about 10 positions a second, and no lidar points (type 0x22, retired on
                        2026-11-15: the app does not show them)
  4. map grows          while mapping, every map slam_toolbox keeps for 1.5 s reaches the phone, none
                        less than a second apart, and the last one equals /map; after leaving mapping
                        the phone has the saved map again
  5. late phone         a second phone gets the map and the position within 0.5 s
  6. frozen robot       while the simulator is frozen, no position is sent: an old one is never
                        presented as current
"""

import json
import math
import os
import signal
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from mode_client import ModeClient  # noqa: E402
from phase7_mode_manager_test import ModeTestDriver, simulator_pids  # noqa: E402
from phase11_gateway_test import Phone  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402
from vacuum_gateway.geometry import rle_decode  # noqa: E402

Result = pb.Reply.Result


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class MapTestDriver(ModeTestDriver):
    """Also records /map and the true pose by simulation stamp."""

    def __init__(self):
        super().__init__()
        self.grid = None
        self.grid_at = 0.0
        self.grids = []  # (time, msg) of every /map whose cells differ from the one before
        self.truth = []  # (stamp, x, y, yaw) of every ground-truth message
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "map", self._on_grid, latched)

    def _on_grid(self, msg):
        self.grid, self.grid_at = msg, time.monotonic()
        if not self.grids or same_map_grids(self.grids[-1][1], msg) is False:
            self.grids.append((self.grid_at, msg))

    def _on_odom(self, msg):
        super()._on_odom(msg)
        s = msg.header.stamp
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.truth.append((s.sec + s.nanosec * 1e-9, p.x, p.y, yaw))

    def truth_at(self, stamp):
        """The true pose recorded nearest to a simulation stamp, and how far off in time it is."""
        stamps = np.fromiter((t[0] for t in self.truth), dtype=float)
        i = int(np.argmin(np.abs(stamps - stamp)))
        return self.truth[i], abs(stamps[i] - stamp)


def same_map_grids(a, b):
    """Do two OccupancyGrids hold the same map (size, origin, cells)?"""
    ia, ib = a.info, b.info
    return ((ia.width, ia.height, ia.resolution, ia.origin.position.x, ia.origin.position.y)
            == (ib.width, ib.height, ib.resolution, ib.origin.position.x, ib.origin.position.y)
            and np.array_equal(np.asarray(a.data, dtype=np.int8), np.asarray(b.data, dtype=np.int8)))


def same_map(phone_map, grid):
    """Does a Map from the phone equal an OccupancyGrid? Returns (equal, why not)."""
    if phone_map is None or grid is None:
        return False, "missing"
    info = grid.info
    if (phone_map.width, phone_map.height) != (info.width, info.height):
        return False, f"size {phone_map.width}x{phone_map.height} vs {info.width}x{info.height}"
    if abs(phone_map.resolution - info.resolution) > 1e-6:
        return False, "resolution"
    if abs(phone_map.origin_x - info.origin.position.x) > 1e-9 or abs(phone_map.origin_y - info.origin.position.y) > 1e-9:
        return False, "origin"
    cells = rle_decode(phone_map.cells_rle)
    expected = np.asarray(grid.data, dtype=np.int8)
    if cells.size != expected.size:
        return False, f"{cells.size} cells decoded, {expected.size} expected"
    differ = int(np.count_nonzero(cells != expected))
    return differ == 0, f"{differ} cells differ"


def main():
    rclpy.init()
    n = MapTestDriver()
    modes = ModeClient(n)
    results = []
    phones = []
    request_ids = iter(range(1, 1000))

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def change(phone, mode, timeout=120.0):
        rid = next(request_ids)
        sent = phone.send(pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=rid, mode=mode))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            n.spin(0.02)
            got = phone.reply_to(rid, sent)
            if got:
                return got[1].result == Result.RESULT_OK, got[1].message
        return False, "no reply"

    def wait(seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            n.spin(0.02)

    def drive(segments):
        """(linear, angular, simulation seconds) one after the other, then stop and settle."""
        for v, w, secs in segments:
            n.hold(v, w, "manual")
            n.wait_sim(secs)
        n.release()
        n.wait_sim(1.0)

    frozen = []
    try:
        if modes.wait_ready(60.0) is None or not modes.wait_for_state("idle", 30.0):
            check("robot ready", False, "the mode manager did not reach idle")
            return results
        end = time.monotonic() + 30.0
        while time.monotonic() < end and (n.sim_time is None or n.odom is None or n.grid is None):
            n.spin(0.05)
        if n.sim_time is None or n.odom is None or n.grid is None:
            check("robot ready", False, "no /clock, /ground_truth/odom or /map")
            return results
        wait(2.0)

        # 1. the map, at once
        phone = Phone("phone").start()
        phones.append(phone)
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not phone.opens():
            n.spin(0.05)
        wait(1.0)
        t_open = phone.opens()[0] if phone.opens() else None
        first_map = phone.first_after(t_open, pb.DATA_MAP) if t_open else None
        equal, why = same_map(first_map[1] if first_map else None, n.grid)
        check("map at once", first_map is not None and first_map[0] - t_open <= 0.5 and equal,
              f"map after {round(first_map[0] - t_open, 3) if first_map else None} s, "
              f"{first_map[1].width if first_map else '?'}x{first_map[1].height if first_map else '?'} cells, "
              f"{len(first_map[1].cells_rle) if first_map else '?'} bytes encoded "
              f"({first_map[1].width * first_map[1].height if first_map else '?'} cells raw); equal to /map: {equal} ({why})")

        # 2-4. drive and compare
        ok, msg = change(phone, "manual")
        if not ok:
            check("position", False, f"manual refused: {msg}")
            return results
        t0 = time.monotonic()
        drive([(0.15, 0.0, 5.0), (0.0, 0.8, 3.9), (0.15, 0.0, 5.0)])
        t1 = time.monotonic()
        poses = phone.messages(pb.DATA_ROBOT_POSE, t0, t1)
        errors, heading_errors, gaps = [], [], []
        for _, p in poses:
            (ts, tx, ty, tyaw), gap = n.truth_at(p.stamp)
            gaps.append(gap)
            errors.append(math.hypot(p.x - tx, p.y - ty))
            heading_errors.append(abs(wrap(p.heading - tyaw)))
        moved = math.hypot(poses[-1][1].x - poses[0][1].x, poses[-1][1].y - poses[0][1].y) if poses else 0.0
        check("position", len(poses) > 50 and max(errors) <= 0.10 and max(gaps) < 0.05,
              f"{len(poses)} positions while driving; distance to the truth at the same moment: median "
              f"{np.median(errors) * 100:.1f} cm, worst {max(errors) * 100:.1f} cm (limit 10); heading worst "
              f"{math.degrees(max(heading_errors)):.1f} deg; the robot ended {moved * 100:.0f} cm from where it started" if poses else "no positions")

        # Positions go out on the computer's clock. The lidar points (type 0x22) are no longer sent.
        secs = t1 - t0
        pose_rate = len(poses) / secs
        lidar = phone.messages(0x22, t0, t1)
        check("rates", 18.0 <= pose_rate <= 22.0 and not lidar,
              f"{pose_rate:.1f} positions per second over {secs:.1f} s; lidar point sets received: {len(lidar)}")

        # 4. the map while mapping
        ok, msg = change(phone, "mapping")
        t_map = time.monotonic()
        drive([(0.15, 0.0, 4.0), (0.0, 0.8, 2.0), (0.15, 0.0, 3.0), (0.0, -0.8, 2.0), (0.15, 0.0, 3.0)])
        wait(4.0)  # slam_toolbox publishes every 2 s, the gateway at most once a second
        # slam_toolbox changes the map only after the robot has moved far enough, so the number
        # of maps depends on the drive. What must hold: every map it kept for 1.5 s or more
        # reached the phone (a change superseded sooner may be skipped: at most one a second goes
        # out), no two maps reached the phone less than a second apart, and the last one is /map.
        t_end = time.monotonic()
        published = [(t, g) for t, g in n.grids if t >= t_map]
        kept = [g for (t, g), nxt in zip(published, published[1:] + [(t_end, None)]) if nxt[0] - t >= 1.5]
        maps = phone.messages(pb.DATA_MAP, t_map)
        missed = sum(1 for g in kept if not any(same_map(m, g)[0] for _, m in maps))
        versions = [m.version for _, m in maps]
        spacing = [round(b[0] - a[0], 2) for a, b in zip(maps, maps[1:])]
        known = [int(np.count_nonzero(rle_decode(m.cells_rle) >= 0)) for _, m in maps]
        last_equal, why_last = same_map(maps[-1][1] if maps else None, n.grid)
        ok_mapping = (ok and len(published) >= 2 and missed == 0 and versions == sorted(set(versions))
                      and min(spacing or [9]) >= 0.95 and known[-1] > known[0] and last_equal)
        ok2, msg2 = change(phone, "idle")
        wait(3.0)
        back = phone.messages(pb.DATA_MAP, t_map)
        back_equal, why_back = same_map(back[-1][1] if back else None, n.grid)
        check("map grows", ok_mapping and ok2 and back_equal,
              f"slam_toolbox published {len(published)} different maps, {len(kept)} of them kept 1.5 s or more; "
              f"the phone got {len(maps)} (versions {versions}), missing {missed} of the kept ones, at least "
              f"{min(spacing or [0])} s apart; known cells {known[0] if known else 0} -> {known[-1] if known else 0}; "
              f"last equal to /map: {last_equal} ({why_last}); after leaving mapping the phone's map equals /map "
              f"(the saved map): {back_equal} ({why_back})")

        # 5. a phone that connects late
        late = Phone("late phone").start()
        phones.append(late)
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not late.opens():
            n.spin(0.05)
        wait(1.0)
        lo = late.opens()[0] if late.opens() else None
        lm = late.first_after(lo, pb.DATA_MAP) if lo else None
        lp = late.first_after(lo, pb.DATA_ROBOT_POSE) if lo else None
        delays = [round(m[0] - lo, 3) if m else None for m in (lm, lp)]
        check("late phone", lm is not None and lp is not None and max(delays) <= 0.5,
              f"map after {delays[0]} s, position after {delays[1]} s")

        # 6. the simulator frozen: no position may be sent
        pids = simulator_pids()
        t_stop = time.monotonic()
        for pid in pids:
            os.kill(pid, signal.SIGSTOP)
        frozen = pids
        time.sleep(3.0)
        t_cont = time.monotonic()
        for pid in pids:
            os.kill(pid, signal.SIGCONT)
        frozen = []
        wait(3.0)
        during = phone.messages(pb.DATA_ROBOT_POSE, t_stop + 0.3, t_cont)
        after = phone.messages(pb.DATA_ROBOT_POSE, t_cont, t_cont + 3.0)
        check("frozen robot", bool(pids) and len(during) == 0 and len(after) >= 10,
              f"simulator frozen 3 s: {len(during)} positions sent while frozen, {len(after)} in the 3 s after")
        # the freeze puts the robot in failsafe (odometry stopped); bring it back for whoever is next
        modes.wait_for_state("failsafe", 5.0)
        change(phone, "idle", 30.0)

        out = os.environ.get("PHASE13_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results}, f)
    finally:
        for pid in frozen:
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        for p in phones:
            p.stop()
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
