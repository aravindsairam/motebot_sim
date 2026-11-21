#!/usr/bin/env python3
"""Phase 10 acceptance test: the robot drives onto its dock by itself.

Needs two things running first, with the robot at its start pose and a saved map present:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase10_docking_test.py

Docking is requested as a mode. Where the robot ends up is compared with the true docked
position from the house layout: centred on the dock, facing it, its front 10 cm from the tower.

Checks:
  1. docks from nearby      from the start position, 12 cm from the docked spot and facing away
  2. parked                 once docked, nobody can drive it and the tag is still in view
  3. leaves the dock        a navigation goal from the docked position succeeds (no special undocking)
  4. docks from afar        from the bedroom, through the doorway
  5. docks from an angle    from close by but off to either side and facing the wrong way
  6. can be called off      asking for idle on the way stops the robot and switches docking off
  7. fails safely           with the tag detector frozen it gives up, says why, and does not drive at the dock blind
  8. works again            once the detector has been restarted, the next docking succeeds

Limits for a good docking: within 2 cm along the approach line, 1.5 cm sideways and 3 degrees,
and the bumper never pressed.
"""

import json
import math
import os
import signal
import sys
import time

import rclpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase1_drive_test import wrap  # noqa: E402
from phase2_sensor_test import zones  # noqa: E402
from phase7_mode_manager_test import find_pids  # noqa: E402
from phase9_dock_detection_test import DOCKED_X, DockTestDriver  # noqa: E402

ALONG_LIMIT, SIDE_LIMIT, ANGLE_LIMIT = 0.02, 0.015, 3.0


def main():
    rclpy.init()
    n = DockTestDriver()
    modes = ModeClient(n)
    results, runs = [], {}

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def where():
        x, y, yaw = n.pose()
        hx, hy = hl.to_house(x, y)
        return hx, hy, yaw

    def dock(name, wall_timeout=240.0):
        """Request docking and wait for it to end, one way or the other."""
        n.bumper_seen, n.monitoring = 0, True
        start_xy = where()[:2]
        t0, wall0 = n.sim_time, time.time()
        ok, why = modes.request("docking")
        steps = []
        while ok and time.time() - wall0 < wall_timeout:
            n.spin(0.05)
            s = modes.state
            if s.state == "docking" and s.reason not in steps:
                steps.append(s.reason)
            if s.state in ("docked", "idle", "failsafe") and not s.transitioning:
                break
        n.wait_sim(1.5)
        n.monitoring = False
        hx, hy, yaw = where()
        r = {
            "accepted": ok, "why": why, "end": modes.state.state, "reason": modes.state.reason, "steps": steps,
            "seconds": n.sim_time - t0, "from": start_xy,
            "along": hx - DOCKED_X, "side": hy - hl.DOCK_Y, "angle": math.degrees(wrap(yaw - math.pi)),
            "gap": hx - hl.ROBOT_RADIUS - hl.DOCK_TOWER[2], "bumper": n.bumper_seen,
        }
        runs[name] = r
        return r

    def docked_well(r):
        return (r["end"] == "docked" and abs(r["along"]) <= ALONG_LIMIT and abs(r["side"]) <= SIDE_LIMIT
                and abs(r["angle"]) <= ANGLE_LIMIT and r["bumper"] == 0)

    def text(r):
        if r["end"] != "docked":
            return f"ended in {r['end']}: {r['reason'] or r['why']}"
        return (f"from house ({r['from'][0]:.1f}, {r['from'][1]:.1f}), docked in {r['seconds']:.0f} s; off by "
                f"{r['along'] * 100:+.1f} cm along the approach, {r['side'] * 100:+.1f} cm sideways, {r['angle']:+.1f} deg; "
                f"front {r['gap'] * 100:.1f} cm from the dock; bumper pressed: {zones(r['bumper'])}")

    try:
        if not n.wait_ready():
            check("docks from nearby", False, "no /odom, /ground_truth/odom or /clock within 60 s; is the simulator running?")
            return results
        st = modes.wait_ready()
        if st is None or not st.current_map:
            check("docks from nearby", False, "the mode manager is not ready or there is no saved map")
            return results
        while n.safety is None:
            n.spin(0.1)

        # 1. from the start position
        r = dock("nearby")
        check("docks from nearby", docked_well(r), text(r) + "; steps: " + ", ".join(s.replace("docking: ", "") for s in r["steps"]))
        if r["end"] != "docked":
            return results

        # 2. parked: nobody drives, the dock is still in view, the docking machinery is switched off
        x0, y0, _ = n.pose()
        n.hold(0.2, 0.3, "manual")
        n.wait_sim(1.5)
        n.hold(0.2, 0.3, "nav")
        n.wait_sim(1.5)
        n.release()
        x1, y1, _ = n.pose()
        moved = math.hypot(x1 - x0, y1 - y0)
        seen = n.measure()
        server, controller = modes.lifecycle("docking_server"), modes.lifecycle("controller_server")
        check("parked",
              moved < 0.005 and n.safety.source == "none" and n.safety.blocked_reason == "docked" and seen["rate"] >= 0.9
              and server == "inactive" and controller == "unconfigured"
              and {"idle", "navigation", "manual", "docking"} <= set(modes.state.allowed_transitions),
              f"commands held on both inputs moved the robot {moved * 1000:.1f} mm (gate: '{n.safety.blocked_reason}'); the tag "
              f"is seen in {seen['rate'] * 100:.0f}% of frames from the docked position; docking_server={server}, "
              f"controller_server={controller}; can go to {list(modes.state.allowed_transitions)}")

        # 3. leave the dock with an ordinary navigation goal, to the bedroom
        ok, why = modes.request("navigation")
        gx, gy = hl.to_world(6.0, 1.6)
        status, secs, _dist = n.navigate(gx, gy, 0.0) if ok else ("not sent", 0.0, 0.0)
        hx, hy, _ = where()
        check("leaves the dock", ok and status == "succeeded" and math.hypot(hx - 6.0, hy - 1.6) < 0.15,
              f"navigation from the docked position to the bedroom: {status} in {secs:.0f} s, robot at house ({hx:.2f}, {hy:.2f})"
              + ("" if ok else f"; could not enter navigation: {why}"))

        # 4. dock from the bedroom
        r = dock("afar")
        check("docks from afar", docked_well(r), text(r))

        # 5. dock from close by, off to one side and then the other, facing the wrong way
        angled = []
        for name, hx, hy, yaw in (("angle, north side", 0.95, 3.15, 0.5), ("angle, south side", 0.95, 1.35, -0.5)):
            modes.request("manual")
            n.go_to(*hl.to_world(hx, hy))
            n.face(yaw)
            angled.append(dock(name))
        check("docks from an angle", all(docked_well(r) for r in angled), " | ".join(text(r) for r in angled))

        # 6. call it off on the way
        modes.request("manual")
        n.go_to(*hl.to_world(2.6, 3.0))
        n.face(0.0)
        ok, why = modes.request("docking")
        n.wait_sim(4.0)
        speed_before = n.speed()
        asked = n.sim_time
        ok_idle, _ = modes.request("idle")
        stop_time = n.stop_time_since(asked)
        x0, y0, _ = n.pose()
        n.wait_sim(3.0)
        x1, y1, _ = n.pose()
        server = modes.lifecycle("docking_server")
        check("can be called off",
              ok and speed_before > 0.05 and ok_idle and stop_time is not None and stop_time < 0.5
              and math.hypot(x1 - x0, y1 - y0) < 0.005 and modes.state.state == "idle" and server == "inactive",
              f"docking on its way at {speed_before:.2f} m/s; idle requested -> at rest "
              f"{'%.2f s' % stop_time if stop_time is not None else 'never'} later, then moved "
              f"{math.hypot(x1 - x0, y1 - y0) * 1000:.1f} mm in 3 s; mode {modes.state.state}, docking_server={server}")

        # 7. the tag detector stops working: docking must give up, not drive at the dock blind
        detector = find_pids("lib/apriltag_ros/apriltag_node")
        for pid in detector:
            os.kill(pid, signal.SIGSTOP)
        try:
            r = dock("blind")
        finally:
            for pid in detector:
                os.kill(pid, signal.SIGCONT)
        check("fails safely",
              len(detector) == 1 and r["accepted"] and r["end"] == "idle" and "could not see the dock" in r["reason"]
              and r["gap"] > 0.4 and r["bumper"] == 0 and n.speed() < 0.02,
              f"tag detector frozen: docking ended in {r['end']} after {r['seconds']:.0f} s, reason: '{r['reason']}'; "
              f"robot stopped {r['gap'] * 100:.0f} cm from the dock; bumper pressed: {zones(r['bumper'])}")

        # 8. and with the detector back it docks. Un-freezing it is not enough: a node that was
        # silent for more than about 20 s has been dropped from the ROS network and stays deaf
        # (measured: frozen 18 s it resumes at once, frozen 27 s never). It has to be restarted,
        # which the launch file does when the process dies.
        for pid in detector:
            os.kill(pid, signal.SIGKILL)
        modes.request("manual")
        n.face_tag()
        wall0 = time.time()
        recovered = False
        while time.time() - wall0 < 40.0 and not recovered:
            recovered = n.measure(1.0)["rate"] > 0.8
        recovery = time.time() - wall0
        r = dock("again")
        check("works again", recovered and docked_well(r),
              f"frozen detector killed; restarted by the launch file and reporting the dock again after {recovery:.0f} s; "
              + text(r))
        modes.request("idle")
    finally:
        try:
            n.release()
            for _ in range(10):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE10_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "runs": runs}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
