#!/usr/bin/env python3
"""Phase 8 acceptance test: the safety gate, end to end.

Needs two things running first, with the robot at its start pose and a saved map present:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase8_safety_test.py

Every check asks the same question: whatever is sent, does only a safe command reach the wheels?
  1. idle is locked      commands on both inputs move nothing; the gate holds the wheels at zero
  2. feedback            a goal sent in idle is answered with a notice in the RViz label, not silence
  3. one driver          manual mode ignores Nav2's topic; navigation mode ignores the manual topic
  4. command timeout     a driver that goes quiet mid-drive (a crashed teleop) stops the robot
  5. cliff, manual       held at full speed towards the stairwell: backs off every time, never falls
  6. cliff, navigation   the same on Nav2's topic in navigation mode
  7. gate crash          killing the safety gate stops the robot and causes failsafe; recovers on request
  8. manager crash       killing the mode manager stops the robot; it comes back in idle, still localized
  9. bumper              full speed into a wall: backs off and ends with the bumper released
"""

import json
import math
import os
import signal
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import UInt8
from visualization_msgs.msg import Marker, MarkerArray

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase2_sensor_test import zones  # noqa: E402
from phase7_mode_manager_test import ModeTestDriver, find_pids  # noqa: E402

FULL_SPEED = 0.5  # asked for; the robot's limit is 0.306 m/s
STAIR_X = (hl.STAIRWELL[0] + hl.STAIRWELL[2]) / 2.0
EDGE_Y = hl.STAIRWELL[1]  # the floor ends here (house frame)
APPROACH = (STAIR_X, 3.9)  # in front of the stairwell, 0.7 m from the edge
DOOR_Y = (hl.DOORWAY[1] + hl.DOORWAY[3]) / 2.0
ROUTE_TO_STAIRS = [(hl.DOORWAY[0] - 0.85, DOOR_Y), (hl.DOORWAY[2] + 0.85, DOOR_Y), APPROACH]
OPEN_FLOOR = (hl.DOORWAY[2] + 0.85, DOOR_Y)  # bedroom, with 2.5 m of free floor to the south
WALL_X = hl.BEDROOM[0]  # east face of the wall between the rooms, west of the approach point


class SafetyTestDriver(ModeTestDriver):
    def __init__(self):
        super().__init__()
        self.label = ""  # text of the mode label shown in RViz
        self.bumper_now = 0
        self.bumper_log = []  # (sim time, mask) on every change
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(MarkerArray, "mode_label", self._on_label, latched)
        self.create_subscription(UInt8, "bumper", self._on_bumper_state, 50)
        self.goal_pub = self.create_publisher(PoseStamped, "goal_pose", 5)

    def _on_label(self, msg):
        lines = [m.text.replace("\u00a0", " ").strip() for m in sorted(msg.markers, key=lambda m: m.id)
                 if m.action == Marker.ADD]
        self.label = " | ".join(line for line in lines if line)

    def _on_bumper_state(self, m):
        if m.data != self.bumper_now:
            self.bumper_log.append((self.sim_time, m.data))
        self.bumper_now = m.data

    def house(self):
        x, y, yaw = self.pose()
        hx, hy = hl.to_house(x, y)
        return hx, hy, yaw

    def moved_during(self, sim_seconds):
        """Distance the robot's centre moves while `sim_seconds` pass (no commands sent by this call)."""
        x0, y0, _ = self.pose()
        self.wait_sim(sim_seconds)
        x1, y1, _ = self.pose()
        return math.hypot(x1 - x0, y1 - y0)

    def wall_until_stopped(self, wall_timeout):
        """Wall-clock seconds until the robot is at rest, or None."""
        t0 = time.time()
        while time.time() - t0 < wall_timeout:
            self.spin()
            if self.speed() < 0.02:
                return time.time() - t0
        return None

    def reflexes_since(self, wall_start):
        """Reasons of every reflex the gate started after `wall_start`."""
        out, active = [], False
        for wall, _sim, status in self.safety_log:
            if status.reflex_active and not active and wall >= wall_start:
                out.append(status.blocked_reason)
            active = status.reflex_active
        return out

    def charge_at_edge(self, speed, source, sim_seconds):
        """Hold a forward command towards the stairwell (robot facing north) and record what happens."""
        wall0 = time.time()
        z0 = self.odom.pose.pose.position.z
        self.max_tilt = 0.0
        peak, max_y, min_z = 0.0, -1e9, z0
        self.hold(speed, 0.0, source)
        t0 = self.sim_time
        while self.sim_time - t0 < sim_seconds:
            self.spin()
            peak = max(peak, self.odom.twist.twist.linear.x)
            max_y = max(max_y, self.house()[1])
            min_z = min(min_z, self.odom.pose.pose.position.z)
        self.release()
        self.wait_sim(1.5)
        return {
            "peak": peak,
            "reflexes": self.reflexes_since(wall0),
            "caster_margin": EDGE_Y - (max_y + hl.CASTER_X),  # floor left ahead of the caster at the furthest point
            "sensor_overshoot": max_y + hl.CLIFF_SENSORS["cliff_front_left"][0] - EDGE_Y,
            "drop": z0 - min_z,
            "tilt": self.max_tilt,
            "at_rest": self.speed() < 0.02,
            "end_gap": EDGE_Y - (self.house()[1] + hl.ROBOT_RADIUS),  # front edge of the robot to the drop, at the end
            "cliff_at_end": self.safety.cliff,
            "fell": self.safety.off_floor,
        }


def edge_ok(r, min_reflexes):
    return (not r["fell"] and r["peak"] > 0.28 and len(r["reflexes"]) >= min_reflexes
            and all("cliff" in why for why in r["reflexes"])
            and r["caster_margin"] > 0.0 and r["drop"] < 0.005 and r["tilt"] < 2.0 and r["at_rest"] and not r["cliff_at_end"])


def edge_text(r):
    return (("THE ROBOT WENT OVER THE EDGE. " if r["fell"] else "")
            + f"reached {r['peak']:.2f} m/s; reflex fired {len(r['reflexes'])} time(s); at the furthest point the cliff "
            f"sensors were {r['sensor_overshoot'] * 100:.1f} cm past the edge and the caster still had "
            f"{r['caster_margin'] * 100:.1f} cm of floor; height change {r['drop'] * 1000:.1f} mm, max tilt {r['tilt']:.2f} deg; "
            f"ends at rest={r['at_rest']} with its front edge {r['end_gap'] * 100:.0f} cm from the drop")


def main():
    rclpy.init()
    n = SafetyTestDriver()
    modes = ModeClient(n)
    results = []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def drive_to(points):
        return all(n.go_to(*hl.to_world(hx, hy)) for hx, hy in points)

    try:
        if not n.wait_ready():
            check("idle is locked", False, "no /odom, /ground_truth/odom or /clock within 60 s; is the simulator running?")
            return results
        st = modes.wait_ready()
        wall0 = time.time()
        while n.safety is None and time.time() - wall0 < 10.0:
            n.spin(0.1)
        if st is None or n.safety is None:
            check("idle is locked", False, "no /robot_state or /safety_status; is the robot stack running?")
            return results
        if st.state != "idle":
            modes.request("idle")
        if not modes.state.current_map:
            check("idle is locked", False, "this test needs a saved map; run phase4_mapping_test.py first")
            return results

        # 1. idle is locked: push on both inputs, nothing may move
        n.hold(0.2, 0.3, "manual")
        moved = n.moved_during(1.5)
        n.hold(0.2, 0.3, "nav")
        moved += n.moved_during(1.5)
        zero, nonzero = n.wheel_rates(1.5)
        n.release()
        s = n.safety
        check("idle is locked",
              moved < 0.005 and zero >= 40.0 and nonzero == 0.0 and s.source == "none" and s.blocked_reason == "idle",
              f"commands held on /cmd_vel and then /cmd_vel_nav: robot moved {moved * 1000:.1f} mm; the gate sent "
              f"{zero:.0f} zero commands/s to the wheels and {nonzero:.0f} others; it reports source='{s.source}', "
              f"reason='{s.blocked_reason}'")

        # 2. feedback: a goal in idle is answered, and the label names the mode
        label_before = n.label
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x, goal.pose.orientation.w = 1.0, 1.0
        wall0 = time.time()
        while time.time() - wall0 < 5.0 and "goal ignored" not in n.label:
            n.goal_pub.publish(goal)
            n.spin(0.2)
        label_notice = n.label
        moved = n.moved_during(1.0)
        check("feedback",
              label_before == "IDLE" and label_notice == "IDLE | goal ignored in idle mode" and moved < 0.005,
              f"RViz label in idle: '{label_before}'; after a goal on /goal_pose: '{label_notice}'; robot moved {moved * 1000:.1f} mm")

        # 3. one driver per mode
        ok_manual, _ = modes.request("manual")
        n.hold(0.2, 0.0, "nav")
        wrong_manual = n.moved_during(1.5)
        n.hold(0.2, 0.0, "manual")
        right_manual = n.moved_during(2.0)
        source_manual, label_manual = n.safety.source, n.label
        n.release()
        n.stop(0.5)
        ok_nav, why_nav = modes.request("navigation")
        n.hold(0.2, 0.0, "manual")
        wrong_nav = n.moved_during(1.5)
        n.hold(-0.2, 0.0, "nav")  # backwards, to end up where this check started
        right_nav = n.moved_during(2.0)
        source_nav, label_nav = n.safety.source, n.label
        n.release()
        n.wait_sim(1.0)
        check("one driver",
              ok_manual and ok_nav and wrong_manual < 0.005 and right_manual > 0.3 and source_manual == "manual"
              and wrong_nav < 0.005 and right_nav > 0.3 and source_nav == "nav2"
              and label_manual.startswith("MANUAL") and label_nav.startswith("NAVIGATION"),
              f"manual mode: Nav2's topic moved the robot {wrong_manual * 1000:.1f} mm, the manual topic {right_manual:.2f} m "
              f"(gate source '{source_manual}'); navigation mode: the manual topic moved it {wrong_nav * 1000:.1f} mm, "
              f"Nav2's topic {right_nav:.2f} m (gate source '{source_nav}')")

        # 4. command timeout: a driver that just stops talking
        modes.request("manual")
        n.run_for(0.25, 0.0, 1.5)
        speed_before = n.speed()
        stop_wall = n.wall_until_stopped(3.0)  # nothing is published from here on, not even a stop
        n.spin(0.3)
        reason = n.safety.blocked_reason
        check("command timeout",
              speed_before > 0.2 and stop_wall is not None and 0.4 < stop_wall < 1.1 and reason == "command timeout",
              f"driving at {speed_before:.2f} m/s when the commands stopped arriving; at rest "
              f"{'%.2f s' % stop_wall if stop_wall is not None else 'never'} later (timeout 0.6 s plus braking); "
              f"gate reason: '{reason}'")

        # 5. cliff in manual mode: full speed at the stairwell, command held for 10 s
        reached = drive_to(ROUTE_TO_STAIRS)
        n.face(math.pi / 2.0)
        r = n.charge_at_edge(FULL_SPEED, "manual", 10.0)
        check("cliff, manual", reached and edge_ok(r, 2), f"forward command held for 10 s: {edge_text(r)}")
        if r["fell"]:
            print("      The robot is off the floor; the remaining checks need a simulator restart.", flush=True)
            return results

        # 6. the same on Nav2's topic, in navigation mode
        drive_to([APPROACH])
        n.face(math.pi / 2.0)
        ok_nav, why_nav = modes.request("navigation")
        r = n.charge_at_edge(0.26, "nav", 10.0) if ok_nav else None
        check("cliff, navigation", ok_nav and r is not None and r["peak"] > 0.2 and edge_ok(dict(r, peak=1.0), 2),
              f"navigation mode, forward command held on Nav2's topic for 10 s: {edge_text(r)}" if r else
              f"could not enter navigation: {why_nav}")
        if r is not None and r["fell"]:
            print("      The robot is off the floor; the remaining checks need a simulator restart.", flush=True)
            return results

        # 7. the safety gate crashes while the robot is being driven
        modes.request("manual")
        reached = drive_to([APPROACH, OPEN_FLOOR])
        n.face(-math.pi / 2.0)
        n.hold(0.15, 0.0, "manual")
        n.wait_sim(1.5)
        speed_before = n.speed()
        victims = find_pids("lib/vacuum_safety/safety_gate")
        for pid in victims:
            os.kill(pid, signal.SIGKILL)
        wall0 = time.time()
        stop_wall = n.wall_until_stopped(4.0)
        in_failsafe = modes.wait_for_state("failsafe", 15.0)
        detect = time.time() - wall0
        reason = modes.state.reason
        # The launch file restarts the gate. Wait for the new one (it reports that nobody may
        # drive, because of the failsafe) and for the manager to see the fault clear.
        while (modes.state.faults or n.safety.source != "none") and time.time() - wall0 < 30.0:
            n.spin(0.1)
        back = time.time() - wall0
        cleared = not modes.state.faults and len(find_pids("lib/vacuum_safety/safety_gate")) == 1
        still = n.moved_during(2.0)  # the command is still held, the gate is back, the robot is in failsafe
        left, left_why = modes.request("idle")
        ok_manual, _ = modes.request("manual")
        n.wait_sim(1.5)
        speed_after = n.speed()
        n.release()
        n.stop(0.5)
        check("gate crash",
              reached and len(victims) == 1 and speed_before > 0.1 and stop_wall is not None and stop_wall < 1.5
              and in_failsafe and "safety gate" in reason and cleared and still < 0.005 and left and ok_manual
              and speed_after > 0.1,
              f"driving at {speed_before:.2f} m/s; gate killed: robot at rest "
              f"{'%.2f s' % stop_wall if stop_wall is not None else 'never'} later (the base stops itself when commands end); "
              f"failsafe after {detect:.1f} s, reason: {reason}; gate restarted and fault cleared after {back:.1f} s; "
              f"with the command still held the robot moved {still * 1000:.1f} mm in failsafe; idle -> {left_why}; "
              f"manual again: driving at {speed_after:.2f} m/s")

        # 8. the mode manager crashes while the robot is being driven
        n.go_to(*hl.to_world(*OPEN_FLOOR))
        n.face(-math.pi / 2.0)
        err_before, _ = n.localization_error()
        n.hold(0.15, 0.0, "manual")
        n.wait_sim(1.5)
        speed_before = n.speed()
        victims = find_pids("lib/vacuum_mode_manager/mode_manager")
        for pid in victims:
            os.kill(pid, signal.SIGKILL)
        wall0 = time.time()
        stop_wall = n.wall_until_stopped(8.0)
        gate_reason = n.safety.blocked_reason
        modes.state = None
        st = modes.wait_ready(90.0)
        back = time.time() - wall0
        still = n.moved_during(2.0)
        n.release()
        n.wait_sim(2.0)
        err_after, yaw_after = n.localization_error()
        check("manager crash",
              len(victims) == 1 and speed_before > 0.1 and stop_wall is not None and stop_wall < 4.0 and st is not None
              and st.state == "idle" and not st.faults and still < 0.005 and err_after < 0.15,
              f"driving at {speed_before:.2f} m/s; manager killed: robot at rest "
              f"{'%.2f s' % stop_wall if stop_wall is not None else 'never'} later (gate reason: '{gate_reason}'); "
              f"manager restarted and ready in {st.state if st else 'nothing'} after {back:.1f} s; command still held: "
              f"robot moved {still * 1000:.1f} mm; AMCL {err_before * 100:.1f} cm from the truth before, "
              f"{err_after * 100:.1f} cm after the restart")

        # 9. bumper: full speed into the wall between the rooms
        modes.request("manual")
        reached = drive_to([OPEN_FLOOR, APPROACH])
        n.face(math.pi)
        n.bumper_log.clear()
        wall0 = time.time()
        peak = 0.0
        n.max_tilt = 0.0
        n.hold(FULL_SPEED, 0.0, "manual")
        t0, seen = n.sim_time, False
        while n.sim_time - t0 < 8.0:
            n.spin()
            peak = max(peak, abs(n.odom.twist.twist.linear.x))
            seen = seen or n.safety.reflex_active
            if seen and not n.safety.reflex_active:
                break  # the reflex has finished
        n.release()
        n.wait_sim(1.5)
        reflexes = n.reflexes_since(wall0)
        pressed = 0
        press_time = 0.0
        last_press = None
        for t, mask in n.bumper_log:
            pressed |= mask
            if mask and last_press is None:
                last_press = t
            elif not mask and last_press is not None:
                press_time += t - last_press
                last_press = None
        gap = n.house()[0] - hl.ROBOT_RADIUS - WALL_X
        err_end, _ = n.localization_error()
        check("bumper",
              reached and peak > 0.28 and len(reflexes) == 1 and "bumper" in reflexes[0] and pressed != 0
              and n.bumper_now == 0 and not n.safety.bumper and gap > 0.02 and n.speed() < 0.02 and press_time < 0.5,
              f"hit the wall at {peak:.2f} m/s; bumper zones pressed: {zones(pressed)}, for {press_time:.2f} s in total; "
              f"reflex: {reflexes}; ends at rest {gap * 100:.1f} cm from the wall with the bumper "
              f"{'released' if n.bumper_now == 0 else 'STILL PRESSED'}; max tilt {n.max_tilt:.2f} deg; "
              f"AMCL {err_end * 100:.1f} cm from the truth afterwards")
        modes.request("idle")
    finally:
        try:
            n.release()
            for _ in range(10):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE8_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
