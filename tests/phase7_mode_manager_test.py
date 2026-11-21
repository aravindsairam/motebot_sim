#!/usr/bin/env python3
"""Phase 7 acceptance test: the mode manager, end to end, without restarting anything.

Needs two things running first, with the robot at its start pose and a saved map present:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase7_mode_manager_test.py

It walks the robot through every mode and three kinds of failure:
  1. starts in idle        localized, Nav2 off, wheels held at zero by the safety gate
  2. bad requests          an unknown mode is refused and nothing changes
  3. mapping and saving    map both rooms, save through the manager, score the saved file
  4. handover              switch to navigation in the bedroom: AMCL starts where mapping left off
  5. emergency stop        failsafe requested mid-drive stops the robot; only idle is allowed next
  6. manual and idle       manual keeps localization; entering idle stops a robot that is being driven
  7. node crash            killing Nav2's controller puts the robot in failsafe; it recovers
  8. simulator freeze      lidar and odometry going silent puts the robot in failsafe; it recovers
  9. one map source        in every mode exactly one of slam_toolbox and AMCL is active
"""

import json
import math
import os
import signal
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from vacuum_interfaces.msg import SafetyStatus

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase1_drive_test import wrap  # noqa: E402
from phase2_sensor_test import zones  # noqa: E402
from phase4_mapping_test import MAP_DIR, score_map  # noqa: E402
from phase6_navigation_test import NavDriver  # noqa: E402

MAP_NAME = "house"
# Both rooms, ending in the bedroom so the switch to navigation happens far from the dock.
# The second pass goes north along the west wall: the direct line from (0.9, 1.2) to the room
# centre clips a leg of the coffee table, and a bump there makes the wheels slip and throws the
# odometry heading off by 45 degrees, which can ruin the map.
TOUR_HOUSE = [
    (2.2, 2.8), (4.1, 3.0), (5.95, 3.0), (6.0, 1.4), (5.6, 3.5), (5.95, 3.0), (4.1, 3.0), (4.3, 1.7),
    (3.3, 2.6), (1.0, 3.3), (0.9, 1.2), (1.0, 3.0), (4.1, 3.0), (5.95, 3.0), (6.0, 1.6),
]


class ModeTestDriver(NavDriver):
    def __init__(self):
        super().__init__()
        self.wheel_commands = []  # (wall time, linear x, angular z) of every wheel command the safety gate sends
        self.safety = None  # latest /safety_status
        self.safety_log = []  # (wall time, sim time, status) whenever the gate's status changes
        self.tf_log = None  # list of (x, y, yaw) of map -> odom while recording
        self._held = None  # (publisher, linear x, angular z) sent 20 times a second while set
        self.last_moving = 0.0  # simulation time at which the robot was last seen moving
        self.nav_cmd_pub = self.create_publisher(Twist, "cmd_vel_nav", 10)  # Nav2's input to the safety gate
        self.create_subscription(Twist, hl.WHEEL_CMD_TOPIC, self._on_wheel_cmd, 100)
        self.create_subscription(SafetyStatus, "safety_status", self._on_safety, 50)
        self.create_timer(0.05, self._hold_tick)

    def _on_wheel_cmd(self, m):
        self.wheel_commands.append((time.time(), m.linear.x, m.angular.z))

    def _on_odom(self, msg):
        super()._on_odom(msg)
        if self.sim_time is not None and self.speed() >= 0.02:
            self.last_moving = self.sim_time

    def stop_time_since(self, sim_start):
        """Simulation seconds from `sim_start` until the robot came to rest, or None if it is still
        moving. Works across a blocking call, unlike waiting for the stop after the call returns."""
        self.wait_sim(0.5)
        if self.speed() >= 0.02:
            return None
        return max(0.0, self.last_moving - sim_start)

    def _on_safety(self, m):
        last = self.safety
        self.safety = m
        key = (m.source, m.blocked_reason, m.cliff, m.bumper, m.off_floor, m.reflex_active, tuple(m.stale))
        if last is None or key != self._safety_key:
            self.safety_log.append((time.time(), self.sim_time, m))
        self._safety_key = key

    def _hold_tick(self):
        if self._held is not None:
            pub, v, w = self._held
            m = Twist()
            m.linear.x, m.angular.z = float(v), float(w)
            pub.publish(m)

    def hold(self, v, w, source="manual"):
        """Keep sending a command in the background, as a held joystick (or Nav2) would, until
        release(). source: "manual" sends on /cmd_vel, "nav" on Nav2's topic /cmd_vel_nav."""
        self._held = (self.pub if source == "manual" else self.nav_cmd_pub, v, w)

    def release(self):
        """Stop sending, without sending a stop command."""
        self._held = None

    def wheel_rates(self, seconds=2.0):
        """(zero, non-zero) wheel commands per second from the safety gate over the next `seconds`."""
        start = time.time()
        while time.time() - start < seconds:
            self.spin(0.05)
        mine = [(v, w) for t, v, w in self.wheel_commands if t >= start]
        zero = sum(1 for v, w in mine if v == 0.0 and w == 0.0)
        return zero / seconds, (len(mine) - zero) / seconds

    def _on_slam_tf(self, msg):
        """While recording, log how far each map -> odom update moves the robot's position on the map.

        One healthy source nudges the robot by a few centimetres per update. (The raw transform
        can step much further: a one degree heading correction seen from 5 m away shifts it 9 cm.)
        """
        before = self.map_to_odom
        super()._on_slam_tf(msg)
        after = self.map_to_odom
        if self.tf_log is None or before is None or after is None or before == after or self.robot_odom is None:
            return
        ox, oy, _ = self.odom_pose()

        def place(t):
            c, s = math.cos(t[2]), math.sin(t[2])
            return t[0] + c * ox - s * oy, t[1] + s * ox + c * oy

        (x0, y0), (x1, y1) = place(before), place(after)
        self.tf_log.append(math.hypot(x1 - x0, y1 - y0))

    def speed(self):
        t = self.odom.twist.twist
        return abs(t.linear.x) + abs(t.angular.z) * 0.1

    def wait_sim(self, seconds):
        """Let simulation time pass without publishing any command."""
        t0 = self.wait_clock()
        wall0 = time.time()
        while self.sim_time - t0 < seconds and time.time() - wall0 < seconds * 5 + 10:
            self.spin()

    def localization_error(self):
        """(metres, radians) between where the map -> odom publisher puts the robot and the truth."""
        tx, ty, tyaw = self.pose()
        ax, ay, ayaw = self.slam_pose()
        return math.hypot(ax - tx, ay - ty), abs(wrap(ayaw - tyaw))

    def send_goal(self, x, y, yaw):
        """Send a Nav2 goal and return without waiting for the result."""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        send = self.nav.send_goal_async(goal)
        while not send.done():
            self.spin()
        return send.result().accepted


def find_pids(pattern):
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    return [int(p) for p in out if int(p) != os.getpid()]


def simulator_pids():
    out = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True, text=True).stdout.splitlines()
    pids = []
    for line in out[1:]:
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[1].startswith(("python", "kit")) and "sim/scripts/run_sim.py" in parts[2]:
            pids.append(int(parts[0]))
    return pids


def main():
    rclpy.init()
    n = ModeTestDriver()
    modes = ModeClient(n)
    results, snapshots = [], []
    frozen = []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def snapshot(mode):
        """Record which map -> odom source is active, asked of the nodes themselves."""
        snap = (mode, modes.lifecycle("slam_toolbox"), modes.lifecycle("amcl"))
        snapshots.append(snap)
        return snap

    def goal_to(hx, hy, yaw_deg):
        gx, gy = hl.to_world(hx, hy)
        status, secs, dist = n.navigate(gx, gy, math.radians(yaw_deg))
        tx, ty, _ = n.pose()
        return status, secs, math.hypot(tx - gx, ty - gy)

    try:
        if not n.wait_ready():
            check("starts in idle", False, "no /odom, /ground_truth/odom or /clock within 60 s; is the simulator running?")
            return results
        st = modes.wait_ready()
        if st is None:
            check("starts in idle", False, "no /robot_state from the mode manager; is the robot stack running?")
            return results

        # 1. starts in idle
        snap = snapshot("idle")
        nav_state = modes.lifecycle("controller_server")
        zero, nonzero = n.wheel_rates(2.0)
        check("starts in idle",
              st.state == "idle" and bool(st.current_map) and {"mapping", "navigation", "manual", "failsafe"} <= set(st.allowed_transitions)
              and snap[1:] == ("unconfigured", "active") and nav_state == "unconfigured" and zero >= 30.0 and nonzero == 0.0
              and n.safety is not None and n.safety.source == "none",
              f"mode={st.state}, map={st.current_map or 'none'}, can go to {list(st.allowed_transitions)}; "
              f"slam_toolbox={snap[1]}, amcl={snap[2]}, controller_server={nav_state}; the safety gate holds the wheels: "
              f"{zero:.0f} zero commands/s, {nonzero:.0f} others")
        if not st.current_map:
            print("      This test needs a saved map. Run phase4_mapping_test.py first.", flush=True)
            return results

        # 2. bad requests
        ok, why = modes.request("flying")
        ok2, why2, _path = modes.save_map("should_not_exist")
        check("bad requests", not ok and "unknown state" in why and modes.state.state == "idle" and not ok2
              and not os.path.exists(os.path.join(MAP_DIR, "should_not_exist.yaml")),
              f"'flying' -> refused: {why}; saving a map while idle -> refused: {why2}")

        # 3. mapping and saving
        t0 = time.time()
        ok, why = modes.request("mapping")
        took_mapping = time.time() - t0
        if not ok:
            check("mapping and saving", False, f"could not enter mapping: {why}")
            return results
        snapshot("mapping")
        n.spin(0.3)
        source_mapping = n.safety.source if n.safety is not None else "no status"
        n.tf_log = []
        n.bumper_seen, n.monitoring = 0, True
        reached = all(n.go_to(*hl.to_world(hx, hy)) for hx, hy in TOUR_HOUSE)
        n.stop(0.5)
        tour_bumps, n.monitoring = n.bumper_seen, False
        n.wait_sim(4.0)  # one more map update from slam_toolbox
        mapping_tf = n.tf_log
        n.tf_log = None
        saved, save_msg, path = modes.save_map(MAP_NAME)
        scores = []
        if saved:
            score_map(os.path.join(MAP_DIR, MAP_NAME), lambda name, ok_, detail: scores.append((name, ok_, detail)))
        check("mapping and saving",
              reached and saved and scores and all(s[1] for s in scores) and source_mapping == "manual" and tour_bumps == 0,
              f"entered mapping in {took_mapping:.1f} s; tour completed={reached}, bumper pressed: {zones(tour_bumps)}; {save_msg}; "
              + "; ".join(f"{s[0]} {'ok' if s[1] else 'FAILED'}" for s in scores)
              + f"; the safety gate lets '{source_mapping}' drive while mapping"
              + ("" if all(s[1] for s in scores) else " | " + " | ".join(s[2] for s in scores if not s[1])))

        # 4. handover: switch to navigation where the robot stands, in the bedroom
        hx, hy = hl.to_house(*n.pose()[:2])
        t0 = time.time()
        ok, why = modes.request("navigation")
        took_nav = time.time() - t0
        if not ok:
            check("handover", False, f"could not enter navigation: {why}")
            return results
        n.wait_sim(1.0)
        snapshot("navigation")
        err, yaw_err = n.localization_error()
        n.tf_log = []
        n.bumper_seen, n.cliff_trips, n.zone_intrusions, n.monitoring = 0, 0, 0, True
        status, secs, goal_err = goal_to(4.0, 3.7, 90.0)
        check("handover", err < 0.15 and yaw_err < math.radians(5) and status == "succeeded" and goal_err < 0.15,
              f"entered navigation in {took_nav:.1f} s at house ({hx:.1f}, {hy:.1f}), not at the dock; AMCL started "
              f"{err * 100:.1f} cm and {math.degrees(yaw_err):.1f} deg from the truth; first goal {status} in {secs:.0f} s, "
              f"{goal_err * 100:.0f} cm off")

        # 5. emergency stop while driving to a goal
        accepted = n.send_goal(*hl.to_world(0.45, 2.25), 0.0)
        n.wait_sim(4.0)
        moving = n.speed()
        asked = n.sim_time
        ok, why = modes.request("failsafe")
        stop_time = n.stop_time_since(asked)
        ctrl = modes.lifecycle("controller_server")
        refused, refused_why = modes.request("mapping")
        left, left_why = modes.request("idle")
        nav_tf = n.tf_log
        n.tf_log = None
        check("emergency stop",
              accepted and moving > 0.1 and ok and stop_time is not None and ctrl != "active" and not refused and left
              and modes.state.state == "idle",
              f"robot was moving at {moving:.2f} m/s; failsafe requested -> {why}; at rest "
              f"{'%.2f s' % stop_time if stop_time is not None else 'never'} after the request; controller_server={ctrl}; "
              f"mapping from failsafe -> refused: {refused_why}; idle -> {left_why}")

        # 6. manual keeps localization, and idle stops a robot even while someone keeps driving it
        ok, why = modes.request("manual")
        snapshot("manual")
        x, y, yaw = n.pose()
        reached = n.go_to(x + 0.8 * math.cos(yaw), y + 0.8 * math.sin(yaw))
        err, yaw_err = n.localization_error()
        source_manual = n.safety.source
        n.hold(0.15, 0.3)  # a joystick held over, and kept there through the mode change
        n.wait_sim(1.0)
        driving = n.speed()
        asked = n.sim_time
        ok_idle, _ = modes.request("idle")
        stop_time = n.stop_time_since(asked)
        zero, nonzero = n.wheel_rates(1.5)
        n.release()
        check("manual and idle",
              ok and reached and err < 0.12 and source_manual == "manual" and driving > 0.1 and ok_idle
              and stop_time is not None and zero >= 30.0 and nonzero == 0.0,
              f"manual: drove 0.8 m, AMCL {err * 100:.1f} cm from the truth; then driving at {driving:.2f} m/s with the "
              f"command held: idle requested -> at rest {'%.2f s' % stop_time if stop_time is not None else 'never'} "
              f"after the request; "
              f"with the command still held the gate sends {zero:.0f} zero commands/s, {nonzero:.0f} others")

        # 7. a Nav2 node crashes while in navigation
        ok, why = modes.request("navigation")
        victims = find_pids("lib/nav2_controller/controller_server")
        for pid in victims:
            os.kill(pid, signal.SIGKILL)
        t0 = time.time()
        in_failsafe = modes.wait_for_state("failsafe", 30.0)
        detect = time.time() - t0
        reason = modes.state.reason
        refused, refused_why = modes.request("navigation")
        t0 = time.time()
        while modes.state.faults and time.time() - t0 < 30.0:
            n.spin(0.1)
        cleared = not modes.state.faults
        left, left_why = modes.request("idle")
        back, back_why = modes.request("navigation")
        status, secs, goal_err = goal_to(0.45, 2.25, 0.0) if back else ("not sent", 0.0, 9.9)
        check("node crash",
              ok and len(victims) == 1 and in_failsafe and "controller_server" in reason and not refused and cleared
              and left and back and status == "succeeded" and goal_err < 0.15,
              f"killed controller_server; failsafe after {detect:.1f} s, reason: {reason}; navigation from failsafe -> "
              f"refused; node restarted and fault cleared={cleared}; idle -> {left_why}; navigation -> {back_why}; "
              f"goal home {status}, {goal_err * 100:.0f} cm off")
        n.monitoring = False
        modes.request("idle")

        # 8. the simulator freezes: lidar and odometry go silent
        frozen[:] = simulator_pids()
        for pid in frozen:
            os.kill(pid, signal.SIGSTOP)
        t0 = time.time()
        in_failsafe = modes.wait_for_state("failsafe", 20.0)
        detect = time.time() - t0
        reason = modes.state.reason
        refused, refused_why = modes.request("idle")
        for _ in range(10):  # the manager answers "busy" until it has finished entering failsafe
            if not refused_why.startswith("busy"):
                break
            time.sleep(0.5)
            refused, refused_why = modes.request("idle")
        for pid in frozen:
            os.kill(pid, signal.SIGCONT)
        frozen[:] = []
        t0 = time.time()
        while modes.state.faults and time.time() - t0 < 30.0:
            n.spin(0.1)
        cleared = not modes.state.faults
        left, left_why = modes.request("idle")
        snapshot("idle")
        check("simulator freeze",
              in_failsafe and ("lidar" in reason or "odometry" in reason) and not refused and "still faulty" in refused_why
              and cleared and left and modes.state.state == "idle",
              f"simulator paused; failsafe after {detect:.1f} s, reason: {reason}; idle while still frozen -> "
              f"refused: {refused_why}; "
              f"simulator resumed, fault cleared={cleared}; idle -> {left_why}")

        # 9. exactly one source of the map -> odom transform in every mode
        one_source = all((s == "active") != (a == "active") for _m, s, a in snapshots)

        jump_map, jump_nav = max(mapping_tf, default=0.0), max(nav_tf, default=0.0)
        check("one map source", one_source and jump_map < 0.10 and jump_nav < 0.10,
              "slam_toolbox / amcl by mode: " + ", ".join(f"{m}: {s} / {a}" for m, s, a in snapshots)
              + f"; largest correction to the robot's position in one update: {jump_map * 100:.1f} cm while mapping "
              f"({len(mapping_tf)} updates), {jump_nav * 100:.1f} cm while navigating ({len(nav_tf)} updates)")
    finally:
        for pid in frozen:  # never leave the simulator frozen
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        out = os.environ.get("PHASE7_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "snapshots": snapshots, "history": modes.history}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
