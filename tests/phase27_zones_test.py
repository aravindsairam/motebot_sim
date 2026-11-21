#!/usr/bin/env python3
"""Phase 27 acceptance test: changing the no-go zones from the app, while the robot navigates.

Needs the simulator and the robot stack running, the robot at its start pose, the house map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase27_zones_test.py

A scripted phone (phase 11's) does what the app does. The robot is sent to the living room in
navigation mode first, so its place on the map is known. Then:
  1. zones sent      the phone has the house map's zones, named for the map ('house'), stairwell included
  2. wrong map       SetNoGoZones for another map is refused, and nothing changes
  3. over the robot  a new zone around the robot is refused with the reason; so is one at the dock
  4. added           a new zone in the bedroom is saved: the answer comes within 5 s, every phone
                     gets the new list within 1 s of it, and the zones file has it
  5. in force        before, Nav2's planner finds a route to the zone's middle; within 3 s of the
                     answer it finds none, without leaving navigation mode (the mask was reloaded live)
  6. cleaning        a preview of an area around the zone keeps every row and hop at least 20 cm
                     out of it (rows keep 25 cm from walls and zones)
  7. deleted         sending the old list back: the route is found again within 3 s, the phone has
                     the old list, the file has the stairwell only
  8. terminal        `ros2 run vacuum_mode_manager zones` lists the zones of the map in use
The zones file is put back exactly as it was at the end (its comments included).
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase7_mode_manager_test import ModeTestDriver  # noqa: E402
from phase11_gateway_test import Phone  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402

Result = pb.Reply.Result
State = pb.NavigationStatus.State
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ZONES_FILE = os.path.join(ROOT, "ros2_ws", "maps", "house_zones.yaml")

LIVING_ROOM = (3.55, 1.25)  # house coordinates: between sofa and armchair
BEDROOM = (6.3, 1.0)  # open floor in the bedroom: the new zone's middle
ZONE_SIZE = 0.8  # m, the new zone's side
ROW_GAP = 0.20  # m: rows and hops keep at least this far out of a zone


def square(x, y, size):
    h = size / 2.0
    return [x - h, y - h, x + h, y - h, x + h, y + h, x - h, y + h]


def zone(name, corners):
    return pb.NoGoZones.Zone(name=name, points_cm=[round(v * 100) for v in corners])


def outside_by(px, py, box):
    """How far (x, y) is outside the rectangle (x0, y0, x1, y1); 0 inside."""
    dx = max(box[0] - px, 0.0, px - box[2])
    dy = max(box[1] - py, 0.0, py - box[3])
    return math.hypot(dx, dy)


def main():
    rclpy.init()
    n = ModeTestDriver()
    modes = ModeClient(n)
    planner = ActionClient(n, ComputePathToPose, "compute_path_to_pose")
    results, phones = [], []
    ids = iter(range(1, 10000))
    backup = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml").name
    shutil.copyfile(ZONES_FILE, backup)

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def ask(phone, msg_type, message, timeout=30.0):
        sent = phone.send(msg_type, message)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            n.spin(0.02)
            got = phone.reply_to(message.request_id, sent)
            if got:
                return got
        return None

    def text(reply):
        return f"{pb.Reply.Result.Name(reply.result)} '{reply.message}'" if reply else "no reply"

    def set_zones(phone, zones, map_name="house"):
        """(time answered, Reply) or (None, None)."""
        got = ask(phone, pb.COMMAND_SET_NO_GO_ZONES, pb.SetNoGoZones(request_id=next(ids), map_name=map_name, zones=zones), 20.0)
        return got if got else (None, None)

    def route_exists(x, y):
        """Nav2's planner: is there a route from the robot to (x, y)? None if it did not answer."""
        goal = ComputePathToPose.Goal(use_start=False)
        goal.goal = PoseStamped()
        goal.goal.header.frame_id = "map"
        goal.goal.pose.position.x, goal.goal.pose.position.y = x, y
        goal.goal.pose.orientation.w = 1.0
        future = planner.send_goal_async(goal)
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not future.done():
            n.spin(0.02)
        if not future.done() or not future.result().accepted:
            return None
        result = future.result().get_result_async()
        while time.monotonic() < end and not result.done():
            n.spin(0.02)
        if not result.done():
            return None
        out = result.result()
        return out.status == GoalStatus.STATUS_SUCCEEDED and len(out.result.path.poses) > 0

    def until(predicate, limit):
        """Seconds until predicate() held (checked every 0.2 s), or None."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit:
            if predicate():
                return round(time.monotonic() - t0, 2)
            n.spin(0.2)
        return None

    def latest_zones(phone, since=0.0):
        got = phone.messages(pb.DATA_NO_GO_ZONES, since)
        return got[-1] if got else None

    def names(msg):
        return [z.name for z in msg.zones] if msg else None

    try:
        if modes.wait_ready(60.0) is None or not modes.wait_for_state("idle", 30.0):
            check("robot ready", False, "the mode manager did not reach idle")
            return results
        end = time.monotonic() + 30.0
        while time.monotonic() < end and (n.sim_time is None or n.odom is None):
            n.spin(0.05)
        phone = Phone("phone").start()
        other = Phone("other phone").start()
        phones += [phone, other]
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not (phone.opens() and other.opens()):
            n.spin(0.05)
        n.wait_sim(2.0)

        # 1. the zones of the map in use
        got = latest_zones(phone)
        first = got[1] if got else None
        check("zones sent", first is not None and first.map_name == "house" and "stairwell" in names(first),
              f"map '{first.map_name if first else '?'}', zones {names(first)}")
        old = list(first.zones) if first else []

        # navigation, and to the living room, so the robot's place on the map is known
        mode = ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(ids), mode="navigation"), 120.0)
        n.wait_sim(2.0)
        gx, gy = hl.to_world(*LIVING_ROOM)
        trip = ask(phone, pb.COMMAND_GO_TO, pb.GoTo(request_id=next(ids), x=gx, y=gy, heading=0.0), 15.0)
        goal_id = trip[1].goal_id if trip else b""
        until(lambda: any(m.goal_id == goal_id and m.state in (State.STATE_REACHED, State.STATE_FAILED)
                          for _, m in phone.messages(pb.DATA_NAVIGATION)), 150.0)
        n.wait_sim(2.0)
        rx, ry, _ = n.pose()
        print(f"  robot at ({rx:.2f}, {ry:.2f}), {math.hypot(rx - gx, ry - gy) * 100:.0f} cm from the living-room goal "
              f"(navigation: {text(mode[1] if mode else None)})", flush=True)

        # 2. another map's zones
        t2 = time.monotonic()
        _, reply = set_zones(phone, old + [zone("wrong", square(*hl.to_world(*BEDROOM), ZONE_SIZE))], map_name="office")
        n.wait_sim(1.0)
        check("wrong map", reply is not None and reply.result == Result.RESULT_REFUSED and not phone.messages(pb.DATA_NO_GO_ZONES, t2),
              f"zones for 'office': {text(reply)}; zones sent since: {len(phone.messages(pb.DATA_NO_GO_ZONES, t2))}")

        # 3. over the robot, at the dock
        _, over = set_zones(phone, old + [zone("over the robot", square(rx, ry, 0.6))])
        _, dock = set_zones(phone, old + [zone("at the dock", square(0.25, 0.0, 0.3))])
        check("over the robot", over is not None and over.result == Result.RESULT_REFUSED
              and dock is not None and dock.result == Result.RESULT_REFUSED,
              f"around the robot: {text(over)}; at the dock: {text(dock)}")

        # 4-5. a new zone in the bedroom
        zx, zy = hl.to_world(*BEDROOM)
        box = (zx - ZONE_SIZE / 2, zy - ZONE_SIZE / 2, zx + ZONE_SIZE / 2, zy + ZONE_SIZE / 2)
        before = route_exists(zx, zy)
        t4 = time.monotonic()
        answered, reply = set_zones(phone, old + [zone("test box", square(zx, zy, ZONE_SIZE))])
        took = (answered - t4) if answered else None
        until(lambda: any("test box" in names(m) for _, m in other.messages(pb.DATA_NO_GO_ZONES, t4)), 3.0)
        got_at = [t for t, m in other.messages(pb.DATA_NO_GO_ZONES, t4) if "test box" in names(m)]
        lag = (got_at[0] - answered) if got_at and answered else None
        on_file = "test box" in open(ZONES_FILE).read()
        check("added", reply is not None and reply.result == Result.RESULT_OK and took is not None and took < 5.0
              and lag is not None and lag < 1.0 and on_file,
              f"{text(reply)} after {took if took is None else round(took, 2)} s; the other phone had it "
              f"{'never' if lag is None else f'{lag:+.2f} s'} from the answer; in the file: {on_file}")
        blocked = until(lambda: route_exists(zx, zy) is False, 3.0) if reply and reply.result == Result.RESULT_OK else None
        state = modes.state.state if modes.state else None
        check("in force", before is True and blocked is not None and state == "navigation",
              f"route to the zone's middle before: {before}; none any more after {blocked} s (limit 3); "
              f"mode {state}")

        # 6. a preview around the zone keeps out of it
        area = [box[0] - 1.0, box[1] - 1.0, box[2] + 1.0, box[1] - 1.0, box[2] + 1.0, box[3] + 1.0, box[0] - 1.0, box[3] + 1.0]
        t6 = time.monotonic()
        job = pb.CleaningJob(what=pb.CleaningJob.WHAT_AREA, area=area)
        got = ask(phone, pb.COMMAND_PREVIEW_CLEANING, pb.PreviewCleaning(request_id=next(ids), job=job), 30.0)
        until(lambda: any(m.job_id == 0 and m.pieces for _, m in phone.messages(pb.DATA_CLEANING_PLAN, t6)), 5.0)
        plans = [m for _, m in phone.messages(pb.DATA_CLEANING_PLAN, t6) if m.job_id == 0 and m.pieces]
        nearest = None
        if plans:
            for p in plans[-1].pieces:
                if p.kind == pb.CleaningPlan.KIND_TRANSIT:
                    continue  # Nav2 plans those around the zone
                a, b = (p.x0_cm / 100, p.y0_cm / 100), (p.x1_cm / 100, p.y1_cm / 100)
                steps = max(1, int(math.dist(a, b) / 0.02))
                for k in range(steps + 1):
                    x = a[0] + (b[0] - a[0]) * k / steps
                    y = a[1] + (b[1] - a[1]) * k / steps
                    d = outside_by(x, y, box)
                    nearest = d if nearest is None else min(nearest, d)
        check("cleaning", got is not None and got[1].result == Result.RESULT_OK and plans and nearest is not None and nearest >= ROW_GAP,
              f"preview {text(got[1] if got else None)}; {len(plans[-1].pieces) if plans else 0} pieces; the closest row "
              f"or hop passes {'?' if nearest is None else f'{nearest * 100:.0f} cm'} from the zone (limit {ROW_GAP * 100:.0f})")

        # 7. delete it again
        t7 = time.monotonic()
        _, reply = set_zones(phone, old)
        clear = until(lambda: route_exists(zx, zy) is True, 3.0) if reply and reply.result == Result.RESULT_OK else None
        last = latest_zones(phone, t7)
        content = open(ZONES_FILE).read()
        check("deleted", reply is not None and reply.result == Result.RESULT_OK and clear is not None
              and last is not None and names(last[1]) == [z.name for z in old] and "test box" not in content and "stairwell" in content,
              f"{text(reply)}; route found again after {clear} s; the phone has {names(last[1]) if last else None}")

        # 8. from the terminal
        out = subprocess.run(["ros2", "run", "vacuum_mode_manager", "zones"], capture_output=True, text=True, timeout=30)
        check("terminal", out.returncode == 0 and "stairwell" in out.stdout,
              f"exit {out.returncode}: {(out.stdout or out.stderr).strip()[:200]}")
        modes.request("idle")
    finally:
        for p in phones:
            p.stop()
        shutil.copyfile(backup, ZONES_FILE)
        os.unlink(backup)
        n.destroy_node()
        rclpy.shutdown()
    return results


if __name__ == "__main__":
    results = main()
    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed")
    out = os.path.join(ROOT, "logs", "phase27_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    sys.exit(0 if results and passed == len(results) else 1)
