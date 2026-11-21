#!/usr/bin/env python3
"""Phase 14 acceptance test: going to a point from the app (step 4 of the Android app).

Needs two things running first, with the robot at its start pose and a saved map:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false     (it starts the gateway)
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase14_go_to_test.py

A scripted phone (phase 11's) does what the app does: asks for navigation mode, then sends GoTo.
  1. not navigating     a goal in idle is answered "not available", with the reason
  2. go to a point      the goal is accepted with an id; the phone sees it on the way with the
                        distance shrinking, then reached; the robot ends within 15 cm of the goal
  3. route              the route arrives while the robot drives, from the robot to the goal
  4. cancel             cancelling a trip brings the robot to rest within 1 s; the phone sees it cancelled
  5. unreachable        a goal inside the stairwell (a no-go zone) is refused at once (the
                        gateway asks the planner for a route first) and the robot does not move
  7. no-go zones        the phone receives the stairwell zone, as written in maps/house_zones.yaml
  6. anyone's goal      a goal sent by someone else (as RViz does) shows up on the phone too, and
                        cancelling every goal from the phone stops it
"""

import json
import math
import os
import sys
import time

import rclpy

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
STATE_NAMES = {v: k.replace("STATE_", "").lower() for k, v in pb.NavigationStatus.State.items()}

LIVING_ROOM = (3.55, 1.25)  # between sofa and armchair (house coordinates)
BEDROOM = (6.0, 1.6)
STAIRWELL = (5.55, 5.0)
FAR_CORNER = (4.0, 3.7)


def main():
    rclpy.init()
    n = ModeTestDriver()
    modes = ModeClient(n)
    results, phones = [], []
    ids = iter(range(1, 10000))

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
                return got[1]
        return None

    def go_to(phone, house_xy):
        """Send the robot to a point the way the app does: facing the way it travels."""
        x, y = hl.to_world(*house_xy)
        rx, ry, _ = n.pose()
        heading = math.atan2(y - ry, x - rx)
        reply = ask(phone, pb.COMMAND_GO_TO, pb.GoTo(request_id=next(ids), x=x, y=y, heading=heading), 15.0)
        return reply, (x, y)

    def statuses(phone, goal_id, since):
        return [(t, m) for t, m in phone.messages(pb.DATA_NAVIGATION, since) if m.goal_id == goal_id]

    def wait_state(phone, goal_id, since, wanted, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            n.spin(0.05)
            seen = statuses(phone, goal_id, since)
            if seen and seen[-1][1].state in wanted:
                return seen[-1][1].state
        seen = statuses(phone, goal_id, since)
        return seen[-1][1].state if seen else None

    def wait_moving(limit=30.0):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            n.spin(0.02)
            if n.speed() > 0.1:
                return True
        return False

    def at_rest_after(t0, limit=5.0):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            n.spin(0.01)
            if n.speed() < 0.02:
                return round(time.monotonic() - t0, 3)
        return None

    def text(reply):
        return f"{pb.Reply.Result.Name(reply.result)} '{reply.message}'" if reply else "no reply"

    try:
        if modes.wait_ready(60.0) is None or not modes.wait_for_state("idle", 30.0):
            check("robot ready", False, "the mode manager did not reach idle")
            return results
        end = time.monotonic() + 30.0
        while time.monotonic() < end and (n.sim_time is None or n.odom is None):
            n.spin(0.05)
        phone = Phone("phone").start()
        phones.append(phone)
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not phone.opens():
            n.spin(0.05)

        # 1. in idle, Nav2 is not running
        reply, _ = go_to(phone, LIVING_ROOM)
        check("not navigating", reply is not None and reply.result == Result.RESULT_NOT_AVAILABLE
              and "navigation first" in reply.message, f"goal in idle: {text(reply)}")

        # 2-3. go to a point
        mode_reply = ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(ids), mode="navigation"), 120.0)
        n.wait_sim(2.0)  # let the costmaps fill in
        t2 = time.monotonic()
        reply, (gx, gy) = go_to(phone, LIVING_ROOM)
        goal = reply.goal_id if reply else b""
        state = wait_state(phone, goal, t2, (State.STATE_REACHED, State.STATE_FAILED, State.STATE_CANCELLED), 150.0)
        n.wait_sim(1.0)
        tx, ty, _ = n.pose()
        error = math.hypot(tx - gx, ty - gy)
        seen = statuses(phone, goal, t2)
        distances = [m.distance_remaining for _, m in seen if m.state == State.STATE_ON_THE_WAY and m.distance_remaining > 0]
        shrinking = len(distances) >= 3 and distances[-1] < distances[0]
        check("go to a point",
              mode_reply is not None and mode_reply.result == Result.RESULT_OK and reply is not None
              and reply.result == Result.RESULT_OK and len(goal) == 16 and state == State.STATE_REACHED
              and shrinking and error < 0.15,
              f"navigation: {text(mode_reply)}; GoTo: {text(reply)}, goal id {len(goal)} bytes; states seen: "
              f"{[STATE_NAMES[s] for s in dict.fromkeys(m.state for _, m in seen)]}; distance left "
              f"{distances[0] if distances else '?'} -> {distances[-1] if distances else '?'} m; ended "
              f"{error * 100:.1f} cm from the goal (limit 15)")

        routes = phone.messages(pb.DATA_ROUTE, t2)
        if routes:
            first = routes[0][1].points_cm
            start_gap = math.hypot(first[0] / 100.0 - hl.to_world(0.45, 2.25)[0], first[1] / 100.0 - hl.to_world(0.45, 2.25)[1])
            last = routes[-1][1].points_cm
            end_gap = math.hypot(last[-2] / 100.0 - gx, last[-1] / 100.0 - gy)
        check("route", bool(routes) and start_gap < 0.5 and end_gap < 0.15,
              f"{len(routes)} routes received, {len(routes[0][1].points_cm) // 2 if routes else 0} points in the first; "
              f"first starts {start_gap * 100 if routes else 0:.0f} cm from where the robot set off, last ends "
              f"{end_gap * 100 if routes else 0:.0f} cm from the goal" if routes else "no route received")

        # 4. cancel a trip
        t4 = time.monotonic()
        reply, _ = go_to(phone, BEDROOM)
        goal = reply.goal_id if reply else b""
        moving = wait_moving()
        cancel = None
        rest = None
        if moving:
            cancel_id = next(ids)
            t_cancel = time.monotonic()
            phone.send(pb.COMMAND_CANCEL_GOAL, pb.CancelGoal(request_id=cancel_id, goal_id=goal))
            rest = at_rest_after(t_cancel)
            got = None
            end = time.monotonic() + 10.0
            while time.monotonic() < end and not got:
                n.spin(0.02)
                got = phone.reply_to(cancel_id, t_cancel)
            cancel = got[1] if got else None
        state = wait_state(phone, goal, t4, (State.STATE_CANCELLED,), 10.0)
        check("cancel", moving and cancel is not None and cancel.result == Result.RESULT_OK and rest is not None
              and rest <= 1.0 and state == State.STATE_CANCELLED,
              f"trip to the bedroom under way: {moving}; CancelGoal: {text(cancel)}; at rest {rest} s after the "
              f"request; the phone sees it {STATE_NAMES.get(state, state)}")

        # 5. a goal inside the stairwell: refused before the robot moves
        n.wait_sim(2.0)
        before = n.pose()
        t5 = time.monotonic()
        reply, _ = go_to(phone, STAIRWELL)
        answered = time.monotonic() - t5
        n.wait_sim(2.0)
        after = n.pose()
        moved = math.hypot(after[0] - before[0], after[1] - before[1])
        check("unreachable", reply is not None and reply.result == Result.RESULT_REFUSED and answered < 3.0 and moved < 0.02,
              f"GoTo into the stairwell: {text(reply)} after {answered:.2f} s; the robot moved {moved * 100:.1f} cm")

        # 6. a goal from someone else, then "cancel every goal" from the phone
        known = {m.goal_id for _, m in phone.messages(pb.DATA_REPLY)}
        t6 = time.monotonic()
        sent = n.send_goal(*hl.to_world(*FAR_CORNER), 1.57)
        theirs = None
        end = time.monotonic() + 15.0
        while time.monotonic() < end and theirs is None:
            n.spin(0.05)
            for _, m in phone.messages(pb.DATA_NAVIGATION, t6):
                if m.goal_id not in known and m.state == State.STATE_ON_THE_WAY:
                    theirs = m
        moving = wait_moving() if theirs else False
        cancel = ask(phone, pb.COMMAND_CANCEL_GOAL, pb.CancelGoal(request_id=next(ids)), 10.0) if moving else None
        rest = at_rest_after(time.monotonic()) if cancel else None
        state = wait_state(phone, theirs.goal_id, t6, (State.STATE_CANCELLED,), 10.0) if theirs else None
        check("anyone's goal", sent and theirs is not None and theirs.distance_remaining >= 0 and cancel is not None
              and cancel.result == Result.RESULT_OK and state == State.STATE_CANCELLED,
              f"goal sent by another client: accepted {sent}; the phone saw it on the way: {theirs is not None} "
              f"({theirs.distance_remaining if theirs else '?':.2f} m left); cancel all: {text(cancel)}; at rest after "
              f"{rest} s; the phone sees it {STATE_NAMES.get(state, state)}" if theirs else
              f"goal sent by another client: accepted {sent}; the phone never saw it")

        # 7. the no-go zones, as in the keepout zones file
        zones = [m for _, m in phone.messages(pb.DATA_NO_GO_ZONES)]
        got = {z.name: list(z.points_cm) for z in zones[-1].zones} if zones else {}
        check("no-go zones", got.get("stairwell") == [455, 200, 565, 200, 565, 330, 455, 330],
              f"zones received: {got if zones else 'none'}")

        ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(ids), mode="idle"), 60.0)
        out = os.environ.get("PHASE14_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results}, f)
    finally:
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
