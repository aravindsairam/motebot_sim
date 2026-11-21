#!/usr/bin/env python3
"""Phase 12 acceptance test: commands from the app (step 2 of the Android app), against the robot.

Needs two things running first, with the robot at its start pose and a saved map:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false     (it starts the gateway)
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase12_app_commands_test.py

Scripted phones (the same as phase 11's) send commands the way the app does: Stop is a change to
idle, Dock a change to docking, Reset a change from failsafe to idle.
  1. ping               answered with the same request number, quickly
  2. refusals           an unknown mode, a mode that cannot be requested, and an unknown command
                        are each answered, the first two with the robot's own reason
  3. only the asker     a second phone receives none of the first phone's replies
  4. stop navigating    Stop while Nav2 drives the robot: at rest within 1 s
  5. stop while busy    Stop sent while the robot is switching to docking is not refused as busy:
                        it waits for the change, then the robot ends in idle without having moved
  6. dock               Dock from the phone ends with the robot docked; the phone sees the progress
  7. reset              failsafe asked for from the phone, then Reset: the robot is back in idle
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
from phase11_gateway_test import Phone, distinct  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402

Result = pb.Reply.Result
RESULT_NAMES = {v: k for k, v in pb.Reply.Result.items()}
GOAL = (3.55, 1.25, math.pi)  # living room, between sofa and armchair (house coordinates)


class Requests:
    """Request numbers, unique across the test, as the app keeps them per connection."""

    def __init__(self):
        self.last = 0

    def next(self):
        self.last += 1
        return self.last


def main():
    rclpy.init()
    n = ModeTestDriver()
    modes = ModeClient(n)
    ids = Requests()
    results = []
    phones = []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def wait_reply(phone, request_id, since, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            n.spin(0.02)
            got = phone.reply_to(request_id, since)
            if got:
                return got
        return None

    def ask(phone, msg_type, message, timeout=30.0):
        """Send and wait. Returns (Reply or None, seconds taken)."""
        sent = phone.send(msg_type, message)
        got = wait_reply(phone, message.request_id, sent, timeout)
        return (got[1], round(got[0] - sent, 3)) if got else (None, None)

    def change(phone, mode, timeout=120.0):
        return ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=ids.next(), mode=mode), timeout)

    def text(reply):
        return f"{RESULT_NAMES.get(reply.result, reply.result)} '{reply.message}'" if reply else "no reply"

    def at_rest_after(t0, limit):
        """Seconds from t0 (monotonic) until the true speed fell below 2 cm/s, or None."""
        end = time.monotonic() + limit
        while time.monotonic() < end:
            n.spin(0.01)
            if n.odom is not None and n.speed() < 0.02:
                return round(time.monotonic() - t0, 3)
        return None

    try:
        if modes.wait_ready(60.0) is None or not modes.wait_for_state("idle", 30.0):
            check("robot ready", False, "the mode manager did not reach idle")
            return results
        end = time.monotonic() + 30.0  # the simulator's clock and the true pose, before anything uses them
        while time.monotonic() < end and (n.sim_time is None or n.odom is None):
            n.spin(0.05)
        if n.sim_time is None or n.odom is None:
            check("robot ready", False, "no /clock or /ground_truth/odom from the simulator")
            return results
        n.wait_sim(1.0)
        phone, other = Phone("phone").start(), Phone("other phone").start()
        phones += [phone, other]
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not (phone.opens() and other.opens()):
            n.spin(0.05)
        if not (phone.opens() and other.opens()):
            check("phones connected", False, "the phones could not connect to the gateway")
            return results
        t_start = time.monotonic()

        # 1. ping
        rtts = []
        for _ in range(5):
            reply, secs = ask(phone, pb.COMMAND_PING, pb.Ping(request_id=ids.next()), 3.0)
            rtts.append(secs if reply and reply.result == Result.RESULT_OK else None)
        check("ping", None not in rtts and max(rtts) < 0.1, f"5 pings answered in {rtts} s")

        # 2. refusals, each with an answer
        r_unknown, _ = change(phone, "flying")
        r_docked, _ = change(phone, "docked")
        bad_id = ids.next()
        r_bad, _ = ask(phone, 0x7E, pb.Ping(request_id=bad_id), 3.0)
        ok = (r_unknown and r_unknown.result == Result.RESULT_REFUSED and "unknown state 'flying'" in r_unknown.message
              and r_docked and r_docked.result == Result.RESULT_REFUSED and "cannot be requested" in r_docked.message
              and r_bad and r_bad.result == Result.RESULT_BAD_REQUEST and r_bad.request_id == bad_id)
        check("refusals", ok, f"'flying': {text(r_unknown)}; 'docked': {text(r_docked)}; command type 0x7e: {text(r_bad)}")

        # 4. stop while Nav2 drives (3 needs the replies of the whole test, so it is checked at the end)
        r_nav, secs_nav = change(phone, "navigation")
        n.wait_sim(2.0)  # let the costmaps fill in
        accepted = r_nav and r_nav.result == Result.RESULT_OK and n.send_goal(*hl.to_world(GOAL[0], GOAL[1]), GOAL[2])
        moving = False
        end = time.monotonic() + 30.0
        while accepted and time.monotonic() < end:
            n.spin(0.02)
            if n.speed() > 0.12:
                moving = True
                break
        speed_before = n.speed()
        r_stop, secs_stop = (None, None)
        rest = None
        if moving:
            t_req = time.monotonic()
            sent = phone.send(pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=ids.next(), mode="idle"))
            rest = at_rest_after(t_req, 5.0)
            got = wait_reply(phone, ids.last, sent, 60.0)
            r_stop, secs_stop = (got[1], round(got[0] - sent, 3)) if got else (None, None)
        check("stop navigating",
              moving and rest is not None and rest <= 1.0 and r_stop and r_stop.result == Result.RESULT_OK
              and modes.wait_for_state("idle", 5.0),
              f"navigation: {text(r_nav)} after {secs_nav} s; robot moving at {speed_before:.2f} m/s; Stop: at rest "
              f"{rest} s after the request; reply {text(r_stop)} after {secs_stop} s")

        # 5. stop while the robot is switching to docking
        n.wait_sim(1.0)
        x0, y0, _ = n.pose()
        moved = 0.0
        dock_id, stop_id = ids.next(), ids.next()
        t_dock = phone.send(pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=dock_id, mode="docking"))
        changing = False
        end = time.monotonic() + 10.0
        while time.monotonic() < end:
            n.spin(0.01)
            if modes.state is not None and modes.state.transitioning:
                changing = True
                break
        t_stop = phone.send(pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=stop_id, mode="idle"))
        dock_reply = stop_reply = None
        end = time.monotonic() + 120.0
        while time.monotonic() < end and not (dock_reply and stop_reply):
            n.spin(0.02)
            x, y, _ = n.pose()
            moved = max(moved, math.hypot(x - x0, y - y0))
            dock_reply = dock_reply or phone.reply_to(dock_id, t_dock)
            stop_reply = stop_reply or phone.reply_to(stop_id, t_stop)
        t_after = time.monotonic()
        while time.monotonic() - t_after < 2.0:  # and a little longer: nothing moves afterwards
            n.spin(0.02)
            x, y, _ = n.pose()
            moved = max(moved, math.hypot(x - x0, y - y0))
        waited =dock_reply and stop_reply and stop_reply[0] >= dock_reply[0]
        ok = (changing and dock_reply and stop_reply and stop_reply[1].result == Result.RESULT_OK
              and waited and modes.state.state == "idle" and not modes.state.transitioning and moved < 0.02)
        check("stop while busy", ok,
              f"Stop sent {round(t_stop - t_dock, 2)} s after Dock, during the change: {changing}; "
              f"Dock: {text(dock_reply[1] if dock_reply else None)}; Stop: {text(stop_reply[1] if stop_reply else None)} "
              f"{round(stop_reply[0] - t_stop, 2) if stop_reply else '?'} s after it was sent, answered after Dock: {bool(waited)}; "
              f"now {modes.state.state}; the robot moved {moved * 100:.1f} cm (limit 2 cm)")

        # 6. dock from the phone
        t6 = time.monotonic()
        r_dock, secs_dock = change(phone, "docking")
        docked = r_dock and r_dock.result == Result.RESULT_OK and modes.wait_for_state("docked", 120.0)
        seen = distinct([m.reason for _, m in phone.messages(pb.DATA_ROBOT_STATE, t6) if m.reason.startswith("dock")])
        check("dock", docked and modes.state.reason.startswith("docked") and len(seen) >= 3,
              f"Dock: {text(r_dock)} after {secs_dock} s; docked {docked} after {round(time.monotonic() - t6, 1)} s; "
              f"the phone saw: {seen}")

        # 7. failsafe, then Reset
        r_fs, _ = change(phone, "failsafe")
        in_fs = modes.wait_for_state("failsafe", 10.0)
        n.wait_sim(1.0)
        allowed = list(modes.state.allowed_transitions)
        r_reset, secs_reset = change(phone, "idle")
        back = modes.wait_for_state("idle", 30.0)
        check("reset", r_fs and r_fs.result == Result.RESULT_OK and in_fs and allowed == ["idle"]
              and r_reset and r_reset.result == Result.RESULT_OK and back,
              f"failsafe: {text(r_fs)}; allowed in failsafe: {allowed}; Reset: {text(r_reset)} after {secs_reset} s; "
              f"now {modes.state.state}")

        # 3. the other phone received no replies at all
        other_replies = other.messages(pb.DATA_REPLY, t_start)
        mine = phone.messages(pb.DATA_REPLY, t_start)
        check("only the asker", len(other_replies) == 0 and len(mine) >= 15,
              f"the phone received {len(mine)} replies, the other phone {len(other_replies)}")

        out = os.environ.get("PHASE12_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "history": modes.history}, f)
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
