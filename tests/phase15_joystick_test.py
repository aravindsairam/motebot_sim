#!/usr/bin/env python3
"""Phase 15 acceptance test: driving with the app's joystick (step 5 of the Android app).

Needs two things running first, with the robot at its start pose:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false     (it starts the gateway)
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase15_joystick_test.py

Scripted phones (phase 11's) send Drive 20 times a second, as the app's joystick does.
  1. not in idle        Drive in idle mode does not move the robot (the safety gate holds it)
  2. speed              in manual mode the commanded forward speed is reached (true speed)
  3. phone goes quiet   a phone that stops sending without saying so (frozen, Wi-Fi gone): the
                        robot is at rest within 1 s
  4. phone leaves       a phone that disconnects mid-drive: at rest within 1 s
  5. one driver         a second phone is refused while the first drives, and may drive once the
                        first has been quiet for 1 s
  6. gateway killed     the gateway dies mid-drive: at rest within 1 s; it is restarted and the
                        phone connects again
The stop checks turn on the spot (1 rad/s), so the robot stays clear of the furniture.
"""

import itertools
import json
import os
import subprocess
import sys
import threading
import time

import rclpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from mode_client import ModeClient  # noqa: E402
from phase7_mode_manager_test import ModeTestDriver  # noqa: E402
from phase11_gateway_test import Phone, gateway_pids  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402

Result = pb.Reply.Result
RESULT_NAMES = {v: k for k, v in pb.Reply.Result.items()}
REQUEST_IDS = itertools.count(1)


class Joystick:
    """A thumb on one phone's joystick: Drive 20 times a second while held."""

    def __init__(self, phone):
        self.phone = phone
        self.sent = []  # request ids
        self._speeds = (0.0, 0.0)
        self._thread = None
        self._running = False

    def _send(self, forward, turn):
        request_id = next(REQUEST_IDS)
        try:
            self.phone.send(pb.COMMAND_DRIVE, pb.Drive(request_id=request_id, forward=forward, turn=turn))
            self.sent.append(request_id)
        except Exception:
            pass  # the connection is gone; a real app keeps trying too

    def _loop(self):
        while self._running:
            self._send(*self._speeds)
            time.sleep(0.05)

    def hold(self, forward, turn):
        self._speeds = (forward, turn)
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def quiet(self):
        """Stop sending without a zero: a frozen app, or Wi-Fi gone."""
        self._running = False
        if self._thread is not None:
            self._thread.join(1.0)

    def release(self):
        """Let go: one zero, as the app sends."""
        self.quiet()
        self._send(0.0, 0.0)

    def replies(self, since=0.0):
        mine = set(self.sent)
        return [m for _, m in self.phone.messages(pb.DATA_REPLY, since) if m.request_id in mine]


def main():
    rclpy.init()
    n = ModeTestDriver()
    modes = ModeClient(n)
    results, phones = [], []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def change(phone, mode, timeout=120.0):
        request_id = next(REQUEST_IDS)
        sent = phone.send(pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=request_id, mode=mode))
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            n.spin(0.02)
            got = phone.reply_to(request_id, sent)
            if got:
                return got[1]
        return None

    def text(reply):
        return f"{RESULT_NAMES.get(reply.result, reply.result)} '{reply.message}'" if reply else "no reply"

    def wait_moving(limit=5.0):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            n.spin(0.02)
            if n.speed() >= 0.05:
                return True
        return False

    def at_rest_after(t0, limit=5.0):
        """Seconds from t0 (monotonic) until the true speed fell below 2 cm/s, or None."""
        end = time.monotonic() + limit
        while time.monotonic() < end:
            n.spin(0.01)
            if n.speed() < 0.02:
                return round(time.monotonic() - t0, 3)
        return None

    def wait_wall(seconds):
        """Wall-clock time: the gateway's one-driver rule runs on the computer's clock."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            n.spin(0.02)

    def connected(name):
        phone = Phone(name).start()
        phones.append(phone)
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not phone.opens():
            n.spin(0.05)
        return phone if phone.opens() else None

    try:
        if modes.wait_ready(60.0) is None or not modes.wait_for_state("idle", 30.0):
            check("robot ready", False, "the mode manager did not reach idle")
            return results
        end = time.monotonic() + 30.0
        while time.monotonic() < end and (n.sim_time is None or n.odom is None):
            n.spin(0.05)
        if n.sim_time is None or n.odom is None:
            check("robot ready", False, "no /clock or /ground_truth/odom from the simulator")
            return results
        phone, other = connected("phone"), connected("other phone")
        if phone is None or other is None:
            check("phones connected", False, "the phones could not connect to the gateway")
            return results
        stick = Joystick(phone)

        # 1. Drive in idle: passed on, but the safety gate does not let it move the robot
        t1 = time.monotonic()
        stick.hold(0.2, 0.0)
        n.wait_sim(1.5)
        moved = wait_moving(0.5)
        stick.release()
        replies = stick.replies(t1)
        check("not in idle", not moved and replies and all(r.result == Result.RESULT_OK for r in replies),
              f"{len(stick.sent)} Drive sent in idle, {len(replies)} answered OK; the robot moved: {moved}")

        # 2. the commanded speed, in manual mode
        r_manual = change(phone, "manual")
        stick.hold(0.2, 0.0)
        n.wait_sim(1.5)
        samples = []
        t_end = n.sim_time + 1.0
        while n.sim_time < t_end:
            n.spin(0.02)
            samples.append(n.odom.twist.twist.linear.x)
        stick.hold(-0.2, 0.0)  # and back to where it was
        n.wait_sim(2.5)
        stick.release()
        mean = sum(samples) / len(samples) if samples else 0.0
        check("speed", r_manual is not None and r_manual.result == Result.RESULT_OK and 0.18 <= mean <= 0.22,
              f"manual: {text(r_manual)}; commanded 0.20 m/s forward, true speed over 1 s {mean:.3f} m/s")
        n.wait_sim(1.0)

        # 3. a phone that stops sending without saying so
        stick.hold(0.0, 1.0)
        moving = wait_moving()
        n.wait_sim(1.0)
        t3 = time.monotonic()
        stick.quiet()
        rest = at_rest_after(t3)
        check("phone goes quiet", moving and rest is not None and rest <= 1.0,
              f"turning: {moving}; the phone stopped sending (no zero, still connected); at rest {rest} s later")

        # 4. a phone that disconnects mid-drive
        leaving = connected("leaving phone")
        wait_wall(1.5)  # the first phone's turn as driver has run out
        stick4 = Joystick(leaving)
        stick4.hold(0.0, 1.0)
        moving = wait_moving()
        n.wait_sim(1.0)
        t4 = time.monotonic()
        stick4.quiet()
        leaving.stop()
        rest = at_rest_after(t4)
        check("phone leaves", moving and rest is not None and rest <= 1.0,
              f"turning: {moving}; the phone disconnected; at rest {rest} s later")

        # 5. one driver at a time
        wait_wall(1.5)
        stick.hold(0.0, 0.8)
        wait_moving()
        second = Joystick(other)
        t5 = time.monotonic()
        second.hold(0.0, -0.8)
        wait_wall(1.0)
        refused = second.replies(t5)
        stick.release()
        wait_wall(0.5)
        still_refused = [r for r in second.replies(time.monotonic() - 0.2) if r.result == Result.RESULT_REFUSED]
        wait_wall(1.0)  # the first phone has now been quiet for more than 1 s
        t5b = time.monotonic()
        wait_wall(0.5)
        taken = second.replies(t5b)
        second.release()
        n.wait_sim(1.0)
        ok = (refused and all(r.result == Result.RESULT_REFUSED and "another phone" in r.message for r in refused)
              and still_refused and taken and all(r.result == Result.RESULT_OK for r in taken))
        check("one driver", ok,
              f"while the first phone drove: {len(refused)} Drive from the second refused "
              f"('{refused[0].message if refused else ''}'); 0.5 s after the first let go: still refused: "
              f"{bool(still_refused)}; after 2 s: {len(taken)} answered, all OK: {all(r.result == Result.RESULT_OK for r in taken)}")

        # 6. the gateway dies mid-drive
        wait_wall(1.5)
        before = gateway_pids()
        stick.hold(0.0, 1.0)
        moving = wait_moving()
        n.wait_sim(1.0)
        t6 = time.monotonic()
        subprocess.run(["pkill", "-f", "lib/vacuum_gateway/gateway"], check=False)
        rest = at_rest_after(t6)
        stick.quiet()
        back = None
        end = time.monotonic() + 20.0
        while time.monotonic() < end and back is None:
            n.spin(0.05)
            o = [t for t in phone.opens() if t > t6]
            back = o[0] if o else None
        after = gateway_pids()
        check("gateway killed", moving and rest is not None and rest <= 1.0 and back is not None,
              f"turning: {moving}; gateway pid {before} killed; at rest {rest} s later; restarted as {after}, "
              f"the phone connected again after {round(back - t6, 2) if back else None} s")

        if back is not None:
            change(phone, "idle", 60.0)
        else:
            modes.request("idle")
        out = os.environ.get("PHASE15_RESULT_JSON")
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
