#!/usr/bin/env python3
"""Phase 16 acceptance test: making a map from the app (step 6 of the Android app).

Needs two things running first, with the robot at its start pose and a saved map 'house':
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false     (it starts the gateway)
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase16_new_map_test.py

A scripted phone (phase 11's) does what the app does: New map (mapping mode), drives with Drive
commands, then Save map, then idle.
  1. mapping            the phone switches the robot to mapping
  2. map grows          while the phone drives, map updates arrive and the known part of the map grows
  3. name taken         saving as 'house' without replace is answered RESULT_ALREADY_EXISTS and
                        the robot is still mapping with nothing written
  4. saved              saving under a new name works; after idle the robot uses the new map
  5. navigation         navigation on the new map: a goal in the mapped part is reached
The new map is saved as 'phone_test'. Afterwards its files are removed and the map in use is set
back to what it was, so the next run (and the user) start from the usual house map.
"""

import glob
import itertools
import json
import math
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
from phase11_gateway_test import Phone  # noqa: E402
from vacuum_gateway import geometry  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402

Result = pb.Reply.Result
RESULT_NAMES = {v: k for k, v in pb.Reply.Result.items()}
REQUEST_IDS = itertools.count(1)
MAPS = os.path.normpath(os.path.join(HERE, "..", "..", "ros2_ws", "maps"))
NEW_MAP = "phone_test"


def remove_new_map():
    for path in glob.glob(os.path.join(MAPS, NEW_MAP + ".*")) + glob.glob(os.path.join(MAPS, NEW_MAP + "_keepout.*")) \
            + glob.glob(os.path.join(MAPS, NEW_MAP + "_zones.yaml")):
        os.remove(path)


class Thumb:
    """Drive 20 times a second while held, as the app's joystick does."""

    def __init__(self, phone):
        self.phone = phone
        self.speeds = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            speeds = self.speeds
            if speeds is not None:
                try:
                    self.phone.send(pb.COMMAND_DRIVE, pb.Drive(request_id=next(REQUEST_IDS), forward=speeds[0], turn=speeds[1]))
                except Exception:
                    pass
            time.sleep(0.05)


def main():
    rclpy.init()
    n = ModeTestDriver()
    modes = ModeClient(n)
    results, phones = [], []
    marker = os.path.join(MAPS, "current_map.txt")
    with open(marker) as f:
        map_before = f.read().strip()

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def ask(phone, msg_type, message, timeout):
        sent = phone.send(msg_type, message)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            n.spin(0.02)
            got = phone.reply_to(message.request_id, sent)
            if got:
                return got[1]
        return None

    def change(phone, mode, timeout=120.0):
        return ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(REQUEST_IDS), mode=mode), timeout)

    def save(phone, name, replace):
        return ask(phone, pb.COMMAND_SAVE_MAP, pb.SaveMap(request_id=next(REQUEST_IDS), name=name, replace=replace), 60.0)

    def text(reply):
        return f"{RESULT_NAMES.get(reply.result, reply.result)} '{reply.message}'" if reply else "no reply"

    def latest(phone, msg_type, since=0.0):
        got = phone.messages(msg_type, since)
        return got[-1][1] if got else None

    try:
        remove_new_map()
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
        if not phone.opens() or n.sim_time is None:
            check("robot ready", False, "no simulator clock, or the phone could not connect")
            return results
        thumb = Thumb(phone)
        # A phone that connects is sent the current map at once (the saved one, in idle). Let that
        # arrive first: it is not a map made while mapping.
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not phone.messages(pb.DATA_MAP):
            n.spin(0.05)
        n.spin(0.5)

        # 1. New map
        t1 = time.monotonic()
        r_map = change(phone, "mapping")
        check("mapping", r_map is not None and r_map.result == Result.RESULT_OK,
              f"mapping: {text(r_map)}")

        # 2. drive: a full turn on the spot, then forward 40 cm and back
        n.wait_sim(2.0)
        for speeds, seconds in (((0.0, 0.8), 8.5), ((0.2, 0.0), 2.0), ((-0.2, 0.0), 2.0)):
            thumb.speeds = speeds
            n.wait_sim(seconds)
        thumb.speeds = None
        n.wait_sim(3.0)  # slam_toolbox publishes its map every couple of seconds
        maps = [m for _, m in phone.messages(pb.DATA_MAP, t1)]
        versions = sorted({m.version for m in maps})
        known = [int((geometry.rle_decode(m.cells_rle) >= 0).sum()) for m in maps]
        check("map grows", len(versions) >= 2 and known and known[-1] > known[0],
              f"{len(maps)} maps received while mapping, {len(versions)} different; known cells "
              f"{known[0] if known else 0} -> {known[-1] if known else 0}")

        # 3. a taken name
        r_taken = save(phone, map_before or "house", False)
        state = latest(phone, pb.DATA_ROBOT_STATE, time.monotonic() - 1.0)
        unchanged = not os.path.exists(os.path.join(MAPS, NEW_MAP + ".yaml"))
        check("name taken", r_taken is not None and r_taken.result == Result.RESULT_ALREADY_EXISTS
              and state is not None and state.state == "mapping" and unchanged,
              f"save as '{map_before or 'house'}' without replace: {text(r_taken)}; robot now "
              f"{state.state if state else '?'}")

        # 4. a new name, then idle
        r_saved = save(phone, NEW_MAP, False)
        r_idle = change(phone, "idle")
        n.wait_sim(1.0)
        state = latest(phone, pb.DATA_ROBOT_STATE, time.monotonic() - 1.0)
        on_disk = os.path.exists(os.path.join(MAPS, NEW_MAP + ".yaml")) and os.path.exists(os.path.join(MAPS, NEW_MAP + ".pgm"))
        check("saved", r_saved is not None and r_saved.result == Result.RESULT_OK and r_idle is not None
              and r_idle.result == Result.RESULT_OK and state is not None and state.current_map == NEW_MAP and on_disk,
              f"save as '{NEW_MAP}': {text(r_saved)}; idle: {text(r_idle)}; map in use: "
              f"'{state.current_map if state else '?'}'; files written: {on_disk}")

        # 5. navigation on the new map, to a point 30 cm ahead (driven over while mapping)
        r_nav = change(phone, "navigation")
        n.wait_sim(2.0)
        pose = latest(phone, pb.DATA_ROBOT_POSE, time.monotonic() - 1.0)
        reached, detail = False, "no robot position from the phone's stream"
        if pose is not None:
            gx, gy = pose.x + 0.3 * math.cos(pose.heading), pose.y + 0.3 * math.sin(pose.heading)
            t5 = time.monotonic()
            r_go = ask(phone, pb.COMMAND_GO_TO, pb.GoTo(request_id=next(REQUEST_IDS), x=gx, y=gy, heading=pose.heading), 15.0)
            final = None
            end = time.monotonic() + 60.0
            while time.monotonic() < end and r_go is not None and r_go.result == Result.RESULT_OK:
                n.spin(0.05)
                for _, m in phone.messages(pb.DATA_NAVIGATION, t5):
                    if m.goal_id == r_go.goal_id and m.state in (pb.NavigationStatus.STATE_REACHED, pb.NavigationStatus.STATE_FAILED):
                        final = m.state
                if final is not None:
                    break
            reached = final == pb.NavigationStatus.STATE_REACHED
            detail = f"go to 30 cm ahead: {text(r_go)}; ended {final}"
        check("navigation", r_nav is not None and r_nav.result == Result.RESULT_OK and reached,
              f"navigation on '{NEW_MAP}': {text(r_nav)}; {detail}")

        change(phone, "idle")
        out = os.environ.get("PHASE16_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results}, f)
    finally:
        for p in phones:
            p.stop()
        # Put the usual map back: the mode manager reads current_map.txt when it starts, so it is
        # restarted (the launch file starts it again) if the map in use changed.
        with open(marker) as f:
            now_using = f.read().strip()
        remove_new_map()
        if now_using != map_before:
            with open(marker, "w") as f:
                f.write(map_before + "\n")
            subprocess.run(["pkill", "-f", "lib/vacuum_mode_manager/mode_manager"], check=False)
            print(f"map in use set back to '{map_before}' (mode manager restarted)", flush=True)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
