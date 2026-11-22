#!/usr/bin/env python3
"""Phase 41 acceptance test: the chat in the home world. Plain requests go to the assistant, as the
app sends them; what really happened is checked with the simulator's true positions and the
robot's own topics, never with the assistant's words alone.

Needs the simulator with the home world and the robot stack running, the home map, and an API key
for the assistant (~/.config/motebot/anthropic_api_key):
    tools/start_sim.sh --world home ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    VACUUM_WORLD=home python3 sim/tests/phase41_assistant_test.py
Each run costs a little: about forty calls to the language model, most with a picture.

  1. rooms learned     asked to learn the rooms, the robot finds six, visits each and names them; each true
                       room is covered by a saved room whose name fits it (kitchen, bathroom, bedroom, ...),
                       at least five of six
  2. go to a room      "go to the kitchen": the robot ends in the true kitchen
  3. go near a thing   "go up to the sofa" (from the living room): it ends within 0.9 m of the sofa, facing it
  4. what it sees      "what do you see?": a picture comes with the answer, and the answer names the sofa
  5. clean a room      "clean the bathroom": a clean of that room starts, finishes, and the robot docks
  6. is it clean       "is the bathroom clean?": answered from the robot's records (the rooms' records, the
                       history or the clean just ended), saying it was just cleaned
  7. stop wins         during "go to the bedroom", the app's Stop (idle) stops the robot within 1 s, and the
                       answer says it stopped
"""

import math
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import cleaning_client as cc  # noqa: E402
from house_layout import WORLD  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus, Rooms  # noqa: E402
from vacuum_interfaces.srv import AskAssistant, SetRooms  # noqa: E402

KINDS = {  # true room -> words, one of which its saved name should hold
    "living_room": ("living", "lounge", "hall", "family", "sitting", "dining"),
    "kitchen": ("kitchen",),
    "corridor": ("hall", "corridor", "passage", "entr", "landing", "lobby"),
    "bedroom_1": ("bed",),
    "bedroom_2": ("bed", "nursery", "kid", "child", "guest"),
    "bathroom": ("bath", "wash", "toilet", "shower", "restroom", "wc"),
}


def inside(poly, x, y):
    hit = False
    for (x0, y0), (x1, y1) in zip(poly, poly[1:] + poly[:1]):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def box_distance(box, x, y):
    x0, y0, x1, y1 = box
    return math.hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))


def main():
    from worlds import home, shapes

    if WORLD != "home":
        print("This test is for the home world: run it with VACUUM_WORLD=home")
        return [{"check": "world", "pass": False, "detail": "VACUUM_WORLD is not home"}]
    run = cc.Run("phase41")
    n, modes, clean = run.n, run.modes, run.clean
    ask_client = n.create_client(AskAssistant, "assistant/ask")
    set_rooms = n.create_client(SetRooms, "rooms/set")
    rooms_now = []
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    n.create_subscription(Rooms, "rooms", rooms_now.append, latched)
    chat = f"phase41-{uuid.uuid4().hex[:6]}"

    def ask(text, timeout=900.0, during=None):
        """The answer to `text`, or None. `during(t0)`, if given, runs while waiting (once)."""
        future = ask_client.call_async(AskAssistant.Request(conversation=chat, text=text))
        t0 = time.monotonic()
        done_during = during is None
        end = t0 + timeout
        while time.monotonic() < end and not future.done():
            run.wait(0.05)
            if not done_during:
                done_during = during(t0)
        answer = future.result() if future.done() else None
        print(f"       you: {text}\n       robot: {answer.reply if answer else '(no answer)'}  "
              f"[{', '.join(answer.tools_used) if answer else ''}; {time.monotonic() - t0:.0f} s]", flush=True)
        return answer

    def true_room():
        x, y, _ = n.pose()
        for name in home.ROOMS:
            x0, y0, x1, y1 = home.room_world(name)
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return ""

    def saved_rooms():
        n.spin(0.2)
        r = rooms_now[-1] if rooms_now else None
        return [(m.name, [(p.x, p.y) for p in m.outline.points]) for m in r.rooms] if r is not None else []

    try:
        if not run.ready():
            return run.results
        if not ask_client.wait_for_service(timeout_sec=10.0):
            run.check("assistant ready", False, "no /assistant/ask service")
            return run.results
        hello = ask("Hello! Are you there?", 120.0)
        if hello is None or not hello.success:
            run.check("assistant ready", False, f"no answer, or no API key: {hello.reply if hello else 'nothing'}")
            return run.results

        # 1. learn the rooms, from none
        if set_rooms.wait_for_service(timeout_sec=5.0):
            future = set_rooms.call_async(SetRooms.Request(map_name="home", rooms=[]))
            clean.spin_until(future.done, 10.0)
        answer = ask("Please learn the rooms of the home.", 1500.0)
        found = saved_rooms()
        named_right = []
        for true_name, words in KINDS.items():
            x0, y0, x1, y1 = home.room_world(true_name)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            best = None
            for name, poly in found:  # the saved room holding most of a grid over the true room
                hits = sum(inside(poly, x0 + (x1 - x0) * (i + 0.5) / 8, y0 + (y1 - y0) * (j + 0.5) / 8) for i in range(8) for j in range(8))
                if best is None or hits > best[1]:
                    best = (name, hits)
            ok = best is not None and best[1] > 0 and any(w in best[0].lower() for w in words)
            named_right.append((true_name, best[0] if best else "none", ok))
        right = sum(ok for _, _, ok in named_right)
        run.check("rooms learned", answer is not None and "learn_rooms" in answer.tools_used and len(found) == 6 and right >= 5,
                  f"{len(found)} rooms saved; {right} of 6 named right: " + ", ".join(f"{t} -> '{s}'" + ("" if ok else " (wrong)")
                                                                              for t, s, ok in named_right))
        names = {t: s for t, s, _ in named_right}

        # 2. go to a room
        answer = ask(f"Go to the {names['kitchen'].lower()}, please.", 600.0)
        run.check("go to a room", answer is not None and "go_to_room" in answer.tools_used and true_room() == "kitchen",
                  f"the robot is in the true {true_room() or 'nowhere'}; tools {list(answer.tools_used) if answer else []}")

        # 3. go near a thing
        sofa = [shapes.footprint(s) for t in home.things() if t.name == "sofa" for s in t.shapes]
        sofa_box = home.to_world(min(b[0] for b in sofa), min(b[1] for b in sofa)) + home.to_world(max(b[2] for b in sofa), max(b[3] for b in sofa))
        answer = ask(f"Go to the {names['living_room'].lower()} and then go up to the sofa.", 900.0)
        x, y, yaw = n.pose()
        gap = box_distance(sofa_box, x, y)
        near = (min(max(x, sofa_box[0]), sofa_box[2]), min(max(y, sofa_box[1]), sofa_box[3]))
        facing = abs(math.degrees(math.atan2(math.sin(math.atan2(near[1] - y, near[0] - x) - yaw), math.cos(math.atan2(near[1] - y, near[0] - x) - yaw))))
        run.check("go near a thing", answer is not None and "go_near" in answer.tools_used and gap <= 0.9 and facing <= 45.0,
                  f"robot centre {gap:.2f} m from the sofa, {facing:.0f} degrees off facing it; tools {list(answer.tools_used) if answer else []}")

        # 4. what it sees
        answer = ask("What do you see now?", 180.0)
        run.check("what it sees", answer is not None and "look" in answer.tools_used and len(answer.picture_jpeg) > 1000
                  and any(w in answer.reply.lower() for w in ("sofa", "couch", "settee")),
                  f"picture {len(answer.picture_jpeg) if answer else 0} bytes")

        # 5. clean a room
        bath = names["bathroom"]
        answer = ask(f"Clean the {bath.lower()} please.", 180.0)
        started = clean.wait_state((CleaningStatus.CLEANING,), 180.0, run.tick)
        target = clean.status.target if clean.status is not None else ""
        final = clean.result(started.job_id, 1500.0, run.tick) if started is not None else None
        docked = modes.wait_for_state("docked", 300.0)
        run.check("clean a room", answer is not None and "start_cleaning" in answer.tools_used and target == bath
                  and final is not None and final.state == CleaningStatus.FINISHED and docked,
                  f"clean of '{target}' {final.state if final else 'not seen'} '{final.message if final else ''}'; docked {docked}")

        # 6. is it clean
        answer = ask(f"Is the {bath.lower()} clean?", 180.0)
        reply = answer.reply.lower() if answer else ""
        looked = answer is not None and bool({"get_rooms", "get_status", "get_cleaning_history"} & set(answer.tools_used))
        run.check("is it clean", looked and any(w in reply for w in ("just", "minute", "today", "a moment ago", "recently")),
                  f"tools {list(answer.tools_used) if answer else []}")

        # 7. Stop wins
        stopped = {}

        def press_stop(t0):
            if n.speed() < 0.08 or time.monotonic() - t0 < 5.0:
                return False
            sim0 = n.sim_time
            ok, _ = modes.request("idle", 120.0)
            stopped["ok"], stopped["rest"] = ok, n.stop_time_since(sim0)
            return True

        answer = ask(f"Go to the {names['bedroom_2'].lower()}.", 600.0, during=press_stop)
        run.check("stop wins", stopped.get("ok") and stopped.get("rest") is not None and stopped["rest"] <= 1.0 and answer is not None
                  and "stop" in answer.reply.lower(),
                  f"Stop pressed while driving: {bool(stopped)}; at rest {stopped.get('rest')} s of robot time later")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
