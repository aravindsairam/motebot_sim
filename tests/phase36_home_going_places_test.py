#!/usr/bin/env python3
"""Phase 36 acceptance test: going places in the home, on its saved map.

Needs the simulator on the home and the robot stack, freshly started, the robot at its start pose,
and the home's map in use (ros2_ws/maps_home/home.*, with its stairs no-go zone):
    tools/start_sim.sh --world home ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase36_home_going_places_test.py
or: tools/run_tests.sh 36

  1. every room        from the dock, a point in each of the six rooms is reached in turn, through
                       the 0.9 m opening, the 1.1 m corridor and the 0.8 m doors: within 15 cm and
                       0.25 rad of the goal by the true position (phase 6's limits)
  2. stairs refused    a goal on the stairs outside the front door, inside the no-go zone, fails
                       and the robot stays out of the zone
  3. dock from far     from the far corner of bedroom 1 (reached first, by Nav2), about 12 m of
                       driving from the dock, docking ends docked
  4. dock from kitchen from the kitchen (reached first), docking ends docked
  5. no hazards        no bumper press, no cliff reading, no safety reflex all the way
  6. tag from afar     standing 2.5 m from the dock's tag (45 cm to the side, past the coffee
                       table), the robot's own tag detector finds it in at least half the pictures
Reported: time and distance of every trip.
"""

import math
import os
import sys
import time

os.environ["VACUUM_WORLD"] = "home"  # before house_layout is imported (by cleaning_client)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

hl = cc.hl
# Goals in the house frame, with a heading, each clear of furniture by 40 cm or more.
ROOM_GOALS = [
    ("living_room", (2.30, 2.80), 0.0),
    ("kitchen", (8.80, 1.40), 0.0),
    ("corridor", (4.60, 4.65), math.pi),
    ("bathroom", (5.50, 6.40), math.pi / 2),
    ("bedroom_1", (7.90, 5.40), 0.0),
    ("bedroom_2", (2.30, 5.60), math.pi),
]
STAIRS_GOAL = (5.70, -0.50)  # in the middle of the landing
FAR_CORNER = ("bedroom_1", (9.90, 7.10), math.pi / 2)  # between the wardrobe and the bedside table, 46 cm clear


def in_room(name, x, y):
    x0, y0, x1, y1 = hl.room_world(name)
    return x0 <= x <= x1 and y0 <= y <= y1


def main():
    run = cc.Run("phase36")
    n, modes = run.n, run.modes
    try:
        if not run.ready():
            return run.results
        if cc.current_map() != "home":
            run.check("home map", False, f"the map in use is '{cc.current_map()}', not 'home'")
            return run.results
        ok, msg = modes.request("navigation", 180.0)
        if not ok:
            run.check("every room", False, f"navigation mode: {msg}")
            return run.results
        ox, oy = run.map_offset()  # map - world, at rest at the dock
        t0 = time.monotonic()
        n.monitoring = True

        def go(house_xy, yaw, timeout=240.0):
            wx, wy = hl.to_world(*house_xy)
            status, secs, dist = n.navigate(wx + ox, wy + oy, yaw, sim_timeout=timeout)
            n.wait_sim(1.0)
            tx, ty, tyaw = n.pose()
            return status, secs, dist, math.hypot(tx - wx, ty - wy), abs(math.atan2(math.sin(tyaw - yaw), math.cos(tyaw - yaw)))

        # 1. every room in turn
        legs, all_ok = [], True
        for room, xy, yaw in ROOM_GOALS:
            status, secs, dist, err, yerr = go(xy, yaw)
            tx, ty, _ = n.pose()
            good = status == "succeeded" and err < 0.15 and yerr < 0.25 and in_room(room, tx, ty)
            all_ok &= good
            legs.append(f"{room}: {status} in {secs:.0f} s, {dist:.1f} m, {err * 100:.0f} cm / {math.degrees(yerr):.0f} deg off")
        run.check("every room", all_ok, "; ".join(legs))

        # 2. the stairs
        p0 = n.pose()
        wx, wy = hl.to_world(*STAIRS_GOAL)
        status, secs, dist = n.navigate(wx + ox, wy + oy, -math.pi / 2, sim_timeout=120.0)
        p1 = n.pose()
        on_stairs = hl.to_house(*p1[:2])[1] < 0.0
        run.check("stairs refused", status in ("aborted", "rejected") and not on_stairs,
                  f"goal on the stairs: {status} after {secs:.0f} s; the robot moved {math.hypot(p1[0] - p0[0], p1[1] - p0[1]) * 100:.0f} cm "
                  f"and is {'ON' if on_stairs else 'not on'} the stairs")

        # 3. dock from the far corner of bedroom 1 (the check counts only if the robot got there)
        room, xy, yaw = FAR_CORNER
        status, secs, dist, err, _ = go(xy, yaw)
        there = status == "succeeded" and in_room(room, *n.pose()[:2])
        t = time.monotonic()
        ok_dock, msg_dock = modes.request("docking", 240.0)
        docked = modes.wait_for_state("docked", 240.0)
        run.check("dock from far", there and docked, f"to the far corner of bedroom 1: {status}, {err * 100:.0f} cm off; "
                  f"docking '{msg_dock}', "
                  f"{'docked' if docked else 'NOT docked: ' + modes.state.state + ' ' + modes.state.reason} "
                  f"after {time.monotonic() - t:.0f} s wall")

        # 4. dock from the kitchen
        ok, msg = modes.request("navigation", 180.0)
        status, secs, dist, err, _ = go(ROOM_GOALS[1][1], 0.0)
        there = ok and status == "succeeded" and in_room("kitchen", *n.pose()[:2])
        t = time.monotonic()
        ok_dock, msg_dock = modes.request("docking", 240.0)
        docked = modes.wait_for_state("docked", 240.0)
        run.check("dock from kitchen", there and docked, f"navigation '{msg}'; to the kitchen: {status}, {err * 100:.0f} cm off; docking '{msg_dock}', "
                  f"{'docked' if docked else 'NOT docked: ' + modes.state.state + ' ' + modes.state.reason} "
                  f"after {time.monotonic() - t:.0f} s wall")

        # 6. the tag from 2.5 m, seen by the robot's own detector (apriltag_ros publishes the tag's
        # frame only while it sees it)
        counts = {"tag": 0, "pictures": 0}
        n.create_subscription(TFMessage, "tf", lambda m: counts.__setitem__(
            "tag", counts["tag"] + sum(1 for t in m.transforms if t.child_frame_id == "dock_tag")), 50)
        n.create_subscription(Image, "camera/image_raw", lambda m: counts.__setitem__("pictures", counts["pictures"] + 1), 5)
        ok, msg = modes.request("navigation", 180.0)
        tag_x, tag_y = hl.DOCK_TAG_HOUSE[:2]
        spot = (tag_x + 2.5 + hl.CAMERA_XYZ[0], tag_y + 0.45)
        facing = math.atan2(tag_y - spot[1], tag_x - spot[0])
        status, secs, dist, err, yerr = go(spot, facing)
        n.wait_sim(1.0)
        counts.update(tag=0, pictures=0)
        n.wait_sim(4.0)
        seen = counts["tag"] / max(1, counts["pictures"])
        run.check("tag from afar", status == "succeeded" and counts["pictures"] > 10 and seen >= 0.5,
                  f"at 2.5 m, {math.degrees(math.atan2(0.45, 2.5)):.0f} deg to the side ({status}, {err * 100:.0f} cm off): "
                  f"the tag was found in {counts['tag']} of {counts['pictures']} pictures")

        bumps, cliffs, reflexes = run.hazards_since(t0)
        run.check("no hazards", (bumps, cliffs, reflexes) == (0, 0, 0) and n.bumper_seen == 0,
                  f"bumper {bumps}, cliff {cliffs}, reflex {reflexes} safety reports; bumper bits seen {n.bumper_seen}")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
