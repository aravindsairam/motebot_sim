#!/usr/bin/env python3
"""Phase 30 acceptance test: the robot maps the whole house by itself (exploration plan, step 3).

Needs the simulator and the robot stack running, freshly started, the robot at its start pose in
front of the dock (the new map's frame is where the odometry starts, as for a map made by hand):
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase30_whole_house_exploration_test.py

  1. explores the house  from idle at the dock, with a new map: the exploration ends by itself as
                         finished (not early) within 12 minutes of robot time
  2. comes back          the robot ends within 30 cm of where it started
  3. no bump, no fall    no bumper press; the robot never leaves the floor
  4. nothing left        a second start on the finished map ends as finished at once: no goal sent,
                         within 30 s of robot time
  5. wall accuracy  )    the map saved from the first exploration passes phase 4's limits, judged
  6. completeness   )    against the surfaces the lidar could see from where the robot really drove
                         (phase 4's own tour never sees, say, the far side of the wardrobe)
  7. both rooms          each room's open floor is at least 90% free on the map
  8. a usable map        on the saved map: a Go to into the bedroom arrives, a docking ends docked,
                         and a cleaning preview plans rows in both rooms
  Reported: time, distance, goals sent and refused, cliff readings, whether the stairwell alcove
  is floor on the map, and where the explorer went on the second start. Kept for a look afterwards
  in logs/: the saved map (phase30_first.*), the map after the second start (phase30_after.*), the
  true track (phase30_track.json). The map in use is set back to what it was, and the test map removed.
"""

import glob
import json
import math
import shutil
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import cleaning_client as cc  # noqa: E402
import exploration_client as ec  # noqa: E402
import house_layout as hl  # noqa: E402
from phase4_mapping_test import load_map, score_map  # noqa: E402
from vacuum_interfaces.msg import CleaningPiece, ExplorationStatus  # noqa: E402
from vacuum_mode_manager import map_files  # noqa: E402

TEST_MAP = "explored_test"
ROBOT_TIME_LIMIT = 12 * 60.0  # s
BEDROOM_GOAL_HOUSE = (6.0, 1.6)  # the bedroom point phase 14 uses: clear of the bed and the wardrobe
LOGS = os.path.join(cc.ROOT, "logs")
NOTHING_LEFT_TIME = 30.0  # s of robot time: two looks (6 to 8 s each), the mode changes, a short trip back


def remove_test_map():
    for path in glob.glob(os.path.join(cc.MAPS, TEST_MAP + ".*")) + glob.glob(os.path.join(cc.MAPS, TEST_MAP + "_*")):
        if os.path.isdir(path):
            subprocess.run(["rm", "-rf", path], check=False)
        else:
            os.remove(path)


def places_driven(track, spacing=0.3):
    """The true track thinned to one point every `spacing` metres: where the lidar looked from."""
    out = []
    for x, y in track:
        if not out or math.hypot(x - out[-1][0], y - out[-1][1]) >= spacing:
            out.append((x, y))
    return out


def room_free_share(stem):
    """Per room, the share of its open floor (inside, clear of furniture, 15 cm from walls) that
    is free on the saved map."""
    img, res, origin = load_map(stem)
    h, w = img.shape
    rows, cols = np.mgrid[0:h, 0:w]
    xs = origin[0] + (cols + 0.5) * res
    ys = origin[1] + (h - rows - 0.5) * res
    boxes, cyls = hl.shapes_at_height(hl.LIDAR_XYZ[2])
    out = {}
    for name in hl.ROOMS:
        x0, y0, x1, y1 = hl.room_world(name)
        inside = (xs > x0 + 0.15) & (xs < x1 - 0.15) & (ys > y0 + 0.15) & (ys < y1 - 0.15)
        for bx0, by0, bx1, by1 in boxes:
            inside &= ~((xs > bx0 - 0.15) & (xs < bx1 + 0.15) & (ys > by0 - 0.15) & (ys < by1 + 0.15))
        for cx, cy, r in cyls:
            inside &= np.hypot(xs - cx, ys - cy) > r + 0.15
        out[name] = float(np.mean(img[inside] > 250)) if inside.any() else 0.0
    return out


def main():
    run = cc.Run("phase30")
    n, modes, clean = run.n, run.modes, run.clean
    ex = ec.Exploring(n)
    marker = os.path.join(cc.MAPS, "current_map.txt")
    with open(marker) as f:
        map_before = f.read().strip()
    remove_test_map()
    report = {}
    explorer_log = os.path.join(LOGS, "frontier_explorer.log")

    def explorer_targets(since_bytes):
        """Where the explorer sent the robot since that point in its log (its own words: "Sending
        frontier goal ... dispatch=(x, y), size=n")."""
        try:
            with open(explorer_log, "rb") as f:
                f.seek(since_bytes)
                lines = f.read().decode(errors="replace").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            if "Sending frontier goal" in line and "dispatch=(" in line:
                out.append(line.split("dispatch=")[1].split(")")[0] + ")" + line.split(")")[-1])
        return out

    try:
        if not run.ready():
            return run.results
        if not ex.spin_until(lambda: ex.status is not None, 15.0):
            run.check("exploration node", False, "no /exploration/status: is the exploration node running?")
            return run.results

        # 1-3. the whole house
        start = n.pose()
        n.monitoring = True
        n.bumper_seen = 0
        n.max_abs_z, n.max_tilt = 0.0, 0.0
        t_wall = time.monotonic()
        answer = ex.start()
        job = answer.job_id if answer is not None and answer.accepted else None
        final = ex.result(job, 1800.0, run.tick) if job else None
        bumps, cliffs, reflexes = run.hazards_since(t_wall)
        end = n.pose()
        back = math.hypot(end[0] - start[0], end[1] - start[1])
        report.update(time=final.exploring_time if final else 0.0, distance=final.distance if final else 0.0,
                      goals=final.goals_sent if final else 0, failed=final.goals_failed if final else 0,
                      cliff_readings=n.cliff_trips, wall_minutes=(time.monotonic() - t_wall) / 60.0)
        finished = final is not None and final.state == ExplorationStatus.FINISHED and "early" not in final.message
        run.check("explores the house",
                  finished and final.exploring_time <= ROBOT_TIME_LIMIT and modes.state.state == "mapping",
                  f"start: '{answer.message if answer else 'no answer'}'; ended {final.state if final else 'never'}: "
                  f"'{final.message if final else ''}' after {final.exploring_time / 60 if final else 0:.1f} min of robot time "
                  f"({report['wall_minutes']:.1f} min wall), {report['distance']:.1f} m, goals {report['goals']} sent, "
                  f"{report['failed']} failed; mode {modes.state.state}")
        run.check("comes back", final is not None and back <= 0.30,
                  f"ends {back * 100:.0f} cm from where it started (true positions)")
        run.check("no bump, no fall", n.bumper_seen == 0 and bumps == 0 and n.max_abs_z < 0.03 and n.max_tilt < 5.0,
                  f"bumper presses {bumps} (bits seen {n.bumper_seen}); cliff readings {n.cliff_trips}, safety reflexes "
                  f"{reflexes}; highest |z| {n.max_abs_z * 100:.1f} cm, largest tilt {n.max_tilt:.1f} deg")
        n.monitoring = False

        # 5-7. the map
        ok, msg, path = modes.save_map(TEST_MAP)
        stem = os.path.join(cc.MAPS, TEST_MAP)
        if ok:
            for ext in (".pgm", ".yaml"):
                shutil.copy(stem + ext, os.path.join(LOGS, "phase30_first" + ext))
            with open(os.path.join(LOGS, "phase30_track.json"), "w") as f:
                json.dump(n.track, f)
            score = score_map(stem, run.check, stops=places_driven(n.track))
            shutil.copy(stem + ".png", os.path.join(LOGS, "phase30_first.png"))
            rooms = room_free_share(stem)
            run.check("both rooms", all(v >= 0.90 for v in rooms.values()),
                      "; ".join(f"{name}: {v * 100:.0f}% of its open floor is free" for name, v in rooms.items()))
            img, res, origin = load_map(stem)
            report["map"] = f"{img.shape[1]}x{img.shape[0]} cells; walls median {score['median_cm']:.1f} cm"
        else:
            run.check("wall accuracy", False, f"save: {msg}")
            run.check("completeness", False, f"save: {msg}")
            run.check("both rooms", False, f"save: {msg}")

        # 4. nothing left: a second start on the finished map
        p0 = n.pose()
        t = time.monotonic()
        log_at = os.path.getsize(explorer_log) if os.path.isfile(explorer_log) else 0
        again = ex.start()
        job2 = again.job_id if again is not None and again.accepted else None
        final2 = ex.result(job2, 240.0, run.tick) if job2 else None
        took = time.monotonic() - t
        targets = explorer_targets(log_at)
        p1 = n.pose()
        moved = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        run.check("nothing left",
                  final2 is not None and final2.state == ExplorationStatus.FINISHED and final2.goals_sent == 0
                  and final2.exploring_time <= NOTHING_LEFT_TIME,
                  f"second start: '{again.message if again else 'no answer'}'; ended {final2.state if final2 else 'never'}: "
                  f"'{final2.message if final2 else ''}' after {final2.exploring_time if final2 else 0:.0f} s of robot time "
                  f"({took:.0f} s wall), goals sent {final2.goals_sent if final2 else '?'}; the robot ended {moved * 100:.0f} cm "
                  f"from where it began; the explorer's targets: {targets or 'none'}")
        after = ex.fresh_map()
        if after is not None and ex.map is not None:
            m = ex.map
            image = map_files.grid_to_image(m.data, m.info.width, m.info.height)
            map_files.write_map(os.path.join(LOGS, "phase30_after"), image, m.info.resolution,
                                (m.info.origin.position.x, m.info.origin.position.y))

        # 8. use it: Go to into the bedroom, dock, plan a clean
        details = []
        usable = ok
        if ok:
            ok_nav, msg_nav = modes.request("navigation", 180.0)
            gx, gy = hl.to_world(*BEDROOM_GOAL_HOUSE)
            status, secs, dist = n.navigate(gx, gy, 0.0, sim_timeout=180.0)
            tx, ty, _ = n.pose()
            arrived = status == "succeeded" and math.hypot(tx - gx, ty - gy) < 0.30
            details.append(f"navigation: '{msg_nav}'; Go to the bedroom: {status} in {secs:.0f} s, ends "
                           f"{math.hypot(tx - gx, ty - gy) * 100:.0f} cm from it (true position)")
            ok_dock, msg_dock = modes.request("docking", 240.0)
            docked = modes.wait_for_state("docked", 180.0)
            details.append(f"docking: '{msg_dock}', docked {docked}")
            reply = clean.preview(cc.job())
            clean.spin_until(lambda: clean.plan is not None and clean.plan.job_id == 0 and len(clean.plan.pieces) > 0, 10.0)
            rows = [p for p in (clean.plan.pieces if clean.plan else []) if p.kind == CleaningPiece.ROW]
            split = hl.to_world((hl.LIVING_ROOM[2] + hl.BEDROOM[0]) / 2.0, 0.0)[0]
            living = sum(1 for p in rows if max(p.x0, p.x1) < split)
            bedroom = sum(1 for p in rows if min(p.x0, p.x1) > split)
            details.append(f"cleaning preview: {reply.message if reply else 'no answer'}; rows in the living room "
                           f"{living}, in the bedroom {bedroom}")
            usable = ok_nav and arrived and ok_dock and docked and reply is not None and reply.success and living > 0 and bedroom > 0
        run.check("a usable map", usable, "; ".join(details) if details else f"not saved: {msg}")
        print("REPORT " + ", ".join(f"{k}: {v:.1f}" if isinstance(v, float) else f"{k}: {v}" for k, v in report.items()),
              flush=True)
    finally:
        results = run.finish()
        # Put the usual map back: the mode manager reads current_map.txt when it starts, so it is
        # restarted (the launch file starts it again) if the map in use changed.
        with open(marker) as f:
            now_using = f.read().strip()
        if now_using != map_before:
            with open(marker, "w") as f:
                f.write(map_before + "\n")
            subprocess.run(["pkill", "-f", "lib/vacuum_mode_manager/mode_manager"], check=False)
            print(f"map in use set back to '{map_before}' (mode manager restarted)", flush=True)
        remove_test_map()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
