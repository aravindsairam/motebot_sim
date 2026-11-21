#!/usr/bin/env python3
"""Phase 35 acceptance test: the robot maps the whole home by itself.

Needs the simulator on the home and the robot stack, freshly started, the robot at its start pose
in front of the dock (the new map's frame is where the odometry starts):
    tools/start_sim.sh --world home ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase35_home_exploration_test.py [--save-as home]
or: tools/run_tests.sh 35

  1. explores the home   from idle at the dock, with a new map: the exploration ends by itself as
                         finished (not early) within 20 minutes of robot time
  2. comes back          the robot ends within 30 cm of where it started
  3. no bump, no fall    no bumper press; the robot never leaves the floor
  4. wall accuracy  )    the saved map passes phase 4's limits, judged against the surfaces the
  5. completeness   )    lidar could see from where the robot really drove
  6. all six rooms       in each room at least 90 % of its open floor is free on the map (open
                         floor: joined to the dock's floor, 15 cm from anything; the inside of the
                         glass shower is not)
  Reported: time, distance, goals sent and refused, cliff readings, where the stairs are on the map.
  Kept in logs/: phase35_map.* and the true track phase35_track.json. The test map is removed and
  the map in use set back, unless --save-as NAME is given: then the map is also saved as NAME and
  becomes the map in use (that is how the home's own map, kept in git, was made).
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

os.environ["VACUUM_WORLD"] = "home"  # before house_layout is imported (by cleaning_client)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import cleaning_client as cc  # noqa: E402
import exploration_client as ec  # noqa: E402
import house_layout as hl  # noqa: E402
from phase4_mapping_test import load_map, score_map  # noqa: E402
from phase30_whole_house_exploration_test import places_driven  # noqa: E402
from vacuum_interfaces.msg import ExplorationStatus  # noqa: E402
from worlds import reach  # noqa: E402

TEST_MAP = "explored_test"
ROBOT_TIME_LIMIT = 20 * 60.0  # s: the exploration node's own time limit (exploration.yaml)
LOGS = os.path.join(cc.ROOT, "logs")


def remove_test_map():
    for path in glob.glob(os.path.join(cc.MAPS, TEST_MAP + ".*")) + glob.glob(os.path.join(cc.MAPS, TEST_MAP + "_*")):
        if os.path.isdir(path):
            subprocess.run(["rm", "-rf", path], check=False)
        else:
            os.remove(path)


def open_floor_world():
    """A function telling, for world-frame points, whether each is open floor: joined to the floor
    the robot starts on (so not inside the glass shower or under a solid base), and at least 15 cm
    from anything the lidar plane crosses."""
    from scipy import ndimage

    x0, y0, blocked = reach.blocked_grid(hl)
    labels, _ = ndimage.label(~blocked)
    sx, sy = hl.START_POSE_HOUSE[:2]
    joined = labels == labels[int((sy - y0) / reach.RES), int((sx - x0) / reach.RES)]
    clear = ndimage.distance_transform_edt(~blocked) * reach.RES > 0.15
    good = joined & clear

    def at(wx, wy):
        hx, hy = wx + hl.START_POSE_HOUSE[0], wy + hl.START_POSE_HOUSE[1]
        ix = np.clip(((hx - x0) / reach.RES).astype(int), 0, good.shape[1] - 1)
        iy = np.clip(((hy - y0) / reach.RES).astype(int), 0, good.shape[0] - 1)
        return good[iy, ix]

    return at


def room_free_share(stem):
    """Per room, the share of its open floor that is free on the saved map."""
    img, res, origin = load_map(stem)
    h, w = img.shape
    rows, cols = np.mgrid[0:h, 0:w]
    xs = origin[0] + (cols + 0.5) * res
    ys = origin[1] + (h - rows - 0.5) * res
    open_floor = open_floor_world()(xs, ys)
    out = {}
    for name in hl.ROOMS:
        x0, y0, x1, y1 = hl.room_world(name)
        inside = open_floor & (xs > x0) & (xs < x1) & (ys > y0) & (ys < y1)
        out[name] = float(np.mean(img[inside] > 250)) if inside.any() else 0.0
    return out


def main(save_as):
    run = cc.Run("phase35")
    n, modes = run.n, run.modes
    ex = ec.Exploring(n)
    marker = os.path.join(cc.MAPS, "current_map.txt")
    map_before = open(marker).read().strip() if os.path.isfile(marker) else ""
    remove_test_map()
    report = {}
    try:
        if not run.ready():
            return run.results
        if not ex.spin_until(lambda: ex.status is not None, 15.0):
            run.check("exploration node", False, "no /exploration/status: is the exploration node running?")
            return run.results

        # 1-3. the whole home
        start = n.pose()
        n.monitoring = True
        n.bumper_seen = 0
        n.max_abs_z, n.max_tilt = 0.0, 0.0
        t_wall = time.monotonic()
        answer = ex.start()
        job = answer.job_id if answer is not None and answer.accepted else None
        final = ex.result(job, 3000.0, run.tick) if job else None
        bumps, cliffs, reflexes = run.hazards_since(t_wall)
        end = n.pose()
        back = math.hypot(end[0] - start[0], end[1] - start[1])
        report.update(time_min=(final.exploring_time if final else 0.0) / 60.0, distance=final.distance if final else 0.0,
                      goals=final.goals_sent if final else 0, failed=final.goals_failed if final else 0,
                      cliff_readings=n.cliff_trips, wall_minutes=(time.monotonic() - t_wall) / 60.0)
        finished = final is not None and final.state == ExplorationStatus.FINISHED and "early" not in final.message
        run.check("explores the home",
                  finished and final.exploring_time <= ROBOT_TIME_LIMIT and modes.state.state == "mapping",
                  f"start: '{answer.message if answer else 'no answer'}'; ended {final.state if final else 'never'}: "
                  f"'{final.message if final else ''}' after {report['time_min']:.1f} min of robot time "
                  f"({report['wall_minutes']:.1f} min wall), {report['distance']:.1f} m, goals {report['goals']} sent, "
                  f"{report['failed']} failed; mode {modes.state.state}")
        run.check("comes back", final is not None and back <= 0.30,
                  f"ends {back * 100:.0f} cm from where it started (true positions)")
        run.check("no bump, no fall", n.bumper_seen == 0 and bumps == 0 and n.max_abs_z < 0.03 and n.max_tilt < 5.0,
                  f"bumper presses {bumps} (bits seen {n.bumper_seen}); cliff readings {n.cliff_trips}, safety reflexes "
                  f"{reflexes}; highest |z| {n.max_abs_z * 100:.1f} cm, largest tilt {n.max_tilt:.1f} deg")
        n.monitoring = False

        # 4-6. the map
        ok, msg, _path = modes.save_map(TEST_MAP)
        stem = os.path.join(cc.MAPS, TEST_MAP)
        if ok:
            with open(os.path.join(LOGS, "phase35_track.json"), "w") as f:
                json.dump(n.track, f)
            score = score_map(stem, run.check, stops=places_driven(n.track))
            for ext in (".pgm", ".yaml", ".png"):
                shutil.copy(stem + ext, os.path.join(LOGS, "phase35_map" + ext))
            rooms = room_free_share(stem)
            run.check("all six rooms", len(rooms) == 6 and all(v >= 0.90 for v in rooms.values()),
                      "; ".join(f"{name}: {v * 100:.0f}% of its open floor is free" for name, v in rooms.items()))
            img, res, origin = load_map(stem)
            report["map"] = f"{img.shape[1]}x{img.shape[0]} cells; walls median {score['median_cm']:.1f} cm"
            if save_as:
                ok2, msg2, _ = modes.save_map(save_as)
                print(f"saved as '{save_as}': {ok2} {msg2}", flush=True)
        else:
            for name in ("wall accuracy", "completeness", "all six rooms"):
                run.check(name, False, f"save: {msg}")
        print("REPORT " + ", ".join(f"{k}: {v:.1f}" if isinstance(v, float) else f"{k}: {v}" for k, v in report.items()),
              flush=True)
    finally:
        results = run.finish()
        if not save_as:
            # Put the map in use back: the mode manager reads current_map.txt when it starts, so it
            # is restarted (the launch file starts it again) if the map in use changed.
            now_using = open(marker).read().strip() if os.path.isfile(marker) else ""
            if now_using != map_before:
                if map_before:
                    with open(marker, "w") as f:
                        f.write(map_before + "\n")
                elif os.path.isfile(marker):
                    os.remove(marker)
                subprocess.run(["pkill", "-f", "lib/vacuum_mode_manager/mode_manager"], check=False)
                print(f"map in use set back to '{map_before or 'none'}' (mode manager restarted)", flush=True)
        remove_test_map()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-as", default="", help="also save the explored map under this name, as the map in use")
    sys.exit(cc.exit_code(main(parser.parse_args().save_as)))
