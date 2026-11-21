#!/usr/bin/env python3
"""Phase 20 acceptance test: the whole house, from the dock back to the dock (coverage step 4).
About 15 to 25 minutes of robot time (20 to 30 minutes on this PC).

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase20_whole_house_test.py

The robot docks first, then cleans everything. Measured with the simulator's true position:
  1. cleaned         the job finishes; >= 95 % of the planned floor cleaned by the true track; the
                     share of ALL open floor (from house_layout.py) is reported
  2. both rooms      >= 95 % of the planned floor of each room
  3. stairwell       the robot's true centre never enters the stairwell no-go zone, and stays at
                     least 15 cm from it on rows (Nav2 keeps the centre out of a zone; rows keep the body out)
  4. no hazards      no bumper press, no cliff reading, no safety reflex, no row skipped
  5. docked          the robot ends on the dock
  6. on time         the cleaning time is within 25 % of the estimate given before the start
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

DIVIDER_X = cc.hl.to_world(5.05, 0.0)[0]  # the dividing wall, map x


def main():
    run = cc.Run("phase20")
    modes, clean = run.modes, run.clean
    try:
        if not run.ready():
            return run.results
        ok, msg = modes.request("docking", 150.0)
        docked = modes.wait_for_state("docked", 150.0)
        if not docked:
            run.check("docked first", False, f"docking: {msg}; mode {modes.state.state} '{modes.state.reason}'")
            return run.results
        preview = clean.preview(cc.job(stay=False))
        estimate = preview.estimate.time if preview and preview.success else 0.0
        run.truth = cc.Truth(run.m)
        t0 = time.monotonic()
        status = run.run_job(cc.job(stay=False), 3600.0)
        plan = clean.job_plan(status.job_id) if status else None
        planned = cc.planned_floor(run.m, plan) if plan is not None else np.zeros(run.m.grid.shape, dtype=bool)
        share = run.truth.share(planned)
        open_floor = run.truth.share(cc.true_floor(run.m), allowance=False)
        run.check("cleaned", status is not None and status.state == CleaningStatus.FINISHED and share >= 95.0,
                  f"{cc.summary_line(status) if status else 'no result'}; true share of the planned floor {run.truth.shares(planned)}, "
                  f"of all open floor {open_floor:.1f}% (reported, not a limit)")

        cols = run.m.grid.origin[0] + (np.arange(run.m.grid.shape[1]) + 0.5) * run.m.grid.resolution
        living = planned & (cols[None, :] < DIVIDER_X)
        bedroom = planned & (cols[None, :] > DIVIDER_X)
        lv, bd = run.truth.share(living), run.truth.share(bedroom)
        run.check("both rooms", lv >= 95.0 and bd >= 95.0, f"living room {lv:.1f}%, bedroom {bd:.1f}% of their planned floor")

        rows = cc.on_rows(run.truth.track, cc.pieces_of(plan)) if plan is not None else []
        on_rows = min((cc.zone_distance(x, y) for (_, x, y, _), _ in rows), default=float("inf"))
        run.check("stairwell", run.truth.closest_zone > 0.0 and on_rows >= 0.15,
                  f"the centre came within {run.truth.closest_zone * 100:.1f} cm of the stairwell zone (must stay outside), "
                  f"{on_rows * 100:.1f} cm on rows (at least 15)")

        bumps, cliffs, reflexes = run.hazards_since(t0)
        skipped = status.rows_skipped if status else -1
        run.check("no hazards", (bumps, cliffs, reflexes, skipped) == (0, 0, 0, 0),
                  f"bumper {bumps}, cliff {cliffs}, reflex {reflexes} safety reports; rows skipped {skipped}")

        docked = modes.wait_for_state("docked", 300.0)
        run.check("docked", docked, f"mode at the end: {modes.state.state} '{modes.state.reason}'")

        took = status.cleaning_time if status else 0.0
        off = abs(took - estimate) / estimate if estimate else 1.0
        run.check("on time", off <= 0.25, f"cleaned for {took / 60:.1f} min of robot time; estimate before the start "
                  f"{estimate / 60:.1f} min ({off * 100:.0f}% off, limit 25%)")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
