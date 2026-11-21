#!/usr/bin/env python3
"""Phase 37 acceptance test: the whole home, from the dock back to the dock.
About 30 to 45 minutes of robot time (45 to 70 minutes on this PC).

Needs the simulator on the home and the robot stack running, the robot at its start pose, the
home's map in use:
    tools/start_sim.sh --world home ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase37_whole_home_test.py
or: tools/run_tests.sh 37

The robot docks first, then cleans everything. Measured with the simulator's true position:
  1. cleaned         the job finishes; >= 95 % of the planned floor cleaned by the true track; the
                     share of ALL open floor (from the layout) is reported
  2. every room      >= 95 % of the planned floor of each of the six rooms, and >= 40 % of each
                     room's true open floor (from the layout), so a room the planner mostly leaves
                     out fails. With edges off the planner covers 49 % (bathroom) to 66 % (corridor)
                     of a room's open floor: the band along walls and furniture is not swept
                     (measured offline on the home's map, 2026-11-20). 40 % is a first limit.
  3. stairs          the robot's true centre never enters the stairs no-go zone, and stays at
                     least 15 cm from it on rows
  4. no hazards      no bumper press, no cliff reading, no safety reflex, no row skipped
  5. docked          the robot ends on the dock
  6. on time         the cleaning time is within 25 % of the estimate given before the start
Reported: the planner's regions against the true rooms (which room each region's rows lie in).
"""

import collections
import os
import sys
import time

os.environ["VACUUM_WORLD"] = "home"  # before house_layout is imported (by cleaning_client)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_interfaces.msg import CleaningPiece, CleaningStatus  # noqa: E402

hl = cc.hl


def room_of(x, y):
    for name in hl.ROOMS:
        x0, y0, x1, y1 = hl.room_world(name)
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return "between rooms"


def regions_against_rooms(plan, offset):
    """For each planner region, the true rooms its rows lie in, by length (rows judged at their middle)."""
    out = collections.defaultdict(collections.Counter)
    for p in plan.pieces:
        if p.kind != CleaningPiece.ROW:
            continue
        mx, my = (p.x0 + p.x1) / 2.0 - offset[0], (p.y0 + p.y1) / 2.0 - offset[1]
        out[p.region][room_of(mx, my)] += float(np.hypot(p.x1 - p.x0, p.y1 - p.y0))
    return out


def main():
    run = cc.Run("phase37")
    modes, clean = run.modes, run.clean
    try:
        if not run.ready():
            return run.results
        if cc.current_map() != "home":
            run.check("home map", False, f"the map in use is '{cc.current_map()}', not 'home'")
            return run.results
        ok, msg = modes.request("docking", 150.0)
        docked = modes.wait_for_state("docked", 150.0)
        if not docked:
            run.check("docked first", False, f"docking: {msg}; mode {modes.state.state} '{modes.state.reason}'")
            return run.results
        offset = run.map_offset()
        preview = clean.preview(cc.job(stay=False))
        estimate = preview.estimate.time if preview and preview.success else 0.0
        run.truth = cc.Truth(run.m)
        t0 = time.monotonic()
        status = run.run_job(cc.job(stay=False), 4800.0)
        plan = clean.job_plan(status.job_id) if status else None
        planned = cc.planned_floor(run.m, plan) if plan is not None else np.zeros(run.m.grid.shape, dtype=bool)
        share = run.truth.share(planned)
        open_floor = run.truth.share(cc.true_floor(run.m), allowance=False)
        run.check("cleaned", status is not None and status.state == CleaningStatus.FINISHED and share >= 95.0,
                  f"{cc.summary_line(status) if status else 'no result'}; true share of the planned floor {run.truth.shares(planned)}, "
                  f"of all open floor {open_floor:.1f}% (reported, not a limit)")

        # 2. each room's planned floor (map cells judged by where they truly are)
        g = run.m.grid
        xs = g.origin[0] + (np.arange(g.shape[1]) + 0.5) * g.resolution - offset[0]
        ys = g.origin[1] + (np.arange(g.shape[0]) + 0.5) * g.resolution - offset[1]
        X, Y = np.meshgrid(xs, ys)
        per_room, of_floor = {}, {}
        true_floor = cc.true_floor(run.m)
        for name in hl.ROOMS:
            x0, y0, x1, y1 = hl.room_world(name)
            inside = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
            mask = planned & inside
            per_room[name] = run.truth.share(mask) if mask.any() else 0.0
            open_room = true_floor & inside
            of_floor[name] = run.truth.share(open_room, allowance=False) if open_room.any() else 0.0
        run.check("every room", len(per_room) == 6 and all(v >= 95.0 for v in per_room.values())
                  and all(v >= 40.0 for v in of_floor.values()),
                  "; ".join(f"{name} {per_room[name]:.1f}% of its planned floor, {of_floor[name]:.0f}% of its open floor"
                            for name in per_room))

        rows = cc.on_rows(run.truth.track, cc.pieces_of(plan)) if plan is not None else []
        on_rows = min((cc.zone_distance(x, y) for (_, x, y, _), _ in rows), default=float("inf"))
        run.check("stairs", run.truth.closest_zone > 0.0 and on_rows >= 0.15,
                  f"the centre came within {run.truth.closest_zone * 100:.1f} cm of the stairs zone (must stay outside), "
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

        if plan is not None:
            regions = regions_against_rooms(plan, offset)
            parts = []
            for region in sorted(regions):
                rooms = regions[region]
                total = sum(rooms.values())
                parts.append(f"region {region}: " + ", ".join(f"{room} {length / total * 100:.0f}%"
                                                               for room, length in rooms.most_common()))
            whole = {room for r in regions.values() for room, length in r.items() if length / sum(r.values()) >= 0.9}
            print(f"REPORT the planner made {len(regions)} regions; " + "; ".join(parts)
                  + f"; rooms that have a region of their own (90 % or more of its rows): {sorted(whole)}", flush=True)
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
