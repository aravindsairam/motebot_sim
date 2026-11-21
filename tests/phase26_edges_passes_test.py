#!/usr/bin/env python3
"""Phase 26 acceptance test: the pass along the edges, and cleaning twice (coverage step 11).

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase26_edges_passes_test.py [--house]

The edges are measured in the living room's north-west corner (cleaning_client.CORNER_AREA), against
two walls; the second pass on the small rectangle. With the simulator's true position:
  1. edges          with edges on, the share of all open floor in the corner that is cleaned rises by
                    at least 5 points over rows only
  2. edges safely   no bumper press, no cliff reading and no safety reflex while driving the edges
  3. second pass    every row of the second pass is at right angles to the first
  4. twice as far   the distance driven is 1.6 to 2.6 times that of one pass
  5. (--house)      one whole-house clean with edges on, reported
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_coverage.grid import polygon_mask  # noqa: E402
from vacuum_coverage.planner import ROW  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402


def main():
    run = cc.Run("phase26")
    clean = run.clean
    try:
        if not run.ready():
            return run.results
        corner_floor = cc.true_floor(run.m) & polygon_mask(run.m.grid, cc.rect(*cc.CORNER_AREA))

        # 1-2. rows only, then rows and edges, in the corner
        shares, statuses, hazards = [], [], None
        for edges in (False, True):
            run.truth = cc.Truth(run.m)
            t0 = time.monotonic()
            status = run.run_job(cc.job("area", area=cc.CORNER_AREA, edges=edges), 900.0)
            shares.append(run.truth.share(corner_floor, allowance=False))
            statuses.append(status)
            if edges:
                hazards = run.hazards_since(t0)
        gain = shares[1] - shares[0]
        done = all(s is not None and s.state == CleaningStatus.FINISHED for s in statuses)
        run.check("edges", done and gain >= 5.0 and statuses[1].edges_total > 0,
                  f"open floor of the corner cleaned: rows only {shares[0]:.1f}%, with edges {shares[1]:.1f}% (+{gain:.1f} points); "
                  f"edges driven {statuses[1].edges_done if statuses[1] else 0}/{statuses[1].edges_total if statuses[1] else 0}, "
                  f"skipped {statuses[1].edges_skipped if statuses[1] else 0}")
        run.check("edges safely", hazards == (0, 0, 0) and statuses[1] is not None and statuses[1].bumps == 0,
                  f"bumper, cliff, reflex safety reports while cleaning with edges: {hazards}")

        # 3-4. one pass, then two
        distances, plans = [], []
        for passes in (1, 2):
            run.truth = cc.Truth(run.m)
            status = run.run_job(cc.job("area", area=cc.SMALL_AREA, passes=passes), 1200.0)
            distances.append(run.truth.distance if status is not None and status.state == CleaningStatus.FINISHED else 0.0)
            plans.append(cc.pieces_of(clean.job_plan(status.job_id)) if status and clean.job_plan(status.job_id) else [])
        rows = [p for p in plans[1] if p.kind == ROW]
        first = [p for p in rows if p.pass_number == 1]
        second = [p for p in rows if p.pass_number == 2]
        horizontal = lambda p: abs(p.start[1] - p.end[1]) < 1e-6  # noqa: E731
        crossed = bool(first) and bool(second) and len({horizontal(p) for p in first}) == 1 and all(
            horizontal(p) != horizontal(first[0]) for p in second)
        run.check("second pass", crossed, f"{len(first)} rows in the first pass, {len(second)} in the second, at right angles: {crossed}")
        ratio = distances[1] / distances[0] if distances[0] else 0.0
        run.check("twice as far", 1.6 <= ratio <= 2.6, f"one pass {distances[0]:.1f} m, two passes {distances[1]:.1f} m (x{ratio:.2f})")

        # 5. the whole house with edges, reported
        if "--house" in sys.argv:
            run.truth = cc.Truth(run.m)
            t0 = time.monotonic()
            status = run.run_job(cc.job(edges=True), 3600.0)
            open_floor = run.truth.share(cc.true_floor(run.m), allowance=False)
            run.check("whole house with edges", status is not None and status.state == CleaningStatus.FINISHED,
                      f"{cc.summary_line(status) if status else 'no result'}; edges {status.edges_done if status else 0}/"
                      f"{status.edges_total if status else 0}; all open floor cleaned {open_floor:.1f}%; "
                      f"bumper, cliff, reflex reports {run.hazards_since(t0)}")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
