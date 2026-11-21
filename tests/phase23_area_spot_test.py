#!/usr/bin/env python3
"""Phase 23 acceptance test: cleaning an area and a spot (coverage step 8).

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase23_area_spot_test.py

  1. area            from its first row on, the robot's true centre stays inside the area, and the
                     true track cleans >= 95 % of the planned floor
  2. spot            a spot clean covers the square around the point the same way
  3. nothing there   an area with no floor the robot can reach (under the bed) is refused at once, with the reason
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_coverage.planner import spot_area  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

# Beside the coffee table, between the sofa's arm and the armchair: house x 3.2..4.6, y 1.4..3.6.
AREA = (2.75, -0.85, 4.15, 1.35)
# In the bedroom between the bed and the wardrobe: house (6.2, 3.0).
SPOT = cc.hl.to_world(6.2, 3.0)
# Under the bed (the bed's frame reaches the floor): house x 7.0..8.0, y 1.0..1.8.
UNDER_THE_BED = (6.55, -1.25, 7.55, -0.45)


def inside_share(track, box, after_time):
    x0, y0, x1, y1 = box
    samples = [(x, y) for t, x, y, _ in track if t >= after_time]
    inside = [x0 - 0.01 <= x <= x1 + 0.01 and y0 - 0.01 <= y <= y1 + 0.01 for x, y in samples]
    return (sum(inside), len(inside))


def outside_report(samples, box):
    """How far outside the box the true centre went, split by what the robot was doing then."""
    x0, y0, x1, y1 = box
    worst = {}
    for x, y, doing in samples:
        d = max(x0 - x, x - x1, y0 - y, y - y1, 0.0)
        if d > 0.01:
            kind = "trip" if doing.startswith("on the way") else ("row" if doing.startswith("row") else doing or "?")
            n, far = worst.get(kind, (0, 0.0))
            worst[kind] = (n + 1, max(far, d))
    return ", ".join(f"{k}: {n} samples, up to {far * 100:.1f} cm out" for k, (n, far) in sorted(worst.items())) or "none outside"


def main():
    run = cc.Run("phase23")
    clean = run.clean
    try:
        if not run.ready():
            return run.results

        for name, goal, box in (("area", cc.job("area", area=AREA), AREA),
                                ("spot", cc.job("spot", spot=SPOT), None)):
            if box is None:
                corners = spot_area(SPOT[0], SPOT[1], 1.5)
                box = (corners[0][0], corners[0][1], corners[2][0], corners[2][1])
            run.truth = cc.Truth(run.m)
            first_row = {"t": None}
            seen = []  # (x, y, what the status said) after the first row

            def tick():
                run.tick()
                s = clean.status
                if first_row["t"] is None and s is not None and s.state == CleaningStatus.CLEANING and s.message.startswith("row"):
                    first_row["t"] = run.n.sim_time
                if first_row["t"] is not None and s is not None and s.state == CleaningStatus.CLEANING and run.n.odom is not None:
                    x, y, _ = run.n.pose()
                    seen.append((x, y, s.message))

            handle = clean.started(goal)
            result = clean.result(handle, 1200.0, tick)
            status = result if result else None
            plan = clean.job_plan(status.job_id) if status else None
            share = run.truth.share(cc.planned_floor(run.m, plan)) if plan is not None else 0.0
            n_in, n_all = inside_share(run.truth.track, box, first_row["t"] if first_row["t"] is not None else float("inf"))
            run.check(name, status is not None and status.state == CleaningStatus.FINISHED and share >= 95.0 and n_all > 0 and n_in == n_all,
                      f"{cc.summary_line(status) if status else 'no result'}; true share {run.truth.shares(cc.planned_floor(run.m, plan)) if plan else '?'}; "
                      f"true centre inside the square {n_in} of {n_all} samples after the first row; outside: "
                      f"{outside_report(seen, box)}")

        answer = clean.start(cc.job("area", area=UNDER_THE_BED))
        run.check("nothing there", answer is not None and not answer.accepted and "no floor" in answer.message
                  and run.modes.state.state == "idle", f"{answer.message if answer else 'no answer'}; mode {run.modes.state.state}")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
