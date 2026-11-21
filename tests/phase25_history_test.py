#!/usr/bin/env python3
"""Phase 25 acceptance test: the record of each clean, the history and the totals (coverage step 10).

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase25_history_test.py

  1. finished record  after a finished job a record exists whose numbers equal the final status and
                      whose picture shows the cleaned floor
  2. stopped record   a stopped job is recorded as stopped, with its part-way numbers
  3. newest first     the history lists the newest first
  4. totals           the totals grew by exactly the two cleans
  5. restart          history and totals are the same after the cleaning node is restarted
"""

import io
import math
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

NUMBERS = ("coverage", "cleaned_area", "planned_area", "distance", "cleaning_time", "rows_done", "rows_total", "rows_skipped", "bumps",
           "pauses")


def equal_numbers(a, b):
    return all(math.isclose(getattr(a, k), getattr(b, k), rel_tol=1e-5, abs_tol=1e-4) for k in NUMBERS)


def tinted_pixels(png: bytes) -> int:
    from PIL import Image

    img = Image.open(io.BytesIO(png)).convert("RGB")
    return sum(1 for p in img.getdata() if p == (150, 205, 240))  # picture.py's cleaned tint


def main():
    run = cc.Run("phase25")
    n, modes, clean = run.n, run.modes, run.clean
    try:
        if not run.ready():
            return run.results
        before = clean.history()
        totals0 = before.totals if before else None

        # 1. a finished job
        spot = cc.job("spot", spot=cc.hl.to_world(2.7, 3.0))  # the living room, between the coffee table and the TV stand
        handle = clean.started(spot)
        result = clean.result(handle, 900.0)
        final = result if result else None
        hist = clean.history()
        newest = hist.records[0] if hist and hist.records else None
        rec = clean.record(newest.id) if newest else None
        tinted = tinted_pixels(bytes(rec.picture)) if rec and rec.success else 0
        run.check("finished record", final is not None and final.state == CleaningStatus.FINISHED and newest is not None
                  and newest.status.state == "finished" and equal_numbers(newest.status, final) and tinted > 0,
                  f"record {newest.id if newest else 'none'}: {cc.summary_line(newest.status) if newest else ''}; "
                  f"numbers equal the final status: {equal_numbers(newest.status, final) if newest and final else False}; "
                  f"picture {len(rec.picture) if rec else 0} bytes with {tinted} tinted pixels")

        # 2. a stopped job
        handle = clean.started(cc.job("area", area=cc.SMALL_AREA))
        clean.wait_state((CleaningStatus.CLEANING,), 120.0)
        run.wait(20.0)
        modes.request("idle", 60.0)
        result = clean.result(handle, 30.0)
        stopped = result if result else None
        hist = clean.history()
        newest = hist.records[0] if hist and hist.records else None
        run.check("stopped record", newest is not None and newest.status.state == "stopped" and stopped is not None
                  and equal_numbers(newest.status, stopped) and 0.0 < newest.status.cleaned_area,
                  f"record {newest.id if newest else 'none'}: {cc.summary_line(newest.status) if newest else ''}")

        # 3. newest first
        starts = [r.started for r in hist.records] if hist else []
        run.check("newest first", len(starts) >= 2 and all(a >= b for a, b in zip(starts, starts[1:])),
                  f"{len(starts)} records, started {[round(s) for s in starts[:4]]} ...")

        # 4. totals
        t = hist.totals if hist else None
        two = hist.records[:2] if hist else []
        area = sum(r.status.cleaned_area for r in two)
        ok = (t is not None and totals0 is not None and t.cleans == totals0.cleans + 2
              and math.isclose(t.area, totals0.area + area, abs_tol=1e-3)
              and math.isclose(t.time, totals0.time + sum(r.status.cleaning_time for r in two), abs_tol=1e-2)
              and math.isclose(t.distance, totals0.distance + sum(r.status.distance for r in two), abs_tol=1e-3))
        run.check("totals", ok, f"before {totals0.cleans if totals0 else '?'} cleans, {totals0.area if totals0 else 0:.2f} m2; "
                  f"after {t.cleans if t else '?'} cleans, {t.area if t else 0:.2f} m2 (the two added {area:.2f} m2)")

        # 5. after a restart
        pids = cc.cleaning_node_pids()
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
        clean.spin_until(lambda: bool(cc.cleaning_node_pids()) and set(cc.cleaning_node_pids()) != set(pids), 30.0)
        run.wait(3.0)
        again = clean.history()
        same = (again is not None and hist is not None and [r.id for r in again.records] == [r.id for r in hist.records]
                and again.totals.cleans == hist.totals.cleans and math.isclose(again.totals.area, hist.totals.area))
        run.check("restart", same, f"after restarting the cleaning node: {len(again.records) if again else 0} records, "
                  f"{again.totals.cleans if again else '?'} cleans")
        n.spin(0.1)
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
