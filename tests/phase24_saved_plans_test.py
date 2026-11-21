#!/usr/bin/env python3
"""Phase 24 acceptance test: saved cleaning plans (coverage step 9).

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase24_saved_plans_test.py

  1. save and load   a plan is saved, listed, and loaded (a preview by name) with the same rows
  2. taken name      saving under a taken name is refused unless replace is set
  3. run by name     running the plan by name drives those rows, and finishes
  4. restart         the plan is still there after the cleaning node is restarted
  5. delete          deleting removes it
A changed map (its rows planned again, and the answer says so) is covered by the unit tests.
"""

import math
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_coverage.planner import ROW  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

NAME = "Phase 24 test"
AREA = cc.job("area", area=cc.SMALL_AREA)


def rows(plan_msg):
    return [p for p in cc.pieces_of(plan_msg) if p.kind == ROW]


def same_rows(a, b):
    return len(a) == len(b) and len(a) > 0 and all(
        math.dist(p.start, q.start) < 1e-6 and math.dist(p.end, q.end) < 1e-6 for p, q in zip(a, b))


def main():
    run = cc.Run("phase24")
    clean = run.clean
    try:
        if not run.ready():
            return run.results
        clean.delete(NAME)  # from an earlier run

        # 1. save, list, load
        clean.preview(AREA)
        clean.spin_until(lambda: clean.plan is not None and clean.plan.job_id == 0 and clean.plan.pieces, 5.0)
        previewed = rows(clean.plan)
        saved = clean.save(NAME, AREA)
        listed = clean.plans()
        names = [p.name for p in listed.plans] if listed else []
        clean.plan = None
        loaded = clean.preview(cc.job("plan", plan=NAME))
        clean.spin_until(lambda: clean.plan is not None and clean.plan.pieces, 5.0)
        loaded_rows = rows(clean.plan) if clean.plan else []
        run.check("save and load", saved is not None and saved.success and NAME in names and loaded is not None and loaded.success
                  and not loaded.replanned and same_rows(previewed, loaded_rows),
                  f"save: {saved.message if saved else 'no answer'}; listed {names}; load: {loaded.message if loaded else 'no answer'}; "
                  f"same {len(previewed)} rows: {same_rows(previewed, loaded_rows)}")

        # 2. a taken name
        again = clean.save(NAME, AREA)
        replaced = clean.save(NAME, AREA, replace=True)
        run.check("taken name", again is not None and not again.success and again.name_taken and replaced is not None and replaced.success,
                  f"again: {again.message if again else 'no answer'}; with replace: {replaced.message if replaced else 'no answer'}")

        # 3. run it by name
        handle = clean.started(cc.job("plan", plan=NAME))
        result = clean.result(handle, 900.0)
        status = result if result else None
        driven = rows(clean.job_plan(status.job_id)) if status is not None and clean.job_plan(status.job_id) else []
        run.check("run by name", status is not None and status.state == CleaningStatus.FINISHED and status.target == NAME
                  and same_rows(previewed, driven), f"{cc.summary_line(status) if status else 'no result'}; target '{status.target if status else ''}'; "
                  f"drove the saved rows: {same_rows(previewed, driven)}")

        # 4. after a restart of the cleaning node
        pids = cc.cleaning_node_pids()
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
        clean.spin_until(lambda: bool(cc.cleaning_node_pids()) and set(cc.cleaning_node_pids()) != set(pids), 30.0)
        run.wait(3.0)
        listed = clean.plans()
        names = [p.name for p in listed.plans] if listed else []
        run.check("restart", NAME in names, f"after restarting the cleaning node ({pids}): {names}")

        # 5. delete
        gone = clean.delete(NAME)
        listed = clean.plans()
        names = [p.name for p in listed.plans] if listed else []
        run.check("delete", gone is not None and gone.success and NAME not in names, f"{gone.message if gone else 'no answer'}; now {names}")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
