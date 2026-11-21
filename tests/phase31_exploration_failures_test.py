#!/usr/bin/env python3
"""Phase 31 acceptance test: exploring when things go wrong (exploration plan, step 4).

Needs the simulator and the robot stack running, the robot at its start pose in front of the dock:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase31_exploration_failures_test.py

  1. time limit        an exploration with a time limit of 20 s of robot time ends as finished
                       early, in mapping, with the map kept. (The plan said two minutes; the whole
                       house is explored in under one, so a two-minute limit is never reached.)
  2. explorer frozen   the explorer program frozen (stopped, not killed): the exploration ends as
                       failed within the watchdog's limit; the frozen program is ended; the robot is
                       in mapping and at rest
  3. planner killed    Nav2's planner killed during an exploration: failsafe; the exploration ends
                       as failed; Reset to mapping keeps the map
A start on a map that is already fully explored is checked by phase 30 ("nothing left"); the
explorer's own failures that need no simulator (a "finished" the second look overturns, the quiet
spell, a trip back that fails) by the exploration node's unit tests.
"""

import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
import exploration_client as ec  # noqa: E402
from vacuum_interfaces.msg import ExplorationStatus  # noqa: E402

SAME_MAP_SHARE = 0.95
# The runner's answer timeout (5 s) plus one state request (1 s) plus the stop sequence's first
# two steps (2 s for the stop request, 3 s for terminate) before the end is reported.
FROZEN_LIMIT = 12.0
TIME_LIMIT = 20.0  # s of robot time: well inside the 45 to 50 s the whole house takes


def main():
    run = cc.Run("phase31")
    n, modes = run.n, run.modes
    ex = ec.Exploring(n)
    try:
        if not run.ready():
            return run.results
        if not ex.spin_until(lambda: ex.status is not None, 15.0):
            run.check("exploration node", False, "no /exploration/status: is the exploration node running?")
            return run.results

        # 1. a time limit of two minutes
        answer = ex.start(minutes=TIME_LIMIT / 60.0)
        job = answer.job_id if answer is not None and answer.accepted else None
        ex.wait_state((ExplorationStatus.EXPLORING,), 90.0, job, run.tick)
        final = ex.wait_state((ExplorationStatus.RETURNING,) + ec.FINAL, 600.0, job, run.tick)
        before = ex.fresh_map()
        points = before.free_points() if before else None
        final = ex.result(job, 400.0, run.tick) if job else None
        modes.wait_for_state("mapping", 30.0)
        after = ex.fresh_map()
        kept = after.still_free(points) if after and points is not None else 0.0
        run.check("time limit",
                  final is not None and final.state == ExplorationStatus.FINISHED and "time limit" in final.message
                  and TIME_LIMIT <= final.exploring_time <= TIME_LIMIT + 8.0 and modes.state.state == "mapping"
                  and kept >= SAME_MAP_SHARE,
                  f"start: '{answer.message if answer else 'no answer'}'; ended {final.state if final else 'never'}: "
                  f"'{final.message if final else ''}' after {final.exploring_time if final else 0:.0f} s of robot time "
                  f"(limit {final.time_limit if final else 0:.0f} s); mode {modes.state.state}; "
                  f"{kept * 100:.1f}% of the floor is still floor")

        # 2. the explorer frozen
        answer = ex.start()
        job = answer.job_id if answer is not None and answer.accepted else None
        ex.wait_state((ExplorationStatus.EXPLORING,), 60.0, job, run.tick)
        ex.spin_until(lambda: bool(ec.explorer_pids()), 20.0, run.tick)
        run.wait(8.0)
        pids = ec.explorer_pids()
        t = time.monotonic()
        for pid in pids:
            os.kill(pid, signal.SIGSTOP)
        final = ex.result(job, 60.0, run.tick) if job else None
        took = time.monotonic() - t
        modes.wait_for_state("mapping", 30.0)
        rest = run.at_rest_after(time.monotonic())
        gone = ex.spin_until(lambda: not ec.explorer_pids(), 10.0, run.tick)
        for pid in pids:  # never leave a frozen program behind, whatever happened
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        run.check("explorer frozen",
                  bool(pids) and final is not None and final.state == ExplorationStatus.FAILED and "answer" in final.message
                  and took <= FROZEN_LIMIT and modes.state.state == "mapping" and rest is not None and gone,
                  f"froze {pids}; the exploration ended {final.state if final else 'never'} after {took:.1f} s: "
                  f"'{final.message if final else ''}'; mode {modes.state.state}; at rest {rest is not None}; "
                  f"frozen program ended {gone}")

        # 3. Nav2's planner killed
        answer = ex.start()
        job = answer.job_id if answer is not None and answer.accepted else None
        ex.wait_state((ExplorationStatus.EXPLORING,), 60.0, job, run.tick)
        run.wait(8.0)
        before = ex.fresh_map()
        points = before.free_points() if before else None
        planners = ec.pids_of("nav2_planner/planner_server")
        t = time.monotonic()
        for pid in planners:
            os.kill(pid, signal.SIGKILL)
        failsafe = modes.wait_for_state("failsafe", 20.0)
        took = time.monotonic() - t
        why = modes.state.reason if modes.state else ""
        final = ex.result(job, 30.0, run.tick) if job else None
        rest = run.at_rest_after(time.monotonic())
        # the planner is restarted by the launch file; failsafe offers mapping once the fault is gone
        offered = ex.spin_until(lambda: modes.state is not None and "mapping" in modes.state.allowed_transitions,
                                60.0, run.tick)
        ok, msg = modes.request("mapping", 120.0)
        after = ex.fresh_map()
        kept = after.still_free(points) if after and points is not None else 0.0
        run.check("planner killed",
                  bool(planners) and failsafe and final is not None and final.state == ExplorationStatus.FAILED
                  and offered and ok and modes.state.state == "mapping" and kept >= SAME_MAP_SHARE and not ec.explorer_pids(),
                  f"killed planner {planners}; failsafe after {took:.1f} s ('{why}'); the exploration ended "
                  f"{final.state if final else 'never'}: '{final.message if final else ''}'; at rest {rest is not None}; "
                  f"mapping offered {offered}; Reset to mapping: '{msg}'; {kept * 100:.1f}% of the floor is still floor; "
                  f"explorers left {ec.explorer_pids()}")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
