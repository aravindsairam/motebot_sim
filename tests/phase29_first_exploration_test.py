#!/usr/bin/env python3
"""Phase 29 acceptance test: the exploration node, and a first run (exploration plan, step 2b).

Needs the simulator and the robot stack running, the robot at its start pose in front of the dock,
the saved map 'house' (for the docking at the start):
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase29_first_exploration_test.py

  1. from docked        /exploration/start on the dock: the robot goes through mapping into
                        exploring, and the explorer's first goal is under way within 45 s of the
                        request; the robot moves
  2. stop               Stop (a request for idle): the robot is at rest within 0.5 s, in mapping,
                        with the map kept; the exploration ends as stopped; no explorer program is
                        left and no Nav2 goal is under way
  3. continue           a second start, from mapping, continues the same map
  4. explorer killed    the explorer program killed: the exploration ends as failed, the robot in
                        mapping and at rest; no explorer left
  5. node killed        the exploration node killed during an exploration: failsafe within 3 s, no
                        explorer program left; the restarted node has no job; Reset to mapping keeps
                        the map
  Reported: goals sent and goals that failed, per exploration.
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


def main():
    run = cc.Run("phase29")
    n, modes = run.n, run.modes
    ex = ec.Exploring(n)
    counts = []

    def goals_line(s):
        return f"goals sent {s.goals_sent}, failed {s.goals_failed}" if s else "no status"

    try:
        if not run.ready():
            return run.results
        if not ex.spin_until(lambda: ex.status is not None, 15.0):
            run.check("exploration node", False, "no /exploration/status: is the exploration node running?")
            return run.results

        # 1. from docked
        ok, msg = modes.request("docking", 180.0)
        docked = modes.wait_for_state("docked", 120.0)
        t = time.monotonic()
        answer = ex.start()
        job = answer.job_id if answer is not None and answer.accepted else None
        exploring = ex.wait_state((ExplorationStatus.EXPLORING,), 90.0, job, run.tick)
        first_goal = ex.spin_until(lambda: ex.first_goal_since(t) is not None, 60.0, run.tick)
        goal_after = (ex.first_goal_since(t) - t) if first_goal else None
        moved = run.moving(30.0)
        seen_modes = [st for st, _ in modes.history]
        through = "mapping" in seen_modes and "exploring" in seen_modes
        run.check("from docked",
                  ok and docked and job is not None and exploring is not None and through and goal_after is not None
                  and goal_after <= 45.0 and moved,
                  f"docking: '{msg}', docked {docked}; start: '{answer.message if answer else 'no answer'}' (job {job}); "
                  f"modes seen {seen_modes[-4:]}; status {ex.status.state if ex.status else 'none'}; "
                  f"first goal under way {goal_after:.1f} s after the request; robot moving {moved}"
                  if goal_after is not None else
                  f"docking: '{msg}', docked {docked}; start: '{answer.message if answer else 'no answer'}' (job {job}); "
                  f"modes seen {seen_modes[-4:]}; status {ex.status.state if ex.status else 'none'}; no goal within 60 s")

        # 2. Stop
        run.wait(10.0)  # let it explore a little
        before = ex.fresh_map()
        points = before.free_points() if before else None
        sim_t0 = n.sim_time
        ok, msg = modes.request("idle", 120.0)
        # From the request until the robot was last seen moving: the call itself takes about a second
        # (Nav2 is switched off before it answers), long after the safety gate has stopped the wheels.
        rest = n.stop_time_since(sim_t0)
        final = ex.result(job, 30.0, run.tick) if job else None
        run.wait(2.0)
        left = ec.explorer_pids()
        active = ex.goals_active()
        after = ex.fresh_map()
        kept = after.still_free(points) if after and points is not None else 0.0
        counts.append(("first", goals_line(final)))
        run.check("stop",
                  ok and rest is not None and rest <= 0.5 and modes.state.state == "mapping" and final is not None
                  and final.state == ExplorationStatus.STOPPED and not left and active == 0 and kept >= SAME_MAP_SHARE,
                  f"idle: '{msg}'; at rest after {rest} s of robot time; mode {modes.state.state}; the exploration ended "
                  f"{final.state if final else 'never'}: '{final.message if final else ''}'; explorer programs left {left}; "
                  f"Nav2 goals under way {active}; {kept * 100:.1f}% of the floor is still floor; {goals_line(final)}")

        # 3. a second start continues the same map
        since = time.monotonic()
        before = ex.fresh_map()
        points = before.free_points() if before else None
        answer = ex.start()
        job = answer.job_id if answer is not None and answer.accepted else None
        exploring = ex.wait_state((ExplorationStatus.EXPLORING,), 60.0, job, run.tick)
        goal = ex.spin_until(lambda: ex.first_goal_since(since) is not None, 60.0, run.tick)
        run.wait(10.0)
        after = ex.fresh_map()
        kept = after.still_free(points) if after and points is not None else 0.0
        touched = ex.slam_touched_since(since)
        run.check("continue",
                  job is not None and "continued" in answer.message and exploring is not None and goal
                  and kept >= SAME_MAP_SHARE and not touched and after.known >= before.known,
                  f"start: '{answer.message if answer else 'no answer'}' (job {job}); goal under way {goal}; "
                  f"{kept * 100:.1f}% of the floor mapped before is still floor; known cells {before.known if before else 0} -> "
                  f"{after.known if after else 0}; slam_toolbox lifecycle changes: {touched or 'none'}")

        # 4. the explorer program killed
        pids = ec.explorer_pids()
        t = time.monotonic()
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        final = ex.result(job, 30.0, run.tick) if job else None
        took = time.monotonic() - t
        modes.wait_for_state("mapping", 30.0)
        rest = run.at_rest_after(time.monotonic())
        run.wait(1.0)
        counts.append(("second", goals_line(final)))
        run.check("explorer killed",
                  bool(pids) and final is not None and final.state == ExplorationStatus.FAILED
                  and "stopped running" in final.message and modes.state.state == "mapping" and rest is not None
                  and not ec.explorer_pids(),
                  f"killed {pids}; the exploration ended {final.state if final else 'never'} after {took:.1f} s: "
                  f"'{final.message if final else ''}'; mode {modes.state.state}; at rest {rest is not None}; "
                  f"explorers left {ec.explorer_pids()}")

        # 5. the exploration node killed during an exploration
        before = ex.fresh_map()
        points = before.free_points() if before else None
        answer = ex.start()
        job = answer.job_id if answer is not None and answer.accepted else None
        ex.wait_state((ExplorationStatus.EXPLORING,), 60.0, job, run.tick)
        ex.spin_until(lambda: bool(ec.explorer_pids()), 20.0, run.tick)
        run.wait(3.0)
        explorers = ec.explorer_pids()
        nodes = ec.exploration_node_pids()
        t = time.monotonic()
        for pid in nodes:
            os.kill(pid, signal.SIGKILL)
        failsafe = modes.wait_for_state("failsafe", 10.0)
        took = time.monotonic() - t
        why = modes.state.reason if modes.state else ""
        rest = run.at_rest_after(t)
        orphans_gone = ex.spin_until(lambda: not ec.explorer_pids(), 3.0, run.tick)
        restarted = ex.spin_until(lambda: bool(ec.exploration_node_pids()) and set(ec.exploration_node_pids()) != set(nodes),
                                  30.0, run.tick)
        ex.status = None
        ex.spin_until(lambda: ex.status is not None, 15.0, run.tick)
        fresh = ex.status is not None and ex.status.state == ExplorationStatus.NONE
        run.wait(1.0)
        offered = list(modes.state.allowed_transitions)
        ok, msg = modes.request("mapping", 120.0)
        after = ex.fresh_map()
        kept = after.still_free(points) if after and points is not None else 0.0
        run.check("node killed",
                  bool(nodes) and bool(explorers) and failsafe and took <= 3.0 and orphans_gone and restarted and fresh
                  and "mapping" in offered and ok and kept >= SAME_MAP_SHARE,
                  f"killed node {nodes} (its explorer {explorers}); failsafe after {took:.1f} s ('{why}'); at rest after "
                  f"{rest} s; explorer gone with it {orphans_gone}; node restarted {restarted}, its status "
                  f"{ex.status.state if ex.status else 'none'}; failsafe offers {offered}; Reset to mapping: '{msg}'; "
                  f"{kept * 100:.1f}% of the floor is still floor")
        print("goals per exploration: " + "; ".join(f"{name}: {line}" for name, line in counts), flush=True)
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
