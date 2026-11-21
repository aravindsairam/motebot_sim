#!/usr/bin/env python3
"""Phase 18 acceptance test: cleaning mode and the preview (coverage step 2).

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase18_cleaning_mode_test.py

  1. preview          answered with an estimate; the plan on /cleaning/plan; neither the mode nor
                      the robot's position changes
  2. no plain mode    a plain request for cleaning mode is refused with the reason, and cleaning is
                      not among the modes that can be requested
  3. a job            a job sent to /cleaning/start puts the robot in cleaning mode, and the safety
                      gate lets Nav2 (and only Nav2) drive
  4. one at a time    a second job while one runs is refused
  5. stop             asking for idle (Stop) ends the job as stopped; the robot is in idle
  6. not in mapping   a job asked for while mapping is refused ("save or discard the map first")
  7. node killed      the cleaning node killed during a job: failsafe within 3 s, also though its
                      replacement starts after 2 s; the replacement starts no job; idle works again
"""

import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

ACTIVE = (CleaningStatus.CLEANING, CleaningStatus.PAUSED)


def main():
    run = cc.Run("phase18")
    n, modes, clean = run.n, run.modes, run.clean
    try:
        if not run.ready():
            return run.results

        # 1. preview: no movement, no mode change
        x0, y0, _ = n.pose()
        reply = clean.preview(cc.job())
        clean.spin_until(lambda: clean.plan is not None and clean.plan.job_id == 0 and len(clean.plan.pieces) > 0, 5.0)
        n.wait_sim(1.0)
        x1, y1, _ = n.pose()
        moved = math.hypot(x1 - x0, y1 - y0)
        e = reply.estimate if reply else None
        run.check("preview", reply is not None and reply.success and e.rows > 0 and clean.plan is not None
                  and len(clean.plan.pieces) > 0 and modes.state.state == "idle" and moved < 0.01,
                  f"answer: {reply.message if reply else 'none'}; {e.rows if e else 0} rows, {e.distance if e else 0:.0f} m, "
                  f"~{(e.time if e else 0) / 60:.0f} min, planned in {e.planning_time if e else 0:.2f} s; "
                  f"plan on the topic: {len(clean.plan.pieces) if clean.plan else 0} pieces; mode {modes.state.state}; moved {moved * 100:.1f} cm")

        # 2. a plain request for the mode
        ok, msg = modes.request("cleaning", 30.0)
        listed = "cleaning" in modes.state.allowed_transitions
        run.check("no plain mode", not ok and "cleaning job" in msg and not listed and modes.state.state == "idle",
                  f"request: {'accepted' if ok else 'refused'} '{msg}'; cleaning offered: {listed}")

        # 3. a job puts the robot in cleaning mode; Nav2 drives
        answer = clean.start(cc.job("area", area=cc.SMALL_AREA))
        job_id = answer.job_id if answer is not None and answer.accepted else None
        status = clean.wait_state(ACTIVE, 120.0, job_id=job_id) if job_id else None
        in_mode = modes.wait_for_state("cleaning", 5.0)
        n.wait_sim(1.0)
        source = n.safety.source if n.safety else "?"
        run.check("a job", job_id is not None and status is not None and status.state in ACTIVE and in_mode and source == "nav2",
                  f"start: {answer.message if answer else 'no answer'} (job {job_id}); status {status.state if status else 'none'} "
                  f"'{status.message if status else ''}'; mode {modes.state.state}; the safety gate lets {source} drive")

        # 4. a second job is refused
        second = clean.start(cc.job())
        run.check("one at a time", second is not None and not second.accepted and "already" in second.message,
                  f"second job: {second.message if second else 'no answer'}")

        # 5. Stop
        t = time.monotonic()
        ok, msg = modes.request("idle", 120.0)
        final = clean.result(job_id, 30.0)
        rest = run.at_rest_after(t)
        run.check("stop", ok and final is not None and final.state == CleaningStatus.STOPPED and modes.state.state == "idle",
                  f"idle: {msg}; the job ended {final.state if final else 'never'}: '{final.message if final else ''}'; at rest after {rest} s")

        # 6. a job asked for while mapping
        ok, msg = modes.request("mapping", 120.0)
        answer = clean.start(cc.job())
        run.check("not in mapping", ok and answer is not None and not answer.accepted and "save or discard" in answer.message,
                  f"mapping: {msg}; start: {answer.message if answer else 'no answer'}")
        modes.request("idle", 120.0)  # leaves mapping without saving: the saved map is untouched
        modes.wait_for_state("idle", 60.0)

        # 7. the cleaning node killed during a job
        job_id = clean.started(cc.job("area", area=cc.SMALL_AREA))
        clean.wait_state((CleaningStatus.CLEANING,), 120.0, job_id=job_id)
        pids = cc.cleaning_node_pids()
        t = time.monotonic()
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        failsafe = modes.wait_for_state("failsafe", 10.0)
        took = time.monotonic() - t
        why = modes.state.reason if modes.state else ""
        rest = run.at_rest_after(t)
        restarted = clean.spin_until(lambda: bool(cc.cleaning_node_pids()) and set(cc.cleaning_node_pids()) != set(pids), 30.0)
        clean.status = None
        clean.spin_until(lambda: clean.status is not None, 15.0)
        run.wait(5.0)
        fresh = clean.status is not None and clean.status.state == CleaningStatus.NONE
        ok, msg = modes.request("idle", 120.0)
        run.check("node killed", bool(pids) and failsafe and took <= 3.0 and restarted and fresh and ok,
                  f"killed {pids}; failsafe after {took:.1f} s ({why}); at rest after {rest} s; "
                  f"restarted {restarted}, its status {clean.status.state if clean.status else 'none'} after 5 s; idle again: {msg}")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
