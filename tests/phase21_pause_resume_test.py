#!/usr/bin/env python3
"""Phase 21 acceptance test: pause, resume, and ending a clean by other means (coverage step 5).
The failure paths (a blocked row retried and skipped, three skipped rows, a failed transit) are
covered by the unit tests in ros2_ws/src/motebot_robot/vacuum_coverage/test/test_job.py.

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase21_pause_resume_test.py

  1. pause            the robot is at rest within 1 s, and stays in cleaning mode
  2. paused time      10 s of pause add less than 1 s to the cleaning time
  3. resume           cleaning continues on the same row; the job finishes with every row done once
  4. pause, then stop the job ends as stopped
  5. manual           asking for manual mode mid-job ends the job as stopped
  6. dock             asking for docking mid-job ends the job as stopped, and the robot docks
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

AREA = cc.job("area", area=cc.SMALL_AREA, stay=True)


def main():
    run = cc.Run("phase21")
    n, modes, clean = run.n, run.modes, run.clean

    def driving_a_row(timeout=150.0):
        clean.wait_state((CleaningStatus.CLEANING,), timeout, run.tick)
        clean.spin_until(lambda: clean.status.message.startswith("row") and n.speed() > 0.1, timeout, run.tick)
        run.wait(2.0)
        return clean.status

    try:
        if not run.ready():
            return run.results
        run.truth = cc.Truth(run.m)

        # 1. pause
        handle = clean.started(AREA)
        before = driving_a_row()
        row, rows_done = before.current_row, before.rows_done
        sim0 = n.sim_time
        reply = clean.pause()
        stop = n.stop_time_since(sim0)
        paused = clean.wait_state((CleaningStatus.PAUSED,), 5.0)
        run.wait(3.0)
        run.check("pause", reply is not None and reply.success and stop is not None and stop <= 1.0 and paused.state == CleaningStatus.PAUSED
                  and modes.state.state == "cleaning",
                  f"paused on row {row}: '{reply.message if reply else 'no answer'}', at rest after {stop} s of robot time; "
                  f"status {paused.state} '{paused.message}'; mode {modes.state.state}")

        # 2. paused time is not counted
        time_before = clean.status.cleaning_time
        run.wait(10.0)
        grew = clean.status.cleaning_time - time_before
        run.check("paused time", grew < 1.0, f"cleaning time grew {grew:.2f} s during 10 s of pause")

        # 3. resume: the same row, and the job finishes with each row once
        reply = clean.resume()
        resumed = clean.wait_state((CleaningStatus.CLEANING,), 5.0)
        same_row = resumed.current_row == row and resumed.rows_done == rows_done
        result = clean.result(handle, 900.0, run.tick)
        status = result if result else None
        run.check("resume", reply is not None and reply.success and same_row and status is not None
                  and status.state == CleaningStatus.FINISHED and status.rows_done + status.rows_skipped == status.rows_total
                  and status.pauses == 1,
                  f"resumed on row {resumed.current_row} (paused on {row}), rows done {resumed.rows_done} (was {rows_done}); "
                  f"then {cc.summary_line(status) if status else 'no result'}; pauses {status.pauses if status else '?'}")

        # 4. pause, then Stop
        handle = clean.started(AREA)
        driving_a_row()
        clean.pause()
        clean.wait_state((CleaningStatus.PAUSED,), 5.0)
        ok, msg = modes.request("idle", 60.0)
        result = clean.result(handle, 30.0)
        run.check("pause, then stop", ok and result is not None and result.state == CleaningStatus.STOPPED,
                  f"idle: {msg}; job {result.state if result else '?'} '{result.message if result else ''}'")

        # 5. manual mid-job
        handle = clean.started(AREA)
        driving_a_row()
        ok, msg = modes.request("manual", 60.0)
        result = clean.result(handle, 30.0)
        run.check("manual", ok and result is not None and result.state == CleaningStatus.STOPPED and modes.state.state == "manual",
                  f"manual: {msg}; job {result.state if result else '?'} '{result.message if result else ''}'; mode {modes.state.state}")
        modes.request("idle", 60.0)

        # 6. dock mid-job
        handle = clean.started(AREA)
        driving_a_row()
        ok, msg = modes.request("docking", 120.0)
        result = clean.result(handle, 30.0)
        docked = modes.wait_for_state("docked", 180.0)
        run.check("dock", ok and result is not None and result.state == CleaningStatus.STOPPED and docked,
                  f"docking: {msg}; job {result.state if result else '?'}; mode {modes.state.state} '{modes.state.reason}'")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
