#!/usr/bin/env python3
"""Phase 19 acceptance test: driving the rows of a small area (coverage step 3).

Needs the simulator and the robot stack running, the robot at its start pose, a saved map:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase19_small_area_test.py

The area is the open rectangle between the coffee table and the TV stand (cleaning_client.SMALL_AREA).
Everything is measured with the simulator's true position (a planned cell counts as cleaned when the
true tool came within 5 cm of it; the figure without that allowance is reported too):
  1. cleaned          the job finishes and the true track cleans >= 95 % of the planned floor
  2. trips not counted the robot's cleaned floor lies along the rows, connectors and edges only (within the
                      tool's radius + 12 cm: a trip may end 10 cm from a row's start), though the robot
                      drove at least 0.5 m of trips outside that band (from the start pose to the area)
  3. rows held        on rows, the true centre is within 3 cm of the row line (median), 8 cm at worst,
                      after removing the constant offset between map and world
  4. no hazards       no bumper press, no cliff reading, no safety reflex
  5. own numbers      the robot's cleaned share is within 3 points of the true one; its distance within 5 %
  6. ends in idle     asked not to dock, the robot ends in idle
  7. turn time        the time per turn on the spot, measured and reported (for coverage.yaml)
  8. stop mid-row     Stop brings the robot to rest within 0.5 s and the job ends as stopped
  9. from a trip      a clean started during a trip in navigation mode cancels the trip and cleans
 10. outside goal     a navigation goal sent from outside during the clean is cancelled, and the clean goes on
"""

import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from action_msgs.msg import GoalStatus  # noqa: E402
from nav2_msgs.action import NavigateToPose  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

FAR_CORNER = cc.hl.to_world(4.0, 3.7)  # living room, far from the small area


def turning_time(track):
    """Seconds spent turning on the spot in a true track (yaw rate over 0.3 rad/s, under 3 cm/s)."""
    total = 0.0
    for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(track, track[1:]):
        dt = t1 - t0
        if dt <= 0 or dt > 0.5:
            continue
        rate = abs(math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))) / dt
        if rate > 0.3 and math.hypot(x1 - x0, y1 - y0) / dt < 0.03:
            total += dt
    return total


def main():
    run = cc.Run("phase19")
    n, modes, clean = run.n, run.modes, run.clean
    nav = ActionClient(n, NavigateToPose, "navigate_to_pose")

    def send_goal(xy):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(xy[0]), float(xy[1])
        goal.pose.pose.orientation.w = 1.0
        if not nav.wait_for_server(timeout_sec=10.0):
            return None
        sent = nav.send_goal_async(goal)
        clean.spin_until(sent.done, 10.0, run.tick)
        handle = sent.result() if sent.done() else None
        return handle if handle is not None and handle.accepted else None

    def goal_status(handle, timeout=15.0):
        future = handle.get_result_async()
        clean.spin_until(future.done, timeout, run.tick)
        return future.result().status if future.done() else None

    try:
        if not run.ready():
            return run.results
        offset = run.map_offset()
        run.truth = cc.Truth(run.m)
        t0 = time.monotonic()
        status = run.run_job(cc.job("area", area=cc.SMALL_AREA, stay=True), 900.0)
        run.truth.on = False
        plan = clean.job_plan(status.job_id) if status else None  # kept: an empty plan replaced it at the end
        planned = cc.planned_floor(run.m, plan) if plan is not None and plan.pieces else None
        share = run.truth.share(planned) if planned is not None else 0.0
        run.check("cleaned", status is not None and status.state == CleaningStatus.FINISHED and share >= 95.0,
                  f"{cc.summary_line(status) if status else 'no result'}; true share of the planned floor "
                  f"{run.truth.shares(planned) if planned is not None else 'no plan seen'}")

        # 2. trips are not cleaning: the robot's cleaned floor (still the finished job's) lies along the plan's pieces
        cleaned = cc.cleaned_cells(clean.cleaned, run.m)
        reach = cc.TOOL + 0.12
        band = cc.planner.cleaned_by(run.m.grid, cc.pieces_of(plan), reach) if planned is not None else None
        outside = int((cleaned & ~band).sum()) if cleaned is not None and band is not None else -1
        trip_m = cc.track_outside(run.truth.track, run.m, band) if band is not None else 0.0
        run.check("trips not counted", outside == 0 and trip_m >= 0.5,
                  f"{outside if outside >= 0 else 'no'} cleaned cells farther than {reach * 100:.0f} cm from a row, connector or edge "
                  f"({int(cleaned.sum()) if cleaned is not None else 0} cleaned in all); the robot drove {trip_m:.1f} m of trips outside that band")

        errors = cc.row_errors(run.truth.track, cc.pieces_of(plan), offset=offset) if planned is not None else []
        med = statistics.median(errors) if errors else float("nan")
        worst = max(errors) if errors else float("nan")
        run.check("rows held", errors and med <= 0.03 and worst <= 0.08,
                  f"{len(errors)} samples on rows: median {med * 100:.1f} cm, worst {worst * 100:.1f} cm (limits 3 and 8 cm); "
                  f"map - world offset ({offset[0] * 100:.1f}, {offset[1] * 100:.1f}) cm removed")

        bumps, cliffs, reflexes = run.hazards_since(t0)
        run.check("no hazards", (bumps, cliffs, reflexes) == (0, 0, 0) and (status is None or status.bumps == 0),
                  f"bumper {bumps}, cliff {cliffs}, reflex {reflexes} safety reports; bumps counted by the robot {status.bumps if status else '?'}")

        own = status.coverage if status else 0.0
        dist_err = abs(status.distance - run.truth.distance) / run.truth.distance if status and run.truth.distance > 0 else 1.0
        run.check("own numbers", status is not None and abs(own - share) <= 3.0 and dist_err <= 0.05,
                  f"robot says {own:.1f}%, truth {share:.1f}%; robot {status.distance if status else 0:.2f} m, "
                  f"truth {run.truth.distance:.2f} m ({dist_err * 100:.1f}% apart)")

        idle = modes.wait_for_state("idle", 60.0)
        run.check("ends in idle", idle, f"mode after the job: {modes.state.state}")

        turns = plan.estimate.turns if plan is not None else 0
        spent = turning_time(run.truth.track)
        per_turn = spent / turns if turns else float("nan")
        took = run.truth.track[-1][0] - run.truth.track[0][0] if len(run.truth.track) > 1 else 0.0
        run.check("turn time", turns > 0, f"{spent:.0f} s turning on the spot over {turns} planned turns: {per_turn:.1f} s per turn "
                  f"(coverage.yaml turn_time 3.0); the job took {took:.0f} s of robot time, estimated {plan.estimate.time if plan else 0:.0f} s")

        # 7. Stop in the middle of a row
        job_id = clean.started(cc.job("area", area=cc.SMALL_AREA, stay=True))
        on_row = clean.wait_state((CleaningStatus.CLEANING,), 120.0, run.tick, job_id=job_id)
        clean.spin_until(lambda: clean.status.message.startswith("row") and n.speed() > 0.1, 120.0, run.tick)
        run.wait(2.0)
        sim0 = n.sim_time
        ok, msg = modes.request("idle", 60.0)
        stop_sim = n.stop_time_since(sim0)
        final = clean.result(job_id, 30.0)
        run.check("stop mid-row", on_row is not None and ok and stop_sim is not None and stop_sim <= 0.5 and final is not None
                  and final.state == CleaningStatus.STOPPED,
                  f"stopped during '{on_row.message if on_row else '?'}': at rest {stop_sim} s of robot time after the request; "
                  f"job {final.state if final else '?'}")

        # 8. a clean started during a trip in navigation mode
        modes.request("navigation", 120.0)
        n.wait_sim(3.0)
        trip = send_goal(FAR_CORNER)
        moving = run.moving(30.0)
        job_id = clean.started(cc.job("area", area=cc.SMALL_AREA, stay=True))
        cleaning = clean.wait_state((CleaningStatus.CLEANING,), 120.0, run.tick, job_id=job_id)
        trip_end = goal_status(trip) if trip else None
        final = clean.result(job_id, 900.0, run.tick)
        run.check("from a trip", trip is not None and moving and cleaning is not None and trip_end == GoalStatus.STATUS_CANCELED
                  and final is not None and final.state == CleaningStatus.FINISHED,
                  f"trip moving {moving}; the trip ended {trip_end} (5 = cancelled); clean {final.state if final else '?'} "
                  f"'{final.message if final else ''}'")

        # 9. a goal sent from outside during a clean
        job_id = clean.started(cc.job("area", area=cc.SMALL_AREA, stay=True))
        clean.wait_state((CleaningStatus.CLEANING,), 120.0, run.tick, job_id=job_id)
        clean.spin_until(lambda: clean.status.message.startswith("row"), 120.0, run.tick)
        stranger = send_goal(FAR_CORNER)
        stranger_end = goal_status(stranger, 10.0) if stranger else None
        noted = clean.spin_until(lambda: "someone else" in clean.status.message, 5.0, run.tick)
        final = clean.result(job_id, 900.0, run.tick)
        run.check("outside goal", stranger is not None and stranger_end == GoalStatus.STATUS_CANCELED and noted and final is not None
                  and final.state == CleaningStatus.FINISHED,
                  f"outside goal ended {stranger_end} (5 = cancelled); the status said so: {noted}; clean {final.state if final else '?'}")
    finally:
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
