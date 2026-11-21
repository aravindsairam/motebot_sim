#!/usr/bin/env python3
"""Phase 22 acceptance test: cleaning from a scripted phone, through the gateway (coverage step 6).

Needs the simulator and the robot stack running (it starts the gateway), robot at its start pose:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase22_cleaning_gateway_test.py

  1. preview          PreviewCleaning is answered with the estimate, and the rows arrive as CleaningPlan
  2. start            StartCleaning for the small rectangle is accepted with a job number (answered
                      before the mode change), and the robot then cleans
  3. one at a time    a second StartCleaning is refused with the robot's reason
  4. same as robot    the status, rows and cleaned area the phone has equal the robot's topics
  5. late phone       a phone that connects mid-job gets all three at once, with the current job's number
  6. no Go to         a Go to during the clean is refused
  7. pause, resume    PauseCleaning pauses and resumes
  8. stop             Stop (ChangeMode idle) ends the job as stopped, and the phone then gets an empty plan
  9. dock             Dock (ChangeMode docking) ends a job and the robot docks
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from phase11_gateway_test import Phone  # noqa: E402
from vacuum_gateway import convert, geometry  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402

Result = pb.Reply.Result
S = pb.CleaningStatus
SMALL = pb.CleaningJob(what=pb.CleaningJob.WHAT_AREA, area=[v for p in cc.rect(*cc.SMALL_AREA) for v in p],
                       return_to_dock=pb.CleaningJob.OPTION_OFF)


def main():
    run = cc.Run("phase22")
    n, modes, clean = run.n, run.modes, run.clean
    phones = []
    ids = iter(range(1, 100000))

    def ask(phone, msg_type, message, timeout=30.0):
        sent = phone.send(msg_type, message)
        got = None

        def done():
            nonlocal got
            got = phone.reply_to(message.request_id, sent)
            return got is not None
        clean.spin_until(done, timeout, run.tick)
        return got[1] if got else None

    def text(reply):
        return f"{Result.Name(reply.result)} '{reply.message}'" if reply else "no reply"

    def latest(phone, msg_type, since=0.0):
        found = phone.messages(msg_type, since)
        return found[-1][1] if found else None

    try:
        if not run.ready():
            return run.results
        phone = Phone("phone").start()
        phones.append(phone)
        clean.spin_until(lambda: bool(phone.opens()), 10.0)

        # 1. preview
        t = time.monotonic()
        reply = ask(phone, pb.COMMAND_PREVIEW_CLEANING, pb.PreviewCleaning(request_id=next(ids), job=SMALL))
        clean.spin_until(lambda: any(m.job_id == 0 and m.pieces for _, m in phone.messages(pb.DATA_CLEANING_PLAN, t)), 5.0)
        plans = [m for _, m in phone.messages(pb.DATA_CLEANING_PLAN, t) if m.pieces]
        run.check("preview", reply is not None and reply.result == Result.RESULT_OK and reply.estimate.rows > 0 and plans,
                  f"{text(reply)}: {reply.estimate.rows if reply else 0} rows, {reply.estimate.time if reply else 0:.0f} s; "
                  f"rows arrived: {len(plans[-1].pieces) if plans else 0} pieces")

        # 2-3. start, and a second start refused
        t = time.monotonic()
        reply = ask(phone, pb.COMMAND_START_CLEANING, pb.StartCleaning(request_id=next(ids), job=SMALL), 30.0)
        took = time.monotonic() - t
        job_id = reply.job_id if reply is not None and reply.result == Result.RESULT_OK else 0
        cleaning = clean.spin_until(lambda: any(m.job_id == job_id and m.state == S.STATE_CLEANING
                                                for _, m in phone.messages(pb.DATA_CLEANING_STATUS, t)), 150.0, run.tick)
        run.check("start", job_id > 0 and reply.estimate.rows > 0 and took < 5.0 and cleaning,
                  f"{text(reply)} after {took:.1f} s, job {job_id}, {reply.estimate.rows if reply else 0} rows; "
                  f"the phone then saw it cleaning: {cleaning}")
        second = ask(phone, pb.COMMAND_START_CLEANING, pb.StartCleaning(request_id=next(ids), job=pb.CleaningJob()), 30.0)
        run.check("one at a time", second is not None and second.result == Result.RESULT_REFUSED and "already" in second.message,
                  f"second start: {text(second)}")

        # 4. the same as the robot's topics: compare while paused, when nothing changes. Wait for some
        # cleaned floor first: the trip to the first row is not cleaning, so the grid starts empty.
        clean.spin_until(lambda: clean.cleaned is not None and any(c >= 100 for c in clean.cleaned.data), 150.0, run.tick)
        run.wait(4.0)
        ask(phone, pb.COMMAND_PAUSE_CLEANING, pb.PauseCleaning(request_id=next(ids)))
        run.wait(3.0)
        mine = latest(phone, pb.DATA_CLEANING_STATUS)
        robot = convert.cleaning_status(clean.status) if clean.status else None
        plan = latest(phone, pb.DATA_CLEANING_PLAN)
        robot_plan = convert.cleaning_plan(clean.plan) if clean.plan else None
        area = latest(phone, pb.DATA_CLEANED_AREA)
        cells = geometry.rle_decode(area.cells_rle).tolist() if area else []
        robot_cells = list(clean.cleaned.data) if clean.cleaned else []
        run.check("same as robot", mine is not None and mine == robot and plan == robot_plan and cells == robot_cells and any(c >= 100 for c in cells),
                  f"status equal {mine == robot}; plan equal {plan == robot_plan} ({len(plan.pieces) if plan else 0} pieces); "
                  f"cleaned area equal {cells == robot_cells} ({sum(1 for c in cells if c == 100)} cleaned cells)")

        # 5. a phone that connects mid-job
        late = Phone("late").start()
        phones.append(late)
        clean.spin_until(lambda: bool(late.opens()), 10.0)
        opened = late.opens()[0] if late.opens() else time.monotonic()
        run.wait(1.0)
        got = {name: late.first_after(opened, t) for name, t in
               (("status", pb.DATA_CLEANING_STATUS), ("plan", pb.DATA_CLEANING_PLAN), ("cleaned", pb.DATA_CLEANED_AREA))}
        delays = {k: round(v[0] - opened, 2) if v else None for k, v in got.items()}
        numbers = {k: v[1].job_id if v else None for k, v in got.items()}
        run.check("late phone", all(v is not None and v <= 1.0 for v in delays.values()) and set(numbers.values()) == {job_id},
                  f"seconds after connecting: {delays}; job numbers {numbers} (the job is {job_id})")

        # 6. Go to during the clean
        x, y, _ = n.pose()
        goto = ask(phone, pb.COMMAND_GO_TO, pb.GoTo(request_id=next(ids), x=x + 0.5, y=y, heading=0.0), 15.0)
        run.check("no Go to", goto is not None and goto.result == Result.RESULT_REFUSED and modes.state.state == "cleaning",
                  f"Go to during the clean: {text(goto)}")

        # 7. pause (above) and resume
        paused = mine is not None and mine.state == S.STATE_PAUSED
        reply = ask(phone, pb.COMMAND_PAUSE_CLEANING, pb.PauseCleaning(request_id=next(ids), resume=True))
        moving = run.moving(30.0)
        run.check("pause, resume", paused and reply is not None and reply.result == Result.RESULT_OK and moving,
                  f"paused: {paused}; resume {text(reply)}; moving again {moving}")

        # 8. Stop
        t = time.monotonic()
        reply = ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(ids), mode="idle"), 120.0)
        clean.spin_until(lambda: any(m.state == S.STATE_STOPPED for _, m in phone.messages(pb.DATA_CLEANING_STATUS, t)), 15.0)
        stopped = [m for _, m in phone.messages(pb.DATA_CLEANING_STATUS, t) if m.state == S.STATE_STOPPED]
        clean.spin_until(lambda: any(not m.pieces for _, m in phone.messages(pb.DATA_CLEANING_PLAN, t)), 5.0)
        emptied = any(not m.pieces and m.job_id == 0 for _, m in phone.messages(pb.DATA_CLEANING_PLAN, t))
        run.check("stop", reply is not None and reply.result == Result.RESULT_OK and stopped and emptied,
                  f"idle: {text(reply)}; the phone saw {stopped[0].message if stopped else 'no stopped status'}; empty plan after: {emptied}")

        # 9. Dock
        reply = ask(phone, pb.COMMAND_START_CLEANING, pb.StartCleaning(request_id=next(ids), job=SMALL), 30.0)
        clean.wait_state(("cleaning",), 150.0, run.tick)
        t = time.monotonic()
        dock = ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(ids), mode="docking"), 200.0)
        clean.spin_until(lambda: any(m.state == S.STATE_STOPPED for _, m in phone.messages(pb.DATA_CLEANING_STATUS, t)), 15.0)
        stopped = any(m.state == S.STATE_STOPPED for _, m in phone.messages(pb.DATA_CLEANING_STATUS, t))
        docked = modes.wait_for_state("docked", 180.0)
        run.check("dock", reply is not None and reply.result == Result.RESULT_OK and dock is not None and dock.result == Result.RESULT_OK
                  and stopped and docked, f"start {text(reply)}; dock {text(dock)}; job stopped {stopped}; mode {modes.state.state}")
    finally:
        for p in phones:
            p.stop()
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
