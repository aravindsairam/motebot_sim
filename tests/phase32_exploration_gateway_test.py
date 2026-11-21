#!/usr/bin/env python3
"""Phase 32 acceptance test: exploring from a scripted phone, through the gateway (exploration plan, step 5).

Needs the simulator and the robot stack running (it starts the gateway), robot at its start pose:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase32_exploration_gateway_test.py

  1. start            StartExploration is accepted at once with the exploration's number, and the
                      phone then sees it exploring, with the robot state saying a map is being made
  2. one at a time    a second StartExploration is refused with the robot's reason
  3. same as robot    the status and the places still to look at the phone has equal the robot's topics
  4. late phone       a phone that connects mid-run gets the status, the places and the growing map
                      within 1 s, with the current exploration's number
  5. no Go to         a Go to while exploring is refused
  6. stop             Stop (ChangeMode idle) ends the exploration as stopped; the robot is in mapping
                      with the map kept
  7. save             SaveMap from the phone saves the explored map, and the robot then uses it
The map in use is set back to what it was afterwards, and the test map removed.
"""

import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
import exploration_client as ec  # noqa: E402
from phase11_gateway_test import Phone  # noqa: E402
from vacuum_gateway import convert  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402

Result = pb.Reply.Result
S = pb.ExplorationStatus
TEST_MAP = "phone_explored"
SAME_MAP_SHARE = 0.95


def remove_test_map():
    for path in glob.glob(os.path.join(cc.MAPS, TEST_MAP + ".*")) + glob.glob(os.path.join(cc.MAPS, TEST_MAP + "_*")):
        if os.path.isdir(path):
            subprocess.run(["rm", "-rf", path], check=False)
        else:
            os.remove(path)


def main():
    run = cc.Run("phase32")
    n, modes = run.n, run.modes
    ex = ec.Exploring(n)
    phones = []
    ids = iter(range(1, 100000))
    marker = os.path.join(cc.MAPS, "current_map.txt")
    with open(marker) as f:
        map_before = f.read().strip()
    remove_test_map()

    def ask(phone, msg_type, message, timeout=30.0):
        sent = phone.send(msg_type, message)
        got = None

        def done():
            nonlocal got
            got = phone.reply_to(message.request_id, sent)
            return got is not None
        ex.spin_until(done, timeout, run.tick)
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
        ex.spin_until(lambda: bool(phone.opens()), 10.0)

        # 1. start
        t = time.monotonic()
        reply = ask(phone, pb.COMMAND_START_EXPLORATION, pb.StartExploration(request_id=next(ids)), 15.0)
        took = time.monotonic() - t
        job = reply.job_id if reply is not None and reply.result == Result.RESULT_OK else 0
        exploring = ex.spin_until(lambda: any(m.job_id == job and m.state == S.STATE_EXPLORING
                                              for _, m in phone.messages(pb.DATA_EXPLORATION_STATUS, t)), 120.0, run.tick)
        state = latest(phone, pb.DATA_ROBOT_STATE)
        run.check("start", job > 0 and took < 2.0 and exploring and state is not None and state.state == "exploring"
                  and state.making_map,
                  f"{text(reply)} after {took:.2f} s, exploration {job}; the phone then saw it exploring: {exploring}; "
                  f"its robot state: {state.state if state else 'none'}, making_map {state.making_map if state else '?'}")

        # 2. a second start
        second = ask(phone, pb.COMMAND_START_EXPLORATION, pb.StartExploration(request_id=next(ids)), 15.0)
        run.check("one at a time", second is not None and second.result == Result.RESULT_REFUSED and "already" in second.message,
                  f"second start: {text(second)}")

        # 3. the same as the robot's topics
        ex.spin_until(lambda: ex.frontiers is not None and len(ex.frontiers.poses) > 0, 60.0, run.tick)
        run.wait(2.0)

        def status_matches():
            mine = latest(phone, pb.DATA_EXPLORATION_STATUS)
            robot = convert.exploration_status(ex.status) if ex.status else None
            if mine is None or robot is None:
                return False
            # the numbers that grow every moment may be one copy apart
            return ((mine.state, mine.message, mine.job_id, mine.time_limit, mine.goals_sent) ==
                    (robot.state, robot.message, robot.job_id, robot.time_limit, robot.goals_sent)
                    and abs(mine.exploring_time - robot.exploring_time) < 1.5 and abs(mine.distance - robot.distance) < 0.5)

        def places_match():
            mine = latest(phone, pb.DATA_FRONTIERS)
            robot = convert.frontiers(ex.frontiers, job) if ex.frontiers is not None else None
            return mine is not None and robot is not None and list(mine.points_cm) == list(robot.points_cm)

        same_status = ex.spin_until(status_matches, 5.0, run.tick)
        same_places = ex.spin_until(places_match, 5.0, run.tick)
        places = latest(phone, pb.DATA_FRONTIERS)
        run.check("same as robot", same_status and same_places,
                  f"status equal {same_status}; places equal {same_places} ({len(places.points_cm) // 2 if places else 0} places, "
                  f"exploration {places.job_id if places else '?'})")

        # 4. a phone that connects mid-run
        late = Phone("late").start()
        phones.append(late)
        ex.spin_until(lambda: bool(late.opens()), 10.0)
        opened = late.opens()[0] if late.opens() else time.monotonic()
        run.wait(1.5)
        got = {name: late.first_after(opened, t) for name, t in
               (("status", pb.DATA_EXPLORATION_STATUS), ("places", pb.DATA_FRONTIERS), ("map", pb.DATA_MAP))}
        delays = {k: round(v[0] - opened, 2) if v else None for k, v in got.items()}
        status_job = got["status"][1].job_id if got["status"] else None
        run.check("late phone", all(v is not None and v <= 1.0 for v in delays.values()) and status_job == job,
                  f"seconds after connecting: {delays}; the status's exploration {status_job} (it is {job}); "
                  f"places' exploration {got['places'][1].job_id if got['places'] else None}")

        # 5. Go to while exploring
        x, y, _ = n.pose()
        goto = ask(phone, pb.COMMAND_GO_TO, pb.GoTo(request_id=next(ids), x=x + 0.5, y=y, heading=0.0), 15.0)
        run.check("no Go to", goto is not None and goto.result in (Result.RESULT_REFUSED, Result.RESULT_NOT_AVAILABLE)
                  and modes.state.state == "exploring", f"Go to while exploring: {text(goto)}; mode {modes.state.state}")

        # 6. Stop
        before = ex.fresh_map()
        points = before.free_points() if before else None
        t = time.monotonic()
        reply = ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(ids), mode="idle"), 120.0)
        ex.spin_until(lambda: any(m.state == S.STATE_STOPPED for _, m in phone.messages(pb.DATA_EXPLORATION_STATUS, t)), 15.0)
        stopped = [m for _, m in phone.messages(pb.DATA_EXPLORATION_STATUS, t) if m.state == S.STATE_STOPPED]
        ex.spin_until(lambda: (latest(phone, pb.DATA_ROBOT_STATE) or pb.RobotState()).state == "mapping", 10.0)
        state = latest(phone, pb.DATA_ROBOT_STATE)
        after = ex.fresh_map()
        kept = after.still_free(points) if after and points is not None else 0.0
        emptied = ex.spin_until(lambda: (latest(phone, pb.DATA_FRONTIERS) or pb.Frontiers(points_cm=[1])).points_cm == [], 5.0)
        run.check("stop", reply is not None and reply.result == Result.RESULT_OK and stopped and state is not None
                  and state.state == "mapping" and state.making_map and kept >= SAME_MAP_SHARE and emptied,
                  f"idle: {text(reply)}; the phone saw '{stopped[0].message if stopped else 'no stopped status'}'; robot "
                  f"{state.state if state else '?'}, making_map {state.making_map if state else '?'}; "
                  f"{kept * 100:.1f}% of the floor is still floor; places emptied {emptied}")

        # 7. save the explored map from the phone, as the app does: SaveMap, then idle
        saved = ask(phone, pb.COMMAND_SAVE_MAP, pb.SaveMap(request_id=next(ids), name=TEST_MAP, replace=True), 60.0)
        idle = ask(phone, pb.COMMAND_CHANGE_MODE, pb.ChangeMode(request_id=next(ids), mode="idle"), 120.0)
        ex.spin_until(lambda: (latest(phone, pb.DATA_ROBOT_STATE) or pb.RobotState()).current_map == TEST_MAP, 10.0)
        state = latest(phone, pb.DATA_ROBOT_STATE)
        on_disk = os.path.isfile(os.path.join(cc.MAPS, TEST_MAP + ".yaml"))
        run.check("save", saved is not None and saved.result == Result.RESULT_OK and idle is not None
                  and idle.result == Result.RESULT_OK and state is not None and state.current_map == TEST_MAP
                  and not state.making_map and on_disk,
                  f"save: {text(saved)}; idle: {text(idle)}; the robot uses '{state.current_map if state else '?'}', "
                  f"making_map {state.making_map if state else '?'}; files written {on_disk}")
    finally:
        for p in phones:
            p.stop()
        results = run.finish()
        with open(marker) as f:
            now_using = f.read().strip()
        if now_using != map_before:
            with open(marker, "w") as f:
                f.write(map_before + "\n")
            subprocess.run(["pkill", "-f", "lib/vacuum_mode_manager/mode_manager"], check=False)
            print(f"map in use set back to '{map_before}' (mode manager restarted)", flush=True)
        remove_test_map()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
