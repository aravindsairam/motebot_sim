#!/usr/bin/env python3
"""Phase 39 acceptance test: two worlds, one switch. The robot software cannot run on the wrong world.

Run with the simulator on the home and NO robot stack (this test starts and stops it):
    tools/start_sim.sh --headless --world home
    python3 sim/tests/phase39_world_switch_test.py
or: tools/run_tests.sh 39

Checks:
  1. script refuses     tools/start_stack.sh world:=house refuses at once and names both worlds
  2. manager refuses    the stack launched by hand for the house (world:=house) ends in failsafe,
                        "wrong world", and Reset (idle) is refused
  3. right world        tools/start_stack.sh with no world given starts for the home, with the
                        home's own maps folder, ready in idle
"""

import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", ".."))
LOG = os.path.join(R, "logs", "stack.log")
SOURCE = f"source /opt/ros/humble/setup.bash && source {R}/ros2_ws/install/setup.bash"


def sh(command, timeout=240):
    out = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout, cwd=R)
    return out.returncode, out.stdout + out.stderr


def stop_stack():
    sh(f"{R}/tools/stop_all.sh --keep-sim", timeout=120)
    time.sleep(2.0)


def robot_state():
    _, out = sh(f"{SOURCE} && timeout 15 ros2 topic echo --once /robot_state vacuum_interfaces/msg/RobotState", 30)
    state = re.search(r"^state: (\S+)", out, re.M)
    return (state.group(1) if state else None), out


def main():
    results = []

    def check(name, ok, detail):
        results.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        stop_stack()
        # 1. the start script asks the simulator, and refuses another world
        t0 = time.time()
        code, out = sh(f"{R}/tools/start_stack.sh rviz:=false world:=house")
        took = time.time() - t0
        _, running = sh("ps -eo args | grep -c '[l]ib/vacuum_mode_manager/mode_manager'")
        check("script refuses", code != 0 and "'home'" in out and "'house'" in out and took < 30.0
              and running.strip() == "0",
              f"exit {code} after {took:.1f} s, mode managers running: {running.strip()}: {out.strip()[:200]}")

        # 2. launched by hand for the wrong world: the mode manager stays in failsafe
        sh(f"{SOURCE} && (nohup ros2 launch vacuum_bringup vacuum.launch.py rviz:=false world:=house > {LOG} 2>&1 &)")
        deadline, state, text = time.time() + 90.0, None, ""
        while time.time() < deadline:
            state, text = robot_state()
            if state == "failsafe":
                break
            time.sleep(2.0)
        code, reset = sh(f"{SOURCE} && timeout 60 ros2 run vacuum_mode_manager mode idle", 90)
        state_after, _ = robot_state()
        reason = re.search(r"^reason: (.*)$", text, re.M)
        check("manager refuses", state == "failsafe" and "wrong world" in text and "REFUSED" in reset
              and state_after == "failsafe",
              f"state {state}, reason {reason.group(1)[:160] if reason else '?'}; Reset: {reset.strip()[:120]}; "
              f"state afterwards {state_after}")
        stop_stack()

        # 3. no world given: the stack follows the simulator
        code, out = sh(f"{R}/tools/start_stack.sh rviz:=false")
        with open(LOG) as f:
            log = f.read()
        started = re.search(r"mode manager started for world '(\w+)'; maps in (\S+),", log)
        maps = started.group(2) if started else "?"
        check("right world", code == 0 and "world: home" in out and started is not None
              and started.group(1) == "home" and maps.endswith("/ros2_ws/maps_home"),
              f"exit {code}; {out.strip().splitlines()[0] if out.strip() else ''}; "
              f"mode manager for world {started.group(1) if started else '?'}, maps in {maps}; "
              f"{'ready in idle' if 'ready in idle' in log else 'NOT ready'}")
    finally:
        stop_stack()

    print(f"SUMMARY {sum(results)}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(res) else 1)
