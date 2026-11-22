#!/usr/bin/env python3
"""Phase 11 acceptance test: the app's gateway (step 1 of the Android app), against the robot.

Needs two things running first, with the robot at its start pose:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false     (it starts the gateway)
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase11_gateway_test.py

A scripted phone connects to ws://127.0.0.1:9090 and speaks the same messages as the app
(proto/vacuum.proto). It reconnects by itself, like the app, after half a second.
  1. hello first        the first message is Hello, with the protocol version and the robot's name
  2. state at once      a phone that connects gets the robot state and safety status within 0.5 s
  3. matches the robot  what the phone receives equals /robot_state and /safety_status
  4. rates              robot state ~2 a second, safety status ~2 a second (not the gate's 10), heartbeat 1 a second
  5. mode change        through idle -> manual -> idle the phone sees every step /robot_state shows, within 0.5 s
  6. second phone       a phone that connects later gets the current state at once
  7. gateway killed     the phone notices within 1 s; the gateway is restarted; the phone is back within 10 s
                        and has the robot state within 5 s of reconnecting. Not 0.5 s as in 2 and 6: a
                        restarted gateway hears nothing until ROS has discovered it, which took 2.6 s in
                        one run and 0.2 s in the next (2026-11-14). Until then the app says "waiting for
                        the robot".
  8. robot unaffected   while the gateway is dead the robot keeps publishing, stays in idle, no faults
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from websockets.asyncio.client import connect

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mode_client import ModeClient  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from vacuum_gateway import framing  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402
from vacuum_interfaces.msg import RobotState, SafetyStatus  # noqa: E402

URL = "ws://127.0.0.1:9090"
DECODERS = {
    pb.DATA_HELLO: pb.Hello,
    pb.DATA_HEARTBEAT: pb.Heartbeat,
    pb.DATA_ROBOT_STATE: pb.RobotState,
    pb.DATA_SAFETY_STATUS: pb.SafetyStatus,
    pb.DATA_MAP: pb.Map,
    pb.DATA_ROBOT_POSE: pb.RobotPose,
    pb.DATA_ROUTE: pb.Route,
    pb.DATA_NAVIGATION: pb.NavigationStatus,
    pb.DATA_NO_GO_ZONES: pb.NoGoZones,
    pb.DATA_CLEANING_STATUS: pb.CleaningStatus,
    pb.DATA_CLEANING_PLAN: pb.CleaningPlan,
    pb.DATA_CLEANED_AREA: pb.CleanedArea,
    pb.DATA_EXPLORATION_STATUS: pb.ExplorationStatus,
    pb.DATA_FRONTIERS: pb.Frontiers,
    pb.DATA_ROBOT_SETTINGS: pb.RobotSettings,
    pb.DATA_REPLY: pb.Reply,
}


class Phone:
    """A scripted phone. Runs in its own thread with its own asyncio loop and records, with the
    time of each (time.monotonic): ("open",), ("close",), and ("msg", type, decoded message).
    send() sends a command from any thread."""

    def __init__(self, name):
        self.name = name
        self.events = []
        self._lock = threading.Lock()
        self._stop = False
        self._ws = None
        self._loop = None
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
        self._thread.join(5.0)

    def _record(self, *event):
        with self._lock:
            self.events.append((time.monotonic(),) + event)

    def snapshot(self):
        with self._lock:
            return list(self.events)

    async def _run(self):
        while not self._stop:
            try:
                async with connect(URL, compression=None, open_timeout=3) as ws:
                    self._ws, self._loop = ws, asyncio.get_running_loop()
                    self._record("open")
                    while not self._stop:
                        try:
                            data = await asyncio.wait_for(ws.recv(), 0.2)
                        except asyncio.TimeoutError:
                            continue
                        frame = framing.decode(data, 1 << 24)
                        decoder = DECODERS.get(frame.type)
                        self._record("msg", frame.type, decoder.FromString(frame.payload) if decoder else None)
            except Exception:
                pass
            if any(e[1] == "open" for e in self.snapshot()):
                self._record("close")
            if not self._stop:
                await asyncio.sleep(0.5)  # the app's first retry pause

    def send(self, msg_type, message):
        """Send a command (request_id set by the caller). Returns the time it was sent."""
        frame = framing.encode(pb.CHANNEL_COMMAND, msg_type, message.SerializeToString(), 1 << 20)
        # The time is taken before the frame leaves. The gateway answers a ping in well under a
        # millisecond, so a time taken afterwards can be later than the reply, and reply_to()
        # would then not count that reply as the answer.
        sent = time.monotonic()
        asyncio.run_coroutine_threadsafe(self._ws.send(frame), self._loop).result(2.0)
        return sent

    def reply_to(self, request_id, since=0.0):
        """(time, Reply) for that request, or None if it has not arrived."""
        found = [m for m in self.messages(pb.DATA_REPLY, since) if m[1].request_id == request_id]
        return found[0] if found else None

    # --- reading the record
    def messages(self, msg_type, since=0.0, until=float("inf")):
        return [(e[0], e[3]) for e in self.snapshot() if e[1] == "msg" and e[2] == msg_type and since <= e[0] <= until]

    def opens(self):
        return [e[0] for e in self.snapshot() if e[1] == "open"]

    def closes(self):
        return [e[0] for e in self.snapshot() if e[1] == "close"]

    def first_after(self, t, msg_type):
        found = self.messages(msg_type, since=t)
        return found[0] if found else None


class TestNode(Node):
    """Records every /robot_state and /safety_status with its arrival time."""

    def __init__(self):
        super().__init__("phase11_gateway_test")
        self.states = []  # (time, state, transitioning, reason, faults)
        self.safety = []  # (time, source)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(RobotState, "robot_state", self._on_state, latched)
        self.create_subscription(SafetyStatus, "safety_status", lambda m: self.safety.append((time.monotonic(), m.source)), 10)

    def _on_state(self, m):
        self.states.append((time.monotonic(), m.state, m.transitioning, m.reason, list(m.faults)))

    def spin(self, seconds=0.02):
        rclpy.spin_once(self, timeout_sec=seconds)

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.spin(0.02)


def distinct(seq):
    out = []
    for item in seq:
        if not out or out[-1] != item:
            out.append(item)
    return out


def gateway_pids():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    return [int(line.split(None, 1)[0]) for line in out[1:] if "lib/vacuum_gateway/gateway" in line and "python" in line]


def main():
    rclpy.init()
    n = TestNode()
    modes = ModeClient(n)
    results = []
    phones = []

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if modes.wait_ready(60.0) is None or not modes.wait_for_state("idle", 30.0):
            check("robot ready", False, "the mode manager did not reach idle")
            return results
        if not gateway_pids():
            check("gateway running", False, "no vacuum_gateway process found (is the stack from this workspace?)")
            return results
        n.wait(2.0)  # let the gateway hear a few messages

        # 1-4: one phone connects and listens for 6 s
        phone = Phone("phone 1").start()
        phones.append(phone)
        n.wait(6.5)
        opened = phone.opens()
        if not opened:
            check("hello first", False, "the phone could not connect to ws://127.0.0.1:9090")
            return results
        t_open = opened[0]
        first = [e for e in phone.snapshot() if e[1] == "msg"][0]
        hello = first[3]
        check("hello first", first[2] == pb.DATA_HELLO and hello.protocol_version == 1 and hello.robot_name,
              f"first message type 0x{first[2]:02x}; protocol {getattr(hello, 'protocol_version', '?')}, "
              f"robot '{getattr(hello, 'robot_name', '?')}', gateway {getattr(hello, 'gateway_version', '?')}")

        s0 = phone.first_after(t_open, pb.DATA_ROBOT_STATE)
        g0 = phone.first_after(t_open, pb.DATA_SAFETY_STATUS)
        delays = [round(m[0] - t_open, 3) if m else None for m in (s0, g0)]
        check("state at once", s0 and g0 and max(delays) <= 0.5,
              f"robot state after {delays[0]} s, safety status after {delays[1]} s (limit 0.5 s)")

        ros_state = modes.state
        ros_source = n.safety[-1][1] if n.safety else None
        got_state, got_safety = s0[1] if s0 else None, g0[1] if g0 else None
        same_state = got_state is not None and (
            got_state.state, got_state.previous_state, got_state.reason, got_state.transitioning,
            list(got_state.allowed_transitions), list(got_state.faults), got_state.current_map) == (
            ros_state.state, ros_state.previous_state, ros_state.reason, ros_state.transitioning,
            list(ros_state.allowed_transitions), list(ros_state.faults), ros_state.current_map)
        check("matches the robot", same_state and got_safety is not None and got_safety.source == ros_source,
              f"phone: {got_state.state if got_state else None} / map '{got_state.current_map if got_state else ''}' / "
              f"allowed {list(got_state.allowed_transitions) if got_state else []} / driver {got_safety.source if got_safety else None}; "
              f"ROS: {ros_state.state} / map '{ros_state.current_map}' / allowed {list(ros_state.allowed_transitions)} / driver {ros_source}")

        window = (t_open + 1.0, t_open + 6.0)
        counts = {name: len(phone.messages(t, *window)) for name, t in
                  (("robot state", pb.DATA_ROBOT_STATE), ("safety status", pb.DATA_SAFETY_STATUS), ("heartbeat", pb.DATA_HEARTBEAT))}
        gate_rate = len([s for s in n.safety if window[0] <= s[0] <= window[1]]) / 5.0
        check("rates", 8 <= counts["robot state"] <= 13 and 8 <= counts["safety status"] <= 13 and 4 <= counts["heartbeat"] <= 6,
              f"in 5 s: robot state {counts['robot state']}, safety status {counts['safety status']} "
              f"(the gate published {gate_rate:.0f} a second), heartbeats {counts['heartbeat']}")

        # 5: a mode change, idle -> manual -> idle
        t_change = time.monotonic()
        ok1, msg1 = modes.request("manual")
        n.wait(1.0)
        ok2, msg2 = modes.request("idle")
        n.wait(1.5)
        t_end = time.monotonic()
        ros_steps = distinct([(s[1], s[2]) for s in n.states if t_change <= s[0] <= t_end])
        phone_msgs = phone.messages(pb.DATA_ROBOT_STATE, t_change, t_end + 0.5)
        phone_steps = distinct([(m.state, m.transitioning) for _, m in phone_msgs])
        lags = []
        for step in ros_steps:
            t_ros = next(s[0] for s in n.states if t_change <= s[0] and (s[1], s[2]) == step)
            t_phone = next((t for t, m in phone_msgs if t >= t_ros - 0.05 and (m.state, m.transitioning) == step), None)
            lags.append(None if t_phone is None else round(max(0.0, t_phone - t_ros), 3))
        manual_driver = any(m.source == "manual" for _, m in phone.messages(pb.DATA_SAFETY_STATUS, t_change, t_end))
        check("mode change", ok1 and ok2 and phone_steps == ros_steps and None not in lags and max(lags) <= 0.5 and manual_driver,
              f"ROS: {ros_steps}; phone: {phone_steps}; phone behind ROS by {lags} s; "
              f"phone saw the gate let manual drive: {manual_driver}")

        # 6: a second phone, connecting later
        phone2 = Phone("phone 2").start()
        phones.append(phone2)
        n.wait(1.5)
        o2 = phone2.opens()
        s2 = phone2.first_after(o2[0], pb.DATA_ROBOT_STATE) if o2 else None
        g2 = phone2.first_after(o2[0], pb.DATA_SAFETY_STATUS) if o2 else None
        d2 = [round(m[0] - o2[0], 3) if m else None for m in (s2, g2)]
        check("second phone", bool(o2) and s2 and g2 and max(d2) <= 0.5 and s2[1].state == modes.state.state,
              f"robot state '{s2[1].state if s2 else None}' after {d2[0]} s, safety status after {d2[1]} s; "
              f"robot is in '{modes.state.state}'")
        phone2.stop()

        # 7-8: kill the gateway
        before = gateway_pids()
        states_before = len(n.states)
        t_kill = time.monotonic()
        for pid in before:
            os.kill(pid, signal.SIGKILL)
        noticed = back = None
        while time.monotonic() - t_kill < 20.0:
            n.spin(0.05)
            if noticed is None:
                c = [t for t in phone.closes() if t >= t_kill]
                noticed = c[0] if c else None
            o = [t for t in phone.opens() if t > t_kill]
            if o and phone.first_after(o[0], pb.DATA_ROBOT_STATE):
                back = o[0]
                break
        after = gateway_pids()
        state_again = phone.first_after(back, pb.DATA_ROBOT_STATE) if back else None
        check("gateway killed",
              noticed is not None and noticed - t_kill <= 1.0 and back is not None and back - t_kill <= 10.0
              and state_again is not None and state_again[0] - back <= 5.0 and after and set(after).isdisjoint(before),
              f"phone noticed after {round(noticed - t_kill, 2) if noticed else None} s, connected again after "
              f"{round(back - t_kill, 2) if back else None} s and had the robot state "
              f"{round(state_again[0] - back, 3) if state_again else None} s later; gateway pid {before} -> {after}")

        down = [s for s in n.states[states_before:] if t_kill <= s[0] <= (back or time.monotonic())]
        unaffected = len(down) >= 2 and all(s[1] == "idle" and not s[4] for s in down)
        check("robot unaffected", unaffected,
              f"{len(down)} robot states while the gateway was away, all idle without faults: {unaffected}")

        out = os.environ.get("PHASE11_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results}, f)
    finally:
        for p in phones:
            p.stop()
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
