#!/usr/bin/env python3
"""The assistant's question set: plain questions a user would type, each with what a good answer
must do. Needs the robot stack running with an API key for the assistant (see vacuum_assistant):
    source ros2_ws/install/setup.bash
    python3 sim/tests/assistant_eval.py            every question, each in a new conversation
    python3 sim/tests/assistant_eval.py 3 5        only these (numbers as listed)

A question passes when the assistant answered, looked at what it had to look at (tools), and its
answer has one of the expected words in it and none of the words it must not say. These questions
only read: doing things (going, cleaning, stopping) is tested in phase 41, where the robot's true
position is checked. Every run costs a
little: each question is one to three calls to the language model.
"""

import sys
import time
import uuid

import rclpy
from rclpy.node import Node

from vacuum_interfaces.srv import AskAssistant

# (question, tools it must use, words of which one must be in the answer, words none of which may be)
CASES = [
    ("What can you do?", [], ["clean", "room"], []),
    ("What are you doing right now?", ["get_status"], ["dock", "idle", "waiting", "clean", "standing", "still"], []),
    ("How did the last clean go?", ["get_cleaning_history"], ["m²", "m2", "square", "never", "no clean", "not cleaned", "haven't"], []),
    ("Is the floor clean?", [], ["clean"], ["dustbin", "filter", "brush"]),
    ("How fast do you clean?", ["get_status"], ["m/s", "metres per second", "meters per second", "cm"], []),
    ("Which rooms do you know?", ["get_rooms"], ["room", "kitchen", "bed"], []),
    ("When was the bathroom last cleaned?", ["get_rooms"], ["bath", "never", "not"], []),
    ("What do you see?", ["look"], [], []),
    ("Please draw a no-go zone around the stairs.", [], ["app", "zone"], []),
    ("Make a new map of the house.", [], ["app", "map"], []),
    ("Are you a real robot?", [], ["simulat"], []),
    ("Is anything wrong with you?", ["get_status"], ["no", "fine", "problem", "fault", "ok", "well"], []),
]


def main():
    chosen = [int(a) for a in sys.argv[1:]] or list(range(1, len(CASES) + 1))
    rclpy.init()
    node = Node("assistant_eval")
    client = node.create_client(AskAssistant, "assistant/ask")
    if not client.wait_for_service(timeout_sec=10.0):
        print("The assistant is not running (no /assistant/ask service).")
        return 1
    passed, times = 0, []
    for number in chosen:
        question, must_use, any_words, no_words = CASES[number - 1]
        t0 = time.monotonic()
        future = client.call_async(AskAssistant.Request(conversation=f"eval-{uuid.uuid4().hex[:8]}", text=question))
        rclpy.spin_until_future_complete(node, future, timeout_sec=120.0)
        took = time.monotonic() - t0
        result = future.result() if future.done() else None
        if result is None:
            print(f"[FAIL] {number}. {question}  no answer in 120 s")
            continue
        times.append(took)
        reply = result.reply.lower()
        problems = []
        if not result.success:
            problems.append("not answered")
        missing = [t for t in must_use if t not in result.tools_used]
        if missing:
            problems.append(f"did not use {', '.join(missing)}")
        if any_words and not any(w.lower() in reply for w in any_words):
            problems.append(f"none of {any_words} in the answer")
        bad = [w for w in no_words if w.lower() in reply]
        if bad:
            problems.append(f"says {bad}")
        ok = not problems
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {number}. {question}  ({took:.1f} s; tools {list(result.tools_used) or 'none'})")
        print(f"       {result.reply}")
        if problems:
            print(f"       problems: {'; '.join(problems)}")
    if times:
        times.sort()
        print(f"SUMMARY {passed}/{len(chosen)} passed; answer time median {times[len(times) // 2]:.1f} s, slowest {times[-1]:.1f} s")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if passed == len(chosen) else 1


if __name__ == "__main__":
    sys.exit(main())
