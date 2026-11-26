"""The simulator's frame pacing (no simulator): each frame, with everything done in it, takes
one sixtieth of a second, so the simulated clock keeps up with the real one.

Run: python3 -m pytest sim/tests/test_frame_pacer.py
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "scripts")))

from frame_pacer import FramePacer  # noqa: E402

PERIOD = 1.0 / 60.0


class Time:
    """A clock the test moves: work takes time, and so does a sleep."""

    def __init__(self):
        self.now = 100.0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        assert seconds > 0.0
        self.now += seconds

    def work(self, seconds):
        self.now += seconds


def pacer(t):
    return FramePacer(60.0, clock=t.clock, sleep=t.sleep)


def test_a_quick_frame_waits_for_the_rest_of_its_time():
    t = Time()
    p = pacer(t)
    t.work(0.010)
    p.wait()
    assert t.now == pytest.approx(100.0 + PERIOD)


def test_the_work_done_after_the_frame_counts_too():
    # The picture and the physics take 13 ms, the sensor messages 3 ms more: still one period.
    t = Time()
    p = pacer(t)
    for _ in range(60):
        t.work(0.013)
        t.work(0.003)
        p.wait()
    assert t.now == pytest.approx(101.0)


def test_a_frame_that_ran_a_little_late_is_made_up_by_the_next():
    t = Time()
    p = pacer(t)
    t.work(0.020)
    p.wait()
    assert t.now == pytest.approx(100.020)  # late already: no waiting
    t.work(0.010)
    p.wait()
    assert t.now == pytest.approx(100.0 + 2 * PERIOD)


def test_after_a_long_stall_it_does_not_rush_to_catch_up():
    t = Time()
    p = pacer(t)
    t.work(0.5)  # the simulator froze for half a second
    p.wait()
    start = t.now
    for _ in range(10):
        t.work(0.001)
        p.wait()
    # At its usual pace again, not ten frames at once: one frame goes at once, nine are held.
    assert t.now - start == pytest.approx(9 * PERIOD)


def test_a_simulator_too_slow_for_the_rate_never_waits():
    t = Time()
    p = pacer(t)
    for _ in range(10):
        t.work(0.020)
        p.wait()
    assert t.now == pytest.approx(100.0 + 10 * 0.020)
