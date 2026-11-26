"""Holds the simulator's loop to its frame rate. No Isaac imports: plain time arithmetic.

Each frame steps the simulated clock by one period (a sixtieth of a second). For that clock to
keep up with the real one, the whole of a frame must take one period: the simulator's own update
and the sensor messages published after it (ros_publishers.py). Isaac Sim's own limiter times only
its update, so with it the messages came on top and the simulation ran at 0.8 of real time.
"""

import time


class FramePacer:
    """Call wait() once at the end of every frame.

    A frame that ran a little late is made up by the next ones, so the average holds. After a
    long stall the loop goes on at its usual pace from where it is: it does not run a burst of
    frames to catch up. A simulator too slow for the rate never waits.
    """

    def __init__(self, rate: float, clock=time.perf_counter, sleep=time.sleep):
        self._period = 1.0 / rate
        self._clock = clock
        self._sleep = sleep
        self._due = clock() + self._period  # when the frame under way should end

    def wait(self) -> None:
        now = self._clock()
        if now < self._due:
            self._sleep(self._due - now)
            now = self._due
        self._due = max(self._due + self._period, now)
