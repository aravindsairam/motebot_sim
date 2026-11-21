"""Helpers for the exploration acceptance tests (phases 29 onward).

Needs the ROS workspace sourced (vacuum_interfaces):
    source ros2_ws/install/setup.bash

Exploring   the exploration node's interfaces: /exploration/start, /exploration/status,
            /exploration/frontiers; and what the tests need to judge it: the map, Nav2's goals,
            slam_toolbox's lifecycle changes, and the explorer and node programs
"""

import math
import os
import subprocess
import time

import numpy as np
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseArray
from lifecycle_msgs.msg import TransitionEvent
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from vacuum_interfaces.msg import ExplorationStatus
from vacuum_interfaces.srv import StartExploration

FINAL = (ExplorationStatus.FINISHED, ExplorationStatus.STOPPED, ExplorationStatus.FAILED)
RUNNING = (ExplorationStatus.STARTING, ExplorationStatus.EXPLORING, ExplorationStatus.RETURNING, ExplorationStatus.ENDING)
_ACTIVE = (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING, GoalStatus.STATUS_CANCELING)


class MapSnapshot:
    """/map as numpy, enough to tell whether a later map is the same map, grown."""

    def __init__(self, msg: OccupancyGrid):
        self.res = msg.info.resolution
        self.ox, self.oy = msg.info.origin.position.x, msg.info.origin.position.y
        self.cells = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.free = (self.cells >= 0) & (self.cells < 50)
        self.known = int(np.count_nonzero(self.cells >= 0))
        self.free_area = float(np.count_nonzero(self.free)) * self.res ** 2

    def free_points(self):
        rows, cols = np.nonzero(self.free)
        return np.stack([self.ox + (cols + 0.5) * self.res, self.oy + (rows + 0.5) * self.res], axis=1)

    def still_free(self, points) -> float:
        if len(points) == 0:
            return 0.0
        cols = np.floor((points[:, 0] - self.ox) / self.res).astype(int)
        rows = np.floor((points[:, 1] - self.oy) / self.res).astype(int)
        ok = (rows >= 0) & (rows < self.cells.shape[0]) & (cols >= 0) & (cols < self.cells.shape[1])
        hit = np.zeros(len(points), dtype=bool)
        hit[ok] = self.free[rows[ok], cols[ok]]
        return float(hit.mean())


class Exploring:
    """The exploration node's interfaces, for a test node with a spin(seconds) method."""

    def __init__(self, node):
        self.node = node
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status = None  # latest ExplorationStatus
        self.history = []  # (monotonic, state, message, job_id) on every change
        self.frontiers = None
        self.map = None
        self.map_at = 0.0
        self.slam_events = []  # (monotonic, lifecycle state reached)
        self.goals = {}  # navigate_to_pose goal id -> last status
        self.goal_first_active = {}  # goal id -> monotonic time it was first seen under way
        self._start = node.create_client(StartExploration, "exploration/start")
        node.create_subscription(ExplorationStatus, "exploration/status", self._on_status, latched)
        node.create_subscription(PoseArray, "exploration/frontiers", lambda m: setattr(self, "frontiers", m), latched)
        node.create_subscription(OccupancyGrid, "map", self._on_map, latched)
        node.create_subscription(TransitionEvent, "slam_toolbox/transition_event",
                                 lambda e: self.slam_events.append((time.monotonic(), e.goal_state.label)), 10)
        node.create_subscription(GoalStatusArray, "navigate_to_pose/_action/status", self._on_goals, latched)

    def _on_status(self, m):
        self.status = m
        key = (m.state, m.message, m.job_id)
        if not self.history or self.history[-1][1:] != key:
            self.history.append((time.monotonic(),) + key)

    def _on_map(self, m):
        self.map, self.map_at = m, time.monotonic()

    def _on_goals(self, m):
        now = time.monotonic()
        for s in m.status_list:
            gid = bytes(bytearray(s.goal_info.goal_id.uuid))
            self.goals[gid] = s.status
            if s.status in _ACTIVE:
                self.goal_first_active.setdefault(gid, now)

    def goals_active(self) -> int:
        return sum(1 for s in self.goals.values() if s in _ACTIVE)

    def first_goal_since(self, t):
        times = [at for at in self.goal_first_active.values() if at >= t]
        return min(times) if times else None

    def spin_until(self, done, timeout, tick=None):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.node.spin(0.05)
            if tick is not None:
                tick()
            if done():
                return True
        return False

    def start(self, minutes=0.0, timeout=15.0):
        """Ask for an exploration. Returns the answer (accepted, message, job_id) or None."""
        if not self._start.wait_for_service(timeout_sec=10.0):
            return None
        for _ in range(2):  # a brand-new client can lose its first reply on Humble
            future = self._start.call_async(StartExploration.Request(time_limit=float(minutes) * 60.0))
            if self.spin_until(future.done, timeout):
                return future.result()
        return None

    def wait_state(self, states, timeout, job_id=None, tick=None):
        def done():
            s = self.status
            return s is not None and s.state in states and (job_id is None or s.job_id == job_id)
        return self.status if self.spin_until(done, timeout, tick) else None

    def result(self, job_id, timeout, tick=None):
        return self.wait_state(FINAL, timeout, job_id, tick)

    def fresh_map(self, timeout=10.0):
        asked = time.monotonic()
        if self.spin_until(lambda: self.map is not None and self.map_at > asked, timeout):
            return MapSnapshot(self.map)
        return None

    def slam_touched_since(self, t):
        return [label for at, label in self.slam_events if at >= t]


def pids_of(marker: str):
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    return [int(line.split(None, 1)[0]) for line in out[1:] if marker in line and "ps -eo" not in line]


def explorer_pids():
    return pids_of("lib/frontier_exploration_ros2/frontier_explorer")


def exploration_node_pids():
    return [p for p in pids_of("lib/vacuum_exploration/exploration_node") if p != os.getpid()]


def distance_from(track, x, y):
    return math.hypot(track[0] - x, track[1] - y)
