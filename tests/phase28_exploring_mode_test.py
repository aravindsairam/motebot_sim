#!/usr/bin/env python3
"""Phase 28 acceptance test: exploring mode, without the explorer (exploration plan, step 2a).

Exploring mode is mapping with Nav2 running: slam_toolbox keeps making the map while Nav2 drives.
This test stands in for the exploration node: it publishes the node's status itself
(/exploration/status), as the node will, and sends Nav2's goals itself, as the explorer will.

Needs the simulator and the robot stack running WITHOUT the exploration node (this test stands in
for it, and two reporters would contradict each other), the robot at its start pose:
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false exploration:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase28_exploring_mode_test.py

  1. no plain mode     a plain request for exploring is refused with the reason, from idle and from
                       mapping, and exploring is never offered
  2. enters exploring  from mapping, while the "exploration node" reports a job that is starting:
                       slam_toolbox active and AMCL not (one source of map -> odom); the safety gate
                       lets Nav2 drive
  3. goal on the map   a Nav2 goal 1.5 m away on the half-made map arrives; afterwards it is the
                       same map, grown, not a new one
  4. unmapped goal     a goal on an unmapped cell one cell beyond the mapped floor is accepted and
                       the robot arrives within 25 cm of it: the planner ends its path at most 15 cm
                       from the goal (its tolerance), and Nav2 stops within 10 cm of the path's end
  5. no spin, no back  a goal that fails (outside the house) moves the robot neither round nor back
  6. stop keeps map    a request for idle while exploring ends in mapping, with the same map
  7. stop mid-change   a request for idle sent during the change into exploring ends in mapping too
  8. node silent       the status going silent in exploring gives failsafe within 3 s; failsafe
                       then offers mapping, and Reset to mapping keeps the same map
  9. making_map        /robot_state says a map is being made in every message from the first
                       mapping to the end, and slam_toolbox was never switched off or on
"""

import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cleaning_client as cc  # noqa: E402
from lifecycle_msgs.msg import TransitionEvent  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from vacuum_interfaces.msg import ExplorationStatus, RobotState  # noqa: E402
from vacuum_interfaces.srv import ChangeState  # noqa: E402

ROBOT_RADIUS = 0.175
# The planner's tolerance (nav2.yaml GridBased.tolerance, 15 cm) plus the goal checker's (10 cm).
# The plan said 15 cm; the first run ended 15.1 cm away (2026-11-19).
UNMAPPED_ARRIVAL = 0.25
SAME_MAP_SHARE = 0.95  # of the floor mapped before, at least this much is still floor afterwards
OUTSIDE = (-1.5, 0.0)  # map metres: west of the dock wall, outside the house and outside the map


class Grid:
    """/map as numpy: cells -1 unknown, 0 free, 100 occupied; row 0 at the bottom."""

    def __init__(self, msg: OccupancyGrid):
        self.res = msg.info.resolution
        self.ox, self.oy = msg.info.origin.position.x, msg.info.origin.position.y
        self.cells = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.free = (self.cells >= 0) & (self.cells < 50)
        self.occupied = self.cells >= 50
        self.unknown = self.cells < 0
        # metres from each cell to the nearest occupied cell (unknown cells do not count)
        self.clearance = ndimage.distance_transform_edt(~self.occupied) * self.res

    def cell(self, x, y):
        return int(math.floor((y - self.oy) / self.res)), int(math.floor((x - self.ox) / self.res))

    def centre(self, r, c):
        return self.ox + (c + 0.5) * self.res, self.oy + (r + 0.5) * self.res

    def inside(self, r, c):
        return 0 <= r < self.cells.shape[0] and 0 <= c < self.cells.shape[1]

    def free_points(self):
        rows, cols = np.nonzero(self.free)
        return np.stack([self.ox + (cols + 0.5) * self.res, self.oy + (rows + 0.5) * self.res], axis=1)

    def still_free(self, points) -> float:
        """Share of these points (metres) that are free on this map."""
        cols = np.floor((points[:, 0] - self.ox) / self.res).astype(int)
        rows = np.floor((points[:, 1] - self.oy) / self.res).astype(int)
        ok = (rows >= 0) & (rows < self.cells.shape[0]) & (cols >= 0) & (cols < self.cells.shape[1])
        hit = np.zeros(len(points), dtype=bool)
        hit[ok] = self.free[rows[ok], cols[ok]]
        return float(hit.mean()) if len(points) else 0.0

    def reachable(self, x, y, margin=0.20):
        """Free cells with room for the robot, connected to the robot's own cell."""
        room = self.free & (self.clearance >= margin)
        labels, _ = ndimage.label(room)
        r, c = self.cell(x, y)
        if not self.inside(r, c):
            return np.zeros_like(room)
        if labels[r, c] == 0:  # the robot's own cell is a little tight: take the nearest roomy cell
            rows, cols = np.nonzero(room)
            if len(rows) == 0:
                return room
            k = np.argmin((rows - r) ** 2 + (cols - c) ** 2)
            r, c = rows[k], cols[k]
        return labels == labels[r, c]


def goal_on_floor(g: Grid, x, y, distance=1.5):
    """A free cell about `distance` from (x, y), as far from obstacles as possible."""
    reach = g.reachable(x, y)
    best, best_clear = None, 0.0
    for r, c in zip(*np.nonzero(reach & (g.clearance >= 0.40))):
        gx, gy = g.centre(r, c)
        if abs(math.hypot(gx - x, gy - y) - distance) <= 0.1 and g.clearance[r, c] > best_clear:
            best, best_clear = (gx, gy), g.clearance[r, c]
    return best, best_clear


def goal_beyond_floor(g: Grid, x, y):
    """An unknown cell next to reachable free floor with room for the robot: the place an explorer
    aims for. The nearest one at least 1 m away. Returns ((x, y), its free neighbour's clearance)."""
    reach = g.reachable(x, y)
    roomy = reach & (g.clearance >= 0.35)
    # unknown cells with a roomy free 4-neighbour
    near = np.zeros_like(roomy)
    near[1:, :] |= roomy[:-1, :]
    near[:-1, :] |= roomy[1:, :]
    near[:, 1:] |= roomy[:, :-1]
    near[:, :-1] |= roomy[:, 1:]
    unknown_neighbours = ndimage.convolve(g.unknown.astype(int), np.ones((3, 3), dtype=int), mode="constant") - g.unknown
    frontier = g.unknown & near & (unknown_neighbours >= 3)
    best, best_d = None, float("inf")
    for r, c in zip(*np.nonzero(frontier)):
        gx, gy = g.centre(r, c)
        d = math.hypot(gx - x, gy - y)
        if 1.0 <= d < best_d:
            best, best_d = (gx, gy), d
    return best, best_d


class Stand:
    """Stands in for the exploration node: publishes /exploration/status five times a second."""

    def __init__(self, node):
        self.state = None  # None: silent
        self.job_id = 0
        self._pub = node.create_publisher(ExplorationStatus, "exploration/status", 10)
        node.create_timer(0.2, self._tick)

    def set(self, state):
        if state == ExplorationStatus.STARTING:
            self.job_id += 1
        self.state = state
        self._tick()

    def _tick(self):
        if self.state is None:
            return
        msg = ExplorationStatus(state=self.state, job_id=self.job_id, message="phase 28 stands in for the exploration node")
        self._pub.publish(msg)


def main():
    run = cc.Run("phase28")
    n, modes = run.n, run.modes
    stand = Stand(n)
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    box = {"map": None, "map_wall": 0.0, "states": [], "slam_events": [], "watch": None, "motion": []}

    def on_map(m):
        box["map"], box["map_wall"] = m, time.monotonic()

    def on_state(m):
        box["states"].append((time.monotonic(), m.state, m.transitioning, m.making_map, m.reason))

    def on_truth(m):
        if box["watch"] is not None:
            p, q = m.pose.pose.position, m.pose.pose.orientation
            box["motion"].append((p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))

    n.create_subscription(OccupancyGrid, "map", on_map, latched)
    n.create_subscription(RobotState, "robot_state", on_state, latched)
    n.create_subscription(TransitionEvent, "slam_toolbox/transition_event",
                          lambda e: box["slam_events"].append((time.monotonic(), e.goal_state.label)), 10)
    n.create_subscription(Odometry, "ground_truth/odom", on_truth, 50)
    change = n.create_client(ChangeState, "change_state")

    def fresh_map(timeout=10.0):
        asked = time.monotonic()
        run.n.spin(0.05)
        end = asked + timeout
        while time.monotonic() < end:
            run.wait(0.1)
            if box["map"] is not None and box["map_wall"] > asked:
                return Grid(box["map"])
        return None

    def where():
        x, y, _ = n.slam_pose()
        return x, y

    def ask_async(state):
        return change.call_async(ChangeState.Request(new_state=state))

    def answer(future, timeout=120.0):
        end = time.monotonic() + timeout
        while not future.done() and time.monotonic() < end:
            run.wait(0.05)
        r = future.result() if future.done() else None
        return (r.success, r.message) if r is not None else (False, "no answer")

    def enter_exploring():
        stand.set(ExplorationStatus.STARTING)
        run.wait(0.5)
        ok, msg = modes.request("exploring", 120.0)
        if ok:
            stand.set(ExplorationStatus.EXPLORING)
        return ok, msg

    def slam_touched_since(t):
        return [label for at, label in box["slam_events"] if at >= t]

    try:
        if not run.ready():
            return run.results
        import exploration_client as ec  # noqa: E402

        if ec.exploration_node_pids():
            run.check("no exploration node", False, "the exploration node is running: start the stack with exploration:=false")
            return run.results

        # 1. a plain request for exploring, from idle and from mapping
        ok_idle, msg_idle = modes.request("exploring", 30.0)
        offered_idle = "exploring" in modes.state.allowed_transitions
        ok_m, msg_m = modes.request("mapping", 120.0)
        mapping_since = time.monotonic()
        run.wait(4.0)  # a few scans and one map from slam_toolbox
        ok_plain, msg_plain = modes.request("exploring", 30.0)
        offered_mapping = "exploring" in modes.state.allowed_transitions
        run.check("no plain mode",
                  not ok_idle and "started by an exploration" in msg_idle and not offered_idle and ok_m
                  and not ok_plain and "started by an exploration" in msg_plain and not offered_mapping
                  and modes.state.state == "mapping",
                  f"from idle: '{msg_idle}', offered {offered_idle}; mapping: '{msg_m}'; from mapping: "
                  f"'{msg_plain}', offered {offered_mapping}; mode {modes.state.state}")

        # 2. into exploring
        fresh_map()  # slam_toolbox has published a map: Nav2's costmap has something to start from
        ok, msg = enter_exploring()
        run.wait(1.0)
        slam, amcl = modes.lifecycle("slam_toolbox"), modes.lifecycle("amcl")
        source = n.safety.source if n.safety else "?"
        st = modes.state
        run.check("enters exploring",
                  ok and st.state == "exploring" and slam == "active" and amcl != "active" and source == "nav2"
                  and st.making_map,
                  f"request: '{msg}'; mode {st.state}, making_map {st.making_map}; slam_toolbox {slam}, amcl {amcl}; "
                  f"the safety gate lets {source} drive")

        # 3. a goal 1.5 m away on the half-made map
        g1 = fresh_map()
        x, y = where()
        goal, clear = goal_on_floor(g1, x, y) if g1 is not None else (None, 0.0)
        before = g1.free_points() if g1 is not None else np.zeros((0, 2))
        known_before = int((~g1.unknown).sum()) if g1 is not None else 0
        status, secs, dist = n.navigate(goal[0], goal[1], 0.0, sim_timeout=90.0) if goal else ("no goal found", 0, 0)
        run.wait(0.5)
        ax, ay = where()
        off = math.hypot(ax - goal[0], ay - goal[1]) if goal else float("nan")
        g2 = fresh_map()
        kept = g2.still_free(before) if g2 is not None else 0.0
        known_after = int((~g2.unknown).sum()) if g2 is not None else 0
        run.check("goal on the map",
                  goal is not None and status == "succeeded" and off <= 0.15 and kept >= SAME_MAP_SHARE
                  and known_after >= known_before and modes.state.state == "exploring",
                  f"goal ({goal[0]:.2f}, {goal[1]:.2f}) {math.hypot(goal[0] - x, goal[1] - y):.2f} m away, {clear:.2f} m "
                  f"from obstacles: {status} in {secs:.1f} s, {dist:.2f} m, ends {off * 100:.1f} cm from it; "
                  f"{kept * 100:.1f}% of the floor mapped before is still floor; known cells {known_before} -> {known_after}"
                  if goal else "no free cell 1.5 m away with 40 cm of room on the map")

        # 4. a goal on an unmapped cell, one cell beyond the floor
        g3 = fresh_map()
        x, y = where()
        goal, d = goal_beyond_floor(g3, x, y) if g3 is not None else (None, 0.0)
        if goal is not None:
            r, c = g3.cell(*goal)
            was = int(g3.cells[r, c])
            status, secs, dist = n.navigate(goal[0], goal[1], 0.0, sim_timeout=150.0)
            run.wait(0.5)
            ax, ay = where()
            off = math.hypot(ax - goal[0], ay - goal[1])
            run.check("unmapped goal", was < 0 and status == "succeeded" and off <= UNMAPPED_ARRIVAL,
                      f"goal ({goal[0]:.2f}, {goal[1]:.2f}), cell value {was} (unknown), {d:.2f} m away: {status} in "
                      f"{secs:.1f} s, {dist:.2f} m driven, ends {off * 100:.1f} cm from it")
        else:
            run.check("unmapped goal", False, "no unknown cell next to roomy reachable floor on the map")

        # 5. a goal that fails: no spin, no back-up
        run.wait(1.0)
        box["motion"], box["watch"] = [], True
        status, secs, _ = n.navigate(OUTSIDE[0], OUTSIDE[1], 0.0, sim_timeout=120.0)
        run.wait(1.0)
        box["watch"] = None
        track = box["motion"]
        if track:
            x0, y0, yaw0 = track[0]
            moved = max(math.hypot(px - x0, py - y0) for px, py, _ in track)
            turned = max(abs(math.atan2(math.sin(a - yaw0), math.cos(a - yaw0))) for _, _, a in track)
        else:
            moved = turned = float("nan")
        run.check("no spin, no back-up",
                  status in ("aborted", "rejected") and moved < 0.03 and math.degrees(turned) < 5.0,
                  f"goal outside the house: {status} after {secs:.1f} s; meanwhile the robot moved at most "
                  f"{moved * 100:.1f} cm and turned at most {math.degrees(turned):.1f} deg")

        # 6. Stop (a request for idle) while exploring
        g4 = fresh_map()
        before = g4.free_points() if g4 is not None else np.zeros((0, 2))
        sim_t0 = n.sim_time
        ok, msg = modes.request("idle", 120.0)
        rest = n.stop_time_since(sim_t0)  # across the blocking call: see phase 29
        stand.set(ExplorationStatus.STOPPED)
        g5 = fresh_map()
        kept = g5.still_free(before) if g5 is not None else 0.0
        slam = modes.lifecycle("slam_toolbox")
        st = modes.state
        run.check("stop keeps the map",
                  ok and "map is kept" in msg and st.state == "mapping" and slam == "active" and kept >= SAME_MAP_SHARE
                  and st.making_map,
                  f"idle: '{msg}'; mode {st.state} ('{st.reason}'), making_map {st.making_map}; slam_toolbox {slam}; "
                  f"at rest after {rest} s of robot time; {kept * 100:.1f}% of the floor is still floor")

        # 7. Stop sent during the change into exploring
        before = g5.free_points() if g5 is not None else np.zeros((0, 2))
        stand.set(ExplorationStatus.STARTING)
        run.wait(0.5)
        into = ask_async("exploring")
        seen_change = run.clean.spin_until(lambda: modes.state is not None and modes.state.transitioning, 10.0)
        stop = ask_async("idle")
        into_ok, into_msg = answer(into)
        stop_ok, stop_msg = answer(stop)
        stand.set(ExplorationStatus.STOPPED)
        modes.wait_for_state("mapping", 30.0)
        g6 = fresh_map()
        kept = g6.still_free(before) if g6 is not None else 0.0
        st = modes.state
        run.check("stop during the change",
                  seen_change and into_ok and stop_ok and st.state == "mapping" and kept >= SAME_MAP_SHARE,
                  f"stop sent while changing: {seen_change}; exploring: '{into_msg}'; idle: '{stop_msg}'; "
                  f"mode {st.state}; {kept * 100:.1f}% of the floor is still floor")

        # 8. the exploration node falls silent in exploring
        ok, msg = enter_exploring()
        run.wait(2.0)
        g7 = fresh_map()
        before = g7.free_points() if g7 is not None else np.zeros((0, 2))
        t = time.monotonic()
        stand.set(None)  # silent
        failsafe = modes.wait_for_state("failsafe", 10.0)
        took = time.monotonic() - t
        st = modes.state
        why, offered, making = st.reason, list(st.allowed_transitions), st.making_map
        rest = run.at_rest_after(t)
        run.wait(1.0)
        offered = list(modes.state.allowed_transitions)
        ok_back, msg_back = modes.request("mapping", 120.0)
        g8 = fresh_map()
        kept = g8.still_free(before) if g8 is not None else 0.0
        slam = modes.lifecycle("slam_toolbox")
        run.check("node silent",
                  ok and failsafe and took <= 3.0 and "exploration node" in why and "mapping" in offered and making
                  and ok_back and modes.state.state == "mapping" and slam == "active" and kept >= SAME_MAP_SHARE,
                  f"exploring: '{msg}'; failsafe after {took:.1f} s ('{why}'), making_map {making}, offers {offered}; "
                  f"at rest after {rest} s; Reset to mapping: '{msg_back}'; slam_toolbox {slam}; "
                  f"{kept * 100:.1f}% of the floor is still floor")

        # 9. making_map throughout, and slam_toolbox never switched
        seen = [(s, tr, mm) for at, s, tr, mm, _ in box["states"] if at >= mapping_since]
        wrong = sorted({(s, tr) for s, tr, mm in seen if not mm})
        touched = slam_touched_since(mapping_since)
        run.check("making_map", seen and not wrong and not touched,
                  f"{len(seen)} robot states since mapping began; making_map false in: {wrong or 'none'}; "
                  f"slam_toolbox lifecycle changes: {touched or 'none'}")
    finally:
        stand.set(None)
        results = run.finish()  # asks for idle: leaves mapping and drops the test's map, unsaved
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
