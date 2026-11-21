"""Helpers for the cleaning acceptance tests (phases 18 to 26).

Needs the ROS workspace sourced (vacuum_interfaces, vacuum_coverage):
    source ros2_ws/install/setup.bash

Cleaning     the robot's cleaning interfaces: the /clean action, the services, the three topics
Truth        what the robot REALLY cleaned and drove, from the simulator's true pose (/ground_truth/odom)
true_floor   the true open floor from sim/scripts/house_layout.py, laid on the saved map's grid

The world frame (the simulator's truth) and the map frame coincide up to the saved map's half-cell
offset (about 2.5 cm), which the limits allow for.
"""

import math
import os
import sys
import time

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger

from vacuum_interfaces.msg import CleaningJob, CleaningPiece, CleaningPlan, CleaningStatus
from vacuum_interfaces.srv import (DeleteCleaningPlan, GetCleaningRecord, ListCleaningHistory, ListCleaningPlans,
                                   PreviewCleaning, SaveCleaningPlan, StartCleaning)

FINAL = (CleaningStatus.FINISHED, CleaningStatus.STOPPED, CleaningStatus.FAILED)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import house_layout as hl  # noqa: E402
from vacuum_coverage import mapdata, planner  # noqa: E402
from vacuum_coverage.grid import dilate, stamp_segment  # noqa: E402
from vacuum_mode_manager import map_files  # noqa: E402

# Each world keeps its own maps (see vacuum.launch.py): ros2_ws/maps for the house, maps_<world> otherwise.
MAPS = os.path.join(ROOT, "ros2_ws", "maps" if hl.WORLD == "house" else f"maps_{hl.WORLD}")
TOOL = 0.15  # half the cleaning width (coverage.yaml)

# The fast loop's small open rectangle in the living room, between the coffee table (house y 2.0)
# and the TV stand (house y 4.05): house x 1.65..3.65, y 2.45..3.65.
SMALL_AREA = (1.2, 0.2, 3.2, 1.4)  # map metres: x0, y0, x1, y1
# The living room's north-west corner, against two walls (phase 26): house x 0.0..1.2, y 3.4..4.5.
# Planned offline on the saved map: rows alone reach 77% of its true floor, with edges 87%.
CORNER_AREA = (-0.45, 1.15, 0.75, 2.25)

_KIND = {CleaningPiece.ROW: planner.ROW, CleaningPiece.CONNECTOR: planner.CONNECTOR,
         CleaningPiece.TRANSIT: planner.TRANSIT, CleaningPiece.EDGE: planner.EDGE}


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def job(what="everything", area=None, spot=None, plan=None, stay=True, passes=1, edges=None, direction=""):
    msg = CleaningJob()
    msg.what = {"everything": CleaningJob.EVERYTHING, "area": CleaningJob.AREA, "spot": CleaningJob.SPOT,
                "plan": CleaningJob.PLAN}[what]
    if area is not None:
        msg.area = [float(v) for p in rect(*area) for v in p]
    if spot is not None:
        msg.spot_x, msg.spot_y = float(spot[0]), float(spot[1])
    if plan is not None:
        msg.plan_name = plan
    msg.direction = direction
    msg.passes = passes
    msg.edges = CleaningJob.OPTION_DEFAULT if edges is None else (CleaningJob.OPTION_ON if edges else CleaningJob.OPTION_OFF)
    msg.return_to_dock = CleaningJob.OPTION_OFF if stay else CleaningJob.OPTION_ON
    return msg


def current_map() -> str:
    with open(os.path.join(MAPS, "current_map.txt")) as f:
        return f.read().strip()


def has_map() -> bool:
    """Is there a map in use? (A world that has never been mapped has none.)"""
    marker = os.path.join(MAPS, "current_map.txt")
    return os.path.isfile(marker) and os.path.isfile(os.path.join(MAPS, current_map() + ".yaml"))


def zones_file() -> str:
    """The no-go zones of the map in use: each saved map has its own file."""
    return map_files.zones_path(MAPS, current_map())


def load_map() -> mapdata.MapData:
    return mapdata.load(os.path.join(MAPS, current_map() + ".yaml"), map_files.read_zones(zones_file()))


def pieces_of(plan_msg: CleaningPlan):
    return [planner.Piece(_KIND[p.kind], (p.x0, p.y0), (p.x1, p.y1), p.number, p.pass_number, p.region) for p in plan_msg.pieces]


def planned_floor(m: mapdata.MapData, plan_msg: CleaningPlan) -> np.ndarray:
    """The plan's planned floor, worked out again from its pieces, as the robot does."""
    return planner.cleaned_by(m.grid, pieces_of(plan_msg), TOOL) & m.grid.floor


def cleaned_cells(msg, m: mapdata.MapData):
    """The robot's /cleaning/cleaned grid as cells of the map (True: cleaned), or None when its size
    is not the map's."""
    if msg is None or (msg.info.height, msg.info.width) != m.grid.shape:
        return None
    return np.asarray(msg.data, dtype=np.int8).reshape(m.grid.shape) >= 100


def track_outside(track, m: mapdata.MapData, band: np.ndarray) -> float:
    """Metres of a true track (sim time, x, y, yaw) driven on cells outside `band`."""
    def outside(x, y):
        r, c = m.grid.cell_of(x, y)
        return not m.grid.contains(r, c) or not band[r, c]
    return sum(math.hypot(x1 - x0, y1 - y0) for (_, x0, y0, _), (_, x1, y1, _) in zip(track, track[1:])
               if outside(x0, y0) and outside(x1, y1))


def true_floor(m: mapdata.MapData) -> np.ndarray:
    """Every cell of the map whose centre is truly open floor: inside the rooms or the doorway, not
    under anything that reaches the floor (walls, furniture bases, table legs, the plant pot, the
    dock), and not in a no-go zone."""
    h, w = m.grid.shape
    res = m.grid.resolution
    ox, oy = m.grid.origin
    xs = ox + (np.arange(w) + 0.5) * res
    ys = oy + (np.arange(h) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in hl.floor_areas():
        (a, b), (c, d) = hl.to_world(x0, y0), hl.to_world(x1, y1)
        inside |= (X >= a) & (X <= c) & (Y >= b) & (Y <= d)
    for _, a, b, c, d in hl.obstacles_world():
        inside &= ~((X >= a) & (X <= c) & (Y >= b) & (Y <= d))
    boxes, cylinders = hl.shapes_at_height(0.05)
    for cx, cy, r in cylinders:
        inside &= (X - cx) ** 2 + (Y - cy) ** 2 > r * r
    return inside & ~m.nogo


def zone_distance(x: float, y: float) -> float:
    """Distance from a point to the nearest no-go zone (0 inside one)."""
    best = float("inf")
    for zone in map_files.read_zones(zones_file()):
        xs = [p[0] for p in zone["polygon"]]
        ys = [p[1] for p in zone["polygon"]]
        dx = max(min(xs) - x, 0.0, x - max(xs))
        dy = max(min(ys) - y, 0.0, y - max(ys))
        best = min(best, math.hypot(dx, dy))
    return best


class Truth:
    """Records the true pose on every spin: the floor the tool really passed over, the distance
    really driven, the closest the body came to a no-go zone, and the track."""

    def __init__(self, m: mapdata.MapData):
        self.m = m
        self.cleaned = np.zeros(m.grid.shape, dtype=bool)
        self.distance = 0.0
        self.track = []  # (sim time, x, y, yaw)
        self.closest_zone = float("inf")  # the closest the robot's centre came to a no-go zone
        self._last = None
        self.on = True

    def add(self, sim_time, x, y, yaw):
        if not self.on:
            self._last = None
            return
        here = (x, y)
        if self._last is None:
            stamp_segment(self.cleaned, self.m.grid, here, here, TOOL)
        elif math.dist(self._last, here) >= 0.002:
            stamp_segment(self.cleaned, self.m.grid, self._last, here, TOOL)
            self.distance += math.dist(self._last, here)
        elif self.track and abs(math.atan2(math.sin(yaw - self.track[-1][3]), math.cos(yaw - self.track[-1][3]))) < 0.02:
            return  # neither moved nor turned: nothing to record (a turn on the spot is still recorded)
        else:
            self.track.append((sim_time, x, y, yaw))
            return
        self._last = here
        self.track.append((sim_time, x, y, yaw))
        self.closest_zone = min(self.closest_zone, zone_distance(x, y))

    def share(self, mask: np.ndarray, allowance: bool = True) -> float:
        """Percent of the cells in `mask` the tool really passed over. With the allowance (the plan's
        section 8), a cell counts when the true tool came within one map cell (5 cm) of it: the saved
        map sits about half a cell off the true world, and localization adds a few centimetres."""
        n = int(mask.sum())
        cleaned = dilate(self.cleaned, self.m.grid.resolution, self.m.grid.resolution) if allowance else self.cleaned
        return 100.0 * float((cleaned & mask).sum()) / n if n else 0.0

    def shares(self, mask: np.ndarray) -> str:
        return f"{self.share(mask):.1f}% (without the 5 cm allowance {self.share(mask, allowance=False):.1f}%)"


def on_rows(track, pieces, spacing=0.25, offset=(0.0, 0.0)):
    """The track samples taken while driving a row: [(sample, sideways distance from the row)]."""
    out = []
    rows = [p for p in pieces if p.kind == planner.ROW and p.length > 0.4]
    for sample in track:
        _, x, y, yaw = sample
        x, y = x + offset[0], y + offset[1]
        best = None
        for r in rows:
            (x0, y0), (x1, y1) = r.start, r.end
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            t = ((x - x0) * dx + (y - y0) * dy) / (length * length)
            if t * length < 0.15 or (1.0 - t) * length < 0.15:
                continue
            side = abs((x - x0) * dy - (y - y0) * dx) / length
            off = abs(math.atan2(math.sin(yaw - math.atan2(dy, dx)), math.cos(yaw - math.atan2(dy, dx))))
            if side < spacing / 2.0 and off < math.radians(10.0) and (best is None or side < best):
                best = side
        if best is not None:
            out.append((sample, best))
    return out


def row_errors(track, pieces, spacing=0.25, offset=(0.0, 0.0)):
    """Sideways distance of the true centre from the row it is driving, for every sample taken on a
    row (within half a row spacing of it, at least 15 cm from its ends, heading along it within 10
    degrees). `offset` (map minus world, measured with the robot standing still) is removed first,
    as the plan's section 8 says. Metres."""
    return [d for _, d in on_rows(track, pieces, spacing, offset)]


class Cleaning:
    """The robot's cleaning interfaces, for a test node with spin(seconds)."""

    def __init__(self, node):
        self.node = node
        self.status = None
        self.statuses = []  # (wall time, CleaningStatus) on every change of state or message
        self.plan = None
        self.job_plans = {}  # job id -> its plan, kept: an empty plan replaces it when the job ends
        self.cleaned = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(CleaningStatus, "cleaning/status", self._on_status, latched)
        node.create_subscription(CleaningPlan, "cleaning/plan", self._on_plan, latched)
        node.create_subscription(OccupancyGrid, "cleaning/cleaned", self._on_cleaned, latched)
        self._clients = {}

    def _on_status(self, m):
        last = self.status
        self.status = m
        if last is None or (last.state, last.message, last.job_id) != (m.state, m.message, m.job_id):
            self.statuses.append((time.monotonic(), m))

    def _on_plan(self, m):
        self.plan = m
        if m.job_id and m.pieces:
            self.job_plans[m.job_id] = m

    def job_plan(self, job_id):
        return self.job_plans.get(job_id) if job_id else None

    def _on_cleaned(self, m):
        self.cleaned = m

    def spin_until(self, done, timeout, tick=None):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.node.spin(0.02)
            if tick is not None:
                tick()
            if done():
                return True
        return False

    # ------------------------------------------------------------ jobs
    def start(self, goal_job):
        """Start a job. Returns the StartCleaning response (None if the service did not answer)."""
        return self.call(StartCleaning, "cleaning/start", StartCleaning.Request(job=goal_job))

    def started(self, goal_job):
        """Start a job; its number, or None if refused."""
        r = self.start(goal_job)
        return r.job_id if r is not None and r.accepted else None

    def result(self, job_id, timeout, tick=None):
        """Wait for the job to end. Returns its final CleaningStatus, or None."""
        if job_id is None:
            return None
        self.wait_state(FINAL, timeout, tick, job_id=job_id)
        s = self.status
        return s if s is not None and s.job_id == job_id and s.state in FINAL else None

    def wait_state(self, states, timeout, tick=None, job_id=None):
        """Wait until /cleaning/status is in one of `states` (for job_id, if given)."""
        def done():
            s = self.status
            return s is not None and s.state in states and (job_id is None or s.job_id == job_id)
        self.spin_until(done, timeout, tick)
        return self.status

    # ------------------------------------------------------------ services
    def call(self, srv_type, name, request, timeout=30.0):
        if name not in self._clients:
            self._clients[name] = self.node.create_client(srv_type, name)
        client = self._clients[name]
        if not client.wait_for_service(timeout_sec=10.0):
            return None
        for _ in range(2):  # a brand-new client can lose its first reply on Humble
            future = client.call_async(request)
            if self.spin_until(future.done, timeout):
                return future.result()
        return None

    def preview(self, goal_job):
        return self.call(PreviewCleaning, "cleaning/preview", PreviewCleaning.Request(job=goal_job))

    def pause(self):
        return self.call(Trigger, "cleaning/pause", Trigger.Request())

    def resume(self):
        return self.call(Trigger, "cleaning/resume", Trigger.Request())

    def save(self, name, goal_job, replace=False):
        return self.call(SaveCleaningPlan, "cleaning/save_plan", SaveCleaningPlan.Request(name=name, job=goal_job, replace=replace))

    def plans(self):
        return self.call(ListCleaningPlans, "cleaning/list_plans", ListCleaningPlans.Request())

    def delete(self, name):
        return self.call(DeleteCleaningPlan, "cleaning/delete_plan", DeleteCleaningPlan.Request(name=name))

    def history(self, max_records=0):
        return self.call(ListCleaningHistory, "cleaning/history", ListCleaningHistory.Request(max_records=max_records))

    def record(self, rid):
        return self.call(GetCleaningRecord, "cleaning/get_record", GetCleaningRecord.Request(id=rid))


def cleaning_node_pids():
    import subprocess

    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    return [int(line.split(None, 1)[0]) for line in out[1:] if "lib/vacuum_coverage/cleaning_node" in line and "python" in line]


def summary_line(s) -> str:
    return (f"{s.state} '{s.message}' {s.coverage:.1f}% of {s.planned_area:.2f} m2, {s.distance:.1f} m, "
            f"{s.cleaning_time:.0f} s, rows {s.rows_done}/{s.rows_total} skipped {s.rows_skipped}, bumps {s.bumps}")


class Run:
    """One acceptance test run: the test node (the true pose, the safety gate's reports, the bumper
    and front cliff sensors), the mode manager, the cleaning node, the checks, and the summary."""

    def __init__(self, name):
        import rclpy

        from mode_client import ModeClient
        from phase7_mode_manager_test import ModeTestDriver

        self.rclpy = rclpy
        rclpy.init()
        self.name = name
        self.n = ModeTestDriver()
        self.modes = ModeClient(self.n)
        self.clean = Cleaning(self.n)
        self.results = []
        self.m = load_map() if has_map() else None  # the test that makes a world's first map starts without one
        self.truth = None  # set a Truth to record what the robot really cleans

    def check(self, name, ok, detail):
        self.results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    def tick(self):
        if self.truth is not None and self.n.odom is not None and self.n.sim_time is not None:
            x, y, yaw = self.n.pose()
            self.truth.add(self.n.sim_time, x, y, yaw)

    def wait(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.n.spin(0.02)
            self.tick()

    def ready(self) -> bool:
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        heard = []
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.n.create_subscription(String, "sim/world", lambda m: heard.append(m.data), latched)
        self.clean.spin_until(lambda: bool(heard), 10.0)
        if not heard or heard[-1] != hl.WORLD:
            self.check("robot ready", False, f"this test is for the world '{hl.WORLD}', the simulator says "
                       f"'{heard[-1] if heard else 'nothing'}' on /sim/world")
            return False
        if self.modes.wait_ready(60.0) is None or not self.modes.wait_for_state("idle", 60.0):
            self.check("robot ready", False, "the mode manager did not reach idle")
            return False
        ok = self.clean.spin_until(lambda: self.n.sim_time is not None and self.n.odom is not None and self.clean.status is not None, 30.0)
        if not ok:
            self.check("robot ready", False, "no /cleaning/status, /ground_truth/odom or /clock: is the cleaning node running?")
        return ok

    def hazards_since(self, t_wall):
        """(bumper presses, front cliff readings, safety reflexes) seen since t_wall."""
        reflexes = sum(1 for t, _, s in self.n.safety_log if t >= t_wall and s.reflex_active)
        bumps = sum(1 for t, _, s in self.n.safety_log if t >= t_wall and s.bumper)
        cliffs = sum(1 for t, _, s in self.n.safety_log if t >= t_wall and s.cliff)
        return bumps, cliffs, reflexes

    def run_job(self, goal_job, timeout):
        """Start a job, record the truth while it runs, wait for its end. Returns its final status or None."""
        return self.clean.result(self.clean.started(goal_job), timeout, self.tick)

    def map_offset(self):
        """(map - world) of the robot's position, read with the robot standing still: the saved map's
        half-cell offset plus localization's. Removed before row tracking is judged (plan section 8)."""
        from rclpy.time import Time
        from tf2_ros import Buffer, TransformListener

        if not hasattr(self, "_tf"):
            self._tf = Buffer()
            self._tf_listener = TransformListener(self._tf, self.n)
        samples = []
        end = time.monotonic() + 5.0
        while time.monotonic() < end and len(samples) < 10:
            self.n.spin(0.1)
            try:
                t = self._tf.lookup_transform("map", "base_link", Time())
            except Exception:
                continue
            x, y, _ = self.n.pose()
            samples.append((t.transform.translation.x - x, t.transform.translation.y - y))
        if not samples:
            print("WARNING: map -> base_link not available within 5 s; taking the map offset as (0, 0)", flush=True)
            return (0.0, 0.0)
        return (sum(s[0] for s in samples) / len(samples), sum(s[1] for s in samples) / len(samples))

    def at_rest_after(self, t0, limit=5.0):
        end = time.monotonic() + limit
        while time.monotonic() < end:
            self.n.spin(0.01)
            self.tick()
            if self.n.speed() < 0.02:
                return round(time.monotonic() - t0, 3)
        return None

    def moving(self, limit=60.0) -> bool:
        end = time.monotonic() + limit
        while time.monotonic() < end:
            self.n.spin(0.02)
            self.tick()
            if self.n.speed() > 0.05:
                return True
        return False

    def finish(self):
        out = os.environ.get(f"{self.name.upper()}_RESULT_JSON")
        if out:
            import json

            with open(out, "w") as f:
                json.dump({"results": self.results}, f)
        try:
            self.modes.request("idle", 120.0)
        finally:
            self.n.destroy_node()
            self.rclpy.shutdown()
        passed = sum(r["pass"] for r in self.results)
        print(f"SUMMARY {passed}/{len(self.results)} checks passed", flush=True)
        return self.results


def exit_code(results) -> int:
    return 0 if results and all(r["pass"] for r in results) else 1
