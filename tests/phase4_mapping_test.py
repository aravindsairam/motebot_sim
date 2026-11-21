#!/usr/bin/env python3
"""Phase 4 acceptance test: map the house with slam_toolbox and score the map against the true layout.

Needs two things running first, with the robot at its start pose:
    1. the simulator:    tools/start_sim.sh
    2. the robot stack:  tools/start_stack.sh rviz:=false
Then:
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase4_mapping_test.py

The test asks the mode manager for the mode it needs (mapping).

The test drives a fixed tour of both rooms (steering by the true pose, standing in for a person
with a keyboard), saves the map through the mode manager, and checks:
  1. mapping runs   slam_toolbox publishes /map and the map -> odom transform
  2. wall accuracy  occupied map cells lie on the surfaces the lidar could see
  3. completeness   those surfaces are all in the map, and the open floor is marked free
  4. localization   slam_toolbox's pose estimate stays close to the truth while wheel odometry drifts

To score a map you made yourself (for example by keyboard driving), without driving again:
    python3 sim/tests/phase4_mapping_test.py --score-only [name]      # reads ros2_ws/maps/<name>.pgm
"""

import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from PIL import Image
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial import cKDTree
from tf2_msgs.msg import TFMessage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import house_layout as hl  # noqa: E402
from mode_client import ModeClient  # noqa: E402
from phase1_drive_test import Driver, wrap, yaw_of  # noqa: E402

MAP_DIR = os.path.join(REPO, "ros2_ws", "maps")
MAP_NAME = os.environ.get("PHASE4_MAP_NAME", "house")

# Tour in house coordinates: through the living room, into the bedroom, back, and home.
TOUR_HOUSE = [
    (2.2, 2.8), (4.1, 3.0), (5.95, 3.0), (6.0, 1.4), (5.6, 3.5), (5.95, 3.0),
    (4.1, 3.0), (4.3, 1.7), (3.3, 2.6), (1.0, 3.3), (0.9, 1.2), (0.45, 2.25),
]


class MappingDriver(Driver):
    def __init__(self):
        super().__init__()
        self.map_msgs = 0
        self.map_to_odom = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "map", self._on_map, latched)
        self.create_subscription(TFMessage, "tf", self._on_slam_tf, 50)

    def _on_map(self, _msg):
        self.map_msgs += 1

    def _on_slam_tf(self, msg):
        for t in msg.transforms:
            if t.header.frame_id == "map" and t.child_frame_id == "odom":
                tr, q = t.transform.translation, t.transform.rotation
                self.map_to_odom = (tr.x, tr.y, yaw_of(q))

    def odom_pose(self):
        p, q = self.robot_odom.pose.pose.position, self.robot_odom.pose.pose.orientation
        return p.x, p.y, yaw_of(q)

    def slam_pose(self):
        """Robot pose in the map frame: (map -> odom) composed with (odom -> base_link)."""
        mx, my, myaw = self.map_to_odom
        ox, oy, oyaw = self.odom_pose()
        c, s = math.cos(myaw), math.sin(myaw)
        return mx + c * ox - s * oy, my + s * ox + c * oy, wrap(myaw + oyaw)


def visible_surface_points(rays=1440, stops=None):
    """Points on walls and furniture that the lidar can see from the stops of the tour (world frame).

    Hidden faces (the outside of the house, the back of the sofa) are left out: no map can show them.
    """
    z = hl.LIDAR_XYZ[2]
    if stops is None:  # this test's own tour; another test passes the places its robot drove past (world frame)
        stops = [hl.to_world(*hl.START_POSE_HOUSE[:2])] + [hl.to_world(x, y) for x, y in TOUR_HOUSE]
    pts = []
    for sx, sy in stops:
        for k in range(rays):
            a = -math.pi + k * 2.0 * math.pi / rays
            r = hl.expected_range(sx, sy, a, z)
            if r < hl.LIDAR_MAX_RANGE:
                pts.append((sx + r * math.cos(a), sy + r * math.sin(a)))
    return np.array(pts)


def rigid_align(points, tree, target, iterations=30, max_pair=0.25):
    """Rotation and shift that best lay `points` onto `target` (plain 2D ICP).

    Returns (aligned points, rotation in degrees, shift x, shift y). A SLAM map is free to sit
    slightly rotated or shifted in the world; this separates that from errors in its shape.
    """
    p = points.copy()
    rot, shift = np.eye(2), np.zeros(2)
    for _ in range(iterations):
        d, idx = tree.query(p)
        keep = d < max_pair
        a, b = p[keep], target[idx[keep]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[1] *= -1
            r = vt.T @ u.T
        t = cb - r @ ca
        p = (r @ p.T).T + t
        rot, shift = r @ rot, r @ shift + t
    return p, math.degrees(math.atan2(rot[1, 0], rot[0, 0])), float(shift[0]), float(shift[1])


def load_map(stem):
    import yaml

    with open(stem + ".yaml") as f:
        meta = yaml.safe_load(f)
    res = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"]]
    img = np.array(Image.open(stem + ".pgm"))
    return img, res, origin


def cell_centers(mask, res, origin):
    rows, cols = np.nonzero(mask)
    h = mask.shape[0]
    return np.stack([origin[0] + (cols + 0.5) * res, origin[1] + (h - rows - 0.5) * res], axis=1)


def score_map(stem, check, stops=None):
    """Checks 2 and 3: compare a saved map (stem.pgm / stem.yaml) with the true house layout."""
    img, res, origin = load_map(stem)
    occupied, free, unknown = img < 100, img > 250, (img >= 100) & (img <= 250)
    h, w = img.shape
    surface = visible_surface_points(stops=stops)
    tree = cKDTree(surface)

    # 2. wall accuracy: occupied cells should sit on visible surfaces
    occ_xy = cell_centers(occupied, res, origin)
    d_raw, _ = tree.query(occ_xy)
    aligned, rot_deg, shift_x, shift_y = rigid_align(occ_xy, tree, surface)
    d_fit, _ = tree.query(aligned)
    med, p95 = float(np.median(d_fit)), float(np.percentile(d_fit, 95))
    stray = float(np.mean(d_fit > 0.15))
    check("wall accuracy", len(occ_xy) > 300 and med < 0.035 and p95 < 0.09 and stray < 0.02
          and abs(rot_deg) < 1.0 and math.hypot(shift_x, shift_y) < 0.06,
          f"{len(occ_xy)} occupied cells at {res * 100:.0f} cm resolution; distance to the true surface after "
          f"aligning the map: median {med * 100:.1f} cm, 95th percentile {p95 * 100:.1f} cm, "
          f"{stray * 100:.1f}% further than 15 cm; the map as saved sits {rot_deg:+.2f} deg and "
          f"({shift_x * 100:+.1f}, {shift_y * 100:+.1f}) cm from the truth (median {np.median(d_raw) * 100:.1f} cm unaligned)")

    # 3. completeness: seen surfaces are mapped, open floor is free
    d_surf, _ = cKDTree(occ_xy).query(surface)
    surf_cov = float(np.mean(d_surf < 0.075))

    interior = [hl.to_world(x0, y0) + hl.to_world(x1, y1) for x0, y0, x1, y1 in hl.floor_areas()]
    rows, cols = np.mgrid[0:h, 0:w]
    all_xy = np.stack([origin[0] + (cols.ravel() + 0.5) * res, origin[1] + (h - rows.ravel() - 0.5) * res], axis=1)
    in_room = np.zeros(len(all_xy), dtype=bool)
    for x0, y0, x1, y1 in interior:
        in_room |= (all_xy[:, 0] >= x0) & (all_xy[:, 0] <= x1) & (all_xy[:, 1] >= y0) & (all_xy[:, 1] <= y1)
    boxes, cyls = hl.shapes_at_height(hl.LIDAR_XYZ[2])
    in_obstacle = np.zeros(len(all_xy), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        in_obstacle |= (all_xy[:, 0] > x0) & (all_xy[:, 0] < x1) & (all_xy[:, 1] > y0) & (all_xy[:, 1] < y1)
    for cx, cy, r in cyls:
        in_obstacle |= np.hypot(all_xy[:, 0] - cx, all_xy[:, 1] - cy) < r
    d_all, _ = tree.query(all_xy)
    # open floor: inside a room, outside every obstacle, and at least 15 cm from a visible surface
    # (the mapped walls are about three cells thick because of lidar noise)
    true_free = in_room & ~in_obstacle & (d_all > 0.15)
    free_flat, unknown_flat = free.ravel(), unknown.ravel()
    free_cov = float(np.mean(free_flat[true_free]))
    free_unknown = float(np.mean(unknown_flat[true_free]))
    sx0, sy0 = hl.to_world(hl.drops()[0][0], hl.drops()[0][1])
    sx1, sy1 = hl.to_world(hl.drops()[0][2], hl.drops()[0][3])
    in_stair = (all_xy[:, 0] > sx0 + 0.15) & (all_xy[:, 0] < sx1 - 0.15) & (all_xy[:, 1] > sy0 + 0.15) & (all_xy[:, 1] < sy1 - 0.15)
    stair_free = float(np.mean(free_flat[in_stair])) if in_stair.any() else float("nan")
    check("completeness", surf_cov > 0.95 and free_cov > 0.90,
          f"{surf_cov * 100:.0f}% of the surfaces the lidar could see are in the map; "
          f"{free_cov * 100:.0f}% of the open floor is marked free ({free_unknown * 100:.0f}% still unknown); "
          f"map is {w}x{h} cells; the stairwell shows as {stair_free * 100:.0f}% free floor, as expected for a lidar")

    # picture for people: 6x enlarged, white free, black walls, grey unknown
    Image.fromarray(img).resize((w * 6, h * 6), Image.NEAREST).save(stem + ".png")
    return {"occupied_cells": len(occ_xy), "median_cm": med * 100, "p95_cm": p95 * 100, "rotation_deg": rot_deg,
            "shift_cm": [shift_x * 100, shift_y * 100], "surface_coverage": surf_cov, "free_coverage": free_cov,
            "map_size": [w, h], "resolution": res, "origin": origin}


def score_only(name):
    results = []

    def check(label, ok, detail):
        results.append({"check": label, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {label}: {detail}", flush=True)

    stem = os.path.join(MAP_DIR, name)
    if not os.path.isfile(stem + ".pgm"):
        sys.exit(f"No map at {stem}.pgm. Save one with: ros2 run vacuum_bringup save_map.sh {name}")
    score_map(stem, check)
    print(f"SUMMARY {sum(r['pass'] for r in results)}/{len(results)} checks passed", flush=True)
    return results


def main():
    rclpy.init()
    n = MappingDriver()
    results, numbers = [], {}

    def check(name, ok, detail):
        results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        if not n.wait_ready():
            check("mapping runs", False, "no /odom, /ground_truth/odom or /clock within 60 s; is run_sim.py playing?")
            return results
        modes = ModeClient(n)
        if modes.wait_ready() is None:
            check("mapping runs", False, "no /robot_state from the mode manager; is the robot stack running?")
            return results
        ok, why = modes.request("mapping")
        if not ok:
            check("mapping runs", False, f"could not enter mapping mode: {why}")
            return results
        n.map_msgs, n.map_to_odom = 0, None  # only count what slam_toolbox publishes from here on
        wall0 = time.time()
        while (n.map_msgs == 0 or n.map_to_odom is None) and time.time() - wall0 < 30.0:
            n.spin(0.1)
        if n.map_msgs == 0 or n.map_to_odom is None:
            check("mapping runs", False,
                  f"in mapping mode but /map received={n.map_msgs > 0}, "
                  f"map->odom transform received={n.map_to_odom is not None}")
            return results

        # drive the tour
        t_start, dist, prev = n.sim_time, 0.0, n.pose()[:2]
        reached = True
        worst_slam = 0.0
        for hx, hy in TOUR_HOUSE:
            wx, wy = hl.to_world(hx, hy)
            reached = n.go_to(wx, wy) and reached
            cur = n.pose()[:2]
            dist += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
            prev = cur
            sx, sy, _ = n.slam_pose()
            worst_slam = max(worst_slam, math.hypot(sx - cur[0], sy - cur[1]))
        n.face(0.0)
        n.run_for(0.0, 0.0, 5.0)  # let slam_toolbox publish a final map update
        tour_time = n.sim_time - t_start
        check("mapping runs", reached and n.map_msgs >= 3,
              f"tour of {len(TOUR_HOUSE)} waypoints completed={reached}, {dist:.1f} m in {tour_time:.0f} s of sim time; "
              f"{n.map_msgs} map updates received")

        # 4. localization (measured now, reported last)
        tx, ty, tyaw = n.pose()
        sx, sy, syaw = n.slam_pose()
        ox, oy, oyaw = n.odom_pose()
        slam_err, slam_yaw = math.hypot(sx - tx, sy - ty), math.degrees(wrap(syaw - tyaw))
        odom_err, odom_yaw = math.hypot(ox - tx, oy - ty), math.degrees(wrap(oyaw - tyaw))

        # save the map
        stem = os.path.join(MAP_DIR, MAP_NAME)
        saved, why, _path = modes.save_map(MAP_NAME)
        if not saved or not all(os.path.isfile(stem + ext) for ext in (".pgm", ".yaml")):
            check("wall accuracy", False, f"saving the map failed: {why}")
            return results

        numbers = score_map(stem, check)

        check("localization", slam_err < 0.10 and abs(slam_yaw) < 3.0 and worst_slam < 0.20,
              f"at the end slam_toolbox places the robot {slam_err * 100:.1f} cm and {slam_yaw:+.1f} deg from the truth "
              f"(worst during the tour {worst_slam * 100:.1f} cm); wheel odometry alone is off by {odom_err * 100:.1f} cm "
              f"and {odom_yaw:+.1f} deg")

        numbers.update({"slam_err_cm": slam_err * 100, "slam_yaw_deg": slam_yaw, "odom_err_cm": odom_err * 100,
                        "odom_yaw_deg": odom_yaw, "tour_m": dist})
    finally:
        try:
            for _ in range(20):
                n.cmd(0.0, 0.0)
                n.spin()
        except Exception:
            pass
        out = os.environ.get("PHASE4_RESULT_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"results": results, "numbers": numbers}, f)
        n.destroy_node()
        rclpy.shutdown()

    passed = sum(r["pass"] for r in results)
    print(f"SUMMARY {passed}/{len(results)} checks passed", flush=True)
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--score-only":
        res = score_only(sys.argv[2] if len(sys.argv) > 2 else MAP_NAME)
    else:
        res = main()
    sys.exit(0 if res and all(r["pass"] for r in res) else 1)
