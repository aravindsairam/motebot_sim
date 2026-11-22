#!/usr/bin/env python3
"""Phase 40 acceptance test: the settings a user changes from the app (cleaning speed, joystick
speed, row spacing, row direction).

Needs the simulator and the robot stack running, the robot at its start pose, the house map, and
the default settings (tools/run_tests.sh 40 starts the stack with a fresh settings file):
    tools/start_sim.sh ; tools/start_stack.sh rviz:=false
    source ros2_ws/install/setup.bash
    python3 sim/tests/phase40_settings_test.py

Everything goes through the gateway, as the app sends it; speeds are the simulator's true ones.
  1. defaults         a phone that connects gets the settings: cleaning 0.20 m/s, joystick 0.30 m/s,
                      rows 25 cm apart, direction auto, with the limits
  2. refused          a cleaning speed of 0.50 m/s is refused with the reason, and nothing changes
  3. saved            new settings are answered OK, every phone gets them, and a restarted settings
                      node still has them (they are kept in the file)
  4. joystick speed   at a joystick speed of 0.15 m/s, the joystick pushed all the way (0.30) drives
                      the robot at 0.15 m/s (true speed within 0.02)
  5. rows at 45°      a clean of a small area plans every row at 45° (or 225°), no more than 15 cm
                      apart; the robot holds them (median 3 cm, worst 8 cm) and cleans >= 90 % of
                      the planned floor
  6. cleaning speed   on those rows the true speed is 0.30 m/s (median, within 10 %); lowered to
                      0.12 m/s during the clean, the robot drives the next rows at 0.12 (within 15 %)
  7. back to auto     with the defaults again, a preview of the same area has rows along x or y only
The test puts the default settings back at the end.
"""

import itertools
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import cleaning_client as cc  # noqa: E402
from phase11_gateway_test import Phone  # noqa: E402
from vacuum_gateway import vacuum_pb2 as pb  # noqa: E402
from vacuum_interfaces.msg import CleaningStatus  # noqa: E402

Result = pb.Reply.Result
RESULT_NAMES = {v: k for k, v in pb.Reply.Result.items()}
REQUEST_IDS = itertools.count(1)
DEFAULTS = dict(cleaning_speed=0.20, manual_speed=0.30, row_spacing=0.25, row_angle_auto=True, row_angle=0.0)
CHANGED = dict(cleaning_speed=0.30, manual_speed=0.15, row_spacing=0.15, row_angle_auto=False, row_angle=45.0)


def settings_pids():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    return [int(line.split()[0]) for line in out.splitlines() if "lib/vacuum_mode_manager/settings_node" in line]


def same(settings, values, tol=1e-4):
    return settings is not None and all(
        (getattr(settings, k) == v) if isinstance(v, bool) else abs(getattr(settings, k) - v) <= tol for k, v in values.items())


def show(s):
    if s is None:
        return "none"
    direction = "auto" if s.row_angle_auto else f"{s.row_angle:g}°"
    return (f"cleaning {s.cleaning_speed:.2f} m/s, joystick {s.manual_speed:.2f} m/s (turn {s.manual_turn_speed:.2f} rad/s), "
            f"rows {s.row_spacing * 100:.0f} cm, direction {direction}")


def read_settings(n, run, timeout=10.0):
    """What /robot_settings holds now, read by a new subscription (the latched copy)."""
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from vacuum_interfaces.msg import RobotSettings

    got = []
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = n.create_subscription(RobotSettings, "robot_settings", got.append, latched)
    end = time.monotonic() + timeout
    while time.monotonic() < end and not got:
        run.wait(0.05)
    n.destroy_subscription(sub)
    if not got:
        return None
    m = got[-1]
    return pb.RobotSettings(cleaning_speed=m.cleaning_speed, manual_speed=m.manual_speed, row_spacing=m.row_spacing,
                            row_angle_auto=m.row_angle_auto, row_angle=m.row_angle, manual_turn_speed=m.manual_speed * m.MANUAL_TURN_PER_SPEED)


def row_speeds(track, pieces, spacing, offset, since=None):
    """True speeds between consecutive samples that are both on a row, at least 30 cm from its ends."""
    rows = cc.on_rows(track, pieces, spacing, offset)
    on = {id(sample) for sample, _ in rows}
    keep = set()
    for sample, _ in rows:
        _, x, y, _ = sample
        x, y = x + offset[0], y + offset[1]
        for r in pieces:
            if r.kind != cc.planner.ROW or r.length < 0.8:
                continue
            (x0, y0), (x1, y1) = r.start, r.end
            dx, dy = x1 - x0, y1 - y0
            t = ((x - x0) * dx + (y - y0) * dy) / (r.length * r.length)
            side = abs((x - x0) * dy - (y - y0) * dx) / r.length
            if side < spacing / 2.0 and 0.3 <= t * r.length <= r.length - 0.3:
                keep.add(id(sample))
                break
    out = []
    for a, b in zip(track, track[1:]):
        if id(a) in on and id(b) in on and id(a) in keep and id(b) in keep and (since is None or a[0] >= since):
            dt = b[0] - a[0]
            if 0.0 < dt <= 0.3:
                out.append(math.dist(a[1:3], b[1:3]) / dt)
    return out


def main():
    run = cc.Run("phase40")
    n, modes, clean = run.n, run.modes, run.clean
    phones = []

    def connected(name):
        phone = Phone(name).start()
        phones.append(phone)
        end = time.monotonic() + 10.0
        while time.monotonic() < end and not phone.opens():
            n.spin(0.05)
        return phone if phone.opens() else None

    def latest_settings(phone, since=0.0, timeout=5.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            got = phone.messages(pb.DATA_ROBOT_SETTINGS, since)
            if got:
                return got[-1][1]
            n.spin(0.05)
        return None

    def command(phone, msg_type, message, timeout=15.0):
        sent = phone.send(msg_type, message)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            run.wait(0.02)
            got = phone.reply_to(message.request_id, sent)
            if got:
                return got[1]
        return None

    def set_settings(phone, values):
        return command(phone, pb.COMMAND_SET_ROBOT_SETTINGS,
                       pb.SetRobotSettings(request_id=next(REQUEST_IDS), settings=pb.RobotSettings(**values)))

    def text(reply):
        return f"{RESULT_NAMES.get(reply.result, reply.result)} '{reply.message}'" if reply else "no reply"

    try:
        if not run.ready():
            return run.results
        phone = connected("phone")
        if phone is None:
            run.check("phone connected", False, "could not connect to the gateway")
            return run.results

        # 1. what a phone gets on connecting
        first = latest_settings(phone)
        limits = first.limits if first is not None else None
        run.check("defaults", same(first, DEFAULTS) and limits is not None and abs(limits.cleaning_speed_max - 0.30) < 1e-4
                  and abs(limits.row_spacing_step - 0.05) < 1e-4 and abs(first.manual_turn_speed - 1.2) < 1e-3,
                  f"{show(first)}; limits cleaning {limits.cleaning_speed_min:.2f}-{limits.cleaning_speed_max:.2f}, rows "
                  f"{limits.row_spacing_min * 100:.0f}-{limits.row_spacing_max * 100:.0f} cm in {limits.row_spacing_step * 100:.0f} cm steps"
                  if limits is not None else f"{show(first)}; no limits")

        # 2. a value outside the limits
        t2 = time.monotonic()
        bad = set_settings(phone, dict(DEFAULTS, cleaning_speed=0.50))
        run.wait(1.0)
        unchanged = not phone.messages(pb.DATA_ROBOT_SETTINGS, t2)
        run.check("refused", bad is not None and bad.result == Result.RESULT_REFUSED and "cleaning speed" in bad.message and unchanged,
                  f"{text(bad)}; settings sent again meanwhile: {not unchanged}")

        # 3. saved, shared, and kept over a restart
        t3 = time.monotonic()
        good = set_settings(phone, CHANGED)
        mine = latest_settings(phone, t3)
        other = connected("other phone")
        theirs = latest_settings(other) if other is not None else None
        pids = settings_pids()
        for pid in pids:
            os.kill(pid, 9)
        restarted = []
        end = time.monotonic() + 30.0
        while time.monotonic() < end and not restarted:  # the launch file starts it again after 2 s
            run.wait(0.5)
            restarted = [p for p in settings_pids() if p not in pids]
        back = read_settings(n, run) if restarted else None
        restarted_same = same(back, CHANGED)
        run.check("saved", good is not None and good.result == Result.RESULT_OK and same(mine, CHANGED) and same(theirs, CHANGED)
                  and len(pids) == 1 and restarted_same,
                  f"{text(good)}; this phone: {show(mine)}; a phone connecting later: {show(theirs)}; settings node killed "
                  f"({len(pids)} found), started again: {bool(restarted)}, and it publishes: {show(back)}")

        # 4. the joystick at a joystick speed of 0.15
        modes.request("manual", 120.0)
        speeds = []

        def drive(forward):
            phone.send(pb.COMMAND_DRIVE, pb.Drive(request_id=next(REQUEST_IDS), forward=forward, turn=0.0))

        t_end = n.sim_time + 1.5
        while n.sim_time < t_end:
            drive(0.30)
            run.wait(0.05)
        t_end = n.sim_time + 1.0
        while n.sim_time < t_end:
            drive(0.30)
            run.wait(0.05)
            speeds.append(n.odom.twist.twist.linear.x)
        t_end = n.sim_time + 2.5
        while n.sim_time < t_end:  # back to where it was
            drive(-0.30)
            run.wait(0.05)
        drive(0.0)
        mean = sum(speeds) / len(speeds) if speeds else 0.0
        run.check("joystick speed", abs(mean - 0.15) <= 0.02,
                  f"joystick pushed to 0.30 m/s with a joystick speed of 0.15: true speed over 1 s {mean:.3f} m/s")
        modes.request("idle", 120.0)

        # 5. and 6. a clean with rows at 45°, 15 cm apart, at 0.30 m/s; slowed to 0.12 on the way
        offset = run.map_offset()
        run.truth = cc.Truth(run.m)
        job_id = clean.started(cc.job("area", area=cc.SMALL_AREA, stay=True))
        clean.wait_state((CleaningStatus.CLEANING,), 120.0, run.tick, job_id=job_id)
        plan = clean.job_plan(job_id)
        pieces = cc.pieces_of(plan) if plan is not None else []
        rows = [p for p in pieces if p.kind == cc.planner.ROW]
        # Lower the speed once a good share of the rows are done.
        clean.spin_until(lambda: clean.status.rows_done >= max(2, len(rows) // 2), 900.0, run.tick)
        t_slow = n.sim_time
        slow = set_settings(phone, dict(CHANGED, cleaning_speed=0.12))
        final = clean.result(job_id, 1200.0, run.tick)
        run.truth.on = False
        planned = cc.planned_floor(run.m, plan) if plan is not None and plan.pieces else None
        share = run.truth.share(planned) if planned is not None else 0.0
        angles = sorted({round(math.degrees(p.heading) % 180.0, 3) for p in rows})
        across = (-math.sin(math.radians(45.0)), math.cos(math.radians(45.0)))
        offsets = sorted({round(p.start[0] * across[0] + p.start[1] * across[1], 4) for p in rows})
        widest = max((b - a for a, b in zip(offsets, offsets[1:])), default=float("nan"))
        errors = cc.row_errors(run.truth.track, pieces, spacing=0.15, offset=offset)
        med = statistics.median(errors) if errors else float("nan")
        worst = max(errors) if errors else float("nan")
        run.check("rows at 45°", final is not None and final.state == CleaningStatus.FINISHED and angles == [45.0]
                  and widest <= 0.15 + 1e-3 and errors and med <= 0.03 and worst <= 0.08 and share >= 90.0
                  and final.job.direction == "45",
                  f"{len(rows)} rows, angles {angles}, widest gap {widest * 100:.1f} cm, job direction '{final.job.direction if final else '?'}'; "
                  f"held: median {med * 100:.1f} cm, worst {worst * 100:.1f} cm; clean {final.state if final else '?'}, true share "
                  f"{run.truth.shares(planned) if planned is not None else 'no plan'}")

        fast = row_speeds([s for s in run.truth.track if s[0] < t_slow], pieces, 0.15, offset)
        later = row_speeds(run.truth.track, pieces, 0.15, offset, since=t_slow + 3.0)
        fast_med = statistics.median(fast) if fast else 0.0
        slow_med = statistics.median(later) if later else 0.0
        run.check("cleaning speed", abs(fast_med - 0.30) <= 0.03 and slow is not None and slow.result == Result.RESULT_OK
                  and abs(slow_med - 0.12) <= 0.018,
                  f"on rows at 0.30: median {fast_med:.3f} m/s ({len(fast)} samples); lowered during the clean ({text(slow)}): "
                  f"median {slow_med:.3f} m/s ({len(later)} samples from 3 s later)")

        # 7. the defaults again: rows along x or y
        reset = set_settings(phone, DEFAULTS)
        run.wait(1.0)
        preview = clean.preview(cc.job("area", area=cc.SMALL_AREA, stay=True))
        run.wait(1.0)
        shown = clean.plan
        straight = sorted({round(math.degrees(p.heading) % 180.0, 3) for p in cc.pieces_of(shown) if p.kind == cc.planner.ROW}) if shown else []
        run.check("back to auto", reset is not None and reset.result == Result.RESULT_OK and preview is not None and preview.success
                  and straight and set(straight) <= {0.0, 90.0},
                  f"{text(reset)}; preview row angles {straight}")
    finally:
        for phone in phones:
            try:
                if phone.opens():
                    set_settings(phone, DEFAULTS)
                    break
            except Exception:
                pass
        for phone in phones:
            phone.stop()
        results = run.finish()
    return results


if __name__ == "__main__":
    sys.exit(cc.exit_code(main()))
