"""Sensor noise settings and models. Every noise number in the simulation lives in this file.

No Isaac Sim imports: the simulator uses these models to corrupt its perfect sensor readings,
and the tests import the same settings to check what actually comes out.

The values are typical of hobby-grade parts, not taken from an iRobot datasheet.
Run the simulator with `run_sim.py --no-noise` to get perfect sensors and ground-truth odometry.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LidarNoise:
    range_std_base: float = 0.010  # m, range error close to the sensor
    range_std_per_m: float = 0.005  # extra error per metre of range (6 cm at 10 m)
    dropout_prob: float = 0.005  # share of beams that return nothing in each scan

    def range_std(self, r):
        return self.range_std_base + self.range_std_per_m * r


@dataclass(frozen=True)
class ImuNoise:
    gyro_noise_std: float = 0.002  # rad/s, jitter on every sample
    gyro_bias_std: float = 0.001  # rad/s, constant offset drawn once per run (about 3.4 deg/min of heading drift)
    accel_noise_std: float = 0.02  # m/s^2
    accel_bias_std: float = 0.03  # m/s^2, constant offset drawn once per run


@dataclass(frozen=True)
class CliffNoise:
    range_std: float = 0.001  # m


@dataclass(frozen=True)
class OdometrySettings:
    # Wheel odometry is dead reckoning from encoder ticks. In this simulator the wheels barely
    # slip, so on its own it is accurate to about 0.1 %. Real robots are not: their wheel sizes
    # and spacing are never known exactly. The three scale factors below model that calibration
    # error (what the robot believes divided by the truth). Set all three to 1.0 for none.
    encoder_ticks_per_rev: float = 508.8  # Create 3 / Roomba wheel encoder
    wheel_radius_scale_left: float = 1.001  # together: about 0.5 degrees of heading error per metre driven
    wheel_radius_scale_right: float = 0.999
    wheel_separation_scale: float = 1.01  # about 3.6 degrees of error per full turn
    velocity_window_s: float = 0.1  # twist is measured over this window to smooth tick quantization


LIDAR = LidarNoise()
IMU = ImuNoise()
CLIFF = CliffNoise()
ODOMETRY = OdometrySettings()


def noisy_ranges(ranges, rng, cfg=LIDAR):
    """Add range noise and random dropouts to a lidar scan. Invalid beams (inf) stay invalid."""
    ranges = np.asarray(ranges, dtype=np.float32)
    out = ranges.copy()
    valid = np.isfinite(ranges)
    std = cfg.range_std_base + cfg.range_std_per_m * ranges[valid]
    out[valid] = ranges[valid] + rng.normal(0.0, 1.0, size=std.shape).astype(np.float32) * std
    out[rng.random(ranges.shape) < cfg.dropout_prob] = np.inf
    return out


class ImuNoiseModel:
    """White noise plus a constant bias per axis; the gyro z bias also makes the heading drift."""

    def __init__(self, rng, cfg=IMU):
        self.cfg = cfg
        self._rng = rng
        self.gyro_bias = rng.normal(0.0, cfg.gyro_bias_std, 3)
        self.accel_bias = rng.normal(0.0, cfg.accel_bias_std, 3)
        self.yaw_drift = 0.0

    def apply(self, accel, gyro, quat_wxyz, dt):
        """Return noisy (accel, gyro, quat_wxyz). The orientation is the true one turned by the
        heading drift a gyro-integrating IMU would have accumulated."""
        cfg = self.cfg
        accel = np.asarray(accel, dtype=float) + self.accel_bias + self._rng.normal(0.0, cfg.accel_noise_std, 3)
        gyro = np.asarray(gyro, dtype=float) + self.gyro_bias + self._rng.normal(0.0, cfg.gyro_noise_std, 3)
        self.yaw_drift += self.gyro_bias[2] * dt
        half = 0.5 * self.yaw_drift
        dw, dz = math.cos(half), math.sin(half)
        w, x, y, z = (float(v) for v in quat_wxyz)
        # world-frame yaw rotation applied on the left: q_out = q_drift * q_true
        quat = (dw * w - dz * z, dw * x - dz * y, dw * y + dz * x, dw * z + dz * w)
        return accel, gyro, quat


class WheelOdometry:
    """Dead reckoning from wheel encoder ticks, the way a real differential-drive robot does it."""

    def __init__(self, wheel_radius, wheel_separation, cfg=ODOMETRY):
        self.cfg = cfg
        self._r_left = wheel_radius * cfg.wheel_radius_scale_left
        self._r_right = wheel_radius * cfg.wheel_radius_scale_right
        self._separation = wheel_separation * cfg.wheel_separation_scale
        self._rad_per_tick = 2.0 * math.pi / cfg.encoder_ticks_per_rev
        self._ticks = None
        self._last_raw = None  # last raw joint angles; the simulator reports them wrapped to one turn
        self._angle = [0.0, 0.0]  # unwrapped wheel angles
        self._history = []  # (time, distance travelled, yaw) samples for the velocity window
        self._distance = 0.0
        self.x = self.y = self.yaw = 0.0
        self.v = self.w = 0.0

    def update(self, left_angle, right_angle, t):
        """Feed the wheel joint angles (rad, may be wrapped) at time t (s). Returns (x, y, yaw, v, w).

        Call it every frame: unwrapping assumes a wheel turns less than half a revolution between calls."""
        raw = (left_angle, right_angle)
        if self._last_raw is not None:
            for i in (0, 1):
                step = raw[i] - self._last_raw[i]
                self._angle[i] += math.atan2(math.sin(step), math.cos(step))  # shortest way round
        self._last_raw = raw
        ticks = (math.floor(self._angle[0] / self._rad_per_tick), math.floor(self._angle[1] / self._rad_per_tick))
        if self._ticks is None:
            self._ticks = ticks
            self._history.append((t, 0.0, 0.0))
            return self.x, self.y, self.yaw, 0.0, 0.0
        d_left = (ticks[0] - self._ticks[0]) * self._rad_per_tick * self._r_left
        d_right = (ticks[1] - self._ticks[1]) * self._rad_per_tick * self._r_right
        self._ticks = ticks
        d_center = 0.5 * (d_left + d_right)
        d_yaw = (d_right - d_left) / self._separation
        self.x += d_center * math.cos(self.yaw + 0.5 * d_yaw)
        self.y += d_center * math.sin(self.yaw + 0.5 * d_yaw)
        self.yaw += d_yaw
        self._distance += d_center

        self._history.append((t, self._distance, self.yaw))
        while len(self._history) > 2 and t - self._history[1][0] >= self.cfg.velocity_window_s:
            self._history.pop(0)
        t0, dist0, yaw0 = self._history[0]
        if t > t0:
            self.v = (self._distance - dist0) / (t - t0)
            self.w = (self.yaw - yaw0) / (t - t0)
        return self.x, self.y, self.yaw, self.v, self.w
