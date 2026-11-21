"""ROS 2 publishers that run inside the Isaac Sim process.

run_sim.py creates one InProcessPublishers and calls step() every frame. It reads the simulator's
perfect sensor values, applies the noise models from sensor_noise.py (unless noise is off), and
publishes:

  (in) /cmd_vel_safe   geometry_msgs/Twist    watched only: if it goes quiet for 0.5 s a stop is commanded
  /odom                nav_msgs/Odometry      wheel odometry (encoder dead reckoning), frame odom
  /tf                  tf2_msgs/TFMessage     odom -> base_link, from the same odometry
  /ground_truth/odom   nav_msgs/Odometry      true pose and velocity, frame world (for tests only)
  /imu                 sensor_msgs/Imu        frame imu_link
  /scan                sensor_msgs/LaserScan  2D lidar from the physics raycast sensor (10 Hz)
  /bumper              std_msgs/UInt8         bit mask of pressed zones (house_layout.BUMPER_ZONES)
  /cliff/front_left    sensor_msgs/Range      distance to the floor under each cliff sensor;
  /cliff/front_right                          +inf means nothing within range. A reading above
  /cliff/side_left                            house_layout.CLIFF_THRESHOLD is a cliff.
  /cliff/side_right
  /tf_static           tf2_msgs/TFMessage     fixed sensor frames under base_link (sent once, latched)

With noise off, /odom and /tf carry the true pose and every sensor is exact.

Isaac Sim has no ROS 2 graph nodes for contact or raycast sensors, and its odometry and IMU nodes
publish perfect values only, so all of these are published from here. Only standard message types
are used: custom messages would have to be rebuilt for Isaac Sim's Python 3.12.
"""

import math

import house_layout as hl
import numpy as np
import omni.usd
import rclpy
import sensor_noise as sn
from geometry_msgs.msg import TransformStamped, Twist
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.physics import IMU, Contact, ContactSensor, IMUSensor, Raycast, RaycastSensor
from nav_msgs.msg import Odometry
from pxr import UsdGeom
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import String, UInt8
from tf2_msgs.msg import TFMessage

# Height of the asset's cliff_* frames above the ray origins, so ranges are reported from the frame.
_CLIFF_FRAME_Z = 0.0192
# Rotation from a ROS body frame (x forward, z up) to its optical frame (z forward, x right, y down).
_OPTICAL_QUAT_XYZW = (-0.5, 0.5, -0.5, 0.5)
LEFT_WHEEL_JOINT = "left_wheel_joint"
RIGHT_WHEEL_JOINT = "right_wheel_joint"


def _to_numpy(value):
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _stamp(header, t):
    header.stamp.sec = int(t)
    header.stamp.nanosec = int((t - int(t)) * 1e9)


def _yaw_of(w, x, y, z):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _diag(values, size):
    """Row-major covariance matrix with the given diagonal."""
    cov = [0.0] * (size * size)
    for i, v in enumerate(values):
        cov[i * size + i] = float(v)
    return cov


class InProcessPublishers:
    def __init__(
        self,
        robot_path,
        base_link_path,
        bumper_path,
        cliff_path,
        imu_path,
        lidar_path=None,
        frame_rate=60.0,
        noise=True,
        seed=0,
        world="house",
    ):
        """lidar_path is the raycast lidar prim, or None when the RTX lidar graph publishes /scan.

        world is the name of the world being simulated. It goes out once on /sim/world (latched),
        so the robot software can refuse to run with another world's maps."""
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("isaac_sim_sensors")
        self._noise = bool(noise)
        self._rng = np.random.default_rng(seed)
        self._base_path = base_link_path
        self._base = RigidPrim(base_link_path)

        # odometry
        self._robot = Articulation(robot_path)
        names = list(self._robot.dof_names)
        missing = [j for j in (LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT) if j not in names]
        if missing:
            raise RuntimeError(f"wheel joints {missing} not found in articulation dofs {names}")
        self._left_i, self._right_i = names.index(LEFT_WHEEL_JOINT), names.index(RIGHT_WHEEL_JOINT)
        self._wheel_odom = sn.WheelOdometry(hl.WHEEL_RADIUS, hl.WHEEL_SEPARATION)
        self._odom_pub = self._node.create_publisher(Odometry, "odom", 10)
        self._truth_pub = self._node.create_publisher(Odometry, "ground_truth/odom", 10)
        self._tf_pub = self._node.create_publisher(TFMessage, "tf", 10)

        # imu
        self._imu = IMUSensor(IMU(imu_path, reset_xform_op_properties=False))
        self._imu_noise = sn.ImuNoiseModel(self._rng)
        self._imu_pub = self._node.create_publisher(Imu, "imu", 10)

        # bumper
        self._bumper_pub = self._node.create_publisher(UInt8, "bumper", 10)
        self._contact = ContactSensor(Contact(bumper_path, reset_xform_op_properties=False))
        self._contact.add_raw_contact_data_to_frame()

        # cliff
        self._cliff_names = list(hl.CLIFF_SENSORS)
        self._cliff_pubs = {
            name: self._node.create_publisher(Range, "cliff/" + name[len("cliff_") :], 10) for name in self._cliff_names
        }
        self._cliff = RaycastSensor(Raycast(cliff_path, reset_xform_op_properties=False))

        # lidar
        self._lidar = None
        if lidar_path:
            self._lidar = RaycastSensor(Raycast(lidar_path, reset_xform_op_properties=False))
            self._scan_pub = self._node.create_publisher(LaserScan, "scan", 10)
            self._scan_every = max(1, int(round(frame_rate / hl.LIDAR_RATE_HZ)))
        self._bumper_every = max(1, int(round(frame_rate / 20.0)))  # bumper also goes out at once on a change

        # Motor-controller watchdog: if wheel commands stop arriving, command a stop ourselves.
        self._wheel_pub = self._node.create_publisher(Twist, hl.WHEEL_CMD_TOPIC, 10)
        self._node.create_subscription(Twist, hl.WHEEL_CMD_TOPIC, self._on_wheel_cmd, 1)  # newest only: no backlog
        self._watchdog_frames = max(1, int(round(hl.BASE_CMD_TIMEOUT * frame_rate)))
        self._frames_since_cmd = 0
        self._wheels_commanded = False  # true while the last command received was non-zero
        self._own_stop_pending = 0

        self._frame = 0
        self._last_t = None
        self._last_mask = None
        self._publish_static_frames()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._world_pub = self._node.create_publisher(String, "sim/world", latched)
        self._world_pub.publish(String(data=world))

    # --- fixed frames -----------------------------------------------------------
    def _publish_static_frames(self):
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._tf_static_pub = self._node.create_publisher(TFMessage, "tf_static", latched)
        stage = omni.usd.get_context().get_stage()
        t = SimulationManager.get_simulation_time()

        def make(parent, child, xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
            m = TransformStamped()
            _stamp(m.header, t)
            m.header.frame_id = parent
            m.child_frame_id = child
            m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in xyz)
            q = m.transform.rotation
            q.x, q.y, q.z, q.w = (float(v) for v in quat_xyzw)
            return m

        def from_prim(name):
            """Local pose of an asset mount frame directly under base_link."""
            prim = stage.GetPrimAtPath(f"{self._base_path}/{name}")
            if not prim.IsValid():
                raise RuntimeError(f"mount frame {name} not found under {self._base_path}")
            mat = UsdGeom.Xformable(prim).GetLocalTransformation()
            tr = mat.ExtractTranslation()
            rot = mat.ExtractRotationQuat()
            im = rot.GetImaginary()
            return make("base_link", name, tr, (im[0], im[1], im[2], rot.GetReal()))

        frames = [
            make("base_link", "laser", hl.LIDAR_XYZ),
            make("base_link", "camera_link", hl.CAMERA_XYZ),
            make("camera_link", "camera_optical_frame", (0.0, 0.0, 0.0), _OPTICAL_QUAT_XYZW),
            from_prim("imu_link"),
        ] + [from_prim(name) for name in self._cliff_names]
        self._tf_static_pub.publish(TFMessage(transforms=frames))

    # --- odometry -----------------------------------------------------------------
    def _publish_odometry(self, t, pos, quat_wxyz):
        w, x, y, z = quat_wxyz
        true_yaw = _yaw_of(w, x, y, z)
        lin, ang = self._base.get_velocities()
        lin, ang = _to_numpy(lin)[0], _to_numpy(ang)[0]
        cos_y, sin_y = math.cos(true_yaw), math.sin(true_yaw)

        truth = Odometry()
        _stamp(truth.header, t)
        truth.header.frame_id = "world"
        truth.child_frame_id = "base_link"
        truth.pose.pose.position.x, truth.pose.pose.position.y, truth.pose.pose.position.z = (float(v) for v in pos)
        q = truth.pose.pose.orientation
        q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
        truth.twist.twist.linear.x = float(cos_y * lin[0] + sin_y * lin[1])
        truth.twist.twist.linear.y = float(-sin_y * lin[0] + cos_y * lin[1])
        truth.twist.twist.angular.z = float(ang[2])
        self._truth_pub.publish(truth)

        # wheel odometry is integrated every frame even when it is not the published source
        angles = _to_numpy(self._robot.get_dof_positions())[0]
        ox, oy, oyaw, ov, ow = self._wheel_odom.update(float(angles[self._left_i]), float(angles[self._right_i]), t)
        if not self._noise:  # perfect odometry: the true planar pose, relative to the world origin start
            ox, oy, oyaw = float(pos[0]), float(pos[1]), true_yaw
            ov, ow = truth.twist.twist.linear.x, truth.twist.twist.angular.z

        odom = Odometry()
        _stamp(odom.header, t)
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = ox
        odom.pose.pose.position.y = oy
        odom.pose.pose.orientation.z = math.sin(0.5 * oyaw)
        odom.pose.pose.orientation.w = math.cos(0.5 * oyaw)
        odom.twist.twist.linear.x = float(ov)
        odom.twist.twist.angular.z = float(ow)
        odom.pose.covariance = _diag([1e-3, 1e-3, 1e6, 1e6, 1e6, 1e-2], 6)
        odom.twist.covariance = _diag([1e-3, 1e-3, 1e6, 1e6, 1e6, 1e-2], 6)
        self._odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = "base_link"
        tf.transform.translation.x = ox
        tf.transform.translation.y = oy
        tf.transform.rotation = odom.pose.pose.orientation
        self._tf_pub.publish(TFMessage(transforms=[tf]))

    # --- imu ------------------------------------------------------------------------
    def _publish_imu(self, t, dt):
        frame = self._imu.get_data()
        accel = _to_numpy(frame["linear_acceleration"]).astype(float)
        gyro = _to_numpy(frame["angular_velocity"]).astype(float)
        quat = tuple(float(v) for v in _to_numpy(frame["orientation"]))  # w, x, y, z
        cfg = sn.IMU
        if self._noise:
            accel, gyro, quat = self._imu_noise.apply(accel, gyro, quat, dt)
            gyro_var, accel_var, yaw_var = cfg.gyro_noise_std**2, cfg.accel_noise_std**2, 1e-2
        else:
            gyro_var, accel_var, yaw_var = 1e-8, 1e-8, 1e-8
        msg = Imu()
        _stamp(msg.header, t)
        msg.header.frame_id = "imu_link"
        msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z = quat
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (float(v) for v in gyro)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (float(v) for v in accel)
        msg.orientation_covariance = _diag([1e-4, 1e-4, yaw_var], 3)
        msg.angular_velocity_covariance = _diag([gyro_var] * 3, 3)
        msg.linear_acceleration_covariance = _diag([accel_var] * 3, 3)
        self._imu_pub.publish(msg)

    # --- bumper -----------------------------------------------------------------
    def _bumper_mask(self, pos, quat_wxyz):
        contacts = self._contact.get_data().get("contacts") or []
        if not contacts:
            return 0
        yaw = _yaw_of(*quat_wxyz)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        mask = 0
        for c in contacts:
            other = c["body1"] if "/base_link" in c["body0"] else c["body0"]
            if "floor" in other.lower():
                continue  # the caster resting on the floor is not a bump
            dx, dy = float(c["position"][0] - pos[0]), float(c["position"][1] - pos[1])
            bx, by = cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy
            if math.hypot(bx, by) < 0.08:
                continue  # under the body, not on the bumper rim
            angle = math.degrees(math.atan2(by, bx))
            for _name, bit, lo, hi in hl.BUMPER_ZONES:
                if lo <= angle < hi:
                    mask |= bit
        return mask

    # --- cliff ------------------------------------------------------------------
    def _publish_cliff(self, t):
        depths = _to_numpy(self._cliff.get_data().get("depths", []))
        for i, name in enumerate(self._cliff_names):
            d = float(depths[i]) if i < len(depths) else float("nan")
            msg = Range()
            _stamp(msg.header, t)
            msg.header.frame_id = name
            msg.radiation_type = Range.INFRARED
            msg.field_of_view = 0.05
            msg.min_range = 0.0
            msg.max_range = hl.CLIFF_MAX_RANGE
            if not math.isfinite(d) or d <= 0.0 or d >= hl.CLIFF_MAX_RANGE:
                msg.range = float("inf")
            else:
                msg.range = d + (_CLIFF_FRAME_Z - hl.CLIFF_RAY_Z)
                if self._noise:
                    msg.range = max(0.0, msg.range + float(self._rng.normal(0.0, sn.CLIFF.range_std)))
            self._cliff_pubs[name].publish(msg)

    # --- lidar ------------------------------------------------------------------
    def _publish_scan(self, t):
        depths = _to_numpy(self._lidar.get_data().get("depths", [])).astype(np.float32)
        if depths.size != hl.LIDAR_BEAMS:
            return  # sensor not ready yet
        ranges = depths.copy()
        ranges[~np.isfinite(ranges) | (ranges < hl.LIDAR_MIN_RANGE) | (ranges >= hl.LIDAR_MAX_RANGE)] = np.inf
        if self._noise:
            ranges = sn.noisy_ranges(ranges, self._rng)
            ranges[ranges < hl.LIDAR_MIN_RANGE] = np.inf
        msg = LaserScan()
        _stamp(msg.header, t)
        msg.header.frame_id = "laser"
        msg.angle_min = -math.pi
        msg.angle_increment = 2.0 * math.pi / hl.LIDAR_BEAMS
        msg.angle_max = msg.angle_min + (hl.LIDAR_BEAMS - 1) * msg.angle_increment
        msg.time_increment = 0.0  # all rays are cast at the same instant
        msg.scan_time = 1.0 / hl.LIDAR_RATE_HZ
        msg.range_min = hl.LIDAR_MIN_RANGE
        msg.range_max = hl.LIDAR_MAX_RANGE
        msg.ranges = ranges.tolist()
        self._scan_pub.publish(msg)

    # --- per frame ----------------------------------------------------------------
    def step(self):
        """Read the sensors and publish. Call once per simulation frame while playing."""
        self._frame += 1
        t = SimulationManager.get_simulation_time()
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        positions, quats = self._base.get_world_poses()
        pos = _to_numpy(positions)[0]
        quat = tuple(float(v) for v in _to_numpy(quats)[0])  # w, x, y, z

        self._publish_odometry(t, pos, quat)
        self._publish_imu(t, dt)

        mask = self._bumper_mask(pos, quat)
        if mask != self._last_mask or self._frame % self._bumper_every == 0:  # a new bump is sent immediately
            self._bumper_pub.publish(UInt8(data=mask))
            self._last_mask = mask
        self._publish_cliff(t)  # every frame: the safety gate's stopping distance depends on this delay
        self._base_watchdog()
        if self._lidar is not None and self._frame % self._scan_every == 0:
            self._publish_scan(t)

    # --- motor-controller watchdog -----------------------------------------------
    def _on_wheel_cmd(self, msg):
        if self._own_stop_pending > 0 and msg.linear.x == 0.0 and msg.angular.z == 0.0:
            self._own_stop_pending -= 1  # our own stop command coming back to us
            return
        self._frames_since_cmd = 0
        self._wheels_commanded = msg.linear.x != 0.0 or msg.angular.z != 0.0

    def _base_watchdog(self):
        """Stop the wheels if a non-zero command was the last thing heard and it has gone quiet."""
        rclpy.spin_once(self._node, timeout_sec=0.0)  # deliver any waiting wheel command
        self._frames_since_cmd += 1
        if self._wheels_commanded and self._frames_since_cmd > self._watchdog_frames:
            self._wheel_pub.publish(Twist())
            self._own_stop_pending += 1
            self._wheels_commanded = False

    def shutdown(self):
        self._node.destroy_node()
