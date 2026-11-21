"""Test helper for talking to the mode manager.

Needs the ROS workspace sourced (for vacuum_interfaces):
    source ros2_ws/install/setup.bash
"""

import time

from lifecycle_msgs.srv import GetState
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from vacuum_interfaces.msg import RobotState
from vacuum_interfaces.srv import ChangeState, SaveMap


class ModeClient:
    """Wraps /change_state, /save_map, /robot_state and lifecycle queries for a test node.

    `node` is any rclpy node with a spin(seconds) method, such as the test Driver classes.
    """

    def __init__(self, node):
        self.node = node
        self.state = None  # latest RobotState
        self.history = []  # every distinct (state, reason) seen
        self._change = node.create_client(ChangeState, "change_state")
        self._save = node.create_client(SaveMap, "save_map")
        self._lifecycle = {}
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(RobotState, "robot_state", self._on_state, latched)

    def _on_state(self, msg):
        self.state = msg
        key = (msg.state, msg.reason)
        if not self.history or self.history[-1] != key:
            self.history.append(key)

    def _call(self, client, request, wall_timeout):
        if not client.wait_for_service(timeout_sec=10.0):
            return None
        future = client.call_async(request)
        end = time.time() + wall_timeout
        while not future.done() and time.time() < end:
            self.node.spin(0.05)
        return future.result() if future.done() else None

    def wait_ready(self, wall_timeout=90.0):
        """Wait until the mode manager has finished starting up. Returns the RobotState or None."""
        end = time.time() + wall_timeout
        while time.time() < end:
            self.node.spin(0.1)
            if self.state is not None and not self.state.transitioning:
                return self.state
        return None

    def request(self, new_state, wall_timeout=120.0):
        """Ask for a mode. Returns (success, message)."""
        response = self._call(self._change, ChangeState.Request(new_state=new_state), wall_timeout)
        if response is None:
            return False, "no answer from /change_state (is the mode manager running?)"
        self.node.spin(0.2)  # pick up the state message that follows the change
        return response.success, response.message

    def save_map(self, name, wall_timeout=40.0):
        """Save the map being built. Returns (success, message, yaml path)."""
        # The tests save over their own maps from earlier runs on purpose.
        response = self._call(self._save, SaveMap.Request(name=name, replace=True), wall_timeout)
        if response is None:
            return False, "no answer from /save_map", ""
        return response.success, response.message, response.path

    def wait_for_state(self, state, wall_timeout=30.0):
        """Wait until /robot_state reports `state` (and no change in progress)."""
        end = time.time() + wall_timeout
        while time.time() < end:
            self.node.spin(0.05)
            if self.state is not None and self.state.state == state and not self.state.transitioning:
                return True
        return False

    def lifecycle(self, node_name, wall_timeout=3.0):
        """Lifecycle state label of a managed node, asked of the node itself; None if no answer."""
        if node_name not in self._lifecycle:
            self._lifecycle[node_name] = self.node.create_client(GetState, f"{node_name}/get_state")
        client = self._lifecycle[node_name]
        if not client.wait_for_service(timeout_sec=wall_timeout):
            return None
        # The first call on a new client can lose its reply on Humble (the server answers before
        # it has finished connecting to the new client), so ask again before giving up.
        for _ in range(3):
            response = self._call(client, GetState.Request(), wall_timeout)
            if response is not None:
                return response.current_state.label
        return None
