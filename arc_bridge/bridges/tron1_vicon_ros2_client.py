from __future__ import annotations

from threading import Thread
from typing import Callable, Optional

import rclpy
from nav_msgs.msg import Odometry

class ViconRos2Client:
    """Subscribe to Vicon topics exposed via ROS2."""

    def __init__(self, odom_topic: str = "/odometry"):
        self.odom_topic = odom_topic
        self.node = None
        self.thread: Optional[Thread] = None
        self._started = False
        self._odom_callback: Optional[Callable[[Odometry], None]] = None

    def register_odom_callback(self, callback: Callable[[Odometry], None]) -> None:
        """Register a callback that will receive the raw Odometry msg."""

        self._odom_callback = callback

    def start(self) -> None:
        if self._started:
            return

        rclpy.init(args=None)
        self.node = rclpy.create_node("arc_bridge_vicon_listener")

        # QoS and subscriptions
        self.sub_odom = self.node.create_subscription(
            Odometry, self.odom_topic, self._odom_cb, 10
        )

        self.thread = Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()
        self._started = True
        print(f"[VICON-ROS2] Subscribing: {self.odom_topic}")

    def close(self) -> None:
        try:
            if self.node is not None:
                self.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        finally:
            self._started = False

    # callback functions
    def _odom_cb(self, msg: Odometry) -> None:
        if self._odom_callback is None:
            return
        try:
            self._odom_callback(msg)
        except Exception as e:
            print(f"[VICON-ROS2] odometry callback failed: {e}")
