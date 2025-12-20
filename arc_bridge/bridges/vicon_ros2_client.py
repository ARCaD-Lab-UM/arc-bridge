from __future__ import annotations

from threading import Thread
from typing import Callable, Optional

import rclpy
from nav_msgs.msg import Odometry


class ViconRos2Client:
    """Subscribe to Vicon topics exposed via ROS2."""

    def __init__(self):
        self.node = None
        self.thread: Optional[Thread] = None
        self._started = False
        self.tron1_topic: Optional[str] = None
        self.slide_object_topic: Optional[str] = None
        self.sub_tron1_topic = None
        self.sub_slide_object_topic = None

    def start(self) -> None:
        if self._started:
            return

        rclpy.init(args=None)
        self.node = rclpy.create_node("arc_bridge_vicon_listener")

        self.thread = Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.thread.start()
        self._started = True
        print("[VICON-ROS2] ROS2 node started")

    def close(self) -> None:
        try:
            if self.node is not None:
                self.node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        finally:
            self.node = None
            self.thread = None
            self._started = False
            self.sub_tron1_topic = None
            self.sub_slide_object_topic = None

    def subscribe_tron1(self, callback: Callable[[Odometry], None], topic: str = "/odometry/tron1") -> None:
        """Start or reconfigure the Tron1 odometry subscription."""
        self.tron1_topic = topic
        if not self._started:
            self.start()
        if self.node is None:
            return
        if self.sub_tron1_topic is not None:
            return
        self.sub_tron1_topic = self.node.create_subscription(Odometry, self.tron1_topic, callback, 10)
        print(f"[VICON-ROS2] Subscribing Tron1 odom: {self.tron1_topic}")

    def subscribe_slide_object(self, callback: Callable[[Odometry], None], topic: str = "/odometry/slide_object") -> None:
        """Start or reconfigure the slide object odometry subscription."""
        self.slide_object_topic = topic
        if not self._started:
            self.start()
        if self.node is None:
            return
        if self.sub_slide_object_topic is not None:
            return
        self.sub_slide_object_topic = self.node.create_subscription(Odometry, self.slide_object_topic, callback, 10)
        print(f"[VICON-ROS2] Subscribing slide object odom: {self.slide_object_topic}")
