import os
from datetime import datetime
from typing import Optional

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class DepthImageSaver(Node):
	def __init__(self) -> None:
		super().__init__("depth_image_saver")

		self.declare_parameter("topic", "/zed/zed_node/depth/depth_registered")
		self.declare_parameter("output_dir", "depth_images")
		self.declare_parameter("save_interval_sec", 2.0)
		self.declare_parameter("max_images", 5)

		self._topic = str(self.get_parameter("topic").value)
		self._output_dir = str(self.get_parameter("output_dir").value)
		self._save_interval_sec = float(self.get_parameter("save_interval_sec").value)
		self._max_images = int(self.get_parameter("max_images").value)

		os.makedirs(self._output_dir, exist_ok=True)

		self._latest_msg: Optional[Image] = None
		self._saved_count = 0

		self.create_subscription(Image, self._topic, self._on_depth, 10)
		self.create_timer(self._save_interval_sec, self._save_latest)

		self.get_logger().info(
			f"Saving {self._max_images} images every {self._save_interval_sec:.1f} sec "
			f"from {self._topic} into {self._output_dir}"
		)

	def _on_depth(self, msg: Image) -> None:
		self._latest_msg = msg

	def _save_latest(self) -> None:
		if self._saved_count >= self._max_images:
			self.get_logger().info("Done saving images. Shutting down.")
			rclpy.shutdown()
			return

		if self._latest_msg is None:
			self.get_logger().warn("No depth image received yet.")
			return

		depth_array = self._image_to_array(self._latest_msg)
		if depth_array is None:
			return

		stamp = self._latest_msg.header.stamp
		stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		filename = f"depth_{self._saved_count:02d}_{timestamp}_{stamp_sec:.3f}.jpg"
		output_path = os.path.join(self._output_dir, filename)
		depth_uint8 = self._normalize_depth_for_jpeg(depth_array)
		cv2.imwrite(output_path, depth_uint8)

		self._saved_count += 1
		self.get_logger().info(f"Saved {output_path}")

	def _image_to_array(self, msg: Image) -> Optional[np.ndarray]:
		if msg.encoding == "32FC1":
			dtype = np.float32
		elif msg.encoding == "16UC1":
			dtype = np.uint16
		else:
			self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
			return None

		expected_len = msg.height * msg.width * np.dtype(dtype).itemsize
		if len(msg.data) < expected_len:
			self.get_logger().warn("Depth image buffer is smaller than expected.")
			return None

		array = np.frombuffer(msg.data, dtype=dtype, count=msg.height * msg.width)
		return array.reshape((msg.height, msg.width))

	def _normalize_depth_for_jpeg(self, depth: np.ndarray) -> np.ndarray:
		depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
		min_val = float(depth_clean.min())
		max_val = float(depth_clean.max())
		if max_val <= min_val:
			return np.zeros(depth_clean.shape, dtype=np.uint8)
		scaled = (depth_clean - min_val) / (max_val - min_val)
		return (scaled * 255.0).clip(0, 255).astype(np.uint8)


def main(args=None) -> None:
	rclpy.init(args=args)
	node = DepthImageSaver()
	try:
		rclpy.spin(node)
	finally:
		node.destroy_node()
		if rclpy.ok():
			rclpy.shutdown()


if __name__ == "__main__":
	main()
