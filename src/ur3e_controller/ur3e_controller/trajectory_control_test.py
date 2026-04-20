import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_srvs.srv import Trigger


class TrajectoryControlTest(Node):
    def __init__(self) -> None:
        super().__init("trajectory_control_test")

        self.declare_parameter("command_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("command_frame", "tool0")
        self.declare_parameter("publish_period", 0.02)
        self.declare_parameter("linear_x", 0.0)
        self.declare_parameter("linear_y", 0.0)
        self.declare_parameter("linear_z", 0.02)
        self.declare_parameter("angular_x", 0.0)
        self.declare_parameter("angular_y", 0.0)
        self.declare_parameter("angular_z", 0.0)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._command_frame = str(self.get_parameter("command_frame").value)
        self._publish_period = float(self.get_parameter("publish_period").value)
        self._linear_x = float(self.get_parameter("linear_x").value)
        self._linear_y = float(self.get_parameter("linear_y").value)
        self._linear_z = float(self.get_parameter("linear_z").value)
        self._angular_x = float(self.get_parameter("angular_x").value)
        self._angular_y = float(self.get_parameter("angular_y").value)
        self._angular_z = float(self.get_parameter("angular_z").value)
        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._start_servo_service = str(self.get_parameter("start_servo_service").value)

        self._twist_pub = self.create_publisher(TwistStamped, self._command_topic, 10)
        self._start_servo_client = self.create_client(Trigger, self._start_servo_service)
        self._start_servo_timer = None
        if self._auto_start_servo:
            self._start_servo_timer = self.create_timer(1.0, self._try_start_servo)

        self.create_timer(self._publish_period, self._publish_twist)

        self.get_logger().info(
            f"TrajectoryControlTest ready. Publishing constant Twist to {self._command_topic}"
        )

    def _try_start_servo(self) -> None:
        if not self._auto_start_servo:
            return
        if not self._start_servo_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for MoveIt Servo start service at {self._start_servo_service}"
            )
            return
        future = self._start_servo_client.call_async(Trigger.Request())
        future.add_done_callback(self._handle_start_servo)

    def _handle_start_servo(self, future: rclpy.task.Future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to call start_servo: {exc}")
            return
        if response.success:
            self.get_logger().info("MoveIt Servo started.")
            if self._start_servo_timer is not None:
                self._start_servo_timer.cancel()
        else:
            self.get_logger().warn(f"MoveIt Servo start failed: {response.message}")

    def _publish_twist(self) -> None:
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = self._command_frame
        twist.twist.linear.x = self._linear_x
        twist.twist.linear.y = self._linear_y
        twist.twist.linear.z = self._linear_z
        twist.twist.angular.x = self._angular_x
        twist.twist.angular.y = self._angular_y
        twist.twist.angular.z = self._angular_z
        self._twist_pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryControlTest()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
