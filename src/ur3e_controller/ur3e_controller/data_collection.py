import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class DataCollection(Node):
    def __init__(self) -> None:
        super().__init__('data_collection')
        self._latest_joint_state: JointState | None = None
        self._joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._on_joint_state,
            10,
        )
        self._print_timer = self.create_timer(1.0, self._print_latest)

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        self._latest_joint_state = msg

    def _print_latest(self) -> None:
        if self._latest_joint_state is None:
            return
        msg = self._latest_joint_state
        if not msg.name or not msg.position:
            return
        joint_pairs = [f"{name}: {pos:.4f}" for name, pos in zip(msg.name, msg.position)]
        self.get_logger().info("Joint angles -> " + ", ".join(joint_pairs))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCollection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
