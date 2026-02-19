import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from control_msgs.msg import JointJog
from ecpmi_gripper.srv import GripperControl


JointWaypoint = List[float]


@dataclass
class Step:
    kind: str  # "waypoint" or "gripper"
    waypoint: Optional[JointWaypoint] = None
    gripper_command: Optional[str] = None
    wait_sec: float = 0.0


class TrajectoryControl(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_control")

        # Parameters
        # MoveIt Servo joint command topic (delta_joint_cmds)
        self.declare_parameter("command_topic", "/servo_node/delta_joint_cmds")
        self.declare_parameter("control_period", 0.01)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("gripper_service", "/gripper_control")
        # Joint-space controller gains and limits (for joint velocities)
        self.declare_parameter("k_p_joint", 0.8)
        self.declare_parameter("max_joint_speed", 0.3)  # rad/s
        self.declare_parameter("joint_tolerance", 0.01)  # rad

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._control_period = float(self.get_parameter("control_period").value)
        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._start_servo_service = str(self.get_parameter("start_servo_service").value)
        self._gripper_service = str(self.get_parameter("gripper_service").value)
        self._k_p_joint = float(self.get_parameter("k_p_joint").value)
        self._max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self._joint_tolerance = float(self.get_parameter("joint_tolerance").value)

        # UR3e joint order used by MoveIt (and Servo)
        self._joint_names: List[str] = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # Hardcoded mixed sequence (waypoints + gripper commands).
        self._steps: List[Step] = self._make_steps()
        self._current_step_index = 0
        self._completed = False
        self._current_joints: Optional[List[float]] = None
        self._name_to_index: Optional[Dict[str, int]] = None
        self._waiting_until_sec: Optional[float] = None
        self._advance_after_wait = False
        self._gripper_future: Optional[rclpy.task.Future] = None
        self._gripper_wait_sec = 0.0

        # ROS interfaces
        self._joint_cmd_pub = self.create_publisher(JointJog, self._command_topic, 10)
        self._joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self._start_servo_client = self.create_client(Trigger, self._start_servo_service)
        self._gripper_client = self.create_client(GripperControl, self._gripper_service)
        self._start_servo_timer = None
        if self._auto_start_servo:
            self._start_servo_timer = self.create_timer(1.0, self._try_start_servo)

        self._control_timer = self.create_timer(self._control_period, self._control_step)

        self.get_logger().info(
            f"TrajectoryControl ready. Publishing joint commands on {self._command_topic} now"
        )

    def _make_steps(self) -> List[Step]:
        """Build mixed waypoint + gripper command sequence."""

        # Waypoints are specified in degrees and converted to radians.
        waypoint_a = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]
        waypoint_b = [72.6, -5.0, 0.0, 0.0, 0.0, 0.0]
        waypoint_c = [0.0, -90.0, 90.0, 0.0, 90.0, 0.0]

        def to_rad(waypoint_deg: JointWaypoint) -> JointWaypoint:
            return [math.radians(angle_deg) for angle_deg in waypoint_deg]

        return [
            Step(kind="waypoint", waypoint=to_rad(waypoint_a)),
            Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
            Step(kind="waypoint", waypoint=to_rad(waypoint_b)),
            Step(kind="gripper", gripper_command="release", wait_sec=1.0),
            Step(kind="waypoint", waypoint=to_rad(waypoint_c)),
            Step(kind="gripper", gripper_command="blow", wait_sec=1.0),
        ]

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

    def _joint_state_callback(self, msg: JointState) -> None:
        # Lazy initialization of name->index mapping using first message
        if self._name_to_index is None:
            self._name_to_index = {name: i for i, name in enumerate(msg.name)}
            missing = [n for n in self._joint_names if n not in self._name_to_index]
            if missing:
                self.get_logger().warn(
                    f"JointState is missing joints: {missing}. Waypoint tracking may fail."
                )

        if self._name_to_index is None:
            return

        # Build current joint vector in configured order
        joints: List[float] = []
        for name in self._joint_names:
            idx = self._name_to_index.get(name)
            if idx is None or idx >= len(msg.position):
                return
            joints.append(msg.position[idx])

        self._current_joints = joints

    def _abort_with_error(self, message: str) -> None:
        self.get_logger().error(message)
        self._completed = True
        self._publish_joint_command([0.0] * len(self._joint_names))
        if self._start_servo_timer is not None:
            self._start_servo_timer.cancel()
        if self._control_timer is not None:
            self._control_timer.cancel()
        rclpy.shutdown()

    def _control_step(self) -> None:
        # If no steps or already completed, send zero joint velocity
        if not self._steps or self._completed:
            self._publish_joint_command([0.0] * len(self._joint_names))
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._waiting_until_sec is not None:
            if now_sec < self._waiting_until_sec:
                self._publish_joint_command([0.0] * len(self._joint_names))
                return
            self._waiting_until_sec = None
            if self._advance_after_wait:
                self._advance_after_wait = False
                self._current_step_index += 1

        if self._current_step_index >= len(self._steps):
            self._completed = True
            self._publish_joint_command([0.0] * len(self._joint_names))
            return

        step = self._steps[self._current_step_index]
        if step.kind == "gripper":
            self._handle_gripper_step(step)
            return

        if step.kind != "waypoint" or step.waypoint is None:
            self._abort_with_error("Invalid step configuration; stopping node.")
            return

        # Need a valid joint state before we can move
        if self._current_joints is None:
            self._publish_joint_command([0.0] * len(self._joint_names))
            return

        target = step.waypoint
        if len(target) != len(self._joint_names):
            self._abort_with_error(
                "Waypoint length does not match number of joints; stopping node."
            )
            return

        # Joint error and norm
        errors: List[float] = []
        max_err = 0.0
        for current, goal in zip(self._current_joints, target):
            e = goal - current
            errors.append(e)
            max_err = max(max_err, abs(e))

        if max_err < self._joint_tolerance:
            self.get_logger().info(
                f"Reached joint waypoint {self._current_step_index + 1}/{len(self._steps)}"
            )
            if step.wait_sec > 0.0:
                self._waiting_until_sec = now_sec + step.wait_sec
                self._advance_after_wait = True
                self._publish_joint_command([0.0] * len(self._joint_names))
                return
            self._current_step_index += 1
            return

        # Proportional joint velocity command
        velocities: List[float] = []
        for e in errors:
            v = self._k_p_joint * e
            # Clamp each joint speed
            if abs(v) > self._max_joint_speed > 0.0:
                v = math.copysign(self._max_joint_speed, v)
            velocities.append(v)

        # self.get_logger().info(f"Publishing joint velocities {velocities}")
        self._publish_joint_command(velocities)

    def _handle_gripper_step(self, step: Step) -> None:
        self._publish_joint_command([0.0] * len(self._joint_names))

        if self._gripper_future is None:
            if not self._gripper_client.wait_for_service(timeout_sec=0.1):
                self.get_logger().warn(
                    f"Waiting for gripper service at {self._gripper_service}"
                )
                return
            request = GripperControl.Request()
            request.command = step.gripper_command or ""
            self._gripper_wait_sec = step.wait_sec
            self._gripper_future = self._gripper_client.call_async(request)
            return

        if not self._gripper_future.done():
            return

        try:
            response = self._gripper_future.result()
        except Exception as exc:  # noqa: BLE001
            self._abort_with_error(f"Gripper service call failed: {exc}")
            return
        finally:
            self._gripper_future = None

        if not response.success:
            self._abort_with_error(f"Gripper command failed: {response.message}")
            return

        if self._gripper_wait_sec > 0.0:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._waiting_until_sec = now_sec + self._gripper_wait_sec
            self._advance_after_wait = True
            return

        self._current_step_index += 1

    def _publish_joint_command(self, velocities: List[float]) -> None:
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self._joint_names
        msg.velocities = velocities
        msg.displacements = []
        msg.duration = 0.0
        self._joint_cmd_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
