import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from std_msgs.msg import UInt8
from control_msgs.msg import JointJog
from ecpmi_gripper.srv import GripperControl
import random


JointWaypoint = List[float]


@dataclass
class Step:
    kind: str  # "waypoint", "gripper", "recorder_start", "recorder_stop"
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
        self.declare_parameter("gripper_state_topic", "/gripper_state")
        self.declare_parameter("recorder_start_service", "/dataset_recorder/start")
        self.declare_parameter("recorder_stop_service", "/dataset_recorder/stop")
        self.declare_parameter("record", True)
        # Joint-space controller gains and limits (for joint velocities)
        self.declare_parameter("k_p_joint", 2.0)
        # self.declare_parameter("max_joint_speed", 1.5)  # rad/s
        self.declare_parameter("max_joint_speed", 0.5)  # rad/s
        self.declare_parameter("joint_tolerance", 0.1)  # rad
        self.declare_parameter("min_joint_speed", 0.02)  # rad/s

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._control_period = float(self.get_parameter("control_period").value)
        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._start_servo_service = str(self.get_parameter("start_servo_service").value)
        self._gripper_service = str(self.get_parameter("gripper_service").value)
        self._gripper_state_topic = str(self.get_parameter("gripper_state_topic").value)
        self._recorder_start_service = str(
            self.get_parameter("recorder_start_service").value
        )
        self._recorder_stop_service = str(self.get_parameter("recorder_stop_service").value)
        self._record_enabled = bool(self.get_parameter("record").value)
        self._k_p_joint = float(self.get_parameter("k_p_joint").value)
        self._max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self._joint_tolerance = float(self.get_parameter("joint_tolerance").value)
        self._min_joint_speed = float(self.get_parameter("min_joint_speed").value)

        # UR3e joint order used by MoveIt (and Servo)
        self._joint_names: List[str] = [
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
            "shoulder_pan_joint",
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
        self._last_vel_log_sec: Optional[float] = None
        self._gripper_state = 0
        self._recorder_start_timer = None
        self._recorder_stop_future: Optional[rclpy.task.Future] = None
        self._shutdown_deadline_sec: Optional[float] = None
        self._shutdown_timer = None

        # ROS interfaces
        self._joint_cmd_pub = self.create_publisher(JointJog, self._command_topic, 10)
        self._gripper_state_pub = self.create_publisher(
            UInt8, self._gripper_state_topic, 10
        )
        self._joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self._start_servo_client = self.create_client(Trigger, self._start_servo_service)
        self._gripper_client = self.create_client(GripperControl, self._gripper_service)
        self._recorder_start_client = None
        self._recorder_stop_client = None
        if self._record_enabled:
            self._recorder_start_client = self.create_client(
                Trigger, self._recorder_start_service
            )
            self._recorder_stop_client = self.create_client(
                Trigger, self._recorder_stop_service
            )
        self._start_servo_timer = None
        if self._auto_start_servo:
            self._start_servo_timer = self.create_timer(1.0, self._try_start_servo)
        if self._record_enabled:
            self._recorder_start_timer = self.create_timer(1.0, self._try_start_recorder)

        self._control_timer = self.create_timer(self._control_period, self._control_step)

        self._publish_gripper_state(self._gripper_state)

        self.get_logger().info(
            f"TrajectoryControl ready. Publishing joint commands on {self._command_topic} now"
        )

    def _publish_gripper_state(self, state: int) -> None:
        msg = UInt8()
        msg.data = int(state)
        self._gripper_state_pub.publish(msg)

    def _set_gripper_state(self, state: int) -> None:
        if self._gripper_state != state:
            self._gripper_state = state
            self._publish_gripper_state(state)

    def _try_start_recorder(self) -> None:
        if not self._record_enabled:
            return
        if self._recorder_start_client is None:
            return
        if not self._recorder_start_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for dataset recorder start service at {self._recorder_start_service}"
            )
            return
        future = self._recorder_start_client.call_async(Trigger.Request())
        future.add_done_callback(self._handle_start_recorder)

    def _handle_start_recorder(self, future: rclpy.task.Future) -> None:
        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"Failed to call recorder start: {exc}")
            return
        if response.success:
            self.get_logger().info("Dataset recorder started.")
            if self._recorder_start_timer is not None:
                self._recorder_start_timer.cancel()
        else:
            self.get_logger().warn(f"Recorder start failed: {response.message}")

    def _make_steps(self) -> List[Step]:
        """Build mixed waypoint + gripper command sequence."""

        """
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "shoulder_pan_joint",
        """

        # Waypoints are specified in degrees and converted to radians.
        home = [-90.00, 0.00, -90.00, 0.00, -0.00, 90.00]
        home_with_noise = [angle + random.uniform(-15, 15) for angle in home]
        
        lb_grip_0 = {
            "lb_gripping_prepare": [-167.59, 69.91, -263.37, -26.39, 0.21, -22.35],
            "lb_gripping": [-150.04, 11.48, -222.11, -43.35, -0.24, -39.32],
            "lb_gripping_pull": [-166.49, 68.98, -263.67, -24.82, 0.34, -20.76],
            "lb_gripper_backoff": [-146.55, 41.70, -256.56, -21.76, 0.62, -17.69],
        }
        
        lb_grip_1 = {
            "lb_gripping_prepare": [-169.24, 69.71, -255.65, -25.30, 0.00, -19.78],
            "lb_gripping": [-151.32, 13.42, -218.94, -40.92, 2.00, -35.43],
            "lb_gripping_pull": [-169.64, 72.54, -257.72, -23.51, -0.37, -17.99],
            "lb_gripper_backoff": [-152.09, 59.24, -259.49, -15.69, -2.99, -10.12],
        }
        
        lb_grip_2 = {
            "lb_gripping_prepare": [-168.18, 78.32, -265.08, -25.37, -0.23, -19.85],
            "lb_gripping": [-156.02, 28.36, -229.23, -44.06, 2.10, -38.59],
            "lb_gripping_pull": [-168.19, 83.05, -268.92, -21.42, -1.19, -15.87],
            "lb_gripper_backoff": [-143.81, 60.50, -265.76, -11.50, -6.38, -5.85],
        }
        
        gripper_choice = random.choice([lb_grip_0, lb_grip_1, lb_grip_2])
        lb_gripping_prepare = gripper_choice["lb_gripping_prepare"]
        lb_gripping = gripper_choice["lb_gripping"]
        lb_gripping_pull = gripper_choice["lb_gripping_pull"]
        lb_gripper_backoff = gripper_choice["lb_gripper_backoff"]
        
        def to_rad(waypoint_deg: JointWaypoint) -> JointWaypoint:
            return [math.radians(angle_deg) for angle_deg in waypoint_deg]

        # return [
        #     Step(kind="waypoint", waypoint=to_rad(lb_gripping_prepare)),
        # ]

        return [
            Step(kind="waypoint", waypoint=to_rad(home)),
            
            # Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            
            # Step(kind="recorder_start"),
            
            # Step(kind="waypoint", waypoint=to_rad(lb_gripping_prepare)),
            # Step(kind="waypoint", waypoint=to_rad(lb_gripping)),
            
            # Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
            # Step(kind="gripper", gripper_command="release", wait_sec=1.0),
            
            # Step(kind="waypoint", waypoint=to_rad(lb_gripping_pull)),
            
            # Step(kind="gripper", gripper_command="blow", wait_sec=1.0),
            
            # # Step(kind="waypoint", waypoint=to_rad(lb_gripper_backoff)),
            # Step(kind="waypoint", waypoint=to_rad(home)),

            # Step(kind="recorder_stop"),
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
        self._finish_and_shutdown()

    def _finish_and_shutdown(self) -> None:
        self._completed = True
        self._publish_joint_command([0.0] * len(self._joint_names))
        self._set_gripper_state(0)
        if self._start_servo_timer is not None:
            self._start_servo_timer.cancel()
        if self._recorder_start_timer is not None:
            self._recorder_start_timer.cancel()
        if self._control_timer is not None:
            self._control_timer.cancel()
        if self._record_enabled:
            self._request_recorder_stop()
        else:
            rclpy.shutdown()

    def _request_recorder_stop(self) -> None:
        if not self._record_enabled:
            return
        if self._recorder_stop_future is not None:
            return
        if self._recorder_stop_client is None:
            self.get_logger().warn(
                "Recorder stop client not initialized. Shutting down anyway."
            )
            rclpy.shutdown()
            return
        if not self._recorder_stop_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Recorder stop service unavailable at {self._recorder_stop_service}. Shutting down anyway."
            )
            rclpy.shutdown()
            return
        self._recorder_stop_future = self._recorder_stop_client.call_async(
            Trigger.Request()
        )
        # Use wall time for shutdown timeout to avoid stalled /clock.
        self._shutdown_deadline_sec = time.time() + 2.0
        if self._shutdown_timer is None:
            self._shutdown_timer = self.create_timer(0.1, self._check_shutdown_ready)

    def _check_shutdown_ready(self) -> None:
        now_sec = time.time()
        if self._recorder_stop_future is not None and self._recorder_stop_future.done():
            try:
                response = self._recorder_stop_future.result()
                if response.success:
                    self.get_logger().info("Dataset recorder stopped.")
                else:
                    self.get_logger().warn(
                        f"Recorder stop failed: {response.message}"
                    )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Failed to stop recorder: {exc}")
            rclpy.shutdown()
            return
        if self._shutdown_deadline_sec is not None and now_sec >= self._shutdown_deadline_sec:
            self.get_logger().warn("Recorder stop timeout. Shutting down.")
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
            self.get_logger().info("All steps complete. Shutting down.")
            self._finish_and_shutdown()
            return

        step = self._steps[self._current_step_index]
        if step.kind == "recorder_start":
            self._handle_recorder_start_step()
            return
        if step.kind == "recorder_stop":
            self._handle_recorder_stop_step()
            return
        if step.kind == "gripper":
            self._handle_gripper_step(step)
            return

        if step.kind != "waypoint" or step.waypoint is None:
            self._abort_with_error("Invalid step configuration; stopping node.")
            return

        self._set_gripper_state(0)

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

        # Joint error and norm (wrapped to shortest angular distance)
        errors: List[float] = []
        max_err = 0.0
        for current, goal in zip(self._current_joints, target):
            e = math.atan2(math.sin(goal - current), math.cos(goal - current))
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
            if 0.0 < self._min_joint_speed <= self._max_joint_speed and abs(v) > 0.0:
                if abs(v) < self._min_joint_speed:
                    v = math.copysign(self._min_joint_speed, v)
            velocities.append(v)
            
        # self.get_logger().info(f"Errors: {errors}")
        # self.get_logger().info(f"Velocities: {velocities}")

        if any(abs(v) > 0.0 for v in velocities):
            if self._last_vel_log_sec is None or now_sec - self._last_vel_log_sec >= 1.0:
                self._last_vel_log_sec = now_sec
                self.get_logger().info(f"Joint velocities: {velocities}")
                self.get_logger().info(f"Joint errors: {errors}")
                self.get_logger().info("--")
        self._publish_joint_command(velocities)

    def _handle_gripper_step(self, step: Step) -> None:
        self._publish_joint_command([0.0] * len(self._joint_names))

        command_map = {"grip": 1, "release": 2, "blow": 3}
        self._set_gripper_state(command_map.get(step.gripper_command or "", 0))

        if self._gripper_future is None:
            if not self._gripper_client.wait_for_service(timeout_sec=0.1):
                self.get_logger().warn(
                    f"Waiting for gripper service at {self._gripper_service}"
                )
                return
            request = GripperControl.Request()
            request.command = step.gripper_command or ""
            self.get_logger().info(
                f"Starting gripper task: {request.command or 'unknown'}"
            )
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

        self.get_logger().info(
            f"Finished gripper task: {step.gripper_command or 'unknown'}"
        )

        if self._gripper_wait_sec > 0.0:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._waiting_until_sec = now_sec + self._gripper_wait_sec
            self._advance_after_wait = True
            return

        self._current_step_index += 1

    def _handle_recorder_start_step(self) -> None:
        if not self._record_enabled:
            self._current_step_index += 1
            return
        if self._recorder_start_client is None:
            self._abort_with_error("Recorder start client not initialized; stopping node.")
            return
        if not self._recorder_start_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for dataset recorder start service at {self._recorder_start_service}"
            )
            return
        future = self._recorder_start_client.call_async(Trigger.Request())
        future.add_done_callback(self._handle_start_recorder)
        self._current_step_index += 1

    def _handle_recorder_stop_step(self) -> None:
        if not self._record_enabled:
            self._current_step_index += 1
            return
        self._request_recorder_stop()
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
