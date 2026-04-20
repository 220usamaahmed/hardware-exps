import math
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import UInt8
from geometry_msgs.msg import TwistStamped
from ecpmi_gripper.srv import GripperControl
import tf2_ros
import random


Position3 = List[float]
Quaternion4 = List[float]


@dataclass
class PoseTarget:
    position: Position3
    orientation: Quaternion4
    linear_tolerance: Optional[float] = None
    angular_tolerance: Optional[float] = None


@dataclass
class Step:
    kind: str  # "pose", "gripper", "recorder_start", "recorder_stop", "wait"
    pose: Optional[PoseTarget] = None
    gripper_command: Optional[str] = None
    wait_sec: float = 0.0


class TrajectoryControl(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_control")

        # Parameters
        # MoveIt Servo twist command topic (delta_twist_cmds)
        self.declare_parameter("command_topic", "/servo_node/delta_twist_cmds")
        self.declare_parameter("control_period", 0.01)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("gripper_service", "/gripper_control")
        self.declare_parameter("gripper_state_topic", "/gripper_state")
        self.declare_parameter("recorder_start_service", "/dataset_recorder/start")
        self.declare_parameter("recorder_stop_service", "/dataset_recorder/stop")
        self.declare_parameter("record", True)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        # Cartesian controller gains and limits (for twist commands)
        self.declare_parameter("k_p_linear", 1.0)
        self.declare_parameter("k_p_angular", 1.0)
        self.declare_parameter("max_linear_speed", 0.2)  # m/s
        self.declare_parameter("max_angular_speed", 1.0)  # rad/s
        self.declare_parameter("linear_tolerance", 0.01)  # m
        self.declare_parameter("angular_tolerance", 0.05)  # rad
        self.declare_parameter("min_linear_speed", 0.005)  # m/s
        self.declare_parameter("min_angular_speed", 0.02)  # rad/s

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
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._ee_frame = str(self.get_parameter("ee_frame").value)
        self._k_p_linear = float(self.get_parameter("k_p_linear").value)
        self._k_p_angular = float(self.get_parameter("k_p_angular").value)
        self._max_linear_speed = float(self.get_parameter("max_linear_speed").value)
        self._max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self._linear_tolerance = float(self.get_parameter("linear_tolerance").value)
        self._angular_tolerance = float(self.get_parameter("angular_tolerance").value)
        self._min_linear_speed = float(self.get_parameter("min_linear_speed").value)
        self._min_angular_speed = float(self.get_parameter("min_angular_speed").value)

        # Hardcoded mixed sequence (waypoints + gripper commands).
        self._steps: List[Step] = self._make_steps()
        self._current_step_index = 0
        self._completed = False
        self._current_pose: Optional[PoseTarget] = None
        self._waiting_until_sec: Optional[float] = None
        self._advance_after_wait = False
        self._gripper_future: Optional[rclpy.task.Future] = None
        self._gripper_wait_sec = 0.0
        self._last_vel_log_sec: Optional[float] = None
        self._last_tf_warn_sec: Optional[float] = None
        self._gripper_state = 0
        self._recorder_start_timer = None
        self._recorder_stop_future: Optional[rclpy.task.Future] = None
        self._shutdown_deadline_sec: Optional[float] = None
        self._shutdown_timer = None

        # ROS interfaces
        self._twist_cmd_pub = self.create_publisher(TwistStamped, self._command_topic, 10)
        self._gripper_state_pub = self.create_publisher(
            UInt8, self._gripper_state_topic, 10
        )
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
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
            f"TrajectoryControl ready. Publishing twist commands on {self._command_topic} now"
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
        # future = self._recorder_start_client.call_async(Trigger.Request())
        # future.add_done_callback(self._handle_start_recorder)

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
        """Build mixed pose target + gripper command sequence."""

        def pose_xyz_rpy_deg(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> PoseTarget:
            qx, qy, qz, qw = self._quat_from_rpy_deg(roll, pitch, yaw)
            return PoseTarget(position=[x, y, z], orientation=[qx, qy, qz, qw])

        # home_pose = pose_xyz_rpy_deg(-90.08, -0.08, -89.20, 0.10, 0.22, -0.22)
        # prepare_to_open = pose_xyz_rpy_deg(0.000, 0.223, 0.694, -90.000, 0.003, 0.002)
        prepare_to_open = pose_xyz_rpy_deg(0.000, 0.223, 0.694, -90.000, 0.003, 0.002)
        
        # approach_pose = pose_xyz_rpy_deg(0.30, 0.10, 0.20, 180.0, 0.0, 10.0)
        # approach_pose.position = [
        #     approach_pose.position[0] + random.uniform(-0.01, 0.01),
        #     approach_pose.position[1] + random.uniform(-0.01, 0.01),
        #     approach_pose.position[2] + random.uniform(-0.01, 0.01),
        # ]

        return [
            Step(kind="pose", pose=prepare_to_open),
            Step(kind="wait", wait_sec=1.0),
            # Step(kind="pose", pose=approach_pose),
            # Step(kind="gripper", gripper_command="grip", wait_sec=0.5),
            # Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            # Step(kind="gripper", gripper_command="blow", wait_sec=0.1),
            # Step(kind="pose", pose=home_pose),
            # Step(kind="wait", wait_sec=0.5),
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

    def _update_current_pose(self) -> bool:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, self._ee_frame, rclpy.time.Time()
            )
        except Exception as exc:  # noqa: BLE001
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if self._last_tf_warn_sec is None or now_sec - self._last_tf_warn_sec >= 2.0:
                self._last_tf_warn_sec = now_sec
                self.get_logger().warn(
                    f"Waiting for TF {self._base_frame} -> {self._ee_frame}: {exc}"
                )
            return False

        t = transform.transform.translation
        r = transform.transform.rotation
        self._current_pose = PoseTarget(
            position=[t.x, t.y, t.z],
            orientation=[r.x, r.y, r.z, r.w],
        )
        return True

    def _abort_with_error(self, message: str) -> None:
        self.get_logger().error(message)
        self._finish_and_shutdown()

    def _finish_and_shutdown(self) -> None:
        self._completed = True
        self._publish_twist_command([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
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
        # if self._recorder_stop_future is not None:
        #     return
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
        # self._shutdown_deadline_sec = time.time() + 2.0
        # if self._shutdown_timer is None:
        #     self._shutdown_timer = self.create_timer(0.1, self._check_shutdown_ready)

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
        # If no steps or already completed, send zero twist
        if not self._steps or self._completed:
            self._publish_twist_command([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self._waiting_until_sec is not None:
            if now_sec < self._waiting_until_sec:
                self._publish_twist_command([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
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
        if step.kind == "wait":
            self._handle_wait_step(step)
            return
        if step.kind == "gripper":
            self._handle_gripper_step(step)
            return

        if step.kind != "pose" or step.pose is None:
            self._abort_with_error("Invalid step configuration; stopping node.")
            return

        self._set_gripper_state(0)

        if not self._update_current_pose() or self._current_pose is None:
            self._publish_twist_command([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            return

        target = step.pose
        linear_error = [
            target.position[i] - self._current_pose.position[i] for i in range(3)
        ]
        angular_error = self._angular_error(self._current_pose.orientation, target.orientation)
        linear_norm = self._vector_norm(linear_error)
        angular_norm = self._vector_norm(angular_error)
        linear_tol = target.linear_tolerance or self._linear_tolerance
        angular_tol = target.angular_tolerance or self._angular_tolerance

        if linear_norm <= linear_tol and angular_norm <= angular_tol:
            self.get_logger().info(
                f"Reached pose target {self._current_step_index + 1}/{len(self._steps)}"
            )
            if step.wait_sec > 0.0:
                self._waiting_until_sec = now_sec + step.wait_sec
                self._advance_after_wait = True
                self._publish_twist_command([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                return
            self._current_step_index += 1
            return

        linear_cmd = [self._k_p_linear * v for v in linear_error]
        angular_cmd = [self._k_p_angular * v for v in angular_error]

        linear_cmd = self._clamp_vector(linear_cmd, self._max_linear_speed)
        angular_cmd = self._clamp_vector(angular_cmd, self._max_angular_speed)
        linear_cmd = self._apply_min_speed(linear_cmd, self._min_linear_speed)
        angular_cmd = self._apply_min_speed(angular_cmd, self._min_angular_speed)

        if self._vector_norm(linear_cmd) > 0.0 or self._vector_norm(angular_cmd) > 0.0:
            if self._last_vel_log_sec is None or now_sec - self._last_vel_log_sec >= 1.0:
                self._last_vel_log_sec = now_sec
                self.get_logger().info(f"Twist linear cmd: {linear_cmd}")
                self.get_logger().info(f"Twist angular cmd: {angular_cmd}")
                self.get_logger().info(
                    f"Pose errors (lin, ang): {[round(v, 4) for v in linear_error]}, {[round(v, 4) for v in angular_error]}"
                )
                self.get_logger().info("--")
        self._publish_twist_command(linear_cmd, angular_cmd)

    def _handle_gripper_step(self, step: Step) -> None:
        self._publish_twist_command([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

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

    def _handle_wait_step(self, step: Step) -> None:
        if step.wait_sec <= 0.0:
            self._current_step_index += 1
            return
        if self._waiting_until_sec is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._waiting_until_sec = now_sec + step.wait_sec
            self._advance_after_wait = True
        self._publish_twist_command([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def _publish_twist_command(self, linear: List[float], angular: List[float]) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self._twist_cmd_pub.publish(msg)

    def _vector_norm(self, vec: List[float]) -> float:
        return math.sqrt(sum(v * v for v in vec))

    def _clamp_vector(self, vec: List[float], max_norm: float) -> List[float]:
        if max_norm <= 0.0:
            return vec
        norm = self._vector_norm(vec)
        if norm <= max_norm or norm == 0.0:
            return vec
        scale = max_norm / norm
        return [v * scale for v in vec]

    def _apply_min_speed(self, vec: List[float], min_norm: float) -> List[float]:
        if min_norm <= 0.0:
            return vec
        norm = self._vector_norm(vec)
        if norm == 0.0 or norm >= min_norm:
            return vec
        scale = min_norm / norm
        return [v * scale for v in vec]

    def _quat_from_rpy_deg(self, roll: float, pitch: float, yaw: float) -> Quaternion4:
        r = math.radians(roll)
        p = math.radians(pitch)
        y = math.radians(yaw)
        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return [qx, qy, qz, qw]

    def _quat_conjugate(self, q: Quaternion4) -> Quaternion4:
        return [-q[0], -q[1], -q[2], q[3]]

    def _quat_multiply(self, a: Quaternion4, b: Quaternion4) -> Quaternion4:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]

    def _angular_error(self, current: Quaternion4, target: Quaternion4) -> List[float]:
        q_err = self._quat_multiply(target, self._quat_conjugate(current))
        q_err = self._normalize_quat(q_err)
        qw = max(min(q_err[3], 1.0), -1.0)
        angle = 2.0 * math.acos(qw)
        if angle > math.pi:
            angle = 2.0 * math.pi - angle
            q_err = [-q_err[0], -q_err[1], -q_err[2], -q_err[3]]
        sin_half = math.sqrt(max(0.0, 1.0 - qw * qw))
        if sin_half < 1e-6 or angle == 0.0:
            return [0.0, 0.0, 0.0]
        axis = [q_err[0] / sin_half, q_err[1] / sin_half, q_err[2] / sin_half]
        return [axis[0] * angle, axis[1] * angle, axis[2] * angle]

    def _normalize_quat(self, q: Quaternion4) -> Quaternion4:
        norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if norm == 0.0:
            return [0.0, 0.0, 0.0, 1.0]
        return [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
