import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from std_msgs.msg import UInt8, Int32
from control_msgs.msg import JointJog
from ecpmi_gripper.srv import GripperControl
import random
import numpy as np


JointWaypoint = List[float]


def _unwrap_goal(current: float, goal: float) -> float:
    return current + math.atan2(math.sin(goal - current), math.cos(goal - current))


def _joint_angle_error(current: float, goal: float) -> float:
    return math.atan2(math.sin(goal - current), math.cos(goal - current))


@dataclass
class _HermiteSegment:
    p0: float
    v0: float
    p1: float
    v1: float
    duration: float


class _SmoothViaPath:
    """C¹ cubic Hermite path: q_start -> q_via -> q_final with zero end velocities."""

    def __init__(
        self,
        q_start: List[float],
        q_via: List[float],
        q_final: List[float],
        max_speed: float,
        duration_sec: Optional[float] = None,
    ) -> None:
        n = len(q_start)
        q_via_u = [_unwrap_goal(q_start[i], q_via[i]) for i in range(n)]
        q_final_u = [_unwrap_goal(q_via_u[i], q_final[i]) for i in range(n)]

        d1 = max(abs(q_via_u[i] - q_start[i]) for i in range(n))
        d2 = max(abs(q_final_u[i] - q_via_u[i]) for i in range(n))
        total_dist = d1 + d2

        if duration_sec is not None and duration_sec > 0.0:
            self.duration = duration_sec
            if total_dist > 0.0:
                self._t1 = duration_sec * (d1 / total_dist)
            else:
                self._t1 = duration_sec * 0.5
            self._t2 = duration_sec - self._t1
        else:
            self._t1 = max(d1 / max_speed, 0.1) if d1 > 0.0 else 0.1
            self._t2 = max(d2 / max_speed, 0.1) if d2 > 0.0 else 0.1
            self.duration = self._t1 + self._t2

        self._seg1: List[_HermiteSegment] = []
        self._seg2: List[_HermiteSegment] = []
        for i in range(n):
            v_via = (q_final_u[i] - q_start[i]) / self.duration
            self._seg1.append(
                _HermiteSegment(q_start[i], 0.0, q_via_u[i], v_via, self._t1)
            )
            self._seg2.append(
                _HermiteSegment(q_via_u[i], v_via, q_final_u[i], 0.0, self._t2)
            )
        self.q_start = list(q_start)
        self.q_via = list(q_via_u)
        self.q_final = q_final_u
        self.path_length = max(total_dist, 0.01)
        self.via_u = self._t1 / self.duration if self.duration > 0.0 else 0.5

    def position_at_u(self, u: float) -> List[float]:
        u = min(max(u, 0.0), 1.0)
        positions, _ = self.evaluate(u * self.duration)
        return positions

    def find_progress_u(self, q_current: List[float], u_min: float = 0.0) -> float:
        """Project q_current onto the path, searching forward from u_min only."""
        u_min = min(max(u_min, 0.0), 1.0)
        best_u = u_min
        best_err = float("inf")
        steps = 64
        for i in range(steps + 1):
            u = u_min + (1.0 - u_min) * i / steps
            q = self.position_at_u(u)
            err = max(
                abs(_joint_angle_error(c, r)) for c, r in zip(q_current, q)
            )
            if err < best_err:
                best_err = err
                best_u = u
        return best_u

    @staticmethod
    def _hermite_eval(seg: _HermiteSegment, t: float) -> Tuple[float, float]:
        duration = seg.duration
        if duration <= 0.0:
            return seg.p1, 0.0
        s = min(max(t / duration, 0.0), 1.0)
        s2 = s * s
        s3 = s2 * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        pos = h00 * seg.p0 + h10 * duration * seg.v0 + h01 * seg.p1 + h11 * duration * seg.v1
        inv_t = 1.0 / duration
        dh00 = (6.0 * s2 - 6.0 * s) * inv_t
        dh10 = 3.0 * s2 - 4.0 * s + 1.0
        dh01 = (-6.0 * s2 + 6.0 * s) * inv_t
        dh11 = 3.0 * s2 - 2.0 * s
        vel = dh00 * seg.p0 + dh10 * seg.v0 + dh01 * seg.p1 + dh11 * seg.v1
        return pos, vel

    def evaluate(self, t: float) -> Tuple[List[float], List[float]]:
        positions: List[float] = []
        velocities: List[float] = []
        if t <= self._t1:
            segments = self._seg1
            local_t = t
        else:
            segments = self._seg2
            local_t = t - self._t1
        for seg in segments:
            pos, vel = self._hermite_eval(seg, local_t)
            positions.append(pos)
            velocities.append(vel)
        return positions, velocities


@dataclass
class Step:
    kind: str  # "waypoint", "smooth-waypoints", "gripper", ...
    waypoint: Optional[JointWaypoint] = None
    waypoint_via: Optional[JointWaypoint] = None
    waypoint_final: Optional[JointWaypoint] = None
    gripper_command: Optional[str] = None
    wait_sec: float = 0.0
    duration_sec: Optional[float] = None
    output_dir: Optional[str] = None
    segment_id: Optional[int] = None


class TrajectoryControl(Node):
    def __init__(self) -> None:
        super().__init__("trajectory_control")

        # Parameters
        # MoveIt Servo joint command topic (delta_joint_cmds)
        self.declare_parameter("command_topic", "/servo_node/delta_joint_cmds")
        self.declare_parameter("command_topic_raw", "/servo_node/delta_joint_cmds_raw")
        self.declare_parameter("control_period", 0.01)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("gripper_service", "/gripper_control")
        self.declare_parameter("gripper_state_topic", "/gripper_state")
        self.declare_parameter("recorder_start_service", "/dataset_recorder/start")
        self.declare_parameter("recorder_stop_service", "/dataset_recorder/stop")
        self.declare_parameter("record", True)
        # Joint-space controller gains and limits (for joint velocities)
        self.declare_parameter("k_p_joint", 4.0)
        # self.declare_parameter("max_joint_speed", 1.5)  # rad/s
        self.declare_parameter("max_joint_speed", 1.0)  # rad/s
        self.declare_parameter("joint_tolerance", 0.01)  # rad
        self.declare_parameter("min_joint_speed", 0.01)  # rad/s
        self.declare_parameter("velocity_noise_std", 0.0)  # rad/s
        self.declare_parameter(
            "smooth_path_lookahead_u", 0.12
        )  # fraction of path ahead for steering target

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._control_period = float(self.get_parameter("control_period").value)
        self._command_topic_raw = str(self.get_parameter("command_topic_raw").value)
        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._start_servo_service = str(self.get_parameter("start_servo_service").value)
        self._gripper_service = str(self.get_parameter("gripper_service").value)
        self._gripper_state_topic = str(self.get_parameter("gripper_state_topic").value)
        self._recorder_start_service = str(
            self.get_parameter("recorder_start_service").value
        )
        self._recorder_stop_service = str(
            self.get_parameter("recorder_stop_service").value
        )
        self._record_enabled = bool(self.get_parameter("record").value)
        self._k_p_joint = float(self.get_parameter("k_p_joint").value)
        self._max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self._joint_tolerance = float(self.get_parameter("joint_tolerance").value)
        self._min_joint_speed = float(self.get_parameter("min_joint_speed").value)
        self._velocity_noise_std = float(self.get_parameter("velocity_noise_std").value)
        self._smooth_path_lookahead_u = float(
            self.get_parameter("smooth_path_lookahead_u").value
        )

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
        self._gripper_state = 0.0
        self._current_segment_id = 0
        self._recorder_start_timer = None
        self._recorder_stop_future: Optional[rclpy.task.Future] = None
        self._shutdown_deadline_sec: Optional[float] = None
        self._shutdown_timer = None
        self.recorder_stopping = False
        self.noise_counter = 0
        self.additive_noise = [0.0] * 6  # Initialize additive noise for each joint
        self.noise_counter_max = 1000
        self._smooth_path: Optional[_SmoothViaPath] = None
        self._smooth_path_u: float = 0.0

        # ROS interfaces
        self._joint_cmd_pub = self.create_publisher(JointJog, self._command_topic, 10)
        self._joint_cmd_raw_pub = self.create_publisher(
            JointJog, self._command_topic_raw, 10
        )
        self._gripper_state_pub = self.create_publisher(
            UInt8, self._gripper_state_topic, 10
        )
        self._segment_pub = self.create_publisher(Int32, "/current_segment", 10)
        self._joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self._start_servo_client = self.create_client(
            Trigger, self._start_servo_service
        )
        self._gripper_client = self.create_client(GripperControl, self._gripper_service)
        self._recorder_start_client = None
        self._recorder_stop_client = None
        self._recorder_param_client = None
        if self._record_enabled:
            self._recorder_start_client = self.create_client(
                Trigger, self._recorder_start_service
            )
            self._recorder_stop_client = self.create_client(
                Trigger, self._recorder_stop_service
            )
            self._recorder_param_client = self.create_client(
                SetParameters, "/dataset_recorder/set_parameters"
            )
        self._start_servo_timer = None
        if self._auto_start_servo:
            self._start_servo_timer = self.create_timer(1.0, self._try_start_servo)
        if self._record_enabled:
            self._recorder_start_timer = self.create_timer(
                1.0, self._try_start_recorder
            )

        self._control_timer = self.create_timer(
            self._control_period, self._control_step
        )

        self._publish_gripper_state(self._gripper_state)
        self._publish_segment(self._current_segment_id)

        self.get_logger().info(
            f"TrajectoryControl ready. Publishing joint commands on {self._command_topic} now"
        )

    def _publish_gripper_state(self, state: int) -> None:
        msg = UInt8()
        msg.data = int(state)
        self._gripper_state_pub.publish(msg)

    def _publish_segment(self, segment_id: int) -> None:
        msg = Int32()
        msg.data = int(segment_id)
        self._segment_pub.publish(msg)

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
        """Build mixed waypoint + gripper command sequence."""

        """
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "shoulder_pan_joint",
        """

        def to_rad(waypoint_deg: JointWaypoint) -> JointWaypoint:
            return [math.radians(angle_deg) for angle_deg in waypoint_deg]

        random.seed(time.time())

        # Waypoints are specified in degrees and converted to radians.
        home_noise_deviations = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        home = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
        home_with_noise = [
            angle + random.uniform(-deviation, deviation)
            for angle, deviation in zip(home, home_noise_deviations)
        ]

        grip_center = {
            "gripping_prepare": [-120.02, -51.74, -98.13, 90.07, 88.82, -72.64],
            "gripping": [-120.04, -68.39, -81.47, 90.08, 88.84, -72.65],
            "gripping_prepare_1": [-61.30, 53.02, -82.03, -89.31, 88.72, 72.04],
            "gripping_1": [-61.13, 70.20, -99.37, -89.32, 88.80, 72.07],
        }

        gripper_choice = random.choice([grip_center])

        gripping_prepare = gripper_choice["gripping_prepare_1"]
        gripping = gripper_choice["gripping_1"]

        # gripping_prepare = gripper_choice["gripping_prepare_1"]
        # gripping = gripper_choice["gripping_1"]

        ## Pick and place left drawer

        lift = [-107.66, -23.45, -139.67, 90.22, 0.21, -64.52]
        lift = [angle + random.uniform(-2, 2) for angle in lift]

        hover_over_left = [-123.76, -23.52, -105.79, 90.23, 0.21, -47.43]
        hover_over_left = [angle + random.uniform(-2, 2) for angle in hover_over_left]

        put_in_left = [-133.61, -26.93, -108.49, 90.23, 0.21, -47.44]
        put_in_left = [angle + random.uniform(-1, 1) for angle in put_in_left]

        ## Pick and place right drawer

        hover_over_right = [-85.24, 70.69, -105.86, -77.93, 89.84, 36.87]
        hover_over_right = [angle + random.uniform(-2, 2) for angle in hover_over_right]

        put_in_right = [-72.68, 70.60, -106.79, -77.93, 89.84, 39.15]
        put_in_right = [angle + random.uniform(-1, 1) for angle in put_in_right]

        # Left 1
        grip = [-128.98, -74.22, -158.44, -31.19, 90.0, -32.55]
        w1 = [-123.18, -84.32, -154.35, -26.44, 90.0, -27.81]
        w2 = [-117.50, -93.28, -151.13, -20.86, 90.65, -22.18]
        w3 = [-110.88, -103.58, -148.70, -11.79, 91.94, -13.12]
        w4 = [-108.18, -107.64, -150.28, -5.93, 94.92, -7.26]
        id1 = [-106.55, -40.54, -120.97, -12.71, 90.00, 24.60]
        id2 = [-110.41, -19.88, -116.55, -10.87, 90.00, 4.19]
        iu1 = [-102.37, -82.28, -131.71, -4.32, 93.39, 6.79]
        iu2 = [-101.53, -77.68, -131.71, -4.33, 93.39, -0.77]

        # Left 2
        # grip = [-131.76, -68.77, -160.65, -30.05, 89.94, -31.39]
        # w1 = [-126.52, -78.09, -156.69, -25.99, 90.06, -27.35]
        # w2 = [-120.94, -87.61, -153.00, -20.55, 90.32, -21.93]
        # w3 = [-113.55, -99.50, -149.69, -10.67, 91.54, -12.06]
        # w4 = [-110.97, -103.52, -151.28, -4.89, 94.60, -6.28]

        # Left 3
        # grip = [-125.47, -79.93, -155.61, -33.94, 89.73, -35.28]
        # w1 = [-119.03, -90.79, -151.31, -28.66, 89.85, -30.02]
        # w2 = [-113.14, -100.13, -148.06, -22.59, 90.07, -23.97]
        # w3 = [-106.15, -110.38, -145.54, -13.27, 90.85, -14.67]
        # w4 = [-102.65, -115.16, -146.02, -6.86, 92.63, -8.27]

        # Right 1
        # grip = [-52.82, 77.41, -28.14, 30.39, 88.09, 30.74]
        # w1 = [-59.32, 88.58, -33.56, 24.85, 88.88, 25.21]
        # w2 = [-64.83, 97.60, -38.28, 19.11, 90.15, 19.47]
        # w3 = [-71.86, 108.52, -46.26, 10.81, 94.31, 11.14]
        # w4 = [-77.05, 116.86, -58.34, 5.60, 103.29, 5.79]

        # Right 2
        # grip = [-56.51, 83.31, -30.63, 32.13, 87.73, 32.49]
        # w1 = [-61.11, 91.04, -34.23, 28.26, 88.25, 28.63]
        # w2 = [-67.18, 100.72, -38.93, 22.22, 89.40, 22.60]
        # w3 = [-75.64, 113.26, -47.09, 12.38, 93.60, 12.72]
        # w4 = [-80.82, 120.87, -56.88, 6.95, 101.03, 7.17]

        # Right 3
        # grip = [-50.55, 72.98, -26.68, 28.42, 88.29, 28.76]
        # w1 = [-54.82, 80.52, -30.49, 24.98, 88.87, 25.32]
        # w2 = [-60.98, 90.99, -36.22, 19.02, 90.36, 19.36]
        # w3 = [-68.37, 102.99, -45.49, 10.74, 95.10, 11.03]
        # w4 = [-72.21, 109.41, -54.23, 6.86, 101.31, 7.04]
        
        # id = random.choice([id1, id2])
        id = random.choice([id1])
        iu = random.choice([iu1, iu2])

        """
        0: Not yet set
        1: Open Drawer Left
        2: Open Drawer Right
        3: Pick
        4: Place Left
        5: Place Right
        """
        
        # return [
        #     Step(kind="waypoint", waypoint=to_rad(home)),
        #     Step(kind="waypoint", waypoint=to_rad(w1)),
        #     # Step(
        #     #     kind="smooth-waypoints",
        #     #     waypoint_via=to_rad(id),
        #     #     waypoint_final=to_rad(w1),
        #     # ),
        # ]
        
        # return [
        #     Step(kind="waypoint", waypoint=to_rad(w4)),
        #     Step(kind="waypoint", waypoint=to_rad(home)),
        #     # Step(
        #     #     kind="smooth-waypoints",
        #     #     waypoint_via=to_rad(iu),
        #     #     waypoint_final=to_rad(home),
        #     # ),
        # ]
        
        return [
            Step(kind="waypoint", waypoint=to_rad(home)),
            
            Step(kind="recorder_start"),
            Step(kind="wait", wait_sec=1.0),
            
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(id),
                waypoint_final=to_rad(w1),
            ),
            Step(kind="waypoint", waypoint=to_rad(grip)),
            Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
            Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            
            Step(kind="waypoint", waypoint=to_rad(w1)),
            Step(kind="waypoint", waypoint=to_rad(w2)),
            Step(kind="waypoint", waypoint=to_rad(w3)),
            Step(kind="waypoint", waypoint=to_rad(w4)),
            
            Step(kind="gripper", gripper_command="blow", wait_sec=0.1),
            
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(iu),
                waypoint_final=to_rad(home),
            ),
            
            Step(
                kind="recorder_stop", output_dir="/home/shokry/ur3e-trajectories/smooth/open_left/open_left"
            ),
        ]

        return [
            Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            Step(kind="start-segment", segment_id=1),  # Open Drawer
            Step(kind="wait", wait_sec=1.0),
            Step(kind="recorder_start"),
            Step(kind="wait", wait_sec=1.0),
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(w1),
                waypoint_final=to_rad(grip),
            ),
            Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
            Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(w1),
                waypoint_final=to_rad(w2),
            ),
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(w2),
                waypoint_final=to_rad(w3),
            ),
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(w3),
                waypoint_final=to_rad(w4),
            ),
            Step(kind="gripper", gripper_command="blow", wait_sec=1.0),
            Step(kind="waypoint", waypoint=to_rad(home)),
            Step(kind="start-segment", segment_id=3),  # Pick
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(gripping_prepare),
                waypoint_final=to_rad(gripping),
            ),
            Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
            Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            Step(kind="waypoint", waypoint=to_rad(home)),
            Step(kind="start-segment", segment_id=4),  # Place Left
            Step(
                kind="smooth-waypoints",
                waypoint_via=to_rad(hover_over_left),
                waypoint_final=to_rad(put_in_left),
            ),
            Step(kind="gripper", gripper_command="blow", wait_sec=0.1),
            Step(kind="waypoint", waypoint=to_rad(home)),
            Step(
                kind="recorder_stop", output_dir="/home/shokry/ur3e-trajectories/st/st"
            ),
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

        def callback(future):
            try:
                response = future.result()
                if response.success:
                    self.get_logger().info("Dataset recorder stopped.")
                    self._current_step_index += 1
                    self.recorder_stopping = False
                else:
                    self.get_logger().warn(f"Recorder stop failed: {response.message}")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Failed to stop recorder: {exc}")

        self._recorder_stop_future.add_done_callback(callback)

        # self._recorder_stop_future = self._recorder_stop_client.call(
        #     Trigger.Request()
        # )
        print("Requested recorder stop returned")
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
                    self.get_logger().warn(f"Recorder stop failed: {response.message}")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Failed to stop recorder: {exc}")
            rclpy.shutdown()
            return
        if (
            self._shutdown_deadline_sec is not None
            and now_sec >= self._shutdown_deadline_sec
        ):
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
        if step.kind == "reset_noise":
            self._reset_noise()
            self._current_step_index += 1
            return
        if step.kind == "hold":
            input("Trajectory complete. Press Enter to continue...")
            self._current_step_index += 1
            return
        if step.kind == "recorder_start":
            self._handle_recorder_start_step()
            return
        if step.kind == "recorder_stop":
            self._handle_recorder_stop_step()
            return
        if step.kind == "wait":
            print("-" * 20)
            print(f"Waiting for {step.wait_sec} seconds...")
            print("-" * 20)
            self._handle_wait_step(step)
            return
        if step.kind == "start-segment":
            self._current_segment_id = (
                step.segment_id if step.segment_id is not None else 0
            )
            self._publish_segment(self._current_segment_id)
            self._current_step_index += 1
            return
        if step.kind == "gripper":
            self._handle_gripper_step(step)
            return
        if step.kind == "smooth-waypoints":
            self._handle_smooth_waypoints_step(step, now_sec)
            return

        if step.kind != "waypoint" or step.waypoint is None:
            self._abort_with_error("Invalid step configuration; stopping node.")
            return

        self._set_gripper_state(self._gripper_state)
        self._publish_segment(self._current_segment_id)

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
        # print("Current joints:", [math.degrees(j) for j in self._current_joints])
        # print("Target waypoint:", [math.degrees(t) for t in target])
        # scale  = 1 / np.exp(0.1 * (30 - self.noise_counter)) if self.noise_counter > 0 else 0.0
        # scale  = self.noise_counter / self.noise_counter_max if self.noise_counter > 0 else 0.0
        # scale = np.exp(-np.square(self.noise_counter - self.noise_counter_max // 2) / 20000) if self.noise_counter > 0 else 0.0
        scale = (
            np.square(np.sin(np.pi * self.noise_counter / self.noise_counter_max))
            if self.noise_counter > 0
            else 0.0
        )
        print(f"Noise counter: {self.noise_counter}, scale: {scale:.2f}")

        for i, (current, goal) in enumerate(zip(self._current_joints, target)):
            e = _joint_angle_error(current, goal)

            added_noise = self.additive_noise[i] * scale
            # print(f"Joint {i}: error={e:.2f} rad, noise_i = {self.noise_counter} scale = {scale:.2f}, noise={added_noise:.2f} rad")

            e += added_noise
            errors.append(e)
            max_err = max(max_err, abs(e))

        self.noise_counter = max(0, self.noise_counter - 1)

        # Check if the target is reached
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

        # Normalize velocities so all joints finish at the same time
        velocities = self._errors_to_velocities(errors, max_err)
        if any(abs(v) > 0.0 for v in velocities):
            if (
                self._last_vel_log_sec is None
                or now_sec - self._last_vel_log_sec >= 1.0
            ):
                self._last_vel_log_sec = now_sec
        self._publish_joint_command(velocities)

    def _errors_to_velocities(
        self, errors: List[float], max_err: float
    ) -> List[float]:
        velocities: List[float] = []
        for e in errors:
            normalized_speed = (
                (abs(e) / max_err) * self._max_joint_speed if max_err > 0 else 0.0
            )
            v = math.copysign(normalized_speed, e)
            if abs(v) > self._max_joint_speed > 0.0:
                v = math.copysign(self._max_joint_speed, v)
            velocities.append(v)
        return velocities

    def _clamp_joint_velocities(self, velocities: List[float]) -> List[float]:
        clamped: List[float] = []
        for v in velocities:
            if abs(v) > self._max_joint_speed > 0.0:
                v = math.copysign(self._max_joint_speed, v)
            clamped.append(v)
        return clamped

    def _clear_smooth_path(self) -> None:
        self._smooth_path = None
        self._smooth_path_u = 0.0

    def _advance_smooth_path_progress(self) -> None:
        """March progress along the path at roughly max joint speed."""
        assert self._smooth_path is not None
        du = self._max_joint_speed * self._control_period / self._smooth_path.path_length
        du = min(du, 0.05)
        self._smooth_path_u = min(1.0, self._smooth_path_u + du)

        projected_u = self._smooth_path.find_progress_u(
            self._current_joints, u_min=self._smooth_path_u
        )
        self._smooth_path_u = max(self._smooth_path_u, projected_u)

    def _handle_smooth_waypoints_step(self, step: Step, now_sec: float) -> None:
        self._set_gripper_state(self._gripper_state)
        self._publish_segment(self._current_segment_id)

        if self._current_joints is None:
            self._publish_joint_command([0.0] * len(self._joint_names))
            return

        q_via = step.waypoint_via
        q_final = step.waypoint_final
        if q_via is None or q_final is None:
            self._abort_with_error(
                "smooth-waypoints step requires waypoint_via and waypoint_final."
            )
            return
        if len(q_via) != len(self._joint_names) or len(q_final) != len(
            self._joint_names
        ):
            self._abort_with_error(
                "Smooth waypoint length does not match number of joints; stopping node."
            )
            return

        if self._smooth_path is None:
            self._smooth_path = _SmoothViaPath(
                self._current_joints,
                q_via,
                q_final,
                self._max_joint_speed,
                step.duration_sec,
            )
            self._smooth_path_u = 0.0
            self.get_logger().info(
                f"Starting smooth-waypoints step {self._current_step_index + 1} "
                f"(path length={self._smooth_path.path_length:.2f} rad, "
                f"via u={self._smooth_path.via_u:.2f})"
            )

        assert self._smooth_path is not None

        self._advance_smooth_path_progress()

        u_target = min(
            self._smooth_path_u + self._smooth_path_lookahead_u, 1.0
        )
        q_target = self._smooth_path.position_at_u(u_target)

        errors: List[float] = []
        max_err = 0.0
        for current, goal in zip(self._current_joints, q_target):
            e = _joint_angle_error(current, goal)
            errors.append(e)
            max_err = max(max_err, abs(e))

        velocities = self._errors_to_velocities(errors, max_err)
        self._publish_joint_command(velocities)

        final_errors = [
            abs(_joint_angle_error(c, g))
            for c, g in zip(self._current_joints, self._smooth_path.q_final)
        ]
        at_final = max(final_errors) < self._joint_tolerance
        path_done = self._smooth_path_u >= 1.0 - 1e-3

        if path_done and at_final:
            self.get_logger().info(
                f"Completed smooth-waypoints step {self._current_step_index + 1}/{len(self._steps)}"
            )
            self._clear_smooth_path()
            if step.wait_sec > 0.0:
                self._waiting_until_sec = now_sec + step.wait_sec
                self._advance_after_wait = True
                self._publish_joint_command([0.0] * len(self._joint_names))
                return
            self._current_step_index += 1

    def _reset_noise(self) -> None:
        print("Resetting velocity noise...")
        self.noise_counter = self.noise_counter_max
        self.additive_noise = [random.uniform(-1, 1) for _ in self._joint_names]

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
            self._abort_with_error(
                "Recorder start client not initialized; stopping node."
            )
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
        if self.recorder_stopping:
            return

        self.recorder_stopping = True

        if not self._record_enabled:
            self._current_step_index += 1
            return
        if self._recorder_param_client is not None:
            step = self._steps[self._current_step_index]
            if step.output_dir:
                if self._recorder_param_client.wait_for_service(timeout_sec=0.1):
                    param = Parameter(
                        "stop_output_dir", Parameter.Type.STRING, step.output_dir
                    )
                    request = SetParameters.Request()
                    request.parameters = [param.to_parameter_msg()]
                    self._recorder_param_client.call_async(request)
        self._request_recorder_stop()
        # self._current_step_index += 1

    def _handle_wait_step(self, step: Step) -> None:
        if step.wait_sec <= 0.0:
            self._current_step_index += 1
            return
        if self._waiting_until_sec is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9
            self._waiting_until_sec = now_sec + step.wait_sec
            self._advance_after_wait = True
        self._publish_joint_command([0.0] * len(self._joint_names))

    def _publish_joint_command(self, velocities: List[float]) -> None:
        raw_msg = self._build_joint_jog(velocities)
        self._joint_cmd_raw_pub.publish(raw_msg)

        noisy_velocities = self._apply_velocity_noise(velocities)
        noisy_msg = self._build_joint_jog(noisy_velocities)
        self._joint_cmd_pub.publish(noisy_msg)

    def _build_joint_jog(self, velocities: List[float]) -> JointJog:
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self._joint_names
        msg.velocities = velocities
        msg.displacements = []
        msg.duration = 0.0
        return msg

    def _apply_velocity_noise(self, velocities: List[float]) -> List[float]:
        if self._velocity_noise_std <= 0.0:
            return velocities
        noisy: List[float] = []
        #  print("Applying velocity noise:", self._velocity_noise_std)
        for v in velocities:
            # noisy_v = v + random.gauss(0.0, self._velocity_noise_std)
            noisy_v = v * (
                1.0
                + random.uniform(-self._velocity_noise_std, self._velocity_noise_std)
            )
            if abs(noisy_v) > self._max_joint_speed > 0.0:
                noisy_v = math.copysign(self._max_joint_speed, noisy_v)
            noisy.append(noisy_v)
        return noisy


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
