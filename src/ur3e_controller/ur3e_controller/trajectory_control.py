import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from std_msgs.msg import UInt8
from control_msgs.msg import JointJog
from ecpmi_gripper.srv import GripperControl
import random
import numpy as np


JointWaypoint = List[float]


@dataclass
class Step:
    kind: str  # "waypoint", "gripper", "recorder_start", "recorder_stop", "wait"
    waypoint: Optional[JointWaypoint] = None
    gripper_command: Optional[str] = None
    wait_sec: float = 0.0
    output_dir: Optional[str] = None


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
        self.declare_parameter("velocity_noise_std", 0.1) # rad/s

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
        self._recorder_stop_service = str(self.get_parameter("recorder_stop_service").value)
        self._record_enabled = bool(self.get_parameter("record").value)
        self._k_p_joint = float(self.get_parameter("k_p_joint").value)
        self._max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self._joint_tolerance = float(self.get_parameter("joint_tolerance").value)
        self._min_joint_speed = float(self.get_parameter("min_joint_speed").value)
        self._velocity_noise_std = float(
            self.get_parameter("velocity_noise_std").value
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
        self._recorder_start_timer = None
        self._recorder_stop_future: Optional[rclpy.task.Future] = None
        self._shutdown_deadline_sec: Optional[float] = None
        self._shutdown_timer = None

        # ROS interfaces
        self._joint_cmd_pub = self.create_publisher(JointJog, self._command_topic, 10)
        self._joint_cmd_raw_pub = self.create_publisher(
            JointJog, self._command_topic_raw, 10
        )
        self._gripper_state_pub = self.create_publisher(
            UInt8, self._gripper_state_topic,10
        )
        self._joint_state_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, 10
        )
        self._start_servo_client = self.create_client(Trigger, self._start_servo_service)
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
        
        random.seed(time.time())

        # Waypoints are specified in degrees and converted to radians.
        home_noise_deviations = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        
        home = [-90.00, 0.00, -90.00, 0.00, 90.00, -0.00]
        # home=[-3.28, -90.76 , 3.69 , -96.56, 2.77 , 92.44]
        home_l = [-90.00, 0.00, -90.00, 0.00, 90.00, 0.00]
        # home = [-90.00, 0.00, -90.00, 0.00, 90.00, 90.00]
        home_with_noise = [angle + random.uniform(-deviation, deviation) for angle, deviation in zip(home, home_noise_deviations)]
        home_with_noise_2 = [angle + random.uniform(-deviation, deviation) for angle, deviation in zip(home, home_noise_deviations)]
        
        grip_0 = {
            "gripping_prepare": [-113.23, -101.83, -144.09, -20.17, 90.70, -20.18],
            "gripping": [-126.14, -80.31, -153.01, -32.99, 91.05, -32.97],
            "gripping_pull_1": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_2": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_3": [-105.65, -113.10, -139.43, -9.46, 89.73, -9.45],
            
            # "gripping_prepare": [-65.48, 96.39, -31.91, 18.97, 88.70, 18.87],
            # "gripping": [-53.43, 76.38, -23.59, 30.72, 88.31, 30.63],
            # "gripping_pull_1": [-58.40, 84.93, -27.24, 26.52, 88.41, 26.42],
            # "gripping_pull_2": [-69.39, 102.38, -34.34, 13.88, 89.07, 13.79],
            # "gripping_pull_3": [-74.46, 109.72, -38.12, 6.46, 90.62, 6.35],
        }
        
        grip_1 = {
            "gripping_prepare": [-113.23, -101.83, -144.09, -20.17, 90.70, -20.18],
            "gripping": [-126.14, -80.31, -153.01, -32.99, 91.05, -32.97],
            "gripping_pull_1": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_2": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_3": [-105.65, -113.10, -139.43, -9.46, 89.73, -9.45],
            
            # "gripping_prepare": [-65.48, 96.39, -31.91, 18.97, 88.70, 18.87],
            # "gripping": [-53.43, 76.38, -23.59, 30.72, 88.31, 30.63],
            # "gripping_pull_1": [-58.40, 84.93, -27.24, 26.52, 88.41, 26.42],
            # "gripping_pull_2": [-69.39, 102.38, -34.34, 13.88, 89.07, 13.79],
            # "gripping_pull_3": [-74.46, 109.72, -38.12, 6.46, 90.62, 6.35],
        }
        
        grip_2 = {
            "gripping_prepare": [-113.23, -101.83, -144.09, -20.17, 90.70, -20.18],
            "gripping": [-126.14, -80.31, -153.01, -32.99, 91.05, -32.97],
            "gripping_pull_1": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_2": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_3": [-105.65, -113.10, -139.43, -9.46, 89.73, -9.45],
            
            # "gripping_prepare": [-65.48, 96.39, -31.91, 18.97, 88.70, 18.87],
            # "gripping": [-53.43, 76.38, -23.59, 30.72, 88.31, 30.63],
            # "gripping_pull_1": [-58.40, 84.93, -27.24, 26.52, 88.41, 26.42],
            # "gripping_pull_2": [-69.39, 102.38, -34.34, 13.88, 89.07, 13.79],
            # "gripping_pull_3": [-74.46, 109.72, -38.12, 6.46, 90.62, 6.35],
        }
        
        
        
        grip_6 = {
            "gripping_prepare": [-129.49, -40.22, -100.18, 90.12, 88.33, -69.73],
            "gripping": [-128.52, -54.32, -87.06, 90.12, 88.36, -69.73],
            
          #  "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-50.88, 42.11, -81.33, -89.56, 92.75, 77.77],
            "gripping_1": [-51.48, 54.90, -93.52, -89.57, 92.81, 77.80],
            
            "gripping_midpoint_2_1": [-93.57, -44.54, -105.36, 74.47, 90.77, -75.8],
            "gripping_midpoint_2_2": [-116.8, -44.59, -99.57, 86.34, 90.77, -74.29],
            "gripping_prepare_2": [-132.32, -35.39, -103.51, 92.75, 93.24, -71.05],
            "gripping_2": [-130.77, -49.93, -90.52, 92.75, 93.27, -71.05],
            
            "gripping_midpoint_3_1": [-109.38, -34.21, -120.06, 82.95, 90.07, -56.82],
           # "gripping_midpoint_3_2": [-116.8, -44.59, -99.57, 86.34, 90.77, -74.29],
            "gripping_prepare_3": [-132.66, -33.10, -105.75, 89.8, 89.98, -70.21],
            "gripping_3": [-130.07, -51.71, -88.30, 91.86, 75.15, -71.05],
            
            
            "gripping_midpoint_4_1": [-120.89, -19.36, -129.08, 91.0, 87.8, -79.74],
           # "gripping_midpoint_3_2": [-116.8, -44.59, -99.57, 86.34, 90.77, -74.29],
            "gripping_prepare_4": [-130.39, -39.77, -100.33, 89.23, 88.62, -69.64],
            "gripping_4": [-129.49, -53.34, -87.66, 89.24, 88.65, -69.65],
            
            
            "gripping_pull_1": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_2": [-116.63, -96.40, -146.23, -24.18, 90.86, -24.18],
            "gripping_pull_3": [-105.65, -113.10, -139.43, -9.46, 89.73, -9.45],
            
            # "gripping_prepare": [-65.48, 96.39, -31.91, 18.97, 88.70, 18.87],
            # "gripping": [-53.43, 76.38, -23.59, 30.72, 88.31, 30.63],
            # "gripping_pull_1": [-58.40, 84.93, -27.24, 26.52, 88.41, 26.42],
            # "gripping_pull_2": [-69.39, 102.38, -34.34, 13.88, 89.07, 13.79],
            # "gripping_pull_3": [-74.46, 109.72, -38.12, 6.46, 90.62, 6.35],
        }
        
        
        grip_7 = {
            
            "gripping_prepare": [-127.31, -44.27, -98.32, 90.11, 97.67, -77.82],
            "gripping": [-126.77, -57.25, -85.87, 90.12, 97.69, -77.83],
            
           # "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-53.08, 43.72, -80.74, -89.56, 87.02, 69.08],
            "gripping_1": [-53.73, 57.57, -93.94, -89.57, 87.08, 69.11],
        }
        
       
       
        
        grip_8 = {
            
            "gripping_prepare": [-107.25, -77.34, -85.32, 90.01, 70.47, -75.42],
            "gripping": [-108.66, -85.41, -75.83, 90.02, 70.47, -75.44],
            
           # "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-72.48, 74.21, -91.82, -89.61, 77.11, 65.42],
            "gripping_1": [-71.01, 84.58, -103.67, -89.62, 77.16, 65.44],
        }
        
        
        grip_9 = {
            
            "gripping_prepare": [-105.92, -75.54, -88.44, 90.01, 89.44, -64.84],
            "gripping": [-107.54, -86.85, -75.50, 90.02, 89.45, -64.86],
            
           # "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-74.02, 77.06, -93.12, -89.61, 92.68, 75.54],
            "gripping_1": [-72.48, 86.67, -104.30, -89.62, 92.73, 75.56],
        }
             
             
        
        grip_10 = {
            
            "gripping_prepare": [-118.54, -56.94, -94.41, 90.06, 91.07, -67.20],
            "gripping": [-119.03, -69.78, -81.08, 90.08, 91.09, -67.21],
            
           # "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-61.72, 58.45, -86.82, -89.58, 90.61, 77.31],
            "gripping_1": [-61.13, 69.82, -98.79, -89.59, 90.66, 77.33],
        }
      
      
        grip_11 = {
            
            "gripping_prepare": [-117.54, -58.86, -93.49, 90.06, 88.037, -77.83],
            "gripping": [-118.23, -71.39, -80.28, 90.08, 88.38, -77.84],
            
           # "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-62.86, 60.37, -87.61, -89.58, 88.76, 66.7],
            "gripping_1": [-62.07, 71.79, -99.82, -89.59, 88.82, 66.71],
        }      
      
 
        grip_12 = {
            
            "gripping_prepare": [-108.49, -73.86, -87.56, 90.06, 91.45, -68.61],
            "gripping": [-109.92, -83.64, -76.34, 90.03, 91.46, -68.63],
            
           # "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-71.73, 73.15, -91.52, -89.61, 89.69, 73.03],
            "gripping_1": [-70.23, 83.84, -103.71, -89.62, 89.74, 73.05],
        }      
      
 
        grip_13 = {
            
            "gripping_prepare": [-117.64, -59.31, -92.95, 90.06, 90.30, -72.54],
            "gripping": [-118.33, -71.19, -80.37, 90.07, 90.31, -72.55],
            
           # "gripping_midpoint_1": [-79.36, 41.53, -76.39, -85.07, 47.32, 53.14],
            "gripping_prepare_1": [-62.77, 60.61, -87.94, -89.58, 88.66, 72.12],
            "gripping_1": [-62.06, 71.24, -99.29, -89.59, 88.71, 72.15],
        }
        
        # grip_14 = {
        #     "gripping_prepare": [-129.15, -40.80, -99.95, 90.12, 93.58, -73.66],
        #     "gripping": [-128.25, -55.03, -86.61, 90.12, 93.61, -73.66],
            
        #     "gripping_prepare_1": [-49.35, 32.96, -73.71, -89.56, 91.81, 72.90],
        #     "gripping_1": [-51.88, 54.19, -92.42, -89.57, 91.89, 72.94],
        # }
          
        grip_15 = {
            "gripping_prepare": [-124.01, -42.06, -103.83, 90.08, 84.71, -67.05],
            "gripping": [-122.68, -64.41, -82.81, 90.09, 84.74, -67.06],
            
            "gripping_prepare_1": [-57.30, 48.24, -81.05, -89.57, 91.97, 78.46],
            "gripping_1": [-57.51, 64.81, -97.40, -89.58, 92.04, 78.50],
        }
        
        grip_16 = {
            "gripping_prepare": [-109.33, -64.95, -95.61, 90.01, 82.37, -62.37],
            "gripping": [-110.74, -83.66, -75.49, 90.03, 82.38, -62.40],
            
            "gripping_prepare_1": [-71.02, 66.09, -85.17, -89.60, 92.93, 79.19],
            "gripping_1": [-69.84, 82.69, -102.95, -89.61, 93.01, 79.22],
        }
        
        grip_17 = {
            "gripping_prepare": [-109.83, -64.36, -95.71, 90.01, 87.60, -78.69],
            "gripping": [-110.87, -81.88, -77.14, 90.03, 87.61, -78.72],
            
            "gripping_prepare_1": [-70.81, 67.16, -86.45, -89.60, 90.92, 62.79],
            "gripping_1": [-69.58, 82.22, -102.74, -89.61, 90.99, 62.82],
        }

        #gripper_choice = random.choice([grip_0, grip_1, grip_2])
        gripper_choice = random.choice([grip_13])
        # print("Selected gripper trajectory:")
        # for key, value in gripper_choice.items():
        #     print(f"  {key}: {value}")
        
        # gripper_choice = grip_0
        
        gripping_prepare = gripper_choice["gripping_prepare"]
        gripping = gripper_choice["gripping"]
        
        # gripping_prepare = gripper_choice["gripping_prepare_1"]
        # gripping = gripper_choice["gripping_1"]
        
   #     gripping_pull_1 = gripper_choice["gripping_pull_1"]
   #     gripping_pull_2 = gripper_choice["gripping_pull_2"]
    #    gripping_pull_3 = gripper_choice["gripping_pull_3"]
        
        reset_pose = grip_0["gripping_pull_3"]
        #reset_pose[5] = 4.0
        reset_pose[5] = 0.0
        ## Pick and place left drawer
        
        lift = [-107.66, -23.45, -139.67, 90.22, 0.21, -64.52]
        lift = [angle + random.uniform(-2, 2) for angle in lift]
        
        hover_over_left = [-123.76, -23.52, -105.79, 90.23, 0.21, -47.43]
        hover_over_left = [angle + random.uniform(-2, 2) for angle in hover_over_left]
        
        put_in_left = [-133.61, -26.93, -108.49, 90.23, 0.21, -47.44]
        
        ## Pick and place right drawer
        
        hover_over_right = [-119.74, -23.53, -110.53, 86.06, 5.38, -100.57]
        hover_over_right = [angle + random.uniform(-2, 2) for angle in hover_over_right]
        
        put_in_right = [-133.95, -23.62, -110.28, 86.05, 5.38, -100.57]
        
        # pick_prepare = [-108.99, -67.52, -81.50, 87.93, -0.11, -71.75] # Middle
        # pick_prepare = [-107.11, -75.61, -86.48, 89.74, 9.77, -63.34] # Left
        pick_prepare = [-106.42, -74.75, -87.63, 89.64, -5.45, -78.59] # Right
        
        # pick = [-120.08, -68.45, -81.19, 84.13, 3.18, -69.55]  # Middle
        # pick = [-108.64, -85.47, -75.10, 89.75, 9.77, -63.36] # Left
        pick = [-108.02, -86.21, -74.57, 89.66, -5.45, -78.61] # Right
        
        def to_rad(waypoint_deg: JointWaypoint) -> JointWaypoint:
            return [math.radians(angle_deg) for angle_deg in waypoint_deg]

        # return [
        #    Step(kind="waypoint", waypoint=to_rad(home)),
        # ]
        
        # drop = [-117.81, -51.64, -88.21, 87.26, 80.07, -44.76]
        drop = [-55.11, 31.30, -74.43, -83.17, 89.97, 42.98]
        drop_noise_deviations = [0, 0, 0, 2, 4, 3]
        drop = [angle + random.uniform(-deviation, deviation) for angle, deviation in zip(drop, drop_noise_deviations)]

        # Left 1            
        # grip = [-128.98, -74.22, -158.44, -31.19, 90.0, -32.55]
        # w1 = [-123.18, -84.32, -154.35, -26.44, 90.0, -27.81]
        # w2 = [-117.50, -93.28, -151.13, -20.86, 90.65, -22.18]
        # w3 = [-110.88, -103.58, -148.70, -11.79, 91.94, -13.12]
        # w4 = [-108.18, -107.64, -150.28, -5.93, 94.92, -7.26]
        
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
        grip = [-50.55, 72.98, -26.68, 28.42, 88.29, 28.76]
        w1 = [-54.82, 80.52, -30.49, 24.98, 88.87, 25.32]
        w2 = [-60.98, 90.99, -36.22, 19.02, 90.36, 19.36]
        w3 = [-68.37, 102.99, -45.49, 10.74, 95.10, 11.03]
        w4 = [-72.21, 109.41, -54.23, 6.86, 101.31, 7.04]
        
        # return [
        #     Step(kind="waypoint", waypoint=to_rad(w1)),
        # ]
            
        return [
            Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            
            Step(kind="recorder_start"),
            Step(kind="wait", wait_sec=1.0),
            
            Step(kind="waypoint", waypoint=to_rad(w1)),
            
            Step(kind="waypoint", waypoint=to_rad(grip)),
            Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
            Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            
            Step(kind="waypoint", waypoint=to_rad(w1)),
            Step(kind="waypoint", waypoint=to_rad(w2)),
            Step(kind="waypoint", waypoint=to_rad(w3)),
            Step(kind="waypoint", waypoint=to_rad(w4)),
            
            Step(kind="gripper", gripper_command="blow", wait_sec=1.0),
            
            Step(kind="waypoint", waypoint=to_rad(home)),
            
            Step(kind="recorder_stop", output_dir="/home/shokry/ur3e-trajectories/open_right/open_right"),
        ]
            
        return [
            Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            
            Step(kind="recorder_start"),
            Step(kind="wait", wait_sec=1.0),
            
            Step(kind="waypoint", waypoint=to_rad(drop)),
            
            Step(kind="gripper", gripper_command="blow", wait_sec=0.5),
            
            Step(kind="recorder_stop", output_dir="/home/shokry/ur3e-trajectories/place_right2/place_right"),
            
            Step(kind="waypoint", waypoint=to_rad(home)),
        ]
        
        return [
            Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            
            Step(kind="recorder_start"),
            Step(kind="wait", wait_sec=1.0),

            Step(kind="waypoint", waypoint=to_rad(gripping_prepare)),

            Step(kind="waypoint", waypoint=to_rad(gripping)),


            Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
            Step(kind="gripper", gripper_command="release", wait_sec=0.1),

            Step(kind="waypoint", waypoint=to_rad(home)),
            
            Step(kind="recorder_stop", output_dir="/home/shokry/ur3e-trajectories/pick13/pick")
        ]

        # return [
            
        #     ### Drawer open
            
        #     # Step(kind="waypoint", waypoint=to_rad(home)),
            
        #     Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            
        #     Step(kind="recorder_start"),
            
        #     Step(kind="wait", wait_sec=1.0),
            
        #     Step(kind="waypoint", waypoint=to_rad(gripping_prepare)),
        #     Step(kind="waypoint", waypoint=to_rad(gripping)),
            
        #     Step(kind="gripper", gripper_command="grip", wait_sec=1.0),
        #     Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            
        #     # Step(kind="waypoint", waypoint=to_rad(gripping_pull_1)),
        #     Step(kind="waypoint", waypoint=to_rad(gripping_pull_2)),
        #     Step(kind="waypoint", waypoint=to_rad(gripping_pull_3)),
            
        #     Step(kind="gripper", gripper_command="blow", wait_sec=0.1),
            
        #     Step(kind="waypoint", waypoint=to_rad(home_l)),
        #     Step(kind="waypoint", waypoint=to_rad(home)),
            
        #     Step(kind="wait", wait_sec=0.5),

        #     Step(kind="recorder_stop", output_dir="/home/siddiquieu1/ur3e-trajectories/open_drawer_left/open_drawer_left"),
            
        #     Step(kind="wait", wait_sec=5.0),
            
        #     # Step(kind="waypoint", waypoint=to_rad(reset_pose)),
        #     # Step(kind="waypoint", waypoint=to_rad(grip_0["gripping"])),
            
        #     ### Drawer pick and place
            
        #     # Step(kind="waypoint", waypoint=to_rad(home_with_noise)),
            
        #     # Step(kind="recorder_start"),
            
        #     # # Step(kind="waypoint", waypoint=to_rad(pick_prepare)),
            
        #     # Step(kind="waypoint", waypoint=to_rad(pick)),
            
        #     # Step(kind="gripper", gripper_command="grip", wait_sec=0.5),
        #     # Step(kind="gripper", gripper_command="release", wait_sec=0.1),
            
        #     # Step(kind="waypoint", waypoint=to_rad(lift)),
            
        #     # Step(kind="recorder_stop", output_dir="/home/siddiquieu1/ur3e-trajectories/pick/pick"),
            
        #     # Step(kind="wait", wait_sec=2.0),
            
        #     # Step(kind="recorder_start"),
            
        #     # # Step(kind="waypoint", waypoint=to_rad(hover_over_left)),
        #     # # Step(kind="waypoint", waypoint=to_rad(put_in_left)),
        #     # Step(kind="waypoint", waypoint=to_rad(hover_over_right)),
        #     # Step(kind="waypoint", waypoint=to_rad(put_in_right)),
            
        #     # Step(kind="gripper", gripper_command="blow", wait_sec=0.5),
            
        #     # Step(kind="waypoint", waypoint=to_rad(home)),
            
        #     # Step(kind="recorder_stop", output_dir="/home/siddiquieu1/ur3e-trajectories/place_right/place_right"),
        # ] * 1

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
        if step.kind == "wait":
            print("-"*20)
            print(f"Waiting for {step.wait_sec} seconds...")
            print("-"*20)
            self._handle_wait_step(step)
            return
        if step.kind == "gripper":
            self._handle_gripper_step(step)
            return

        if step.kind != "waypoint" or step.waypoint is None:
            self._abort_with_error("Invalid step configuration; stopping node.")
            return
        
        self._set_gripper_state(self._gripper_state)

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
        print("Current joints:", [math.degrees(j) for j in self._current_joints])
        print("Target waypoint:", [math.degrees(t) for t in target])
        for current, goal in zip(self._current_joints, target):
            e = math.atan2(math.sin(goal - current), math.cos(goal - current))
            errors.append(e)
            max_err = max(max_err, abs(e))

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
        velocities: List[float] = []
        for e in errors:
            normalized_speed = (abs(e) / max_err) * self._max_joint_speed if max_err > 0 else 0.0
            v = math.copysign(normalized_speed, e)

            # Clamp each joint speed
            if abs(v) > self._max_joint_speed > 0.0:
                v = math.copysign(self._max_joint_speed, v)
            velocities.append(v)

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
        self._current_step_index += 1

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
            noisy_v = v * (1.0 + random.uniform(-self._velocity_noise_std, self._velocity_noise_std))
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
