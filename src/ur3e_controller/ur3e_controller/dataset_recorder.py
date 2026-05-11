import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import UInt8
from std_srvs.srv import Trigger
from control_msgs.msg import JointJog


@dataclass
class LatestMsg:
    stamp_sec: float
    msg: object


class DatasetRecorder(Node):
    def __init__(self) -> None:
        super().__init__("dataset_recorder")

        self.declare_parameter("sample_rate_hz", 15.0)
        self.declare_parameter("sync_tolerance_sec", 0.3)
        self.declare_parameter("output_dir", "/home/siddiquieu1/ur3e-trajectories")
        self.declare_parameter("stop_output_dir", "")
        self.declare_parameter("joint_names", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("joint_command_topic", "/servo_node/delta_joint_cmds_raw")
        self.declare_parameter("gripper_state_topic", "/gripper_state")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("start_service", "/dataset_recorder/start")
        self.declare_parameter("stop_service", "/dataset_recorder/stop")

        self._sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self._sync_tolerance_sec = float(
            self.get_parameter("sync_tolerance_sec").value
        )
        self._output_dir = str(self.get_parameter("output_dir").value)
        self._stop_output_dir = str(self.get_parameter("stop_output_dir").value)
        self._joint_names = list(self.get_parameter("joint_names").value)
        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._joint_command_topic = str(
            self.get_parameter("joint_command_topic").value
        )
        self._gripper_state_topic = str(
            self.get_parameter("gripper_state_topic").value
        )
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._start_service = str(self.get_parameter("start_service").value)
        self._stop_service = str(self.get_parameter("stop_service").value)

        self._latest_joint_state: Optional[LatestMsg] = None
        self._latest_joint_cmd: Optional[LatestMsg] = None
        self._latest_gripper_state: Optional[LatestMsg] = None
        self._latest_depth: Optional[LatestMsg] = None
        self._name_to_index: Optional[Dict[str, int]] = None

        self._recording = False
        self._session_dir: Optional[str] = None
        self._frame_index = 0
        self._observations: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._timestamps: List[float] = []
        self._depth_frames: List[np.ndarray] = []

        self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_state, 10
        )
        self.create_subscription(
            JointJog, self._joint_command_topic, self._on_joint_command, 10
        )
        self.create_subscription(
            UInt8, self._gripper_state_topic, self._on_gripper_state, 10
        )
        self.create_subscription(
            Image, self._depth_topic, self._on_depth, 10
        )

        self._start_srv = self.create_service(Trigger, self._start_service, self._on_start)
        self._stop_srv = self.create_service(Trigger, self._stop_service, self._on_stop)

        period = 1.0 / self._sample_rate_hz if self._sample_rate_hz > 0 else 0.0333
        self._timer = self.create_timer(period, self._sample)

        self.get_logger().info("DatasetRecorder ready. Recording at {:.2f} Hz with sync tolerance of {:.2f} sec.".format(
            self._sample_rate_hz, self._sync_tolerance_sec
        ))

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        if self._name_to_index is None:
            self._name_to_index = {name: i for i, name in enumerate(msg.name)}
        self._latest_joint_state = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)

    def _on_joint_command(self, msg: JointJog) -> None:
        self._latest_joint_cmd = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)

    def _on_gripper_state(self, msg: UInt8) -> None:
        
        print("Gripper State", msg)
        
        self._latest_gripper_state = LatestMsg(self._now_sec(), msg)

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)

    def _on_start(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._recording:
            response.success = True
            response.message = "Already recording."
            return response

        # index_in_folder = 0
        # while True:
        #     session_dir = os.path.join(self._output_dir, f"{self._stop_output_dir}_{index_in_folder:04d}")
        #     if not os.path.exists(session_dir):
        #         break
        #     index_in_folder += 1
        
        # self._session_dir = os.path.join(self._output_dir, f"{self._stop_output_dir}_{index_in_folder:04d}")
        # os.makedirs(self._session_dir, exist_ok=True)
        self._frame_index = 0
        self._observations.clear()
        self._actions.clear()
        self._timestamps.clear()
        self._depth_frames.clear()
        self._recording = True

        response.success = True
        response.message = f"Recording..."
        self.get_logger().info(response.message)
        return response

    def _on_stop(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._recording:
            response.success = True
            response.message = "Not recording."
            return response

        self._recording = False
        # if self._session_dir is None:
        #     response.success = False
        #     response.message = "No active session directory."
        #     return response

        obs = np.stack(self._observations) if self._observations else np.zeros((0, 9))
        act = np.stack(self._actions) if self._actions else np.zeros((0, 9))
        ts = np.asarray(self._timestamps, dtype=np.float64)
        depth_frames = (
            np.stack(self._depth_frames)
            if self._depth_frames
            else np.zeros((0, 0, 0))
        )

        stop_output_dir = str(self.get_parameter("stop_output_dir").value).strip()
        
        index_in_folder = 0
        while True:
            session_dir = os.path.join(self._output_dir, f"{stop_output_dir}_{index_in_folder:04d}")
            if not os.path.exists(session_dir):
                break
            index_in_folder += 1
        
        self._session_dir = os.path.join(self._output_dir, f"{stop_output_dir}_{index_in_folder:04d}")
        os.makedirs(self._session_dir, exist_ok=True)
    
        output_path = os.path.join(self._session_dir, "dataset.npz")
    
        np.savez(
            output_path,
            observations=obs,
            actions=act,
            timestamps=ts,
            depth_frames=depth_frames,
        )
        
        self._depth_frames.clear()  # Clear depth frames from memory after saving
        self._observations.clear()
        self._actions.clear()
        self._timestamps.clear()

        response.success = True
        response.message = f"Saved dataset to {output_path}"
        self.get_logger().info(response.message)
        return response

    def _sample(self) -> None:
        # print("Sampling dataset...")
        
        if not self._recording:
            # print("Not recording, skipping sample.")
            return

        now_sec = self._now_sec()
        if not self._inputs_ready(now_sec):
            print("Inputs not ready or not synchronized, skipping sample.")
            return

        joint_state = self._latest_joint_state.msg
        joint_cmd = self._latest_joint_cmd.msg
        gripper_state = self._latest_gripper_state.msg
        depth = self._latest_depth.msg

        observation = self._build_observation(joint_state, gripper_state)
        action = self._build_action(joint_cmd, gripper_state)

        depth_array = self._image_to_array(depth)
        if depth_array is None:
            return
        
        # print(f"Recorded frame {self._frame_index:06d} at time {now_sec:.3f} sec")
        print(f"  Joint positions: {observation[:6]}")

        self._observations.append(observation)
        self._actions.append(action)
        self._timestamps.append(now_sec)
        self._depth_frames.append(depth_array)

    def _inputs_ready(self, now_sec: float) -> bool:
        if (
            self._latest_joint_state is None
            or self._latest_joint_cmd is None
            or self._latest_gripper_state is None
            or self._latest_depth is None
            or self._name_to_index is None
        ):
            print("One or more inputs are not yet available.")
            print(f"  Latest joint state: {self._latest_joint_state is not None}")
            print(f"  Latest joint command: {self._latest_joint_cmd is not None}")
            print(f"  Latest gripper state: {self._latest_gripper_state is not None}")
            print(f"  Latest depth: {self._latest_depth is not None}")
            print(f"  Name to index mapping: {self._name_to_index is not None}")
            return False

        for latest in (
            self._latest_joint_state,
            self._latest_joint_cmd,
            # self._latest_gripper_state,
            self._latest_depth,
        ):
            if abs(now_sec - latest.stamp_sec) > self._sync_tolerance_sec:
                print(f"Latest {type(latest.msg).__name__} timestamp difference: {abs(now_sec - latest.stamp_sec):.3f} sec")
                return False
        return True

    def _build_observation(self, joint_state: JointState, gripper_state: UInt8) -> np.ndarray:
        joints = self._extract_joint_vector(joint_state)
        gripper_onehot = self._gripper_onehot(gripper_state.data)
        return np.concatenate([joints, gripper_onehot])

    def _build_action(self, joint_cmd: JointJog, gripper_state: UInt8) -> np.ndarray:
        velocities = self._extract_velocity_vector(joint_cmd)
        gripper_onehot = self._gripper_onehot(gripper_state.data)
        return np.concatenate([velocities, gripper_onehot])

    def _extract_joint_vector(self, joint_state: JointState) -> np.ndarray:
        positions = np.zeros(len(self._joint_names), dtype=np.float32)
        for i, name in enumerate(self._joint_names):
            idx = self._name_to_index.get(name)
            if idx is None or idx >= len(joint_state.position):
                continue
            positions[i] = joint_state.position[idx]
        return positions

    def _extract_velocity_vector(self, joint_cmd: JointJog) -> np.ndarray:
        velocities = np.zeros(len(self._joint_names), dtype=np.float32)
        name_to_index: Dict[str, int] = {name: i for i, name in enumerate(joint_cmd.joint_names)}
        for i, name in enumerate(self._joint_names):
            idx = name_to_index.get(name)
            if idx is None or idx >= len(joint_cmd.velocities):
                continue
            velocities[i] = joint_cmd.velocities[idx]
        return velocities

    def _gripper_onehot(self, state: int) -> np.ndarray:
        onehot = np.zeros(3, dtype=np.float32)
        if state == 1:
            onehot[0] = 1.0
        elif state == 2:
            onehot[1] = 1.0
        elif state == 3:
            onehot[2] = 1.0
        return onehot

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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
