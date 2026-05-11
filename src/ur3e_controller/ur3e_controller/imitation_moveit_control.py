import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger, SetBool
from control_msgs.msg import JointJog
from ecpmi_gripper.srv import GripperControl
from concurrent.futures import Future, ThreadPoolExecutor
import torch
import torch.nn as nn
import math
from einops import rearrange
from pymoveit2 import MoveIt2
import threading
from collections import deque

np.set_printoptions(suppress=True)
torch.set_printoptions(precision=4, sci_mode=False)

class Flatten(nn.Module):
    r"""Copied from torch 1.9."""
    __constants__ = ["start_dim", "end_dim"]
    start_dim: int
    end_dim: int

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return "start_dim={}, end_dim={}".format(self.start_dim, self.end_dim)


class SimpleCNN(nn.ModuleList):
    def __init__(self, in_channels, input_shape, out_channels) -> None:
        super().__init__()

        self.extend(
            [
                nn.Conv2d(in_channels, 32, 8, stride=4),
                nn.ReLU(True),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(True),
                nn.Conv2d(64, 32, 3, stride=1),
                Flatten(),
            ]
        )

        # Infer the final output resolution
        with torch.no_grad():
            x = torch.zeros(1, in_channels, *input_shape)
            dim = self.forward(x).size(-1)
        self.extend([nn.Linear(dim, out_channels), nn.ReLU(True)])

        self.reset_parameters()

    def forward(self, x):
        for m in self:
            x = m(x)
        return x

    def reset_parameters(self):
        for m in self:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, nn.init.calculate_gain("relu")
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)       # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                      # (d_model/2)

        pe[:, 0::2] = torch.sin(position * div_term)           # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)           # Apply cos to odd indices
        pe = pe.unsqueeze(0)                                   # Shape: (1, max_len, d_model)

        self.register_buffer('pe', pe)  # Not a parameter, but saved with the model

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model) with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :].to(x.device)

# Sinusoidal Timestep Embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / half_dim))
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# Cross-Attention Block
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        b, n_decoder, _ = x.shape
        b_context, n_context, _ = context.shape
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b n_decoder (h d) -> b h n_decoder d', h=h)
        k = rearrange(k, 'b n_context (h d) -> b h n_context d', h=h)
        v = rearrange(v, 'b n_context (h d) -> b h n_context d', h=h)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn_scores.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n_decoder d -> b n_decoder (h d)')
        return self.to_out(out)

# Transformer Block with Cross Attention
class DiffusionTransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim, heads=8, dim_head=128):
        super().__init__()
        self.attn = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True,dim_feedforward=256)
     #   self.atten_cond = nn.TransformerEncoderLayer(d_model=cond_dim, nhead=heads, batch_first=True,dim_feedforward=256)
        self.cross_attn = CrossAttention(dim, cond_dim, heads, dim_head)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, cond):
        x = self.attn(x)
      #  cond = self.atten_cond(cond)
        x = self.norm(x + self.cross_attn(x, cond))
        return x

# Conditional Diffusion Model
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, action_dim=7, output_dim=10,sensor_dim=7,depth_features_dim=512, hidden_dim=256, num_layers=2):
        super().__init__()

        #self.visual_feature_extractor = Feat_ext
        self.action_input_proj = nn.Linear(action_dim , hidden_dim)
        self.depth_projection= nn.Linear(depth_features_dim, hidden_dim)

        self.non_visual_obs_projection= nn.Linear(sensor_dim, hidden_dim)

        self.depth_encoder = SimpleCNN(1, (120,300), depth_features_dim)


        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),

        )
      #  self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        self.transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear( hidden_dim,action_dim )
        
        self.decoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=21)  # Action position embedding
        self.encoder_position_embedding=SinusoidalPositionalEncoding(hidden_dim, max_len=16)  # Sensor position embedding


    def forward(self, depth_images , non_visual_obs, noisy_action, t):        
        batch_size=non_visual_obs.shape[0]
        context_length=non_visual_obs.shape[1]

        noisy_action=self.action_input_proj(noisy_action.to(device=depth_images.device, dtype=torch.float32))
        
        depth_images=depth_images.reshape(batch_size*context_length, depth_images.shape[-3], depth_images.shape[-2], depth_images.shape[-1])

        depth_features=self.depth_encoder(depth_images.to(torch.float32))
        depth_features=depth_features.reshape(batch_size, context_length, -1)
       
        depth_features=self.depth_projection(depth_features.to(torch.float32))

        non_visual_obs=self.non_visual_obs_projection(non_visual_obs.to(torch.float32))

        t=self.time_mlp(t.to(torch.float32))  # Time embedding
        
        t = t.unsqueeze(1).repeat(non_visual_obs.shape[0], 1, 1)  # Shape: (batch_size, 1, hidden_dim)

        encoder_input=torch.zeros((batch_size, (context_length*2)+1, non_visual_obs.shape[-1])).to(torch.float32).to(non_visual_obs.device)

        encoder_input[:,0:-2:2,:]=depth_features
        encoder_input[:,1:-1:2,:]=non_visual_obs
        encoder_input[:, -1, :]=t.squeeze(1)     

        encoder_input=self.encoder_position_embedding(encoder_input)  # Apply sensor position embedding

        decoder_input=torch.cat((t, noisy_action), dim=1)
        
        decoder_input = self.decoder_position_embedding(decoder_input)  # Apply action position embedding

        for block in self.transformer_blocks:
            decoder_input = block(decoder_input, encoder_input)
        out=self.output_proj(decoder_input)
        out=out[:,1:,:]

        return out
    
@dataclass
class LatestMsg:
    stamp_sec: float
    msg: object
    
@dataclass
class Observation:
    joints: np.ndarray
    depth: np.ndarray
    
    
class ImitationMoveitControl(Node):
    
    def __init__(self):
        super().__init__('imitation_moveit_control')
        
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
        
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("obs_window", 5)
        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("joint_names", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])
        self.declare_parameter("gripper_state_service", "/set_observation_gripper_state")
        self.declare_parameter("gripper_service", "/gripper_control")
        
        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._obs_window = int(self.get_parameter("obs_window").value)
        self._control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._joint_names = [str(name) for name in self.get_parameter("joint_names").value]
        self._latest_joint_state: Optional[LatestMsg] = None
        self._latest_depth: Optional[LatestMsg] = None
        self._name_to_index: Optional[Dict[str, int]] = None
        self._obs_queue: Deque[Observation] = deque(maxlen=self._obs_window)
        self._inference_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._gripper_state_service = str(
            self.get_parameter("gripper_state_service").value
        )
        self._gripper_service = str(self.get_parameter("gripper_service").value)
        
        self.joint_observations = deque(maxlen=1000)
        self.depth_observations = deque(maxlen=30)
        
        self.motion_start_time = None
        self.motion_end_time = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
        
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self._joint_names,
            base_link_name="base_link",
            end_effector_name="tool0",
            group_name="ur_manipulator",
            use_move_group_action=True,
        )
        
        print("Initializing Robot Framework...")
        time.sleep(2.0)
        
        def to_rad(waypoint_deg):
            return [math.radians(angle_deg) for angle_deg in waypoint_deg]
        
        home = to_rad([-0.00, -90.00, 0.00, -90.00, 0.00, 90.00])
        
        self.moveit2.move_to_configuration(home, self._joint_names, tolerance=0.005)
        self.moveit2.wait_until_executed()
        
        print("Robot Framework initialized. Subscribing to topics and starting control loop...")
        
        self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_state, 10
        )
        self.create_subscription(Image, self._depth_topic, self._on_depth, 10)
        
        self._gripper_client = self.create_client(GripperControl, self._gripper_service)
        
        print(f"Subscribed to {self._joint_states_topic} and {self._depth_topic}")
        
        self._gripper_state_srv = self.create_service(
            SetBool, self._gripper_state_service, self._set_observation_gripper_state
        )
        
        period = 1.0 / self._control_rate_hz if self._control_rate_hz > 0 else 0.0667
        self._control_timer = self.create_timer(period, self._control_step)
        
        self.executing_actions = False
        self._gripper_sequence_active = False
        self._next_checkpoint: List[float] | None = None
        self._prev_gripper_state: bool | None = None
        self._next_gripper_state: bool | None = None
        self._gripper_release_timer = None
        
        # Initial gripper state, set to 1 for placing models
        self._obs_gripper_state = 0.0

        
    def load_model(self):
        self.flow_matching_policy=ConditionalDiffusionModel()
        
        # check_point=torch.load('/home/shokry/ur3e-trajectories/weights/flow_matching_real_world_data_pick3_processed_without_data_augmentation_epoch_320.pt', map_location=self.device)
        # check_point=torch.load('/home/shokry/ur3e-trajectories/weights/flow_matching_real_world_data_pick3_with_incermental_joint_actions_processed_with_data_augmentation_epoch_3000.pt', map_location=self.device)
        # check_point=torch.load('/home/shokry/ur3e-trajectories/weights/weights_5_may/flow_matching_real_world_data_all_pick_data_with_incermental_joint_actions_processed_without_data_augmentation_epoch_6400.pt', map_location=self.device)        
        # check_point=torch.load("/home/shokry/ur3e-trajectories/weights/flow_matching_real_world_data_open_left_data_with_incermental_joint_actions_processed_without_data_augmentation_epoch_9950.pt", map_location=self.device)
        
        check_point=torch.load("/home/shokry/ur3e-trajectories/weights/flow_matching_real_world_data_all_skills_with_incermental_joint_actions_processed_without_data_augmentation_epoch_4000.pt", map_location=self.device)
        
        print("Loading flow matching policy from checkpoint")
        
        self.flow_matching_policy.load_state_dict(check_point['model'])
        self.flow_matching_policy.to(self.device)
        self.flow_matching_policy.eval()
        for p in self.flow_matching_policy.parameters():
            p.requires_grad_(False)
        
        
    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _extract_joint_vector(self, joint_state: JointState) -> np.ndarray:
        positions = np.zeros(len(self._joint_names), dtype=np.float32)
        for i, name in enumerate(self._joint_names):
            idx = self._name_to_index.get(name)
            if idx is None or idx >= len(joint_state.position):
                continue
            positions[i] = joint_state.position[idx]
        return positions
    
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

    def _on_joint_state(self, msg: JointState) -> None:
        if not msg.name or not msg.position:
            return
        if self._name_to_index is None:
            self._name_to_index = {name: i for i, name in enumerate(msg.name)}
        self._latest_joint_state = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)
        self.joint_observations.append((self._latest_joint_state.stamp_sec, self._extract_joint_vector(msg)))
       # print("---- Latest joint observation: ", self.joint_observations[-1])

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)
        self.depth_observations.append((self._latest_depth.stamp_sec, self._image_to_array(msg)))

    def _set_observation_gripper_state(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        self._obs_gripper_state = 1.0 if request.data else 0.0
        
        if self._obs_gripper_state == 0.0:
            self._next_gripper_state = False
            self._execute_gripper_action_if_ready()
        
        response.success = True
        response.message = (
            f"Observation gripper state set to {self._obs_gripper_state:.1f}."
        )
        return response

    def _control_step(self):
        
        # print("---- Latest joint observation: ", self.joint_observations[-1] if self.joint_observations else "None")
        
        if len(self.joint_observations) == 0 or len(self.depth_observations) == 0:
            self.get_logger().debug("Waiting for initial observations...")
            return
        
        if not self.executing_actions:
            if self.motion_start_time is None and self.motion_end_time is None:

                self.motion_start_time = self.joint_observations[0][0]
                self.motion_end_time = self.joint_observations[-1][0]
            
            self._maybe_start_inference()
            self._maybe_collect_inference()
            self._execute_waypoint_if_ready()
            self._execute_gripper_action_if_ready()
        
    def _maybe_start_inference(self):
        if self._inference_future is not None:
            return
        
        assert self.motion_start_time is not None and self.motion_end_time is not None, "Motion start and end times must be set before starting inference."
        
        start_time = self.motion_start_time
        end_time = self.motion_end_time
        print(f"Motion start time in inference: {start_time}, Motion end time: {end_time}")
        
        duration = end_time - start_time
        time_delta = duration / 10
        print("time delta: ", time_delta)
        
        self._obs_queue.clear()
        for t in np.arange(start_time + 5 * time_delta, end_time, time_delta):
            closest_joint_obs = min(self.joint_observations, key=lambda obs: abs(obs[0] - t))
            closest_depth_obs = min(self.depth_observations, key=lambda obs: abs(obs[0] - t))
            
            print(f"{t}, {closest_joint_obs[0]}, {closest_depth_obs[0]}")
            
            # if abs(closest_joint_obs[0] - t) > time_delta or abs(closest_depth_obs[0] - t) > time_delta:
            #     self.get_logger().warn(f"No close observation found for time {t:.2f}. Skipping this timestamp.")
            #     continue
            
            joints = closest_joint_obs[1]
            print("Closest joint observation: ", joints)
            joints = np.append(joints, self._obs_gripper_state)
            
            depth = closest_depth_obs[1]
            depth = depth[70:190, 190:490]
            depth = np.nan_to_num(depth, nan=10.0)
            depth = np.clip(depth, 0, 0.8)

                        
            self._obs_queue.append(Observation(joints, depth))

        observations = list(self._obs_queue)[-self._obs_window:]
        self._inference_future = self._executor.submit(self._run_model, observations)
        
    def _maybe_collect_inference(self):
        if self._inference_future is None:
            return
        if not self._inference_future.done():
            return
        
        print("Model inference completed, collecting results...")

        try:
            actions = self._inference_future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Model inference failed: {exc}")
            actions = None
        finally:
            self._inference_future = None

        if actions is None:
            return
        
        actions = np.array(actions)
        actions = actions[:10, :]
        joint_actions = actions[:, :6] * np.pi / 180.0
        gripper_actions = actions[:, 6]
        
        self._next_gripper_state = (gripper_actions > 0.9).any()
        
        print("Model Actions --------------")
        print(actions)
        print(f"Gripper actions: {gripper_actions}, Next gripper state: {self._next_gripper_state}")
        print("----------------------------")
        
        print(f"Model inference completed. Actions shape: {actions.shape}")
        
        # Integrate actions over time to get the actual joint positions to execute
        # joint_deltas = joint_actions / self._control_rate_hz
        joint_deltas = joint_actions
        # self._next_checkpoint = np.sum(joint_deltas, axis=0) + self._obs_queue[-1].joints[:6]
        self._next_checkpoint = np.sum(joint_deltas, axis=0) + self.joint_observations[-1][1][:6]
        
        print(f"Actions: {np.sum(joint_deltas, axis=0)*180.0/np.pi}")
      #  print(f"Current joint state: {self._obs_queue[-1].joints[:6]}")
        print(f"Current joint state: {self.joint_observations[-1][1][:6]*180.0/np.pi}")
        print(f"Next checkpoint: {self._next_checkpoint*180.0/np.pi}")
        
    def _execute_gripper_action_if_ready(self):
        if self._next_gripper_state is None:
            # print("No gripper action ready for execution.")
            return
        
        if self._gripper_sequence_active:
            # print("Gripper sequence already active, waiting for it to complete before executing next gripper action.")
            return
        
        if self._next_gripper_state == self._prev_gripper_state:
            # print("Gripper state has not changed since last execution, skipping gripper command.")
            return
        
        self._prev_gripper_state = self._next_gripper_state
        
        if self._next_gripper_state:
            self._start_grip_sequence()
        else:
            self._send_gripper_command("blow")
            
    def _start_grip_sequence(self) -> None:
        # print("Starting gripper sequence: GRIP")
        
        self._gripper_sequence_active = True
        
        if not self._send_gripper_command("grip"):
            self._prev_gripper_state = None
            return

        if self._gripper_release_timer is not None:
            self._gripper_release_timer.cancel()
        self._gripper_release_timer = self.create_timer(1.0, self._release_grip)

    def _release_grip(self) -> None:
        if self._gripper_release_timer is not None:
            self._gripper_release_timer.cancel()
            self._gripper_release_timer = None

        self._send_gripper_command("release")
        self._gripper_sequence_active = False
        
    def _send_gripper_command(self, command: str) -> bool:
        
        # print(f"Sending gripper command: {command}")
        
        if not self._gripper_client.wait_for_service(timeout_sec=0.1):
            self.get_logger().warn(
                f"Waiting for gripper service at {self._gripper_service}"
            )
            self._gripper_sequence_active = False
            return False

        request = GripperControl.Request()
        request.command = command
        self._gripper_future = self._gripper_client.call_async(request)
        return True
        
    def _execute_waypoint_if_ready(self):        
        if self.executing_actions:
            return
            
        if self._next_checkpoint is None:
            # print("No checkpoint ready for execution.")
            return
        
        print(f"Executing actions")

        checkpoint = self._next_checkpoint
        self._next_checkpoint = None

        self.executing_actions = True

        def move_to_checkpoint():            
            self.get_logger().info("Moving to next checkpoint...")
            self.motion_start_time = time.time()
            print("start time before motion: ", self.motion_start_time)
            print("joints before motion: ", self.joint_observations[-1][1][:6]*180.0/np.pi)
            print("checkpoint: ", checkpoint*180.0/np.pi)
            self.moveit2.move_to_configuration(checkpoint, self._joint_names, tolerance=0.001)
            self.get_logger().info("Waiting for movement to complete...")
            
            if not self.moveit2.wait_until_executed():
                self.get_logger().error("Failed to execute movement to checkpoint.")
            # time.sleep(2)
            self.motion_end_time = time.time()
            print("end time after motion: ", self.motion_end_time)
            joints_after_motion = self.joint_observations[-1][1][:6]*180.0/np.pi
            print(f"Reached checkpoint after action. Current joint state: {joints_after_motion}")
            print(f"Difference from checkpoint: {(joints_after_motion - checkpoint*180.0/np.pi)}")
            self.executing_actions = False
            
            # Dummy motion
            # self.motion_start_time = time.time()
            # time.sleep(2.0)
            # self.motion_end_time = time.time()
            # self.executing_actions = False
            
            self.get_logger().info("Movement to checkpoint completed.")

        threading.Thread(target=move_to_checkpoint).start()
    
        
    @torch.no_grad()
    def _run_model(self, observations: Sequence[Observation]) -> List[List[float]]:                
        device = self.device
        
        depth_images = torch.stack([torch.from_numpy(obs.depth).unsqueeze(0) for obs in observations], dim=0).unsqueeze(0)  # Shape: (obs_window, 1, H, W)
        non_visual_obs = torch.stack([torch.from_numpy(obs.joints) for obs in observations], dim=0)  # Shape: (obs_window, num_joints)
        
        print("Model input:")
        print(non_visual_obs)
        
        num_predicted_actions = 1
        action_sequence_length = 20
        num_steps = 100
        action_dim = 7
            
        depth_images = depth_images.to(device=device, dtype=torch.float32)
        non_visual_obs = non_visual_obs.to(device=device, dtype=torch.float32)
        
        depth_images = depth_images.repeat(num_predicted_actions, 1,1,1,1)
        non_visual_obs = non_visual_obs.repeat(num_predicted_actions, 1,1)     
        
        x = torch.randn((num_predicted_actions, action_sequence_length, action_dim), device=device, dtype=torch.float32)
        
        dt = 1.0 / num_steps
        
        for k in range(num_steps):
            t_k = k * dt
            t_k_tensor = torch.tensor(t_k, device=device, dtype=torch.float32).unsqueeze(0)
            
            v_k = self.flow_matching_policy(depth_images, non_visual_obs, x, t_k_tensor)
                        
            x_pred = x + dt * v_k
            
            t_k1 = (k + 1) * dt
            t_k1_tensor = torch.tensor(t_k1, device=device, dtype=torch.float32).unsqueeze(0)
            
            v_k1 = self.flow_matching_policy(depth_images, non_visual_obs, x_pred, t_k1_tensor)
            
            x = x + 0.5 * dt * (v_k + v_k1)

        actions = x.squeeze(0).cpu().numpy()[:, :7]

        return actions.tolist()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImitationMoveitControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    

if __name__ == "__main__":
    main()