import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
from control_msgs.msg import JointJog
from ecpmi_gripper.srv import GripperControl
from concurrent.futures import Future, ThreadPoolExecutor
import torch
import torch.nn as nn
import math
from einops import rearrange

np.set_printoptions(suppress=True)


@dataclass
class LatestMsg:
    stamp_sec: float
    msg: object


@dataclass
class Observation:
    stamp_sec: float
    joints: np.ndarray
    depth: np.ndarray


############################################
#### flow matching model class ########


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
        depth_images = torch.nan_to_num(depth_images, nan=10.0)
        depth_images = torch.clip(depth_images, 0, 0.8)
        
        # print("depth_images shape == " , depth_images.shape)
        
        # print("-"*50)
        # print(depth_images)
        # print("-"*50)
        
        # print("depth image min: ", depth_images.min().item(), " max: ", depth_images.max().item())
        # print("non-visual obs min: ", non_visual_obs.min().item(), " max: ", non_visual_obs.max().item())

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

############################################################

















class ImitationControl(Node):
    def __init__(self) -> None:
        super().__init__("imitation_control")

        self.declare_parameter("command_topic", "/servo_node/delta_joint_cmds")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("sync_tolerance_sec", 0.2)
        self.declare_parameter("obs_window", 5)
        self.declare_parameter("action_horizon", 20)
        self.declare_parameter("action_execute_count", 10)
        self.declare_parameter("max_joint_speed", 1.0)
        self.declare_parameter("auto_start_servo", True)
        self.declare_parameter("start_servo_service", "/servo_node/start_servo")
        self.declare_parameter("gripper_service", "/gripper_control")
        self.declare_parameter("command_timeout_sec", 0.5)
        self.declare_parameter("joint_names", [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ])

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self._sync_tolerance_sec = float(
            self.get_parameter("sync_tolerance_sec").value
        )
        self._obs_window = int(self.get_parameter("obs_window").value)
        self._action_horizon = int(self.get_parameter("action_horizon").value)
        self._action_execute_count = int(
            self.get_parameter("action_execute_count").value
        )
        self._max_joint_speed = float(self.get_parameter("max_joint_speed").value)
        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._start_servo_service = str(
            self.get_parameter("start_servo_service").value
        )
        self._gripper_service = str(self.get_parameter("gripper_service").value)
        self._command_timeout_sec = float(
            self.get_parameter("command_timeout_sec").value
        )
        self._joint_names = list(self.get_parameter("joint_names").value)

        self._latest_joint_state: Optional[LatestMsg] = None
        self._latest_depth: Optional[LatestMsg] = None
        self._name_to_index: Optional[Dict[str, int]] = None
        self._obs_queue: Deque[Observation] = deque(maxlen=self._obs_window)
        self._action_queue: Deque[List[float]] = deque()
        self._inference_future: Optional[Future] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_action_time_sec: Optional[float] = None
        self._last_drop_log_sec: Optional[float] = None
        self._last_collect_log_sec: Optional[float] = None
        self._gripper_state = 0.0
        self._last_gripper_signal: Optional[int] = None
        self._gripper_sequence_active = False
        self._gripper_future: Optional[rclpy.task.Future] = None
        self._gripper_release_timer = None
        self._collect_observations_enabled = True
        self._has_received_model_action = False

        self._joint_cmd_pub = self.create_publisher(JointJog, self._command_topic, 10)
        self.create_subscription(
            JointState, self._joint_states_topic, self._on_joint_state, 10
        )
        self.create_subscription(Image, self._depth_topic, self._on_depth, 10)
        self._gripper_client = self.create_client(GripperControl, self._gripper_service)

        self._start_servo_client = self.create_client(Trigger, self._start_servo_service)
        self._start_servo_timer = None
        if self._auto_start_servo:
            self._start_servo_timer = self.create_timer(1.0, self._try_start_servo)

        period = 1.0 / self._control_rate_hz if self._control_rate_hz > 0 else 0.0667
        self._control_timer = self.create_timer(period, self._control_step)

        self.get_logger().info(
            f"ImitationControl ready. Publishing joint commands on {self._command_topic} at {self._control_rate_hz:.2f} Hz"
        )
        
        self.previous_obs = None
        self.previous_act = None
        
        
        
        #############################################
        ##### initiate the flow matching policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.flow_matching_policy=ConditionalDiffusionModel()
        ## add the path for the model checkpoint and the device
        check_point=torch.load('/home/shokry/ur3e-trajectories/weights/flow_matching_real_world_data_pick3_processed_without_data_augmentation_epoch_320.pt', map_location=self.device)
        # check_point=torch.load('/home/siddiquieu1/ur3e-trajectories/pick/weights/flow_matching_real_world_data_pick_epoch_250.pt', map_location=self.device)
        
        print("Loading flow matching policy from checkpoint:", check_point.keys())
        
        self.flow_matching_policy.load_state_dict(check_point['model'])
        self.flow_matching_policy.to(self.device)
        self.flow_matching_policy.eval()
        for p in self.flow_matching_policy.parameters():
            p.requires_grad_(False)
            
        ##########################################
        
            

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

    def _on_depth(self, msg: Image) -> None:
        self._latest_depth = LatestMsg(self._stamp_to_sec(msg.header.stamp), msg)

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

    def _control_step(self) -> None:
        now_sec = self._now_sec()
        self._maybe_append_observation(now_sec)
        self._maybe_collect_inference()
        self._maybe_start_inference()
        self._publish_next_action(now_sec)

    def _maybe_append_observation(self, now_sec: float) -> None:
        if not self._collect_observations_enabled:
            # print("not collecting observations")
            self._rate_limited_collect_log(
                now_sec, "Skipping observation: waiting for model-driven actions."
            )
            return
        if (
            self._latest_joint_state is None
            or self._latest_depth is None
            or self._name_to_index is None
        ):
            print("missing joint state or depth or name_to_index")
            return

        joint_age = abs(now_sec - self._latest_joint_state.stamp_sec)
        depth_age = abs(now_sec - self._latest_depth.stamp_sec)
        if joint_age > self._sync_tolerance_sec or depth_age > self._sync_tolerance_sec:
            self._rate_limited_drop_log(
                now_sec, "Inputs are too old; skipping observation."
            )
            return

        if abs(self._latest_joint_state.stamp_sec - self._latest_depth.stamp_sec) > self._sync_tolerance_sec:
            self._rate_limited_drop_log(
                now_sec, "Inputs are not synchronized; skipping observation."
            )
            return

        joints = self._extract_joint_vector(self._latest_joint_state.msg)
        joints = np.append(joints, float(self._gripper_state))
        # joints = np.append(joints, 0.0)
        
        # print("observation", joints)
        
        depth = self._image_to_array(self._latest_depth.msg)
        if depth is None:
            return

        # Crop depth image to match the training data window.
        # depth = depth[70:220, 165:515]
        depth = depth[70:190, 190:490]

        self._obs_queue.append(Observation(now_sec, joints, depth))
        
        return joints
    
    # def _control_step_2(self) -> None:
    #     now_sec = self._now_sec()
    #     obs_joints = self._maybe_append_observation(now_sec)
    #     self._maybe_collect_inference()
    #     self._maybe_start_inference()
    #     # act_joints = self._publish_next_action(now_sec)
    
    #     if obs_joints is None:
    #         self._publish_next_action(now_sec)
    #         return
        
    #     if self.previous_obs is not None:
    #         previous_obs = self.previous_obs[:6]
    #         previous_act = np.array(self.previous_act[:6])
    #         expected_next_obs = previous_obs + previous_act[:6] / self._control_rate_hz
    #         next_obs = obs_joints[:6]
            
    #         error = expected_next_obs - next_obs
            
    #         def rad_to_deg(arr):
    #             return np.round(arr * (180 / math.pi) * 100) / 100
            
    #         print("Previous Obs", rad_to_deg(previous_obs))
    #         print("Previous Act", rad_to_deg(previous_act))
    #         print("Excepted Next obs", rad_to_deg(expected_next_obs))
    #         print("Actual Next obs", rad_to_deg(next_obs))
    #         print("Error", rad_to_deg(error))
    #         print("Error (radians)", error)
    #         print("-" * 50)
            
    #         if (np.abs(error) > 0.02).any():
    #             correction_actions = self.previous_act
    #             correction_actions[:6] = error * self._control_rate_hz
                
    #             self._action_queue.appendleft(correction_actions)
    #             self._obs_queue.popleft()
    #             self._publish_next_action(now_sec)
                
    #             return
        
    #     act_joints = self._publish_next_action(now_sec)
        
    #     self.previous_obs = obs_joints
    #     self.previous_act = act_joints

    def _rate_limited_drop_log(self, now_sec: float, message: str) -> None:
        if self._last_drop_log_sec is None or now_sec - self._last_drop_log_sec >= 1.0:
            self._last_drop_log_sec = now_sec
            self.get_logger().warn(message)

    def _rate_limited_collect_log(self, now_sec: float, message: str) -> None:
        if (
            self._last_collect_log_sec is None
            or now_sec - self._last_collect_log_sec >= 2.0
        ):
            self._last_collect_log_sec = now_sec
            self.get_logger().info(message)

    def _maybe_start_inference(self) -> None:
        
        # print("maybe start inference", len(self._obs_queue), len(self._action_queue))
        
        if self._inference_future is not None:
            return
        if len(self._obs_queue) < self._obs_window:
            return
        if self._action_queue:
            return

        observations = list(self._obs_queue)[-self._obs_window:]
        self._inference_future = self._executor.submit(self._run_model, observations)

    def _maybe_collect_inference(self) -> None:
        if self._inference_future is None:
            return
        if not self._inference_future.done():
            return

        try:
            actions = self._inference_future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Model inference failed: {exc}")
            actions = None
        finally:
            self._inference_future = None

        if actions is None:
            return

        normalized = self._normalize_actions(actions)
        if not normalized:
            self.get_logger().warn("Model returned no valid actions.")
            return
        
        normalized = actions
        
        # print("Combined actions")
        # print(np.sum(normalized[:self._action_execute_count], axis=0))

        for action in normalized[: self._action_execute_count]:
            # print("queuing action", action)
            self._action_queue.append(action)

    def _normalize_actions(self, actions: Sequence[Sequence[float]]) -> List[List[float]]:
        if len(actions) < self._action_execute_count:
            self.get_logger().warn(
                f"Expected at least {self._action_execute_count} actions, got {len(actions)}."
            )
            return []

        normalized: List[List[float]] = []
        for idx, action in enumerate(actions[: self._action_horizon]):
            if len(action) != len(self._joint_names) + 1:
                self.get_logger().warn(
                    f"Action {idx} length {len(action)} does not match {len(self._joint_names) + 1} actions."
                )
                return []
            joint_values = [float(v) for v in action[: len(self._joint_names)]]
            gripper_value = action[len(self._joint_names)]
            normalized.append(joint_values + [gripper_value])
        return normalized

    def _publish_next_action(self, now_sec: float) -> None:
        action_from_model = False
        if self._action_queue:
            action = self._action_queue.popleft()
            action_from_model = True
        else:
            # Temporary fallback: publish zeros when inference is late.
            # Replace this with a hold-last-action policy if needed.
            action = [0.0] * (len(self._joint_names) + 1)
            action[-1] = 1 if self._gripper_state == 1 else -1

        if self._last_action_time_sec is not None:
            if now_sec - self._last_action_time_sec > self._command_timeout_sec:
                action = [0.0] * (len(self._joint_names) + 1)
                action[-1] = 1 if self._gripper_state == 1 else -1
                action_from_model = False

        if len(action) >= len(self._joint_names) + 1:
            velocities = action[: len(self._joint_names)]
            gripper_value = action[len(self._joint_names)]
            print("publishing action", velocities, "gripper", gripper_value)
        else:
            velocities = action[: len(self._joint_names)]
            gripper_value = 0.0
            print("publishing action", velocities, "gripper", gripper_value)

        velocities = self._clamp_velocities(velocities)
        
        self._publish_joint_command(velocities)
        self._maybe_handle_gripper_action(gripper_value)
        self._last_action_time_sec = now_sec
        if action_from_model:
            self._has_received_model_action = True
        if self._has_received_model_action:
            self._collect_observations_enabled = action_from_model
            
        return velocities

    def _maybe_handle_gripper_action(self, gripper_value: float) -> None:
        if self._gripper_sequence_active:
            return

        if gripper_value > 0.9:
            signal = 1
        elif gripper_value < -0.9:
            signal = -1
        else:
            return

        if signal == self._last_gripper_signal:
            return

        self._last_gripper_signal = signal

        if signal == 1:
            self._start_grip_sequence()
        else:
            if self._send_gripper_command("blow"):
                self._gripper_state = 0.0
            else:
                self._last_gripper_signal = None

    def _start_grip_sequence(self) -> None:
        if not self._send_gripper_command("grip"):
            self._last_gripper_signal = None
            return

        self._gripper_sequence_active = True
        self._gripper_state = 1.0

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

    def _clamp_velocities(self, velocities: Sequence[float]) -> List[float]:
        clamped: List[float] = []
        for v in velocities:
            if abs(v) > self._max_joint_speed > 0.0:
                v = np.sign(v) * self._max_joint_speed
            # if abs(v) < 0.05 and v != 0.0:
            #     v = np.sign(v) * 0.05
                
            clamped.append(float(v))
        return clamped

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

    @torch.no_grad()
    def _run_model(self, observations: Sequence[Observation]) -> List[List[float]]:                
        device = self.device
        
        depth_images = torch.stack([torch.from_numpy(obs.depth).unsqueeze(0) for obs in observations], dim=0).unsqueeze(0)  # Shape: (obs_window, 1, H, W)
        non_visual_obs = torch.stack([torch.from_numpy(obs.joints) for obs in observations], dim=0)  # Shape: (obs_window, num_joints)
        
        num_predicted_actions=1
        action_sequence_length=20
        num_steps=100
        action_dim=7
            
        depth_images = depth_images.to(device=device, dtype=torch.float32)
        non_visual_obs = non_visual_obs.to(device=device, dtype=torch.float32)
        
        print("model input (2 decimal places)", torch.round(non_visual_obs * 100) / 100)
        # print("model input (degrees)", torch.round(non_visual_obs[:, :6] * 100) / 100 * (180 / math.pi))
        
        ## depth image shape should be (5,1,150,350)
        ## repeat the depth image and non-visual observations by the number of predicted actions
        depth_images = depth_images.repeat(num_predicted_actions, 1,1,1,1)
        non_visual_obs = non_visual_obs.repeat(num_predicted_actions, 1,1)     
        
        ## generate noisy actions
        x = torch.randn((num_predicted_actions, action_sequence_length, action_dim), device=device, dtype=torch.float32)
        
        ## time per denoising step, to be multiplied by the noise velocity vector and get the value of the noise 
        dt = 1.0 / num_steps
        
        ## start the denoising process
        for k in range(num_steps):
            ## calculate the current denoising step (normalized between 0 and 1)
            t_k = k * dt
            t_k_tensor = torch.tensor(t_k, device=device, dtype=torch.float32).unsqueeze(0)
            
            ## calculate the noise vector from the flow matching model    
            v_k = self.flow_matching_policy(depth_images, non_visual_obs, x, t_k_tensor)
                        
            x_pred = x + dt * v_k
            
            t_k1 = (k + 1) * dt
            t_k1_tensor = torch.tensor(t_k1, device=device, dtype=torch.float32).unsqueeze(0)
            
            v_k1 = self.flow_matching_policy(depth_images, non_visual_obs, x_pred, t_k1_tensor)
            
            x = x + 0.5 * dt * (v_k + v_k1)

        actions = x.squeeze(0).cpu().numpy()[:, :7]  # Shape: (action_sequence_length, action_dim)

        # print("model output", actions[:, 6])

        return actions.tolist()

        # Placeholder: replace with your model inference. Expected output shape: [20, 7].

    def _publish_joint_command(self, velocities: Sequence[float]) -> None:
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self._joint_names
        msg.velocities = list(velocities)
        msg.displacements = []
        msg.duration = 0.0
        self._joint_cmd_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImitationControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
