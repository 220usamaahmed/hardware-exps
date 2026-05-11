- Start robot

- Select program: external_control1.urp

- Start real hardware
```bash
ros2 launch ur_bringup ur3e.launch.py ur_setup:=ur3e launch_rviz:=false robot_ip:=192.168.1.102
```

- Press play

- Start Gripper
```bash
ros2 launch ecpmi_gripper suction_gripper.launch.py

# Control gripper
ros2 run ecpmi_gripper gripper_client grip
ros2 run ecpmi_gripper gripper_client release
ros2 run ecpmi_gripper gripper_client blow
```

- Start camera
```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i
```

- Start RViz and Move it
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e launch_servo:=true launch_rviz:=true
```

- Change controller
```bash
ros2 control switch_controllers --activate forward_position_controller --deactivate scaled_joint_trajectory_controller
```

forward_position_controller for servo control
scaled_joint_trajectory_controller for rviz and maybe move group

- Camera parameters
```bash
ros2 run rqt_reconfigure rqt_reconfigure
```

- Controller
```bash
ros2 run ur3e_controller dataset_recorder
ros2 run ur3e_controller trajectory_control
ros2 run ur3e_controller imitation_moveit_control
```

- New Code
Add here ``src/ur3e_controller/ur3e_controller/new_node.py``

Run ones:
```

source install/setup.bash 
```

Gripper state (manual input)
```bash
ros2 service call /set_observation_gripper_state std_srvs/srv/SetBool "{data: true}"
```