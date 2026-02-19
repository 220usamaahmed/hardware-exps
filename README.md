```bash

# Start fake hardware
ros2 launch ur_bringup ur3e.launch.py ur_setup:=ur3e use_fake_hardware:=true launch_rviz:=false robot_ip:=0.0.0.0

# Start real hardware
ros2 launch ur_bringup ur3e.launch.py ur_setup:=ur3e launch_rviz:=false robot_ip:=192.168.1.102

# Launch RViz
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur3e launch_servo:=true launch_rviz:=true

# Change controller, forward_position_controller for Moveit Servo, scaled_joint_trajectory_controller for RViz
ros2 control switch_controllers --deactivate forward_position_controller --activate scaled_joint_trajectory_controller

# Start controller
ros2 run ur3e_controller trajectory_control

# Start gripper
ros2 launch ecpmi_gripper suction_gripper.launch.py

# Control gripper
ros2 run ecpmi_gripper gripper_client grip
ros2 run ecpmi_gripper gripper_client release
ros2 run ecpmi_gripper gripper_client blow

```