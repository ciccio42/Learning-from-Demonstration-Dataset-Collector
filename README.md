# Learning-from-Demonstration-Dataset-Collector

# 1. Dependencies
```bash
python3 -m pip install pyquaternion
python3 -m pip install ~/Desktop/multi_task_lfd/Multi-Task-LFD-Framework/repo/Multi-Task-LFD-Training-Framework/training
```

# 1. Run ur-drivers
```bash
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.1.100 kinematics_config:="/home/ciccio/.ros/real_robot_calibration.yaml" use_tool_communication:=true tool_voltage:=24 tool_parity:=0 tool_baud_rate:=115200 tool_stop_bits:=1 tool_rx_idle_chars:=1.5 tool_tx_idle_chars:=3.5 tool_device_name:=/tmp/ttyUR robot_description_file:="/home/ciccio/Desktop/catkin_ws/src/Ur5e-2f-85f/ur5e_2f_85_description/launch/load_ur5e_2f_85.launch"
```

# 2. Run teleoperation node
```bash
roslaunch ur5e_2f_85_teleoperation ur5e_teleoperation.launch
```

# 3. Run camera
```bash
roslaunch zed_camera_controller zed_camera_controller.launch
```

# 4. Run dataset collector

```bash
roslaunch ur5e_2f_85_controller controller.launch 
```

# 5. Run dataset collector

```bash
roslaunch dataset_collector dataset_collector.launch 
```


# To Test saved trajectories

# 1. Run ur-drivers
```bash
roslaunch ur_robot_driver ur5e_bringup.launch robot_ip:=192.168.1.100 kinematics_config:="/home/ciccio/.ros/real_robot_calibration.yaml" use_tool_communication:=true tool_voltage:=24 tool_parity:=0 tool_baud_rate:=115200 tool_stop_bits:=1 tool_rx_idle_chars:=1.5 tool_tx_idle_chars:=3.5 tool_device_name:=/tmp/ttyUR robot_description_file:="/home/ciccio/Desktop/catkin_ws/src/Ur5e-2f-85f/ur5e_2f_85_description/launch/load_ur5e_2f_85.launch"
```

# 2.1. Run gripper driver
```bash
rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /tmp/ttyUR
```

# 2.2. Run controller
```bash
roslaunch ur5e_2f_85_controller controller.launch 
```

# 2.3. Run rviz
```bash
roslaunch ur5e_2f_85_camera_table_moveit_config moveit_rviz.launch 
```

# 3. Run rollout trajectory

```bash
roslaunch dataset_collector rollout_trajectory.launch file_path:=/media/ciccio/Sandisk/real-world-dataset/pick_place/task_01/traj015.pkl
```