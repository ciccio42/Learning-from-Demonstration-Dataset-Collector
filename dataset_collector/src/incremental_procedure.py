#!/usr/bin/env python3
import rospy
from trajectory_collector import TrajectoryCollector
from multi_task_il.datasets.savers import Trajectory
import os

def collect_correct_sample(model_name:str, task_name: str, task_id:int, start_trj_cnt: int, traj: Trajectory):
    # 1. Get conf parameters from parameter server
    gripper_state_topic = rospy.get_param("/gripper_state_topic")
    joint_state_topic = rospy.get_param("/joint_state_topic")
    trj_state_topic = rospy.get_param("/trj_state_topic")
    tcp_frame_name = rospy.get_param("tcp_frame_name")
    frame_rate = rospy.get_param("frame_rate")
    env_camera_topic = rospy.get_param("env_camera_topic")
    env_camera_name = rospy.get_param("env_camera_name")
    robot_home = rospy.get_param("home_position")
    collect_with_joy = rospy.get_param("collect_with_joy")
    
    saving_folder = os.path.join(rospy.get_param("saving_folder"), "dagger", model_name)
    
    trajectory_collector = TrajectoryCollector(env_camera_name=env_camera_name,
                                        gripper_state_topic=gripper_state_topic,
                                        env_camera_topic_name=env_camera_topic,
                                        joint_state_topic=joint_state_topic,
                                        trj_state_topic=trj_state_topic,
                                        tcp_frame_name=tcp_frame_name,
                                        frame_rate=frame_rate,
                                        saving_folder=saving_folder,
                                        task_name=task_name,
                                        task_id=task_id,
                                        start_trj_cnt=start_trj_cnt,
                                        home_pos=robot_home,
                                        collect_with_joy=collect_with_joy,
                                        incremental=False)
    
    trajectory_collector.run(preliminary_traj=traj)