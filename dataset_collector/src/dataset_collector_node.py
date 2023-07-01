#!/usr/bin/env python3
import rospy
from trajectory_collector import TrajectoryCollector


def main():
    rospy.init_node("dataset_collector", log_level=rospy.INFO)

    # 1. Get conf parameters from parameter server
    gripper_state_topic = rospy.get_param("/gripper_state_topic")
    joint_state_topic = rospy.get_param("/joint_state_topic")
    trj_state_topic = rospy.get_param("/trj_state_topic")
    tcp_frame_name = rospy.get_param("tcp_frame_name")
    frame_rate = rospy.get_param("frame_rate")
    saving_folder = rospy.get_param("saving_folder")
    task_name = rospy.get_param("task_name")
    task_id = rospy.get_param("task_id")
    start_trj_cnt = rospy.get_param("start_trj_cnt")
    env_camera_topic = rospy.get_param("env_camera_topic")
    env_camera_name = rospy.get_param("env_camera_name")
    robot_home = rospy.get_param("home_position")

    # 2. Init TrajectoryCollector
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
                                               home_pos=robot_home)

    trajectory_collector.run()


if __name__ == '__main__':
    main()
