#!/usr/bin/env python3
import rospy
from trajectory_collector import TrajectoryCollector, TrajectoryCollectorHuman
from pynput import keyboard


def main():
    rospy.init_node("dataset_collector", log_level=rospy.INFO)
    # import debugpy
    # debugpy.listen(('0.0.0.0', 5678))
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()

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
    collect_with_joy = rospy.get_param("collect_with_joy")

    human = rospy.get_param("human")

    if not human:
        rospy.loginfo("Running TrajectoryCollector")
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
                                                   home_pos=robot_home,
                                                   collect_with_joy=collect_with_joy)
    else:
        # 2. Init TrajectoryCollector
        rospy.loginfo("Running TrajectoryCollectorHuman")
        trajectory_collector = TrajectoryCollectorHuman(env_camera_name=env_camera_name,
                                                        env_camera_topic_name=env_camera_topic,
                                                        trj_state_topic=trj_state_topic,
                                                        frame_rate=frame_rate,
                                                        saving_folder=saving_folder,
                                                        task_name=task_name,
                                                        task_id=task_id,
                                                        start_trj_cnt=start_trj_cnt)

    trajectory_collector.run()


if __name__ == '__main__':
    main()
