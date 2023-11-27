#!/usr/bin/env python3
import tf2_ros
import rospy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from multi_task_il.datasets.savers import Trajectory
import numpy as np
import math
import os
import time
import pickle as pkl
from camera_controller.srv import *
from ur5e_2f_85_controller.msg import *
from ur5e_2f_85_teleoperation.msg import TrajectoryState
from cv_bridge import CvBridge
from ur5e_2f_85_controller.srv import GoToJoint, GoToJointRequest
import cv2


class TrajectoryCollector():

    # Obs keys
    IMAGE = 'image'
    JOINT_POS = 'joint_pos'
    JOINT_VEL = 'joint_vel'
    EEF_POS = 'eef_pos'
    EEF_QUAT = 'eef_quat'
    GRIPPER_QPOS = 'gripper_qpos'
    GRIPPER_QVEL = 'gripper_qvel'
    EE_AA = 'ee_aa'

    def __init__(self, env_camera_name: list, env_camera_topic_name: str, gripper_state_topic: str, joint_state_topic: str, trj_state_topic: str, tcp_frame_name: str, frame_rate: int, saving_folder: str, task_name: str, task_id: str, start_trj_cnt: int, home_pos: list):

        # Set rostopic name parameters
        self._gripper_state_topic = gripper_state_topic
        self._joint_state_topic = joint_state_topic
        self._trj_state_topic = trj_state_topic
        self._tcp_frame_name = tcp_frame_name
        self._frame = frame_rate
        self._trajectory_cnt = start_trj_cnt

        # moveit service
        self._home_pos = home_pos
        rospy.loginfo(self._home_pos)
        self._moveit_service_client = rospy.ServiceProxy(
            "/go_to_joint", GoToJoint, persistent=True)

        # camera client service
        rospy.loginfo("---- Waiting for env camera service ----")
        self._env_camera_service_client = rospy.ServiceProxy(
            env_camera_topic_name, GetFrames, True)
        self._bridge = CvBridge()
        self._show_image = False
        self._env_camera_name = env_camera_name

        # Init tf listener for the TCP Pose
        self._tfBuffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tfBuffer)
        exception = True
        while exception:
            try:
                # Get TCP Pose
                tcp_pose = self._tfBuffer.lookup_transform(
                    'base_link', self._tcp_frame_name, rospy.Time())
                rospy.logdebug(f"TCP Pose {tcp_pose}")
                exception = False
            except ((tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)) as e:
                exception = True
                rospy.logerr(e)

        # Trajectory cnt, saving folder
        self._saving_folder = saving_folder
        self._task_name = task_name
        self._task_id = '{:02}'.format(task_id)

        try:
            rospy.loginfo(f"Running Task {task_name} - ID {self._task_id}")

            self._saving_folder = os.path.join(
                self._saving_folder, task_name, f"task_{self._task_id}")
            os.makedirs(self._saving_folder)
        except Exception as e:
            print(e)

    def _save_step(self, env_frames: GetFramesResponse, tcp_pose: TransformStamped, joint_state: JointState, state: str, gripper_action: int, gripper_q_pos: int, done: int, reward: int):

        # fill observations
        obs = dict()
        obs[TrajectoryCollector.EEF_POS] = np.array(
            [tcp_pose.transform.translation.x, tcp_pose.transform.translation.y, tcp_pose.transform.translation.z])
        obs[TrajectoryCollector.EEF_QUAT] = np.array(
            [tcp_pose.transform.rotation.x, tcp_pose.transform.rotation.y, tcp_pose.transform.rotation.z, tcp_pose.transform.rotation.w])
        obs[TrajectoryCollector.JOINT_POS] = np.array(joint_state.position)
        obs[TrajectoryCollector.JOINT_VEL] = np.array(joint_state.velocity)
        obs[TrajectoryCollector.EE_AA] = self._quat2axisangle(
            obs[TrajectoryCollector.EEF_QUAT])
        obs[TrajectoryCollector.GRIPPER_QPOS] = gripper_q_pos

        # take env_frames
        for j, (color_msg, depth_msg) in enumerate(zip(env_frames.color_frames, env_frames.depth_frames)):
            color_cv_image = self._bridge.imgmsg_to_cv2(
                color_msg, desired_encoding='rgba8')
            color_cv_image = cv2.cvtColor(
                color_cv_image, cv2.COLOR_RGBA2RGB)
            # if j == 3:
            #     cv2.imshow("robot hand camera", color_cv_image)
            #     cv2.waitKey(1)
            # cv2.destroyAllWindows()
            depth_cv_image = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
            if self._show_image and self._env_camera_name[j] == 'camera_front':
                cv2.imshow("Color image", color_cv_image)
                # cv2.imshow("Depth image", depth_cv_image)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()
            obs[f"{self._env_camera_name[j]}_image"] = color_cv_image
            obs[f"{self._env_camera_name[j]}_depth"] = depth_cv_image
        # fill info
        # rospy.loginfo(f"Teleoperation state {state}")
        info = {'status': state}
        rospy.loginfo(state)
        # create action
        action = np.array([tcp_pose.transform.translation.x,
                           tcp_pose.transform.translation.y,
                           tcp_pose.transform.translation.z,
                           tcp_pose.transform.rotation.x,
                           tcp_pose.transform.rotation.y,
                           tcp_pose.transform.rotation.z,
                           tcp_pose.transform.rotation.w,
                           gripper_action])

        self._trajectory.append(obs=obs, reward=reward,
                                done=done, info=info, action=action)

        self._step += 1

        # if the task is completed save the trajectory
        if done == 1:
            rospy.loginfo("Saving Trajectory.....")
            self._save_trajectory()
            return True

        return False

    def _save_trajectory(self):
        file_name = os.path.join(
            self._saving_folder, 'traj{:03d}.pkl'.format(self._trajectory_cnt))
        self._trajectory_cnt += 1
        pkl.dump({
            'traj': self._trajectory,
            'len': len(self._trajectory),
            'env_type': self._task_name,
            'task_id': self._task_id}, open(file_name, 'wb'))
        rospy.loginfo(f"Saved file {file_name}")
        del self._trajectory

    def _quat2axisangle(self, quat):
        """
        Converts quaternion to axis-angle format.
        Returns a unit vector direction scaled by its angle in radians.
        Args:
            quat (np.array): (x,y,z,w) vec4 float angles
        Returns:
            np.array: (ax,ay,az) axis-angle exponential coordinates
        """
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)

        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

    def init_trj(self):
        self._step = 0
        self._trajectory = Trajectory()
        # wait for going in home position
        enter = None
        while enter != "":
            rospy.loginfo("Press enter to go to gome: ")
            enter = input()
        # ask for going to home position
        home_joint_pos_req = GoToJointRequest()
        home_joint_pos_req.joint_goal_pos = np.array(self._home_pos)
        home_joint_pos_req.stop_controllers = True
        result = self._moveit_service_client.call(home_joint_pos_req)
        if result.success:
            rospy.loginfo_once(
                "Robot in home position, ready to get a new trajectory")

    def get_message(self):
        exception = True
        while exception:
            try:
                # Get TCP Pose
                tcp_pose = self._tfBuffer.lookup_transform(
                    'base_link', self._tcp_frame_name, rospy.Time())
                exception = False
            except ((tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException)) as e:
                exception = True
                rospy.logerr(e)

        # Get joint states
        joint_state = rospy.wait_for_message(
            topic=self._joint_state_topic, topic_type=JointState)

        # Get env_frames
        env_frames = self._env_camera_service_client()

        # Get gripper state
        gripper_state = rospy.wait_for_message(
            topic=self._gripper_state_topic, topic_type=GripperState)

        # Get the trajectory state
        trajectory_state = rospy.wait_for_message(
            topic=self._trj_state_topic, topic_type=TrajectoryState)

        # Compute the done signal
        done = trajectory_state.trajectory_state == TrajectoryState.TRAJECTORY_END

        # Compute reward
        reward = 1 if trajectory_state.trajectory_state == TrajectoryState.TRAJECTORY_END else 0

        # Save step if trajectory state is not in idle
        if trajectory_state.trajectory_state != TrajectoryState.TRAJECTORY_IDLE:
            rospy.logdebug(f"Saving step {self._step}")
            return self._save_step(env_frames=env_frames,
                                   tcp_pose=tcp_pose,
                                   joint_state=joint_state,
                                   state=trajectory_state.trajectory_state,
                                   gripper_action=1-gripper_state.gripper_open,
                                   gripper_q_pos=gripper_state.finger_position,
                                   done=done,
                                   reward=reward)
        rospy.loginfo("Teleoperator in idle mode")
        return False

    def run(self):

        while not rospy.is_shutdown():
            # press enter to start the collection of a new trajectory
            enter = None
            while enter != "":
                rospy.loginfo("Press enter to collect a new trajectory: ")
                enter = input()

            self.init_trj()
            rospy.loginfo(
                f"Collecting task {self._task_name} - Sub task {self._task_id} - Counter {self._trajectory_cnt}")
            trajectory_completed = False
            while not trajectory_completed:
                # time_begin = rospy.Time.now()
                trajectory_completed = self.get_message()
                # time_end = rospy.Time.now()
                # duration = time_end - time_begin
                # rospy.loginfo(f"Wait for {duration.to_sec()} secs")
