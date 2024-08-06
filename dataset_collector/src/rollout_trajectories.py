#!/usr/bin/env python3
import rospy
import pickle as pkl

import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from controller.robotiq2f_85 import Robotiq2f85
import cv2
import numpy as np
from utils import *
import glob
import os

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

PKL_FILE_PATH = "/media/ciccio/Sandisk/real-world-dataset/pick_place/task_00/traj000.pkl"

T_aruco_table = np.array([[-1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.06],
                          [0.0, 0.0, 1.0, 0.0],
                          [0, 0, 0, 1]])
T_aruco_bl = T_aruco_table  @ np.array([[-1, 0.0, 0, 0.00],
                                        [0.0, -1.0, 0, 0.612],
                                        [0, 0, 1, 0.120],
                                        [0, 0, 0, 1]])

camera_intrinsic = np.array([[345.2712097167969, 0.0, 337.5007629394531],
                             [0.0, 345.2712097167969,
                              179.0137176513672],
                             [0, 0, 1]])


def _quat2axisangle(quat):
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


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveGroupPythonInterfaceTutorial(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()

        # BEGIN_SUB_TUTORIAL setup
        ##
        # First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("planning_example_node", anonymous=True)

        # Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        # kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        # Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        # for getting, setting, and updating the robot's internal understanding of the
        # surrounding world:
        scene = moveit_commander.PlanningSceneInterface()

        # Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        # to a planning group (group of joints).  In this tutorial the group is the primary
        # arm joints in the Panda robot, so we set the group's name to "panda_arm".
        # If you are using a different robot, change this value to the name of your robot
        # arm planning group.
        # This interface can be used to plan and execute motions:
        group_name = "tcp_group"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        # Create a `DisplayTrajectory`_ ROS publisher which is used to display
        # trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        # END_SUB_TUTORIAL

        # BEGIN_SUB_TUTORIAL basic_info
        ##
        # Getting Basic Information
        # ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        move_group.set_pose_reference_frame('base_link')
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        # END_SUB_TUTORIAL

        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        # Gripper initialization
        self._gripper = Robotiq2f85()

    def go_to_joint_state(self, joint_goal_pos=[]):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        # BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        # Planning to a Joint Goal
        # ^^^^^^^^^^^^^^^^^^^^^^^^
        # The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        # thing we want to do is move it to a slightly better configuration.
        # We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = joint_goal_pos[0]
        joint_goal[1] = joint_goal_pos[1]
        joint_goal[2] = joint_goal_pos[2]
        joint_goal[3] = joint_goal_pos[3]
        joint_goal[4] = joint_goal_pos[4]
        joint_goal[5] = joint_goal_pos[5]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        # END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self, position=[], orientation=[], gripper_pos=-1, ):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        # BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        # Planning to a Pose Goal
        # ^^^^^^^^^^^^^^^^^^^^^^^
        # We can plan a motion for this group to a desired pose for the
        # end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = orientation[0]
        pose_goal.orientation.y = orientation[1]
        pose_goal.orientation.z = orientation[2]
        pose_goal.orientation.w = orientation[3]
        pose_goal.position.x = position[0]
        pose_goal.position.y = position[1]
        pose_goal.position.z = position[2]

        move_group.set_pose_target(pose_goal)

        # Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        self._gripper.send_command(
            command='s', position=gripper_pos, force=100, speed=255)
        success = move_group.go(wait=True)
        rospy.loginfo(f"Move group result {success}")
        if success:
            # Calling `stop()` ensures that there is no residual movement
            move_group.stop()
            # It is always good to clear your targets after planning with poses.
            # Note: there is no equivalent function for clear_joint_value_targets().
            move_group.clear_pose_targets()

            # END_SUB_TUTORIAL

            # For testing:
            # Note that since this section of code will not be included in the tutorials
            # we use the class variable rather than the copied state variable
            # current_pose = self.move_group.get_current_pose().pose
            # rospy.loginfo(f"{self.move_group.get_current_pose()}")
            # return all_close(pose_goal, current_pose, 0.1)
        return success


if __name__ == '__main__':

    myargv = rospy.myargv(argv=sys.argv)

    import debugpy
    debugpy.listen(('0.0.0.0', 5678))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()

    movegroup_interface = MoveGroupPythonInterfaceTutorial()
    # rospy.init_node("rollout_node", anonymous=True)

    file_path = myargv[1]
    rospy.loginfo(f"Reading file {file_path}")
    file_paths = [file_path]
    file_paths = glob.glob(os.path.join(file_path, "traj015.pkl"))

    for trj_path in file_paths:
        rospy.loginfo(f"trj_path")
        with open(trj_path, "rb") as f:
            sample = pkl.load(f)

        cv2.namedWindow("Camera view", cv2.WINDOW_NORMAL)

        trj = sample['traj']
        # print(trj.get(0)['obs'].keys())

        # cv2.imshow(f'Frame {0}', trj[0]['obs'].get('camera_front_image'))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        camera_quat = ENV_OBJECTS['camera_orientation']["camera_front"]
        r_aruco_camera = quat2mat(
            np.array(camera_quat))
        p_aruco_camera = ENV_OBJECTS['camera_pos']["camera_front"]

        camera_quat = ENV_OBJECTS['camera_orientation']["camera_front"]
        r_camera_aruco = quat2mat(
            np.array(camera_quat)).T
        p_camera_aruco = -np.matmul(r_camera_aruco, np.array(
            [ENV_OBJECTS['camera_pos']["camera_front"]]).T)
        T_camera_aruco = np.append(
            r_camera_aruco, p_camera_aruco, axis=1)

        for t in range(len(trj)):
            pos = trj.get(t)['obs']['eef_pos']
            quat = trj.get(t)['obs']['eef_quat']
            ee_aa = trj.get(t)['obs']['ee_aa']
            gripper_pos = trj.get(t)['obs']['gripper_qpos']

            img = np.array(trj.get(
                t)['obs'][f'camera_front_image'])

            # convert gripper_pos to pixel
            gripper_pos_bl = np.array(
                [trj[t]['action'][:3]]).T
            gripper_quat_bl = np.array(
                trj[t]['action'][3:-1])
            gripper_aa_bl = _quat2axisangle(gripper_quat_bl)
            rospy.loginfo(
                f"Gripper position {pos} - Gripper orientation {ee_aa} ")
            rospy.loginfo(
                f"Gripper action pos {gripper_pos_bl} - Gripper action orientation {gripper_aa_bl} ")

            # cv2.imshow(f'Frame {t}', trj[t]['obs'].get('camera_front_image'))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            gripper_rot_bl = quat2mat(
                np.array(gripper_quat_bl))
            T_gripper_bl = np.concatenate(
                (gripper_rot_bl, gripper_pos_bl), axis=1)
            T_gripper_bl = np.concatenate(
                (T_gripper_bl, np.array([[0, 0, 0, 1]])), axis=0)

            print(f"TCP_bl\n{gripper_pos_bl}")
            TCP_aruco = T_aruco_bl @ T_gripper_bl
            print(f"TCP_aruco:\n{TCP_aruco}")
            print(f"T_camera_aruco\n{T_camera_aruco}")
            tcp_camera = np.array(
                [(T_camera_aruco @ TCP_aruco)[:3, -1]]).T
            tcp_camera = tcp_camera/tcp_camera[2][0]
            print(f"TCP camera:\n{tcp_camera}")
            tcp_pixel_cord = np.array(
                camera_intrinsic @ tcp_camera, dtype=np.uint32)
            print(f"Pixel coordinates\n{tcp_pixel_cord}")

            # plot point
            img = cv2.circle(
                img, (tcp_pixel_cord[0][0], tcp_pixel_cord[1][0]), radius=1, color=(255, 0, 0), thickness=-1)
            cv2.imshow("Camera view", img)
            cv2.waitKey(500)
            if t == 0:
                home_pos = pos
                home_quat = quat
                home_gripper_pos = gripper_pos

            rospy.loginfo(f"Position {pos} - Gripper pos {gripper_pos}")

            success = movegroup_interface.go_to_pose_goal(
                position=pos, orientation=quat, gripper_pos=gripper_pos)
            if success:
                rospy.loginfo("Next waypoint")
                rospy.sleep(3)
                # input("Press enter to go to next waypoint")
            else:
                rospy.logerr("Failed to move")
                rospy.logerr("Error during motion")
                exit(-1)

        rospy.loginfo("Rollout completed")
        pos[2] = pos[2] + 0.10
        success = movegroup_interface.go_to_pose_goal(
            position=pos, orientation=quat, gripper_pos=gripper_pos)
        input("Press enter to go to home position")
        success = movegroup_interface.go_to_pose_goal(
            position=home_pos, orientation=home_quat, gripper_pos=home_gripper_pos)
    exit(1)
