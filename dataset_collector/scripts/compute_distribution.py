import pickle
import os
import cv2
import debugpy
import logging
from PIL import Image
import numpy as np
import random
import glob
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import resized_crop
import matplotlib.pyplot as plt
from collections import OrderedDict
from robosuite.utils.transform_utils import quat2axisangle
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("Log")
logger.setLevel(logging.INFO)

MAX_DIST = 10
MIN_DIST = 0.01

# object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
object_to_id = {"greenbox": 0, "yellowbox": 1, "bluebox": 2, "redbox": 3}

# x, y, z, e_x, e_y, e_z
NORM_RANGES = np.array([[-0.35,  0.35],
                        [-0.35,  0.35],
                        [0.60,  1.20],
                        [-3.14,  3.14911766],
                        [-3.14911766, 3.14911766],
                        [-3.14911766,  3.14911766]])

ACTION_DISTRIBUTION = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
}

max_len_trj = 0


def normalize_action(action, n_action_bin, action_ranges, continous=False):
    half_action_bin = int(n_action_bin/2)
    norm_action = action.copy()
    # normalize between [-1 , 1]
    if action.shape[0] == 7:
        norm_action[:-1] = (2 * (norm_action[:-1] - action_ranges[:, 0]) /
                            (action_ranges[:, 1] - action_ranges[:, 0])) - 1

    else:
        norm_action = (2 * (norm_action - action_ranges[:, 0]) /
                       (action_ranges[:, 1] - action_ranges[:, 0])) - 1
    if continous:
        return norm_action
    else:
        # action discretization
        return (norm_action * half_action_bin).astype(np.int32).astype(np.float32) / half_action_bin


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', default="/", help="Path to task")
    parser.add_argument("--debug", action='store_true',
                        help="whether or not attach the debugger")
    args = parser.parse_args()

    camera_name = "image"

    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        logger.info("Waiting for debugger attach")
        debugpy.wait_for_client()

    logger.info(f"Task path: {args.task_path}")
    i = 0
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    task_name = args.task_path.split('/')[-1]
    task_paths = glob.glob(os.path.join(args.task_path, 'task_*'))

    task_distribution = OrderedDict()
    for task_var, dir in enumerate(sorted(task_paths)):

        task_distribution[task_var] = OrderedDict()
        print(dir)

        if os.path.isdir(os.path.join(args.task_path, dir)):
            # assert len(trjs) == 100, print(f"{os.path.join(args.task_path, dir)} does not have 100 trjs")
            trj_paths = glob.glob(os.path.join(dir, 'traj*.pkl'))
            norm_action = []
            for j, trj in enumerate(sorted(trj_paths)):

                task_distribution[task_var][j] = list()
                print(trj)

                with open(trj, "rb") as f:
                    sample = pickle.load(f)
                    logger.debug(f"Sample keys {sample.keys()}")
                    logger.debug(sample)
                    # print(f"Sample command: {sample['command']}")
                    if i == 0:
                        i += 1
                        logger.debug(sample)
                        logger.debug(
                            f"Observation keys: {sample['traj'][0]['obs'].keys()}")
                        logger.debug(
                            f"{sample['traj'][0]['obs']['ee_aa']}")

                    # take the Trajectory obj from the trajectory
                    trajectory_obj = sample['traj']
                    i = 0
                    obj_in_hand = 0
                    start_moving = 0
                    end_moving = 0
                    for t in range(len(trajectory_obj)):
                        if t > 0:
                            logger.debug(f"Time-step {t}")
                            try:
                                action_t = trajectory_obj.get(t)[
                                    'action']
                                task_distribution[task_var][j].append(
                                    normalize_action(action=action_t,
                                                     n_action_bin=256,
                                                     action_ranges=NORM_RANGES))
                                norm_action.append(normalize_action(action=action_t,
                                                                    n_action_bin=256,
                                                                    action_ranges=NORM_RANGES))
                                # for dim, action_label in enumerate(action_t):
                                #     ACTION_DISTRIBUTION[dim].append(
                                #         action_label)
                                # print(f"Norm action{action_t[:3]}")
                                # print(
                                #     f"Denorm action {denormalize_action(action_t)[:3]}")
                            except KeyError:
                                print("error")

    # Compute mean and std-deviation
    norm_action_matrix = np.array(norm_action).reshape(len(norm_action), 7)
    print(
        f"Standard deviation {np.std(np.array(norm_action_matrix), axis=0)}")
    print(f"Standard mean {np.mean(np.array(norm_action_matrix), axis=0)}")
    for variation in task_distribution.keys():
        print(variation)
        # Plot y-axis trajectories for each variation
        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(
            16, 9))
        fig.tight_layout(pad=5.0)
        for trj in task_distribution[variation].keys():
            trajectory = np.array(
                [action_t[:3] for action_t in task_distribution[variation][trj]])

            t = np.array([i for i in range(len(trajectory))])
            ax1.plot(t, trajectory[:, 0], color='b', alpha=0.5)
            ax1.set_title(f"Trajectory distribution along x axis")
            ax1.set_xlabel("Timestamp t")
            ax1.set_ylabel("scaled x value")

            ax2.plot(t, trajectory[:, 1], color='b', alpha=0.5)
            ax2.set_title(f"Trajectory distribution along y axis")
            ax2.set_xlabel("Timestamp t")
            ax2.set_ylabel("scaled y value")

            ax3.plot(t, trajectory[:, 2], color='b', alpha=0.5)
            ax3.set_title(f"Trajectory distribution along z axis")
            ax3.set_xlabel("Timestamp t")
            ax3.set_ylabel("scaled z value")

        plt.savefig(f"{task_name}_variation_{variation}.png")

    # An "interface" to matplotlib.axes.Axes.hist() method
    # for dim, key in enumerate(ACTION_DISTRIBUTION.keys()):
    #     n, bins, patches = plt.hist(x=ACTION_DISTRIBUTION[key], bins=256, color='#0504aa',
    #                                 alpha=0.7, rwidth=0.85)
    #     plt.grid(axis='y', alpha=0.75)
    #     plt.xlabel('Value')
    #     plt.ylabel('Frequency')
    #     plt.title('Action Distribution')
    #     maxfreq = n.max()
    #     # Set a clean upper y-axis limit.
    #     plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq %
    #              10 else maxfreq + 10)
    #     plt.savefig(f"dim_{dim}.png")
    #     plt.clf()
