import os
import pickle
import cv2
import re
import glob
from PIL import Image
import numpy as np
# img=cv2.imread('/home/ciccio/Pictures/conf_1_v3.png')
# cv2.imshow('Window',img)
# cv2.destroyAllWindows()
import torch
from torchvision.transforms import Normalize
import json
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms.functional import resized_crop
import multi_task_il
from torchvision.ops import box_iou
from multi_task_il.datasets.savers import Trajectory

STATISTICS_CNTRS = {'reach_correct_obj': 0,
                    'reach_wrong_obj': 0,
                    'pick_correct_obj': 0,
                    'pick_wrong_obj': 0,
                    'pick_correct_obj_correct_place': 0,
                    'pick_correct_obj_wrong_place': 0,
                    'pick_wrong_obj_correct_place': 0,
                    'pick_wrong_obj_wrong_place': 0,
                    }


def find_number(name):
    # return int(re.search(r"\d+", name).group())
    # regex = r'(\d+)_(\d+)'
    regex = r'(\d+)'
    res = re.search(regex, name)
    return res.group()


def sort_key(file_name):
    # Extract the number X from the file name using a regular expression
    pkl_name = file_name.split('/')[-1].split('.')[0]
    match = find_number(pkl_name)
    if match:
        return match
    else:
        return 0  # Return 0 if the file name doesn't contain a number


def sample_command(context):
    demo_t = 4
    selected_frames = list()
    for i in range(demo_t):
        # get first frame
        if i == 0:
            n = 1
        # get the last frame
        elif i == demo_t - 1:
            n = len(context) - 1
        elif i == 1:
            obj_in_hand = 0
            # get the first frame with obj_in_hand and the gripper is closed
            for t in range(1, len(context)):
                state = context.get(t)['info']['status']
                trj_t = context.get(t)
                gripper_act = trj_t['action'][-1]
                if state == 'obj_in_hand' and gripper_act == 1:
                    obj_in_hand = t
                    n = t
                    break
        elif i == 2:
            # get the middle moving frame
            start_moving = 0
            end_moving = 0
            for t in range(obj_in_hand, len(context)):
                state = context.get(t)['info']['status']
                if state == 'moving' and start_moving == 0:
                    start_moving = t
                elif state != 'moving' and start_moving != 0 and end_moving == 0:
                    end_moving = t
                    break
            n = start_moving + int((end_moving-start_moving)/2)
        selected_frames.append(n)

    if isinstance(context, (list, tuple)):
        return [context[i] for i in selected_frames]
    elif isinstance(context, Trajectory):
        return [context[i]['obs'][f"camera_front_image"] for i in selected_frames]


def create_video_for_each_trj(base_path="/", task_name="pick_place"):
    from omegaconf import DictConfig, OmegaConf

    adjust = False if "Real" in base_path else True
    flip_channels = False if "Real" in base_path else True

    trj_files = glob.glob(os.path.join(base_path, "traj*.pkl"))

    trj_files.sort(key=sort_key)
    print(trj_files)

    try:
        print("Creating folder {}".format(
            os.path.join(base_path, "video")))
        video_path = os.path.join(base_path, "video")
        os.makedirs(video_path)
    except:
        pass

    if len(trj_files) != 0:
        for traj_file in trj_files:
            print(f"Open {traj_file}")
            with open(traj_file, "rb") as f:
                traj_data = pickle.load(f)['traj']

            traj_frames = dict()
            traj_frames_depth = dict()
            bb_frames = []
            for t, step in enumerate(traj_data):
                # "camera_lateral_left", "camera_lateral_right", "eye_in_hand"
                for camera, camera_name in enumerate(['camera_front']):
                    if t == 0:
                        traj_frames[camera_name] = []
                        traj_frames_depth[camera_name] = []

                    if step["obs"].get(f'{camera_name}_image_full_size') is not None:
                        img = cv2.imdecode(step["obs"][f'{camera_name}_image_full_size'], cv2.IMREAD_COLOR)
                    else:
                        img = step["obs"][f'{camera_name}_image']
                    # depth = np.array((np.nan_to_num(
                    #     step["obs"][f'{camera_name}_depth'], 2.0)/2.0)*256, dtype=np.uint8)

                    if len(img.shape) != 3:
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

                    traj_frames[camera_name].append(img)
                    # traj_frames_depth[camera_name].append(depth)

            #  "camera_lateral_left", "camera_lateral_right", "eye_in_hand"
            for camera, camera_name in enumerate(['camera_front']):
                out = None
                traj_height, traj_width, _ = traj_frames[camera_name][0].shape

                trj_number = find_number(
                    traj_file.split('/')[-1].split('.')[0])

                if len(traj_data) >= 3:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    output_width = traj_width
                    output_height = traj_height
                    # out_path = f"{task_name}_step_{step}_demo_{context_number}_traj_{trj_number}.mp4"
                    out_path = f"traj_{trj_number}_{camera_name}.mp4"
                    print(f"Out-Path {out_path}")
                    out = cv2.VideoWriter(os.path.join(
                        video_path, out_path), fourcc, 30, (output_width, output_height))

                    # out_path_depth = f"traj_{trj_number}_{camera_name}_depth.mp4"
                    # print(f"Out-Path {out_path_depth}")
                    # out_depth = cv2.VideoWriter(os.path.join(
                    #     video_path, out_path_depth), fourcc, 30, (output_width, output_height))

                else:
                    out_path = os.path.join(
                        video_path, f"traj_{trj_number}.png")

                # create the string to put on each frame
                for i, traj_frame in enumerate(traj_frames[camera_name]):
                    # and len(bb_frames) >= i+1:
                    # if len(bb_frames) != 0 and i > 0 and len(bb_frames) >= i+1:

                    #     bb = bb_frames[i -
                    #                    1]['camera_front'][0].cpu().numpy()

                    # traj_frame = np.array(cv2.rectangle(
                    #     traj_frame,
                    #     (int(bb[0]),
                    #         int(bb[1])),
                    #     (int(bb[2]),
                    #         int(bb[3])),
                    #     (0, 0, 255), 1))

                    cv2.imwrite("frame.png", traj_frame)
                    if out is not None:
                        out.write(traj_frame)
                        # out_depth.write(cv2.cvtColor(
                        #     traj_frames_depth[camera_name][i], cv2.COLOR_GRAY2BGR))
                    else:
                        cv2.imwrite(out_path, traj_frame)
                if out is not None:
                    out.release()
                    # out_depth.release()


if __name__ == '__main__':
    import argparse
    import debugpy
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/",
                        help="Path to checkpoint folder")
    parser.add_argument('--task', type=str,
                        default="pick_place", help="Task name")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        debugpy.listen(('0.0.0.0', 5678))
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
    # 1. create video
    create_video_for_each_trj(
        base_path=args.base_path, task_name=args.task)
