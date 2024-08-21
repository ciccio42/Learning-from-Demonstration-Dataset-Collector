#!/bin/python3
import pickle as pkl

import sys
import cv2
import argparse
from utils import *
import logging
import glob
import os
import debugpy

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")


PKL_FILE_PATH = "/media/ciccio/Sandisk/real-world-dataset/pick_place/task_00/traj000.pkl"
NUM_OBJ = 4

object_loc = []
cnt = 0
press = False

# T_aruco_table @ T_table_bl
# T_aruco_table = np.array([[-1.0, 0.0, 0.0, 0.0],
#                           [0.0, -1.0, 0.0, 0.00],
#                           [0.0, 0.0, 1.0, 0.0],
#                           [0, 0, 0, 1]])
# T_aruco_bl = T_aruco_table  @ np.array([[-1, 0.0, 0, 0.01],
#                                         [0.0, -1.0, 0, 0.612],
#                                         [0, 0, 1, 0.120],
#                                         [0, 0, 0, 1]])
T_table_bl = np.array([[-1, 0.0, 0, 0.01],
                        [0.0, -1.0, 0, 0.612],
                        [0, 0, 1, 0.120],
                        [0, 0, 0, 1]])

camera_intrinsic = np.array([[345.2712097167969, 0.0, 337.5007629394531],
                             [0.0, 345.2712097167969,
                              179.0137176513672],
                             [0, 0, 1]])

film_px_offset = np.array([[337.5007629394531],
                           [179.0137176513672]])

OBJ_DEPTH = np.zeros(NUM_OBJ)


def mouse_drawing(event, x, y, flags, params):
    global object_loc, cnt, press
    if event == cv2.EVENT_LBUTTONDOWN:
        object_loc.append([x, y])
        cnt += 1
        press = True


def check_pick(step):
    logger.debug("Checking picking condition")
    if step['action'][-1] != 0:
        return True
    else:
        return False


def check_init(step):
    if step['obs'].get("obj_bb", None) is None or len(step['obs']['obj_bb']) == 0:
        return True
    else:
        return False


def plot_bb(img, obj_bb, show_image=False):

    # draw bb
    indx = 0
    for obj_name in obj_bb.keys():
        logger.debug(
            f"Obj name {obj_name} - BB ({obj_bb[obj_name]['center']},{obj_bb[obj_name]['upper_left_corner']},{obj_bb[obj_name]['bottom_right_corner']})")
        center = obj_bb[obj_name]['center']
        upper_left_corner = obj_bb[obj_name]['upper_left_corner']
        bottom_right_corner = obj_bb[obj_name]['bottom_right_corner']
        img = cv2.circle(
            img, center, radius=1, color=(0, 0, 255), thickness=-1)
        img = cv2.rectangle(
            img, upper_left_corner,
            bottom_right_corner, (255, 0, 0), 1)
        indx += 1
    assert indx == NUM_OBJ, "Number of bounding box must be equal to number of objects"
    if show_image:
        cv2.imwrite("test_bb.png", img)
        cv2.imshow("Test", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--task_path", type=str, default="//media/ciccio/Sandisk/real-world-dataset/only_frontal/reduced_space/pick_place/task_01")
    argParser.add_argument(
        "--debug", action='store_true')

    args = argParser.parse_args()

    if args.debug:
        import debugpy
        print("Wait for debbugger")
        debugpy.listen(5678)
        debugpy.wait_for_client()

    task_path = args.task_path
    task_name = args.task_path.split('/')[-2]
    logger.info(f"Task name {task_name}")

    files_path = glob.glob(os.path.join(args.task_path, "*.pkl"))
    for indx, file_path in enumerate(files_path):
        if True:  # indx == 0:
            logger.info(f"Reading file {file_path}")
            object_loc = []
            with open(file_path, "rb") as f:
                sample = pkl.load(f)

            trj = sample['traj']
            # trj._data.pop(0)
            obj_bb_novel = dict()
            sample['env_type']
            sample['task_id']
            traj_bb = []
            # for t in range(len(trj)):
            #     if t == 0:
            #         img_t = trj[t]['obs'].get('camera_front_image')
            #         cv2.imshow(f'Frame {t}', img_t)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()
            # print(trj.get(0)['obs'].keys())
            
            # traj = sample['traj']
            # traj._data.pop(0)
            # pickle.dump({
            #     'traj': traj,
            #     'len': len(traj),
            #     'env_type': sample['env_type'],
            #     'task_id': sample['task_id']}, open(file_path, 'wb'))
            
            for t in range(len(trj)):
                
                OBJ_PICKED = False
                logger.debug(f"Timestamp {t}")
                # ENV_OBJECTS['camera_names']:
                for camera_name in ['camera_front']:
                    # 1. Compute rotation_camera_to_world ()
                    camera_quat = ENV_OBJECTS['camera_orientation'][camera_name]
                    r_table_camera = quat2mat(
                        np.array(camera_quat))
                    p_table_camera = ENV_OBJECTS['camera_pos'][camera_name]
                    
                    camera_quat = ENV_OBJECTS['camera_orientation'][camera_name]
                    r_camera_table = quat2mat(
                        np.array(camera_quat)).T
                    p_camera_table = -np.matmul(r_camera_table, np.array(
                        [ENV_OBJECTS['camera_pos'][camera_name]]).T)
                    T_camera_table = np.append(
                        r_camera_table, p_camera_table, axis=1)

                    obj_bb_novel[camera_name] = dict()

                    # define target object-id
                    if t == 0:
                        if "pick_place" in task_path:
                            try:
                                task_id = int(file_path.split(
                                    '/')[-2].split('_')[-1].lstrip('0'))
                            except:
                                task_id = 0
                            target_obj_id = int(task_id/4)
                            logger.info(f"Target object id {target_obj_id}")

                    object_name_list = ENV_OBJECTS[task_name]['obj_names']

                    # Iterate over trajectory
                    traj = sample['traj']
                    print(traj.get(t)['action'][-1])
                    # for each object in the observation get the position
                    obj_positions = dict()
                    if task_name == 'pick_place':
                        try:
                            obs = traj.get(t)['obs']
                        except:
                            _img = traj._data[t][0]['camera_front_image']
                            okay, im_string = cv2.imencode(
                                '.jpg', _img)
                            traj._data[t][0]['camera_front_image'] = im_string
                            obs = traj.get(t)['obs']

                    logger.debug(obs.keys())
                    # get object position
                    for obj_name in object_name_list:
                        if obj_name != 'bin':
                            # obj_positions[obj_name] = obs[f"{obj_name}_pos"]
                            if t > 0 and obj_name == ENV_OBJECTS[task_name]["id_to_obj"][target_obj_id]:
                                # check if the robot has pick the object
                                if check_pick(trj[t]):
                                    logger.debug("Object picked")
                                    OBJ_PICKED = True
                                    # convert gripper_pos to pixel
                                    gripper_pos_bl = np.array(
                                        [trj[t]['action'][:3]]).T
                                    
                                    if '00' in task_path or '08' in task_path:
                                        gripper_pos_bl[0] = gripper_pos_bl[0] - 0.02
                                    if '01' in task_path or '04' in task_path or '05' in task_path:
                                        gripper_pos_bl[0] = gripper_pos_bl[0] + 0.02
                                    
                                    
                                    gripper_quat_bl = np.array(
                                        trj[t]['action'][3:-1])
                                    gripper_rot_bl = quat2mat(
                                        np.array(gripper_quat_bl))
                                    T_gripper_bl = np.concatenate(
                                        (gripper_rot_bl, gripper_pos_bl), axis=1)
                                    T_gripper_bl = np.concatenate(
                                        (T_gripper_bl, np.array([[0, 0, 0, 1]])), axis=0)

                                    logger.debug(f"TCP_bl\n{gripper_pos_bl}")
                                    TCP_table = T_table_bl @ T_gripper_bl
                                    logger.debug(f"TCP_table:\n{TCP_table}")
                                    logger.debug(
                                        f"T_camera_table\n{T_camera_table}")
                                    tcp_camera = np.array(
                                        [(T_camera_table @ TCP_table)[:3, -1]]).T
                                    tcp_camera_scaled = tcp_camera / \
                                        tcp_camera[2][0]
                                    tcp_camera_scaled[0][0] = - tcp_camera_scaled[0][0]
                                    logger.debug(
                                        f"TCP camera:\n{tcp_camera_scaled}")
                                    tcp_pixel_cord = np.array(
                                        camera_intrinsic @ tcp_camera_scaled, dtype=np.uint32)
                                    logger.debug(
                                        f"Pixel coordinates\n{tcp_pixel_cord}")
                                    # convert to pixel
                                    object_loc[target_obj_id][0] = tcp_pixel_cord[0][0]
                                    object_loc[target_obj_id][1] = tcp_pixel_cord[1][0]
                                    # image = cv2.circle(
                                    #     obs['camera_front_image'], (tcp_pixel_cord[0][0], tcp_pixel_cord[1][0]), 3, (0, 0, 255), 3)
                                    # cv2.imshow(f'Frame {t}', image)
                                    # cv2.waitKey(500)
                                    # cv2.destroyAllWindows()

                                else:
                                    logger.debug("Object not picked yet")

                            elif t == 0 and check_init(trj[t]):
                                if obj_name == ENV_OBJECTS[task_name]["id_to_obj"][target_obj_id]:
                                    logger.info(
                                        f"Get position for target object {obj_name}")
                                    img_t = trj[t]['obs'].get(
                                        'camera_front_image')
                                else:
                                    logger.info(
                                        f"Get position for object {obj_name}")
                                    img_t = trj[t]['obs'].get(
                                        'camera_front_image')
                                cv2.imshow(f'Frame {t}', img_t)
                                cv2.setMouseCallback(
                                    f'Frame {t}', mouse_drawing)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()

                            elif t == 0 and not check_init(trj[t]):
                                logger.debug("Initialization not needed")
                                object_loc_dict = trj[t]['obs'].get(
                                    'obj_bb').get('camera_front')

                                object_loc.append(
                                    object_loc_dict[obj_name]['center'])

                        else:
                            # obj_positions[obj_name] = ENV_OBJECTS[task_name]['bin_position']
                            pass

                    cnt = 0
                    logger.debug(
                        f"Starting to compute bounding-boxes for timestamp {t}")

                    # for each object create bb
                    for obj_id in ENV_OBJECTS[task_name]["id_to_obj"].keys():
                        obj_name = ENV_OBJECTS[task_name]["id_to_obj"][obj_id]
                        # get depth of object
                        if t == 0:
                            OBJ_DEPTH[obj_id] = obs[f"{camera_name}_depth"][object_loc[obj_id][1],
                                                                            object_loc[obj_id][0]]
                        if t != 0 and OBJ_PICKED:
                            OBJ_DEPTH[target_obj_id] = tcp_camera[2][0]

                        pixel_location = np.array(
                            [[object_loc[obj_id][0],
                              object_loc[obj_id][1],
                              1]]).T

                        # Convert pixels into camera coordinates
                        continuos_pixel = (np.linalg.inv(
                            camera_intrinsic) @ pixel_location)*OBJ_DEPTH[obj_id]

                        obj_bb_novel[camera_name][obj_name] = dict()

                        logger.debug(f"\nObject: {obj_name}")
                        # convert obj pos in camera coordinate
                        obj_pos = np.array(
                            [continuos_pixel[0], continuos_pixel[1], continuos_pixel[2]])

                        logger.debug(f"Object {obj_name} - Position {obj_pos}")
                        # cv2.imshow(f'Frame {t}', img_t)
                        # cv2.waitKey(0)

                        # 2. Define the object position with respect to the world
                        T_table_camera = np.concatenate(
                            (r_table_camera, np.array([p_table_camera]).T), axis=1)
                        T_table_camera = np.concatenate(
                            (T_table_camera, np.array([[0, 0, 0, 1]])), axis=0)
                        # logger.debug(T_camera_world)
                        p_camera_object = np.expand_dims(
                            np.insert(obj_pos, 3, 1), 0).T
                        p_table_object = (T_table_camera @ p_camera_object)
                        logger.debug(
                            f"\nP_table_object:\n{p_table_object} - \nP_camera_object:\n {p_camera_object}")

                        p_x_center = object_loc[obj_id][0]
                        p_y_center = object_loc[obj_id][1]
                        logger.debug(
                            f"\nImage coordinate: px {p_x_center}, py {p_y_center}")

                        p_x_corner_list = []
                        p_y_corner_list = []
                        # 3.1 create a box around the object
                        for i in range(8):
                            if i == 0:  # upper-left front corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 1:  # upper-right front corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 2:  # bottom-left front corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])
                            elif i == 3:  # bottom-right front corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])
                            elif i == 4:  # upper-left back corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[-ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 5:  # upper-right back corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[-ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2+OFFSET],
                                            [0]])
                            elif i == 6:  # bottom-left back corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[-ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2-OFFSET],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])
                            elif i == 7:  # bottom-right back corner
                                p_table_object_corner = p_table_object + \
                                    np.array(
                                        [[-ENV_OBJECTS[task_name]
                                          ['obj_dim'][obj_name][2]/2],
                                            [ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][0]/2+OFFSET],
                                            [-ENV_OBJECTS[task_name]
                                         ['obj_dim'][obj_name][1]/2-OFFSET],
                                            [0]])

                            p_camera_object_corner = T_camera_table @ p_table_object_corner
                            logger.debug(
                                f"\nP_table_object_upper_left:\n{p_table_object_corner} -   \nP_camera_object_upper_left:\n {p_camera_object_corner}")

                            # 3.1 Upper-left corner and bottom right corner in pixel coordinate
                            p_camera_object_corner = p_camera_object_corner / \
                                p_camera_object_corner[2][0]

                            corner_pixel_cords = np.array(
                                camera_intrinsic @ p_camera_object_corner, dtype=np.uint32)

                            p_x_corner = corner_pixel_cords[0][0]

                            p_y_corner = corner_pixel_cords[1][0]

                            logger.debug(
                                f"\nImage coordinate upper_left corner: px {p_x_corner}, py {p_y_corner}")

                            p_x_corner_list.append(p_x_corner)
                            p_y_corner_list.append(p_y_corner)

                        x_min = min(p_x_corner_list)
                        y_min = min(p_y_corner_list)
                        x_max = max(p_x_corner_list)
                        y_max = max(p_y_corner_list)
                        # save bb
                        obj_bb_novel[camera_name][obj_name]['center'] = [
                            p_x_center, p_y_center]
                        obj_bb_novel[camera_name][obj_name]['upper_left_corner'] = [
                            x_max, y_max]
                        obj_bb_novel[camera_name][obj_name]['bottom_right_corner'] = [
                            x_min, y_min]
                        if obj_name == 'bin':
                            print(obj_bb_novel)
                    # draw center
                    if camera_name == 'camera_front':
                        img = np.array(traj.get(
                            t)['obs'][f'{camera_name}_image'])

                        # convert gripper_pos to pixel
                        gripper_pos_bl = np.array(
                            [trj[t]['action'][:3]]).T
                        if '00' in task_path or '08' in task_path:
                            gripper_pos_bl[0] = gripper_pos_bl[0] - 0.02
                        if '01' in task_path or '04' in task_path or '05' in task_path:
                            gripper_pos_bl[0] = gripper_pos_bl[0] + 0.02
                        gripper_quat_bl = np.array(
                            trj[t]['action'][3:-1])
                        gripper_rot_bl = quat2mat(
                            np.array(gripper_quat_bl))
                        T_gripper_bl = np.concatenate(
                            (gripper_rot_bl, gripper_pos_bl), axis=1)
                        T_gripper_bl = np.concatenate(
                            (T_gripper_bl, np.array([[0, 0, 0, 1]])), axis=0)

                        logger.debug(f"TCP_bl\n{gripper_pos_bl}")
                        TCP_table = T_table_bl @ T_gripper_bl
                        logger.debug(f"TCP_table:\n{TCP_table}")
                        logger.debug(f"T_camera_table\n{T_camera_table}")
                        tcp_camera = np.array(
                            [(T_camera_table @ TCP_table)[:3, -1]]).T
                        tcp_camera = tcp_camera/tcp_camera[2][0]
                        tcp_camera[0][0] = -tcp_camera[0][0]
                        logger.debug(f"TCP camera:\n{tcp_camera}")
                        tcp_pixel_cord = np.array(
                            camera_intrinsic @ tcp_camera, dtype=np.uint32)
                        logger.debug(f"Pixel coordinates\n{tcp_pixel_cord}")

                        #plot point
                        img = cv2.circle(
                            img, (tcp_pixel_cord[0][0], tcp_pixel_cord[1][0]), radius=1, color=(255, 0, 0), thickness=-1)

                        plot_bb(img=img,
                                obj_bb=obj_bb_novel[camera_name],
                                show_image= False)#t == 0 or OBJ_PICKED)

                    # print(obj_bb)
                    traj_bb.append(copy.deepcopy(obj_bb_novel))

            overwrite_pkl_file(pkl_file_path=file_path,
                               sample=sample,
                               traj_obj_bb=traj_bb)
