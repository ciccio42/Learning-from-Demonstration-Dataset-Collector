import cv2
import pickle
import numpy as np
import copy
import logging
import math

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger("BB-Creator")

PI = np.pi
EPS = np.finfo(float).eps * 4.0


ENV_OBJECTS = {
    'pick_place': {
        'obj_names': ['greenbox', 'yellowbox', 'bluebox', 'redbox', 'bin'],
        'bin_position': [0.18, 0.00, 0.75],
        'obj_dim': {'greenbox': [0.05, 0.055, 0.045],  # W, H, D
                    'yellowbox': [0.05, 0.055, 0.045],
                    'bluebox': [0.05, 0.055, 0.045],
                    'redbox': [0.05, 0.055, 0.045],
                    'bin': [0.6, 0.06, 0.15]},

        "id_to_obj": {0: "greenbox",
                      1: "yellowbox",
                      2: "bluebox",
                      3: "redbox"}
    },

    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    },

    'camera_names': {'camera_front', 'camera_lateral_right', 'camera_lateral_left'},

    'camera_fovx': 345.27,
    'camera_fovy': 345.27,

    'camera_pos': {'camera_front': [-0.002826249197217832,
                                    0.45380661695322316,
                                    0.5322894621129393],
                   'camera_lateral_right': [-0.3582777207605626,
                                            -0.44377700364575223,
                                            0.561009214792732],
                   'camera_lateral_left': [-0.32693157973832665,
                                           0.4625646268626449,
                                           0.5675614538972504]},

    'camera_orientation': {'camera_front': [-0.00171609,
                                            0.93633855,
                                            -0.35105349,
                                            0.00535055],
                           'camera_lateral_right': [0.8623839571785069,
                                                    -0.3396500629838305,
                                                    0.12759260213488172,
                                                    -0.3530607214016715],
                           'camera_lateral_left': [-0.305029713753832,
                                                   0.884334094984367,
                                                   -0.33268049448458464,
                                                   0.11930536771213586]}
}

OFFSET = 0.0  # CM


def _compress_obs(obs):
    for key in obs.keys():
        if 'image' in key:
            okay, im_string = cv2.imencode('.jpg', obs[key])
            assert okay, "image encoding failed!"
            obs[key] = im_string
        if 'depth_norm' in key:
            assert len(
                obs[key].shape) == 2 and obs[key].dtype == np.uint8, "assumes uint8 greyscale depth image!"
            depth_im = np.tile(obs[key][:, :, None], (1, 1, 3))
            okay, depth_string = cv2.imencode('.jpg', depth_im)
            assert okay, "depth encoding failed!"
            obs[key] = depth_string
    return obs


def overwrite_pkl_file(pkl_file_path, sample, traj_obj_bb):
    # get trajectory from sample
    new_sample = copy.deepcopy(sample)

    traj = new_sample['traj']

    # modify trajectory observation
    for t in range(len(traj)):
        # logger.info(t)
        try:
            obs = traj.get(t)['obs']
        except:
            _img = traj._data[t][0]['camera_front_image']
            okay, im_string = cv2.imencode(
                '.jpg', _img)
            traj._data[t][0]['camera_front_image'] = im_string
            obs = traj.get(t)['obs']

        obs['obj_bb'] = traj_obj_bb[t]
        obs = _compress_obs(obs)
        traj.change_obs(t, obs)
        logger.debug(obs.keys())

    pickle.dump({
        'traj': traj,
        'len': len(traj),
        'env_type': sample['env_type'],
        'task_id': sample['task_id']}, open(pkl_file_path, 'wb'))


def convert_quat(q, to="xyzw"):
    """
    Converts quaternion from one convention to another.
    The convention to convert TO is specified as an optional argument.
    If to == 'xyzw', then the input is in 'wxyz' format, and vice-versa.

    Args:
        q (np.array): a 4-dim array corresponding to a quaternion
        to (str): either 'xyzw' or 'wxyz', determining which convention to convert to.
    """
    if to == "xyzw":
        return q[[1, 2, 3, 0]]
    if to == "wxyz":
        return q[[3, 0, 1, 2]]
    raise Exception(
        "convert_quat: choose a valid `to` argument (xyzw or wxyz)")


def quat2mat(quaternion):
    """
    Converts given quaternion to matrix.

    Args:
        quaternion (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: 3x3 rotation matrix
    """
    # awkward semantics for use with numba
    inds = np.array([3, 0, 1, 2])
    q = np.asarray(quaternion).copy().astype(np.float32)[inds]

    n = np.dot(q, q)
    if n < EPS:
        return np.identity(3)
    q *= math.sqrt(2.0 / n)
    q2 = np.outer(q, q)
    return np.array(
        [
            [1.0 - q2[2, 2] - q2[3, 3], q2[1, 2] - q2[3, 0], q2[1, 3] + q2[2, 0]],
            [q2[1, 2] + q2[3, 0], 1.0 - q2[1, 1] - q2[3, 3], q2[2, 3] - q2[1, 0]],
            [q2[1, 3] - q2[2, 0], q2[2, 3] + q2[1, 0], 1.0 - q2[1, 1] - q2[2, 2]],
        ]
    )
