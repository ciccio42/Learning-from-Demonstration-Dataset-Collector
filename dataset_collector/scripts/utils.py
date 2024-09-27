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
                    'bin': [0.6, 0.06, 0.15],
                    'bin_0': [0.15, 0.06, 0.15],
                    'bin_1': [0.15, 0.06, 0.15],
                    'bin_2': [0.15, 0.06, 0.15],
                    'bin_3': [0.15, 0.06, 0.15]},

        "id_to_obj": {0: "greenbox",
                      1: "yellowbox",
                      2: "bluebox",
                      3: "redbox",
                      4: "bin_0",
                      5: "bin_1",
                      6: "bin_2",
                      7: "bin_3"}
    },

    'nut_assembly': {
        'obj_names': ['nut0', 'nut1', 'nut2'],
        'ranges': [[0.10, 0.31], [-0.10, 0.10], [-0.31, -0.10]]
    },

    'camera_names': {'camera_front', 'camera_lateral_right', 'camera_lateral_left'},

    'camera_fovx': 345.27,
    'camera_fovy': 345.27,

    # w.r.t table_1 frame
    'camera_pos': {'camera_front': [0.022843813138628592,
                                    -0.43800020977692405,
                                    0.5643843146648674],
                   'camera_lateral_right': [-0.3582777207605626,
                                            -0.44377700364575223,
                                            0.561009214792732],
                   'camera_lateral_left': [-0.32693157973832665,
                                           0.4625646268626449,
                                           0.5675614538972504]},

    'camera_orientation': {'camera_front': [0.3603325062389276,
                                            0.015749675284185274,
                                            -0.0008269422755895826,
                                            0.9326905965230317],
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
            if len(obs[key].shape) == 3:
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


def overwrite_pkl_file(pkl_file_path, sample, traj_obj_bb, ):
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


def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def mat2quat(rmat):
    """
    Converts given rotation matrix to quaternion.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (x,y,z,w) float quaternion angles
    """
    M = np.asarray(rmat).astype(np.float32)[:3, :3]

    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array(
        [
            [m00 - m11 - m22, np.float32(0.0), np.float32(0.0), np.float32(0.0)],
            [m01 + m10, m11 - m00 - m22, np.float32(0.0), np.float32(0.0)],
            [m02 + m20, m12 + m21, m22 - m00 - m11, np.float32(0.0)],
            [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22],
        ]
    )
    K /= 3.0
    # quaternion is Eigen vector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    inds = np.array([3, 0, 1, 2])
    q1 = V[inds, np.argmax(w)]
    if q1[0] < 0.0:
        np.negative(q1, q1)
    inds = np.array([1, 2, 3, 0])
    return q1[inds]