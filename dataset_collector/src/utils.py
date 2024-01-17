import numpy as np
import math
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
