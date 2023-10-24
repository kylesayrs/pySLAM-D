import numpy


def get_translation(pose: numpy.ndarray) -> numpy.ndarray:
    return pose[0:3, 3]


def set_translation(pose: numpy.ndarray, translation: numpy.ndarray) -> numpy.ndarray:
    pose[0:3, 3] = translation
    return pose


def set_rotation(pose: numpy.ndarray, rotation: numpy.ndarray) -> numpy.ndarray:
    pose[0:3, 0:3] = rotation
    return pose


def get_rotation(pose: numpy.ndarray) -> numpy.ndarray:
    return pose[0:3, 0:3]


def get_pose(rotation: numpy.ndarray, translation: numpy.ndarray) -> numpy.ndarray:
    pose = numpy.eye(4)
    pose[0:3, 0:3] = rotation
    pose[0:3, 3] = translation

    return pose


def orientation_to_rotation(orientation: numpy.ndarray) -> numpy.ndarray:
    """
    Use 3-2-1 YRP standard used by Arducopter

    :param orientation: _description_
    :return: _description_
    """
    yaw = numpy.radians(orientation[0])
    roll = numpy.radians(orientation[1])
    pitch = numpy.radians(orientation[2])

    x_rotation = numpy.array([
        [1, 0, 0],
        [0, numpy.cos(pitch), -1 * numpy.sin(pitch)],
        [0, numpy.sin(pitch), numpy.cos(pitch)]
    ])

    y_rotation = numpy.array([
        [numpy.cos(roll), 0, numpy.sin(roll)],
        [0, 1, 0],
        [-1 * numpy.sin(roll), 0, numpy.cos(roll)]
    ])

    z_rotation = numpy.array([
        [numpy.cos(yaw), -1 * numpy.sin(yaw), 0],
        [numpy.sin(yaw), numpy.cos(yaw), 0],
        [0, 0, 1]
    ])

    return x_rotation @ y_rotation @ z_rotation
