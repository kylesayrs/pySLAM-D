from typing import List, Tuple, Any

import os
import cv2
import sys
import parse
import numpy
import contextlib

from pyslamd.Settings import Settings, CameraSettings


def get_image_paths(image_dir: str) -> List[str]:
    file_names = [
        file_name
        for file_name in os.listdir(image_dir)
        if os.path.splitext(file_name)[1].lower() == ".jpg"
    ]

    file_numbers = [
        parse.parse("IMG_{}.JPG", file_name)[0]
        for file_name in file_names
    ]

    file_names_sorted = list(zip(
        *sorted(
            zip(file_names, file_numbers),
            key=lambda pair: pair[0]
        )
    ))[0]

    file_paths = [
        os.path.join(image_dir, file_name)
        for file_name in file_names_sorted
    ]

    return file_paths


def make_keypoint_detector(settings: Settings) -> cv2.Feature2D:
    return cv2.ORB_create(
        nfeatures=settings.keypoints.num_features,
        scaleFactor=settings.keypoints.scale_factor,
        nlevels=settings.keypoints.num_levels,
        fastThreshold=settings.keypoints.fast_threshold
    )


def make_keypoint_matcher(settings: Settings) -> cv2.DescriptorMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def get_translation(pose: numpy.ndarray) -> numpy.ndarray:
    return pose[0:3, 3]


def get_rotation(pose: numpy.ndarray) -> numpy.ndarray:
    return pose[0:3, 0:3]


def get_pose(rotation: numpy.ndarray, translation: numpy.ndarray) -> numpy.ndarray:
    pose = numpy.eye(4)
    pose[0:3, 0:3] = rotation
    pose[0:3, 3] = translation

    return pose


def blocks(image_shape: Tuple[int, int], num_rows: int, num_columns: int):
    ys = numpy.linspace(0, image_shape[0], num=num_rows + 1)
    xs = numpy.linspace(0, image_shape[1], num=num_columns + 1)

    for y_start, y_end in zip(ys[:-1], ys[1:]):
        for x_start, x_end in zip(xs[:-1], xs[1:]):
            yield int(y_start), int(y_end), int(x_start), int(x_end)

def all_none(list: List[Any]):
    return all(value is None for value in list)