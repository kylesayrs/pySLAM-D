from typing import Tuple, List

import cv2
import numpy

from pyslamd.Settings import MatcherSettings, KeypointSettings
from pyslamd.Frame import Frame


def make_keypoint_detector(settings: KeypointSettings) -> cv2.Feature2D:
    return cv2.ORB_create(
        nfeatures=settings.num_features,
        scaleFactor=settings.scale_factor,
        nlevels=settings.num_levels,
        fastThreshold=settings.fast_threshold
    )


def make_keypoint_matcher(settings: MatcherSettings) -> cv2.DescriptorMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def blocks(image_shape: Tuple[int, int], num_rows: int, num_columns: int):
    ys = numpy.linspace(0, image_shape[0], num=num_rows + 1)
    xs = numpy.linspace(0, image_shape[1], num=num_columns + 1)

    for y_start, y_end in zip(ys[:-1], ys[1:]):
        for x_start, x_end in zip(xs[:-1], xs[1:]):
            yield int(y_start), int(y_end), int(x_start), int(x_end)


def get_world_keypoints(frame: Frame) -> List[Tuple[float, float, float]]:
    return [
        frame.image_to_world_point(*keypoint.pt)
        for keypoint in frame.keypoints
    ]


def get_matched_points(
    query_points: List[cv2.KeyPoint],
    train_points: List[cv2.KeyPoint],
    matches: List[cv2.DMatch]
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    query_points = numpy.array([
        query_points[match.queryIdx]
        for match in matches
    ])
    
    train_points = numpy.array([
        train_points[match.trainIdx]
        for match in matches
    ])

    return query_points, train_points