from typing import Union, List, Tuple

import cv2
import numpy

from Frame import Frame


def get_world_overlap(frame_one: Frame, frame_two: Frame) -> Union[numpy.ndarray, None]:
    """
    Special algorithm

    :param frame_one: first frame
    :param frame_two: second frame
    :return: array of overlap vertices if overlap exists, None otherwise
    """
    corners_one = frame_one.get_world_corners()
    corners_two = frame_two.get_world_corners()

    vertices = []
    for corner in corners_one:
        if _convex_polygon_contains(corners_two, corner):
            vertices.append(corner)

    for corner in corners_two:
        if _convex_polygon_contains(corners_one, corner):
            vertices.append(corner)

    if len(vertices) < 3:
        return None

    return vertices


def filter_by_overlap(
    keypoints: List[cv2.KeyPoint],
    descriptors,
    overlap: numpy.ndarray
) -> Tuple[List[cv2.KeyPoint]]:
    raise NotImplementedError()



def _convex_polygon_contains(vertices: List[Tuple[float, float]], point: Tuple[float, float]) -> True:
    pass