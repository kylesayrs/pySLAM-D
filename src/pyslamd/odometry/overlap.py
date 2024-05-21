from typing import Union, List, Tuple

import cv2
import numpy
import shapely

from Frame import Frame
from pyslamd.utils.pose import get_pose


def get_overlap(frame: Frame, key_frame: Frame, origin_frame: Frame) -> Union[shapely.Polygon, None]:
    """
    :param frame_one: first frame
    :param frame_two: second frame
    :return: overlap polygon if overlap exists, None otherwise
    """
    frame_corners = frame.get_global_footprint(origin_frame)
    key_frame_corners = key_frame.get_global_footprint(origin_frame)

    frame_footprint = shapely.Polygon(frame_corners)
    key_frame_footprint = shapely.Polygon(key_frame_corners)

    overlap = shapely.intersection(frame_footprint, key_frame_footprint)

    if overlap.is_empty:
        return None

    return overlap


def get_overlap_masks(
    frame_world_points: List[Tuple[float, float, float]],
    frame: Frame,
    key_frame_world_points: List[Tuple[float, float, float]],
    key_frame: Frame,
    overlap: shapely.Polygon,
    origin_frame: Frame
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    TODO: This is pretty unreadable
    TODO: This is also very slow

    :param frame_world_points: TODO
    :param frame: TODO
    :param key_frame_world_points: TODO
    :param key_frame: TODO
    :param overlap: TODO
    :param origin_frame: TODO
    """
    frame_mask = [
        overlap.contains(
            shapely.Point(
                frame.world_to_global_point(world_point, origin_frame)
            )
        )
        for world_point in frame_world_points
    ]

    key_frame_mask = [
        overlap.contains(
            shapely.Point(
                key_frame.world_to_global_point(world_point, origin_frame)
            )
        )
        for world_point in key_frame_world_points
    ]

    return numpy.array(frame_mask), numpy.array(key_frame_mask)
