from typing import Union, List, Tuple

import cv2
import numpy
import shapely

from Frame import Frame
from pyslamd.utils.pose import get_pose


def get_overlap(frame: Frame, reference: Frame) -> Union[shapely.Polygon, None]:
    """
    :param frame_one: first frame
    :param frame_two: second frame
    :return: overlap polygon if overlap exists, None otherwise
    """
    corners = [
        (0, 0),
        (frame.settings.camera.width, 0),
        (frame.settings.camera.width, frame.settings.camera.height),
        (0, frame.settings.camera.height)
    ]

    extrinsic = get_pose(
        frame.get_imu_rotation(reference),
        frame.get_gps_translation(reference)
    )

    # TODO: if these are compuated with global reference, then they can be cached
    # which reduces runtime
    frame_corners = numpy.array([
        (extrinsic @ numpy.append(frame.image_to_world_point(*corner), 1))[:2]  # project to flat plane
        for corner in corners
    ])

    reference_corners = numpy.array([
        reference.image_to_world_point(*corner)[:2]  # project to flat plane
        for corner in corners
    ])

    frame_footprint = shapely.Polygon(frame_corners)
    reference_footprint = shapely.Polygon(reference_corners)

    overlap = shapely.intersection(frame_footprint, reference_footprint)

    if overlap.is_empty:
        return None

    return overlap


def get_overlap_masks(
    frame_world_points: List[Tuple[float, float, float]],
    frame: Frame,
    reference_world_points: List[Tuple[float, float, float]],
    reference: Frame,
    overlap: shapely.Polygon
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    TODO: This is pretty unreadable
    TODO: This is also very slow

    :param frame_world_points: 
    :param frame: 
    :param reference_world_points: 
    :param reference: 
    :param overlap: 
    """
    # do gps_imu pose translation
    extrinsic = get_pose(
        frame.get_imu_rotation(reference),
        frame.get_gps_translation(reference)
    )

    # create frame mask
    frame_mask = []
    for world_point in frame_world_points:
        # move to reference frame
        referenced_point = extrinsic @ numpy.append(world_point, 1)

        # project to flat plane
        point = shapely.Point(referenced_point[:2])

        # check in overlap
        frame_mask.append(overlap.contains(point))

    # create keyframe (reference) mask
    reference_mask = []
    for world_point in reference_world_points:
        # project to flat plane
        point = shapely.Point(world_point[:2])

        # check in overlap
        reference_mask.append(overlap.contains(point))

    return numpy.array(frame_mask), numpy.array(reference_mask)
