from typing import Tuple, Union, List

import os
import cv2
import numpy
import pymap3d

from pyslamd.Frame import Frame
from pyslamd.PoseOptimizer import PoseOptimizerTeaser
from pyslamd.Settings import Settings
from pyslamd.helpers import (
    make_keypoint_detector,
    make_keypoint_matcher,
    get_translation,
    get_rotation
)


RelativePosesType = Union[List[numpy.ndarray], None]
OutliersType = Union[List["cv2.DMatch"], None]


class FrameMatcher():
    """
    Class which implements the odometry frame matching process

    :param settings: settings which define keypoint and visualization settings
    """
    def __init__(self, settings: Settings):
        self.settings = settings

        self.keypoint_matcher = make_keypoint_matcher(settings)
    

    def match_frames(self, frame: Frame, key_frames: List[Frame]) -> Tuple[List[RelativePosesType], List[OutliersType]]:
        relative_poses = []
        outliers = []

        if self.settings.matcher.overlap is not None and self.settings.matcher.gps_match_bound is not None:
            raise ValueError()

        for key_frame in key_frames:
            # TODO: Use georeference to find which frames overlap at all
            distance = numpy.linalg.norm(pymap3d.geodetic2enu(*key_frame.gps_coords, *frame.gps_coords))
            if (
                (
                    self.settings.matcher.gps_match_bound is not None and
                    distance < self.settings.matcher.gps_match_bound
                ) or 
                (
                    self.settings.matcher.overlap is not None and
                    key_frame.key_frame_num == len(key_frames) - 1
                )
            ):
                _relative_poses, _outliers = self._match_frame(frame, key_frame)
            else:
                _relative_poses, _outliers = None, None

            relative_poses.append(_relative_poses)
            outliers.append(_outliers)

        return relative_poses, outliers


    def _match_frame(self, frame: Frame, key_frame: Frame) -> Tuple[RelativePosesType, OutliersType]:
        settings = self.settings.matcher

        # TODO: filter descriptors to only include keypoints in the overlap
        #       make sure to preserve order so they can be collated with positions

        # get matches (query, train)
        matches = self.keypoint_matcher.match(frame.descriptors, key_frame.descriptors)
        total_possible_matches = min(len(frame.descriptors), len(key_frame.descriptors))

        # print number of matches
        print(
            f"frame {frame.frame_num} -> "
            f"key_frame {key_frame.frame_num} "
            f"| {len(matches):04d} / "
            f"{total_possible_matches} "
            "matches | ",
            end=""
        )

        # only use top k matches
        if settings.top_k_matches is not None:
            matches = sorted(matches, key=lambda x:x.distance)
            matches = matches[:settings.top_k_matches]

        # check for minimum number of matches
        if len(matches) < settings.min_num_matches:
            print("failure")
            return None, None

        # debug
        if settings.debug_matches:
            _draw_matches(frame, key_frame, matches)

        # transform to georeferenced points
        key_frame_points = numpy.array([
            key_frame.georeference_point(*key_frame.keypoints[match.trainIdx].pt)
            for match in matches
        ])

        frame_points = numpy.array([
            frame.georeference_point(*frame.keypoints[match.queryIdx].pt)
            for match in matches
        ])

        # use optimizer to find pose transform from key_frame to frame
        optimizer = PoseOptimizerTeaser()
        relative_pose = optimizer.solve(key_frame_points.T, frame_points.T)

        # check optimization
        if relative_pose is None:
            print("optimization failure")
            return None, None

        # check translation threshold
        total_translation = numpy.linalg.norm(get_translation(relative_pose))
        print(f"translation {int(total_translation):04d} | ", end="")
        if total_translation > settings.max_translation:
            print("translation failure")
            return None, None

        # TODO: print rotation in degrees and add a threshold

        # TODO: compute outliers
        outliers = matches

        print("success")
        return relative_pose, outliers


    def reproject_outliers(self, frame: Frame, key_frame_outliers: List[Union[List["cv2.DMatch"], None]]):
        for key_frame, outliers in zip(self.key_frames, key_frame_outliers):
            if outliers is not None:
                raise NotImplementedError()


def _draw_matches(frame: Frame, key_frame: Frame, matches: List[cv2.DMatch]):
    """
    Debug function used to visualize keypoint matching
    """
    frame_image = cv2.imread(frame.image_path)
    key_frame_image = cv2.imread(key_frame.image_path)

    matches_image = cv2.drawMatches(
        frame_image,
        frame.keypoints,
        key_frame_image,
        key_frame.keypoints,
        matches,
        None
    )

    file_name = f"{frame.frame_num}_{key_frame.frame_num}.png"
    file_path = os.path.join("matches", file_name)
    cv2.imwrite(file_path, matches_image)
